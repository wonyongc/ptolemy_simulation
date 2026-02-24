from __future__ import annotations

import json
from pathlib import Path

import pytest

from ptolemy_simulation.config.loader import load_study_bundle
from ptolemy_simulation.config.validation import validate_detector, validate_simulator
from ptolemy_simulation.pipeline.matrix import expand_study


@pytest.fixture()
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_study(
    tmp_path: Path,
    repo_root: Path,
    matrix: dict | None = None,
    *,
    run_name_template: str | None = None,
    overrides: dict | None = None,
) -> Path:
    study = {
        "schema_version": "1.0",
        "study_name": "test_study",
        "detector_config": str((repo_root / "config/examples/detector.minimal.yaml").resolve()),
        "sim_config": str((repo_root / "config/examples/sim.cst.minimal.yaml").resolve()),
        "run_config": str((repo_root / "config/examples/run.local.yaml").resolve()),
        "output_root": str((tmp_path / "generated/studies").resolve()),
        "matrix": matrix or {},
        "run_name_template": run_name_template,
        "overrides": overrides or {"detector": {}, "sim": {}, "run": {}},
    }
    path = tmp_path / "study.yaml"
    path.write_text(json.dumps(study))
    return path


def test_load_and_validate_bundle(repo_root: Path, tmp_path: Path) -> None:
    study_path = _write_study(tmp_path, repo_root)
    bundle = load_study_bundle(study_path)
    assert bundle.study.raw["study_name"] == "test_study"
    assert bundle.simulator.raw["simulator"] == "cst"


def test_matrix_expansion_count(repo_root: Path, tmp_path: Path) -> None:
    matrix = {"theta": [10, 20], "phi": [0, 90, 180]}
    study_path = _write_study(tmp_path, repo_root, matrix=matrix)
    bundle = load_study_bundle(study_path)
    variants = expand_study(bundle)
    assert len(variants) == 6
    assert variants[0].run_id.startswith("theta-")


def test_matrix_override_and_naming(repo_root: Path, tmp_path: Path) -> None:
    matrix = {"theta_deg": [20], "phi_deg": [45]}
    study_path = _write_study(
        tmp_path,
        repo_root,
        matrix=matrix,
        run_name_template="th{theta_deg}_phi{phi_deg}",
        overrides={
            "detector": {
                "/sources/singles/0/pitches/0": "$theta_deg",
                "/sources/singles/0/phis/0": "$phi_deg",
            },
            "sim": {},
            "run": {},
        },
    )
    bundle = load_study_bundle(study_path)
    variants = expand_study(bundle)
    assert variants[0].run_id == "th20_phi45"
    assert variants[0].detector["sources"]["singles"][0]["pitches"][0] == 20
    assert variants[0].detector["sources"]["singles"][0]["phis"][0] == 45


def test_invalid_toggle_type_raises() -> None:
    payload = {
        "schema_version": "1.0",
        "simulator": "cst",
        "toggles": {"run_solver_after_build": "yes"},
    }
    with pytest.raises(Exception):
        validate_simulator(payload)


def test_duplicate_geometry_id_rejected() -> None:
    detector = {
        "schema_version": "1.0",
        "experiment": {"name": "dup"},
        "geometry": [
            {"id": "same", "type": "lngs_magnet"},
            {"id": "same", "type": "target", "z_span_m": [0, 1]},
        ],
        "sources": {"singles": [], "rings": []},
    }
    with pytest.raises(Exception):
        validate_detector(detector)
