from __future__ import annotations

import json
from pathlib import Path

from ptolemy_simulation.adapters.registry import build_default_registry
from ptolemy_simulation.config.loader import load_study_bundle
from ptolemy_simulation.pipeline.compile import compile_variants
from ptolemy_simulation.pipeline.matrix import expand_study


def _study_file(tmp_path: Path, root: Path) -> Path:
    study = {
        "schema_version": "1.0",
        "study_name": "compile_study",
        "detector_config": str((root / "config/examples/detector.minimal.yaml").resolve()),
        "sim_config": str((root / "config/examples/sim.cst.minimal.yaml").resolve()),
        "run_config": str((root / "config/examples/run.local.yaml").resolve()),
        "output_root": str((tmp_path / "out").resolve()),
        "matrix": {},
        "overrides": {"detector": {}, "sim": {}, "run": {}},
    }
    path = tmp_path / "study.yaml"
    path.write_text(json.dumps(study))
    return path


def test_compile_generates_macro_and_manifest(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    bundle = load_study_bundle(_study_file(tmp_path, root))
    variants = expand_study(bundle)

    artifacts = compile_variants(
        build_default_registry(),
        study_name=bundle.study.raw["study_name"],
        output_root=Path(bundle.study.raw["output_root"]),
        variants=variants,
    )

    assert len(artifacts) == 1
    build_macro = artifacts[0].files["macro_build"]
    run_macro = artifacts[0].files["macro_run"]
    manifest = artifacts[0].files["manifest"]
    assert build_macro.exists()
    assert run_macro.exists()
    assert manifest.exists()
    build_text = build_macro.read_text()
    run_text = run_macro.read_text()

    assert "Sub Main()" in build_text
    assert "RunSolveAndPostprocess" not in build_text
    assert "TrackingSolver.Start" not in build_text
    assert '.Name "Epoxy resin"' in build_text

    assert 'RunMacro("PTOLEMY_build")' in run_text
    assert "RunSolveAndPostprocess" in run_text
    assert "With TrackingSolver" in run_text
    assert "Function WaitForTrackingSolver() As Boolean" in run_text
    assert "TrackingSolver.GetSolverStatus" not in run_text
    assert "Dim n As Long" in run_text
    assert "Dim i As Long" in run_text
    assert "Dim avgE As Double" in run_text
    assert "ParticleTrajectoryReader.LoadTrajectoryData()" in run_text
    assert "Mesh.GetClosestPtIndex" in run_text
    assert 'combinedPath = outputDir & "\\\\trajectories.csv"' in run_text
    assert "Sub ExportFieldMaps()" in run_text
    assert "Sub ExportFieldRegion(" in run_text

    artifacts_second = compile_variants(
        build_default_registry(),
        study_name=bundle.study.raw["study_name"],
        output_root=Path(bundle.study.raw["output_root"]),
        variants=variants,
    )
    assert artifacts_second[0].files["macro_build"].read_text() == build_text
    assert artifacts_second[0].files["macro_run"].read_text() == run_text


def test_compile_respects_solver_toggle_override(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    study = {
        "schema_version": "1.0",
        "study_name": "compile_no_solver",
        "detector_config": str((root / "config/examples/detector.minimal.yaml").resolve()),
        "sim_config": str((root / "config/examples/sim.cst.minimal.yaml").resolve()),
        "run_config": str((root / "config/examples/run.local.yaml").resolve()),
        "output_root": str((tmp_path / "out").resolve()),
        "matrix": {},
        "overrides": {"detector": {}, "sim": {"/toggles/run_solver_after_build": False}, "run": {}},
    }
    study_path = tmp_path / "study_no_solver.yaml"
    study_path.write_text(json.dumps(study))

    bundle = load_study_bundle(study_path)
    variants = expand_study(bundle)
    artifacts = compile_variants(
        build_default_registry(),
        study_name=bundle.study.raw["study_name"],
        output_root=Path(bundle.study.raw["output_root"]),
        variants=variants,
    )

    run_text = artifacts[0].files["macro_run"].read_text()
    assert "Function WaitForTrackingSolver() As Boolean" not in run_text
    assert "With TrackingSolver" not in run_text
    assert 'RunMacro("PTOLEMY_build")' in run_text


def test_compile_respects_field_export_toggle_override(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    study = {
        "schema_version": "1.0",
        "study_name": "compile_field_exports",
        "detector_config": str((root / "config/examples/detector.minimal.yaml").resolve()),
        "sim_config": str((root / "config/examples/sim.cst.minimal.yaml").resolve()),
        "run_config": str((root / "config/examples/run.local.yaml").resolve()),
        "output_root": str((tmp_path / "out").resolve()),
        "matrix": {},
        "overrides": {"detector": {}, "sim": {"/toggles/export_field_maps": True}, "run": {}},
    }
    study_path = tmp_path / "study_field_exports.yaml"
    study_path.write_text(json.dumps(study))

    bundle = load_study_bundle(study_path)
    variants = expand_study(bundle)
    artifacts = compile_variants(
        build_default_registry(),
        study_name=bundle.study.raw["study_name"],
        output_root=Path(bundle.study.raw["output_root"]),
        variants=variants,
    )

    run_text = artifacts[0].files["macro_run"].read_text()
    assert "\tExportFieldMaps" in run_text
    assert "ExportFieldRegion \"B\"" in run_text
    assert "ExportFieldRegion \"E\"" in run_text
    assert "ExportFieldRegion \"POTENTIAL\"" in run_text
