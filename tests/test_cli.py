from __future__ import annotations

import json
from pathlib import Path

from ptolemy_simulation.cli import main


def _study(tmp_path: Path, root: Path) -> Path:
    payload = {
        "schema_version": "1.0",
        "study_name": "cli_study",
        "detector_config": str((root / "config/examples/detector.minimal.yaml").resolve()),
        "sim_config": str((root / "config/examples/sim.cst.minimal.yaml").resolve()),
        "run_config": str((root / "config/examples/run.local.yaml").resolve()),
        "output_root": str((tmp_path / "out").resolve()),
        "matrix": {},
        "overrides": {"detector": {}, "sim": {}, "run": {}},
    }
    path = tmp_path / "study.yaml"
    path.write_text(json.dumps(payload))
    return path


def test_cli_validate_and_compile(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    study = _study(tmp_path, root)
    assert main(["validate", "--study", str(study)]) == 0
    assert main(["compile", "--study", str(study)]) == 0
