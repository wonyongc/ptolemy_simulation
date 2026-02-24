"""Load and validate top-level study bundles."""

from __future__ import annotations

from pathlib import Path

from .io import load_mapping
from .models import DetectorConfig, RunConfig, SimulatorConfig, StudyBundle, StudyConfig
from .validation import validate_detector, validate_run, validate_simulator, validate_study


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_study_bundle(study_path: Path) -> StudyBundle:
    study_path = study_path.resolve()
    raw_study = validate_study(load_mapping(study_path))

    base_dir = study_path.parent
    detector_path = _resolve(base_dir, raw_study["detector_config"])
    sim_path = _resolve(base_dir, raw_study["sim_config"])
    run_path = _resolve(base_dir, raw_study["run_config"])

    detector = DetectorConfig(path=detector_path, raw=validate_detector(load_mapping(detector_path)))
    simulator = SimulatorConfig(path=sim_path, raw=validate_simulator(load_mapping(sim_path)))
    run = RunConfig(path=run_path, raw=validate_run(load_mapping(run_path)))
    study = StudyConfig(path=study_path, raw=raw_study)

    detector.raw["__config_dir"] = str(detector_path.parent)
    simulator.raw["__config_dir"] = str(sim_path.parent)
    run.raw["__config_dir"] = str(run_path.parent)

    sim_override = raw_study.get("simulator")
    if sim_override and sim_override != simulator.raw["simulator"]:
        simulator.raw["simulator"] = sim_override

    return StudyBundle(study=study, detector=detector, simulator=simulator, run=run)
