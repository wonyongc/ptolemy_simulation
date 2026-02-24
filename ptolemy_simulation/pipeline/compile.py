"""Compilation pipeline for study variants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ptolemy_simulation.adapters.registry import AdapterRegistry
from ptolemy_simulation.pipeline.models import CompileArtifact, RunVariant
from ptolemy_simulation.utils.hash import stable_sha256


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def compile_variants(
    registry: AdapterRegistry,
    study_name: str,
    output_root: Path,
    variants: List[RunVariant],
) -> List[CompileArtifact]:
    study_root = output_root / study_name
    runs_root = study_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    artifacts: List[CompileArtifact] = []
    manifest_rows: List[Dict[str, Any]] = []

    for variant in variants:
        run_root = runs_root / variant.run_id
        config_root = run_root / "effective_config"
        config_root.mkdir(parents=True, exist_ok=True)

        detector_path = config_root / "detector.json"
        sim_path = config_root / "sim.json"
        run_path = config_root / "run.json"
        detector_path.write_text(json.dumps(variant.detector, indent=2))
        sim_path.write_text(json.dumps(variant.sim, indent=2))
        run_path.write_text(json.dumps(variant.run, indent=2))

        hashes = {
            "detector_sha256": stable_sha256(variant.detector),
            "sim_sha256": stable_sha256(variant.sim),
            "run_sha256": stable_sha256(variant.run),
        }
        _write_json(config_root / "hashes.json", hashes)

        adapter = registry.get(variant.simulator)
        artifact = adapter.compile(variant, run_root / "artifacts")
        artifacts.append(artifact)

        run_manifest = {
            "index": variant.index,
            "run_id": variant.run_id,
            "simulator": variant.simulator,
            "variables": variant.variables,
            "hashes": hashes,
            "artifact_files": {k: str(v) for k, v in artifact.files.items()},
        }
        _write_json(run_root / "run_manifest.json", run_manifest)
        manifest_rows.append(run_manifest)

    _write_json(
        study_root / "study_manifest.json",
        {
            "study_name": study_name,
            "run_count": len(variants),
            "runs": manifest_rows,
        },
    )

    jsonl = "\n".join(json.dumps(row) for row in manifest_rows)
    (study_root / "study_manifest.jsonl").write_text(jsonl + ("\n" if jsonl else ""))

    return artifacts
