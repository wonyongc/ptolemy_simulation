"""End-to-end one-click pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ptolemy_simulation.adapters.registry import build_default_registry
from ptolemy_simulation.config.loader import load_study_bundle
from ptolemy_simulation.pipeline.compile import compile_variants
from ptolemy_simulation.pipeline.matrix import expand_study
from ptolemy_simulation.pipeline.stage import stage_della, stage_local


def run_study(study_path: Path, *, dry_run: bool = False) -> Dict[str, Any]:
    bundle = load_study_bundle(study_path)
    variants = expand_study(bundle)

    study_name = bundle.study.raw["study_name"]
    output_root = Path(str(bundle.study.raw.get("output_root", "generated/studies")))

    registry = build_default_registry()
    artifacts = compile_variants(registry, study_name=study_name, output_root=output_root, variants=variants)

    run_mode = bundle.run.raw["mode"]
    if run_mode == "local":
        stage_result = stage_local(study_name, bundle.run.raw, variants, artifacts)
    else:
        stage_result = stage_della(study_name, bundle.run.raw, variants, artifacts, dry_run=dry_run)

    return {
        "study_name": study_name,
        "run_count": len(variants),
        "mode": run_mode,
        "artifacts": [
            {
                "run_id": artifact.run_id,
                "simulator": artifact.simulator,
                "files": {k: str(v) for k, v in artifact.files.items()},
            }
            for artifact in artifacts
        ],
        "stage": stage_result,
    }
