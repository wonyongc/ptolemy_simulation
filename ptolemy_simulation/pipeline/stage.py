"""Staging and submission for local and Della targets."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ptolemy_simulation.pipeline.models import CompileArtifact, RunVariant

TEMPLATE_DIR_NAME = "PTOLEMY_CST_TEMPLATE"
TEMPLATE_CST_NAME = "PTOLEMY_CST_TEMPLATE.cst"
TEMPLATE_BUILD_MACRO_PATH = Path("Model/3D/PTOLEMY_build.mcs")
TEMPLATE_RUN_MACRO_PATH = Path("Model/3D/PTOLEMY_run.mcr")


class StageError(RuntimeError):
    """Raised for stage failures."""


def _artifact_macro_paths(artifact: CompileArtifact) -> Tuple[Optional[Path], Optional[Path]]:
    build_macro = artifact.files.get("macro_build") or artifact.files.get("macro")
    run_macro = artifact.files.get("macro_run") or artifact.files.get("macro")
    return build_macro, run_macro


def _unpack_template(zip_path: Path, unpack_dir: Path) -> Path:
    if not zip_path.is_file():
        raise StageError(f"template zip not found: {zip_path}")
    unpack_dir.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(dir=unpack_dir))
    shutil.unpack_archive(str(zip_path), str(tmpdir))
    if not (tmpdir / TEMPLATE_DIR_NAME).is_dir() or not (tmpdir / TEMPLATE_CST_NAME).is_file():
        raise StageError("template archive missing expected CST template contents")
    return tmpdir


def stage_local(
    study_name: str,
    run_cfg: Dict[str, object],
    variants: List[RunVariant],
    artifacts: List[CompileArtifact],
) -> Dict[str, object]:
    base_dir = Path(str(run_cfg.get("__config_dir", ".")))

    project_root_raw = Path(str(run_cfg["project_root"]))
    project_root = (
        (project_root_raw if project_root_raw.is_absolute() else base_dir / project_root_raw).resolve() / study_name
    )

    unpack_dir_raw = Path(str(run_cfg["unpack_dir"]))
    unpack_dir = (unpack_dir_raw if unpack_dir_raw.is_absolute() else (base_dir / unpack_dir_raw)).resolve()

    template_zip_raw = Path(str(run_cfg["template_zip"]))
    template_zip = (template_zip_raw if template_zip_raw.is_absolute() else (base_dir / template_zip_raw)).resolve()

    project_root.mkdir(parents=True, exist_ok=True)
    staged: List[Dict[str, str]] = []

    by_run: Dict[str, CompileArtifact] = {artifact.run_id: artifact for artifact in artifacts}

    for variant in variants:
        artifact = by_run[variant.run_id]
        build_macro_path, run_macro_path = _artifact_macro_paths(artifact)
        if not build_macro_path:
            continue
        if run_macro_path is None:
            run_macro_path = build_macro_path

        run_dir = project_root / variant.run_id
        cst_path = project_root / f"{variant.run_id}.cst"
        if not run_dir.exists() or not cst_path.exists():
            tmpdir = _unpack_template(template_zip, unpack_dir)
            shutil.move(str(tmpdir / TEMPLATE_DIR_NAME), run_dir)
            shutil.move(str(tmpdir / TEMPLATE_CST_NAME), cst_path)

        build_macro_target = run_dir / TEMPLATE_BUILD_MACRO_PATH
        run_macro_target = run_dir / TEMPLATE_RUN_MACRO_PATH
        build_macro_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(build_macro_path, build_macro_target)
        shutil.copy2(run_macro_path, run_macro_target)

        staged.append(
            {
                "run_id": variant.run_id,
                "project_dir": str(run_dir),
                "project_cst": str(cst_path),
                "build_macro": str(build_macro_target),
                "run_macro": str(run_macro_target),
            }
        )

    stage_manifest = {
        "mode": "local",
        "study_name": study_name,
        "stage_only": True,
        "runs": staged,
    }
    manifest_path = Path("generated") / "studies" / study_name / "local_stage_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(stage_manifest, indent=2))
    return {"manifest": str(manifest_path), "runs": staged}


def _runs_table(
    variants: Iterable[RunVariant], artifacts: Dict[str, CompileArtifact]
) -> List[Tuple[int, str, str, str]]:
    rows: List[Tuple[int, str, str, str]] = []
    for variant in variants:
        build_macro, run_macro = _artifact_macro_paths(artifacts[variant.run_id])
        if build_macro is None:
            continue
        if run_macro is None:
            run_macro = build_macro
        rows.append((variant.index, variant.run_id, Path(build_macro).name, Path(run_macro).name))
    return rows


def _render_remote_scripts(
    study_name: str, run_cfg: Dict[str, object], rows: List[Tuple[int, str, str, str]]
) -> Dict[str, str]:
    remote_root = Path(str(run_cfg["project_root"])) / study_name
    template_zip = str(run_cfg["template_zip"])
    cst_bin = str(run_cfg["cst_bin"])
    slurm = run_cfg.get("slurm", {})
    submit = bool(run_cfg.get("submit", True))

    prep = f"""#!/bin/bash
set -euo pipefail
study_root=\"{remote_root}\"
runs_file=\"$study_root/runs.tsv\"
template_zip=\"{template_zip}\"
mkdir -p \"$study_root\"

while IFS=$'\\t' read -r idx run_id build_macro_name run_macro_name; do
  [ -z \"$run_id\" ] && continue
  run_dir=\"$study_root/$run_id\"
  run_cst=\"$study_root/$run_id.cst\"
  if [ ! -d \"$run_dir\" ] || [ ! -f \"$run_cst\" ]; then
    tmpdir=$(mktemp -d)
    unzip \"$template_zip\" -d \"$tmpdir\"
    mv \"$tmpdir/{TEMPLATE_DIR_NAME}\" \"$run_dir\"
    mv \"$tmpdir/{TEMPLATE_CST_NAME}\" \"$run_cst\"
  fi
  mkdir -p \"$run_dir/Model/3D\"
  cp \"$study_root/$build_macro_name\" \"$run_dir/{TEMPLATE_BUILD_MACRO_PATH.as_posix()}\"
  cp \"$study_root/$run_macro_name\" \"$run_dir/{TEMPLATE_RUN_MACRO_PATH.as_posix()}\"
done < \"$runs_file\"
"""

    array_max = max(0, len(rows) - 1)
    array = f"""#!/bin/bash
#SBATCH --job-name={study_name}
#SBATCH --output={remote_root}/slurm/%A_%a.out
#SBATCH --error={remote_root}/slurm/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={slurm.get('cpus', 4)}
#SBATCH --mem={slurm.get('mem', '32G')}
#SBATCH --time={slurm.get('time', '24:00:00')}
#SBATCH --mail-type=ALL
#SBATCH --mail-user={slurm.get('mail_user', '')}
#SBATCH --array=0-{array_max}

set -euo pipefail
study_root=\"{remote_root}\"
runs_file=\"$study_root/runs.tsv\"
status_dir=\"$study_root/status\"
mkdir -p \"$status_dir\" \"$study_root/slurm\"
line=$(sed -n \"$((SLURM_ARRAY_TASK_ID+1))p\" \"$runs_file\")
IFS=$'\\t' read -r idx run_id build_macro_name run_macro_name <<< \"$line\"
run_cst=\"$study_root/$run_id.cst\"
macro=\"$study_root/$run_id/{TEMPLATE_RUN_MACRO_PATH.as_posix()}\"

if \"{cst_bin}\" -r \"$macro\" \"$run_cst\"; then
  echo 0 > \"$status_dir/$run_id.rc\"
else
  echo 1 > \"$status_dir/$run_id.rc\"
fi
exit 0
"""

    summary = f"""#!/bin/bash
#SBATCH --job-name={study_name}_summary
#SBATCH --output={remote_root}/slurm/%j.summary.out
#SBATCH --error={remote_root}/slurm/%j.summary.err

set -euo pipefail
study_root=\"{remote_root}\"
runs_file=\"$study_root/runs.tsv\"
status_dir=\"$study_root/status\"
summary_file=\"$study_root/summary.json\"

success=0
failed=0
missing=0

while IFS=$'\\t' read -r idx run_id build_macro_name run_macro_name; do
  [ -z \"$run_id\" ] && continue
  rc_file=\"$status_dir/$run_id.rc\"
  if [ ! -f \"$rc_file\" ]; then
    missing=$((missing+1))
  elif [ \"$(cat \"$rc_file\")\" = \"0\" ]; then
    success=$((success+1))
  else
    failed=$((failed+1))
  fi
done < \"$runs_file\"

cat > \"$summary_file\" <<JSON
{{
  \"study\": \"{study_name}\",
  \"success\": $success,
  \"failed\": $failed,
  \"skipped\": $missing
}}
JSON

cat \"$summary_file\"
"""

    submit_script = f"""#!/bin/bash
set -euo pipefail
study_root=\"{remote_root}\"
cd \"$study_root\"
bash prep.sh
array_job=$(sbatch --parsable array.slurm)
echo \"array_job=$array_job\"
summary_job=$(sbatch --parsable --dependency=afterany:${{array_job}} summary.slurm)
echo \"summary_job=$summary_job\"
"""

    result_sync = run_cfg.get("result_sync", {}) if isinstance(run_cfg.get("result_sync", {}), dict) else {}
    patterns = result_sync.get("patterns", ["*.txt", "*.npz", "*.png"])
    include_lines = "\n".join(f"--include='{pattern}' \\" for pattern in patterns)

    fetch = f"""#!/bin/bash
set -euo pipefail
REMOTE=\"{run_cfg.get('user', os.environ.get('USER', ''))}@{run_cfg['host']}:{remote_root}\"
LOCAL=\"{result_sync.get('local_dir', f'generated/studies/{study_name}/fetched')}\"
mkdir -p \"$LOCAL\"
rsync -av --prune-empty-dirs --include='*/' \\
{include_lines}
--exclude='*' \"$REMOTE/\" \"$LOCAL/\"
"""

    return {
        "prep.sh": prep,
        "array.slurm": array,
        "summary.slurm": summary,
        "submit.sh": submit_script,
        "fetch_results.sh": fetch,
        "fetch_enabled": "true" if bool(result_sync.get("enabled", False)) else "false",
        "submit_enabled": "true" if submit else "false",
    }


def stage_della(
    study_name: str,
    run_cfg: Dict[str, object],
    variants: List[RunVariant],
    artifacts: List[CompileArtifact],
    dry_run: bool,
) -> Dict[str, object]:
    if not variants:
        raise StageError("no variants to stage")

    by_run = {artifact.run_id: artifact for artifact in artifacts}
    rows = _runs_table(variants, by_run)

    study_local_root = Path("generated") / "studies" / study_name
    della_dir = study_local_root / "della"
    della_dir.mkdir(parents=True, exist_ok=True)

    runs_tsv_path = della_dir / "runs.tsv"
    runs_tsv_path.write_text(
        "\n".join(
            f"{idx}\t{run_id}\t{build_macro_name}\t{run_macro_name}"
            for idx, run_id, build_macro_name, run_macro_name in rows
        )
        + "\n"
    )

    scripts = _render_remote_scripts(study_name, run_cfg, rows)
    script_paths: Dict[str, str] = {}
    for name, content in scripts.items():
        if name in {"submit_enabled", "fetch_enabled"}:
            continue
        path = della_dir / name
        path.write_text(content)
        path.chmod(path.stat().st_mode | 0o111)
        script_paths[name] = str(path)

    host = str(run_cfg["host"])
    user = str(run_cfg.get("user", os.environ.get("USER", "")))
    prefix = f"{user}@" if user else ""
    remote_root = f"{run_cfg['project_root']}/{study_name}"
    remote = f"{prefix}{host}:{remote_root}"

    copied = []
    if not dry_run:
        subprocess.run(["ssh", f"{prefix}{host}", f"mkdir -p {remote_root}"], check=True)

        for _, run_id, build_macro_name, run_macro_name in rows:
            build_macro_path, run_macro_path = _artifact_macro_paths(by_run[run_id])
            if build_macro_path is None:
                raise StageError(f"missing build macro artifact for run_id={run_id}")
            if run_macro_path is None:
                run_macro_path = build_macro_path

            subprocess.run(["scp", str(build_macro_path), f"{remote}/{build_macro_name}"], check=True)
            copied.append(str(build_macro_path))
            subprocess.run(["scp", str(run_macro_path), f"{remote}/{run_macro_name}"], check=True)
            copied.append(str(run_macro_path))

        subprocess.run(["scp", str(runs_tsv_path), f"{remote}/runs.tsv"], check=True)
        copied.append(str(runs_tsv_path))

        for name, path in script_paths.items():
            subprocess.run(["scp", path, f"{remote}/{name}"], check=True)
            copied.append(path)

        subprocess.run(["ssh", f"{prefix}{host}", f"bash {remote_root}/prep.sh"], check=True)
        if bool(run_cfg.get("submit", False)):
            subprocess.run(["ssh", f"{prefix}{host}", f"bash {remote_root}/submit.sh"], check=True)

    stage_manifest = {
        "mode": "della",
        "study_name": study_name,
        "remote_root": remote_root,
        "host": host,
        "submitted": bool(run_cfg.get("submit", False)) and not dry_run,
        "result_sync_enabled": bool(run_cfg.get("result_sync", {}).get("enabled", False)),
        "dry_run": dry_run,
        "copied": copied,
        "rows": [
            {"index": i, "run_id": rid, "build_macro": build_macro, "run_macro": run_macro}
            for i, rid, build_macro, run_macro in rows
        ],
        "scripts": script_paths,
    }

    manifest_path = study_local_root / "della_stage_manifest.json"
    manifest_path.write_text(json.dumps(stage_manifest, indent=2))
    return {"manifest": str(manifest_path), "rows": rows, "remote_root": remote_root}
