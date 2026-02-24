from __future__ import annotations

import json
import os
from pathlib import Path

from ptolemy_simulation.pipeline.models import CompileArtifact, RunVariant
from ptolemy_simulation.pipeline.stage import stage_della


def test_stage_della_dry_run_writes_scripts(tmp_path: Path) -> None:
    study_name = "stage_test"
    variant = RunVariant(
        index=0,
        run_id="run-0000",
        variables={},
        simulator="cst",
        detector={},
        sim={"simulator": "cst"},
        run={"mode": "della"},
    )
    build_macro = tmp_path / "run-0000.build.mcs"
    run_macro = tmp_path / "run-0000.run.mcr"
    build_macro.write_text("Sub Main()\nEnd Sub\n")
    run_macro.write_text("Sub Main()\nEnd Sub\n")
    artifact = CompileArtifact(
        run_id="run-0000",
        simulator="cst",
        output_dir=tmp_path,
        files={"macro_build": build_macro, "macro_run": run_macro},
    )

    run_cfg = {
        "schema_version": "1.0",
        "mode": "della",
        "host": "example.org",
        "user": "tester",
        "project_root": "/scratch/test",
        "template_zip": "/scratch/template.zip",
        "cst_bin": "/usr/bin/cst",
        "slurm": {"cpus": 4, "mem": "32G", "time": "01:00:00", "mail_user": "a@b.com"},
        "submit": True,
    }

    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        result = stage_della(
            study_name=study_name,
            run_cfg=run_cfg,
            variants=[variant],
            artifacts=[artifact],
            dry_run=True,
        )
    finally:
        os.chdir(cwd)

    manifest = (tmp_path / result["manifest"]).resolve()
    assert manifest.exists()
    payload = json.loads(manifest.read_text())
    assert payload["mode"] == "della"
    assert (manifest.parent / "della" / "runs.tsv").exists()
    runs_tsv = (manifest.parent / "della" / "runs.tsv").read_text().strip()
    assert runs_tsv == "0\trun-0000\trun-0000.build.mcs\trun-0000.run.mcr"
    array_script = (manifest.parent / "della" / "array.slurm").read_text()
    assert "#SBATCH --array=0-0" in array_script
    assert "PTOLEMY_run.mcr" in array_script
    submit_script = (manifest.parent / "della" / "submit.sh").read_text()
    assert "afterany" in submit_script
