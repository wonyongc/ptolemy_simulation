"""Kassiopeia adapter stub output."""

from __future__ import annotations

import json
from pathlib import Path

from ptolemy_simulation.adapters.base import Adapter
from ptolemy_simulation.pipeline.models import CompileArtifact, RunVariant


class KassiopeiaAdapter(Adapter):
    """Validated stub for future Kassiopeia adapter completion."""

    name = "kassiopeia"

    def compile(self, variant: RunVariant, output_dir: Path) -> CompileArtifact:
        output_dir.mkdir(parents=True, exist_ok=True)
        xml_path = output_dir / f"{variant.run_id}.kassiopeia.xml"
        manifest_path = output_dir / "kassiopeia_manifest.json"

        xml = (
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            "<kassiopeia_stub>\n"
            f"  <run id=\"{variant.run_id}\"/>\n"
            "  <note>This is a validated adapter stub. Full Kassiopeia generation is not implemented.</note>\n"
            f"  <geometry_elements count=\"{len(variant.detector.get('geometry', []))}\"/>\n"
            "</kassiopeia_stub>\n"
        )
        xml_path.write_text(xml)

        manifest = {
            "simulator": self.name,
            "run_id": variant.run_id,
            "status": "stub",
            "warning": "Kassiopeia output is a non-physics stub placeholder.",
            "files": {"xml": str(xml_path)},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

        artifact = CompileArtifact(
            run_id=variant.run_id,
            simulator=self.name,
            output_dir=output_dir,
            files={"xml": xml_path, "manifest": manifest_path},
            metadata=manifest,
        )
        return artifact
