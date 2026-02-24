"""Pipeline dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class RunVariant:
    index: int
    run_id: str
    variables: Dict[str, Any]
    simulator: str
    detector: Dict[str, Any]
    sim: Dict[str, Any]
    run: Dict[str, Any]


@dataclass
class CompileArtifact:
    run_id: str
    simulator: str
    output_dir: Path
    files: Dict[str, Path] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
