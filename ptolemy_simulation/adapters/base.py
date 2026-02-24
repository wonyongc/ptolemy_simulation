"""Adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ptolemy_simulation.pipeline.models import CompileArtifact, RunVariant


class Adapter(ABC):
    """Simulator adapter interface."""

    name: str

    @abstractmethod
    def compile(self, variant: RunVariant, output_dir: Path) -> CompileArtifact:
        """Compile one run variant into simulator-specific artifacts."""
        raise NotImplementedError
