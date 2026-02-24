"""Typed config models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class DetectorConfig:
    path: Path
    raw: Dict[str, Any]


@dataclass(frozen=True)
class SimulatorConfig:
    path: Path
    raw: Dict[str, Any]


@dataclass(frozen=True)
class RunConfig:
    path: Path
    raw: Dict[str, Any]


@dataclass(frozen=True)
class StudyConfig:
    path: Path
    raw: Dict[str, Any]


@dataclass(frozen=True)
class StudyBundle:
    study: StudyConfig
    detector: DetectorConfig
    simulator: SimulatorConfig
    run: RunConfig
