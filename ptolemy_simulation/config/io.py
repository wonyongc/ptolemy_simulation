"""Config file input/output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_mapping(path: Path) -> Dict[str, Any]:
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config extension: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError(f"Top-level config must be a mapping: {path}")
    return data
