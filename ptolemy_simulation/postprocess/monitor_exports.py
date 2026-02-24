"""Parse monitor export text outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_transmission(path: Path) -> np.ndarray:
    """Return columns [z, unique_count, avg_energy]."""
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def load_pitch(path: Path) -> np.ndarray:
    """Return columns [z, pid, x, y, px, py, pz, e]."""
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data
