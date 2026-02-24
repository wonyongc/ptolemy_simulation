"""Pitch analysis plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def aggregate_pitch_files(input_files: list[Path], x_window_abs: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    by_z: dict[float, list[float]] = {}
    for path in input_files:
        z, _pid, x, _y, px, py, pz, _e = np.loadtxt(path, delimiter=",").T
        p = np.stack((px, py, pz), axis=1)
        costh = np.sum(p * np.array([1, 0, 0]), axis=1)
        pitch = np.arccos(np.clip(np.abs(costh), 0, 1)) * 180.0 / np.pi
        for zi, xi, pitchi in zip(z, x, pitch):
            if abs(xi) > x_window_abs:
                continue
            by_z.setdefault(float(zi), []).append(float(pitchi))

    if not by_z:
        return np.array([]), np.array([])

    z_sorted = np.array(sorted(by_z.keys()))
    avg_pitch = np.array([np.mean(by_z[zi]) for zi in z_sorted])
    return z_sorted, avg_pitch


def plot_pitch(z: np.ndarray, avg_pitch: np.ndarray, output: Path, title: str = "Pitch distributions") -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.scatter(z, avg_pitch, s=10)
    ax.set_xlabel("z")
    ax.set_ylabel("pitch (deg)")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
