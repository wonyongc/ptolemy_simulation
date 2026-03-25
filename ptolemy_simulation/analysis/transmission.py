"""Transmission analysis"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


_PATTERN = re.compile(r"th(?P<theta>-?\d+)_phi(?P<phi>-?\d+)\.txt$")


def scan_theta_phi_files(input_dir: Path) -> Dict[int, Dict[int, Path]]:
    mapping: Dict[int, Dict[int, Path]] = {}
    for path in sorted(input_dir.glob("th*_phi*.txt")):
        m = _PATTERN.match(path.name)
        if not m:
            continue
        theta = int(m.group("theta"))
        phi = int(m.group("phi"))
        mapping.setdefault(theta, {})[phi] = path
    return mapping


def aggregate_theta(mapping: Dict[int, Dict[int, Path]], particles_per_run: int = 10200) -> Dict[int, Dict[str, np.ndarray]]:
    result: Dict[int, Dict[str, np.ndarray]] = {}
    for theta, phi_map in mapping.items():
        total_counts = None
        total_energy = None
        z_axis = None
        for path in phi_map.values():
            z, counts, avg_e = np.loadtxt(path, delimiter=",").T
            if total_counts is None:
                total_counts = counts / particles_per_run * 100
                total_energy = avg_e
                z_axis = z
            else:
                min_len = min(len(total_counts), len(counts))
                total_counts = total_counts[:min_len] + counts[:min_len] / particles_per_run * 100
                total_energy = total_energy[:min_len] + avg_e[:min_len]
                z_axis = z_axis[:min_len]
        if total_counts is None:
            continue
        result[theta] = {
            "z": z_axis,
            "count_percent_sum": total_counts,
            "avg_energy": total_energy / max(1, len(phi_map)),
            "phi_count": np.array([len(phi_map)]),
        }
    return result


def aggregate_all(mapping: Dict[int, Dict[int, Path]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_counts = None
    total_energy = None
    z_axis = None
    runs = 0
    for phi_map in mapping.values():
        for path in phi_map.values():
            z, counts, avg_e = np.loadtxt(path, delimiter=",").T
            if total_counts is None:
                total_counts = counts
                total_energy = avg_e
                z_axis = z
            else:
                min_len = min(len(total_counts), len(counts))
                total_counts = total_counts[:min_len] + counts[:min_len]
                total_energy = total_energy[:min_len] + avg_e[:min_len]
                z_axis = z_axis[:min_len]
            runs += 1
    if total_counts is None:
        return np.array([]), np.array([]), np.array([])
    return z_axis, total_counts, total_energy / max(1, runs)


def plot_aggregate(
    theta_data: Dict[int, Dict[str, np.ndarray]],
    output_path: Path,
    title: str = "Transmission by theta",
) -> None:
    fig, (ax_count, ax_energy) = plt.subplots(1, 2, figsize=(12, 4))
    for theta, payload in sorted(theta_data.items()):
        z = payload["z"]
        ax_count.plot(z, payload["count_percent_sum"], label=f"theta={theta}")
        ax_energy.plot(z, payload["avg_energy"], label=f"theta={theta}")

    ax_count.set_xlabel("z")
    ax_count.set_ylabel("count sum (%)")
    ax_count.grid(True)
    ax_count.legend()

    ax_energy.set_xlabel("z")
    ax_energy.set_ylabel("avg KE (eV)")
    ax_energy.grid(True)
    ax_energy.legend()

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
