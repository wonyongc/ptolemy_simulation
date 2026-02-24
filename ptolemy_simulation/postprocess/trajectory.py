"""Trajectory conversion and GCS reduction."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def process_cst_traj(infile: Path, outfile: Path) -> None:
    data = np.loadtxt(infile, delimiter=",")
    (
        t_bins,
        x_bins,
        y_bins,
        z_bins,
        vx_bins,
        vy_bins,
        vz_bins,
        phi_bins,
        ex_bins,
        ey_bins,
        ez_bins,
        bx_bins,
        by_bins,
        bz_bins,
    ) = data.T
    np.savez(
        outfile,
        t_bins=t_bins,
        x_bins=x_bins,
        y_bins=y_bins,
        z_bins=z_bins,
        vx_bins=vx_bins,
        vy_bins=vy_bins,
        vz_bins=vz_bins,
        phi_bins=phi_bins,
        ex_bins=ex_bins,
        ey_bins=ey_bins,
        ez_bins=ez_bins,
        bx_bins=bx_bins,
        by_bins=by_bins,
        bz_bins=bz_bins,
    )


def process_cst_traj_no_fields(infile: Path, outfile: Path) -> None:
    data = np.loadtxt(infile, delimiter=",")
    t_bins, x_bins, y_bins, z_bins, vx_bins, vy_bins, vz_bins, phi_bins = data.T
    np.savez(
        outfile,
        t_bins=t_bins,
        x_bins=x_bins,
        y_bins=y_bins,
        z_bins=z_bins,
        vx_bins=vx_bins,
        vy_bins=vy_bins,
        vz_bins=vz_bins,
        phi_bins=phi_bins,
    )


def _kinetic_components(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mass = 9.109e-31
    c = 299_792_458
    ev = 1.60217e-19

    beta_x = vx / c
    beta_y = vy / c
    beta_z = vz / c

    gamma_x = np.divide(1, np.sqrt(1 - beta_x**2))
    gamma_y = np.divide(1, np.sqrt(1 - beta_y**2))
    gamma_z = np.divide(1, np.sqrt(1 - beta_z**2))

    ke_x = (gamma_x * mass * c**2 - mass * c**2) / ev
    ke_y = (gamma_y * mass * c**2 - mass * c**2) / ev
    ke_z = (gamma_z * mass * c**2 - mass * c**2) / ev
    return ke_x, ke_y, ke_z


def _gcs_bucket_splits(vy: np.ndarray, vz: np.ndarray, prefix: int) -> np.ndarray:
    vz_signs = np.diff(np.sign(vz[prefix:]))
    vy_signs = np.diff(np.sign(vy[prefix:]))
    signsums = np.abs(np.cumsum(vy_signs)) + np.abs(np.cumsum(vz_signs))
    return np.where(np.diff(np.sign(signsums)) == -1)[0]


def save_gcs_traj(infile: Path, outfile: Path, prefix: int = 3) -> None:
    raw = np.load(infile)

    t = raw["t_bins"]
    x = raw["x_bins"]
    y = raw["y_bins"]
    z = raw["z_bins"]
    vx = raw["vx_bins"]
    vy = raw["vy_bins"]
    vz = raw["vz_bins"]
    phi = raw["phi_bins"]

    with_fields = "ex_bins" in raw
    if with_fields:
        ex = raw["ex_bins"]
        ey = raw["ey_bins"]
        ez = raw["ez_bins"]
        bx = raw["bx_bins"]
        by = raw["by_bins"]
        bz = raw["bz_bins"]

    ke_x, ke_y, ke_z = _kinetic_components(vx, vy, vz)
    splits = _gcs_bucket_splits(vy, vz, prefix)

    if len(splits) < 2:
        np.savez(outfile, t=np.array([]), x=np.array([]), y=np.array([]), z=np.array([]), vx=np.array([]), vy=np.array([]), vz=np.array([]), phi=np.array([]), KEx=np.array([]), KEy=np.array([]), KEz=np.array([]), radius=np.array([]))
        return

    out = {
        "t": [],
        "x": [],
        "y": [],
        "z": [],
        "vx": [],
        "vy": [],
        "vz": [],
        "phi": [],
        "KEx": [],
        "KEy": [],
        "KEz": [],
        "radius": [],
    }
    if with_fields:
        out.update({"ex": [], "ey": [], "ez": [], "bx": [], "by": [], "bz": []})

    for i in range(len(splits) - 1):
        c1 = splits[i]
        c2 = splits[i + 1]

        out["t"].append(np.mean(t[c1:c2]))
        out["x"].append(np.mean(x[c1:c2]))
        out["y"].append(np.mean(y[c1:c2]))
        out["z"].append(np.mean(z[c1:c2]))
        out["vx"].append(np.mean(vx[c1:c2]))
        out["vy"].append(np.mean(vy[c1:c2]))
        out["vz"].append(np.mean(vz[c1:c2]))
        out["phi"].append(np.mean(phi[c1:c2]))
        out["KEx"].append(np.mean(ke_x[c1:c2]))
        out["KEy"].append(np.mean(ke_y[c1:c2]))
        out["KEz"].append(np.mean(ke_z[c1:c2]))
        out["radius"].append((abs(np.min(y[c1:c2]) - np.max(y[c1:c2])) + abs(np.min(z[c1:c2]) - np.max(z[c1:c2]))) / 4)

        if with_fields:
            out["ex"].append(np.mean(ex[c1:c2]))
            out["ey"].append(np.mean(ey[c1:c2]))
            out["ez"].append(np.mean(ez[c1:c2]))
            out["bx"].append(np.mean(bx[c1:c2]))
            out["by"].append(np.mean(by[c1:c2]))
            out["bz"].append(np.mean(bz[c1:c2]))

    np.savez(outfile, **{k: np.asarray(v) for k, v in out.items()})


def derive_output_path(inp: Path, output_dir: Optional[Path], suffix: str) -> Path:
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{inp.stem}{suffix}"
    return inp.with_name(f"{inp.stem}{suffix}")
