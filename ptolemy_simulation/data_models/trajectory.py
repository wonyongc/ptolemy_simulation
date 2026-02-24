"""Reusable trajectory and field models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def normed(b: np.ndarray) -> Tuple[np.ndarray, float]:
    bmax = float(np.max(b))
    if bmax == 0:
        return b.copy(), 0.0
    return b / bmax, bmax


def normed_flat_before_bmax(b: np.ndarray) -> Tuple[np.ndarray, float]:
    bmax_idx = int(np.argmax(b))
    bmax = float(b[bmax_idx])
    out = b.copy()
    if bmax != 0:
        out[:bmax_idx] = bmax
        out = out / bmax
    return out, bmax


def normed_to_z(z: np.ndarray, b: np.ndarray, threshold: float = -0.1) -> Tuple[np.ndarray, float]:
    idx_candidates = np.where(z > threshold)[0]
    if len(idx_candidates) == 0:
        return b.copy(), 0.0
    ref = float(b[idx_candidates[0]])
    if ref == 0:
        return b.copy(), 0.0
    return b / ref, ref


def normed_to_value(_z: np.ndarray, b: np.ndarray, maxval: float) -> Tuple[np.ndarray, float]:
    if maxval == 0:
        return b.copy(), 0.0
    return b / maxval, maxval


@dataclass(frozen=True)
class RawTrajectory:
    attrs: Dict[str, Any]
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    phi: np.ndarray
    ex: np.ndarray | None
    ey: np.ndarray | None
    ez: np.ndarray | None
    bx: np.ndarray | None
    by: np.ndarray | None
    bz: np.ndarray | None

    @classmethod
    def from_npz(cls, path: Path, attrs: Dict[str, Any] | None = None) -> "RawTrajectory":
        data = np.load(path)
        return cls(
            attrs=attrs or {},
            t=data["t_bins"],
            x=data["x_bins"],
            y=data["y_bins"],
            z=data["z_bins"],
            vx=data["vx_bins"],
            vy=data["vy_bins"],
            vz=data["vz_bins"],
            phi=data["phi_bins"],
            ex=data["ex_bins"] if "ex_bins" in data else None,
            ey=data["ey_bins"] if "ey_bins" in data else None,
            ez=data["ez_bins"] if "ez_bins" in data else None,
            bx=data["bx_bins"] if "bx_bins" in data else None,
            by=data["by_bins"] if "by_bins" in data else None,
            bz=data["bz_bins"] if "bz_bins" in data else None,
        )

    @property
    def kinetic_components_eV(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mass = 9.109e-31
        c = 299_792_458
        ev = 1.60217e-19

        beta_x = self.vx / c
        beta_y = self.vy / c
        beta_z = self.vz / c
        gamma_x = np.divide(1, np.sqrt(1 - beta_x**2))
        gamma_y = np.divide(1, np.sqrt(1 - beta_y**2))
        gamma_z = np.divide(1, np.sqrt(1 - beta_z**2))

        ke_x = (gamma_x * mass * c**2 - mass * c**2) / ev
        ke_y = (gamma_y * mass * c**2 - mass * c**2) / ev
        ke_z = (gamma_z * mass * c**2 - mass * c**2) / ev
        return ke_x, ke_y, ke_z


@dataclass(frozen=True)
class GCSTrajectory:
    attrs: Dict[str, Any]
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    phi: np.ndarray
    radius: np.ndarray
    ke_x: np.ndarray
    ke_y: np.ndarray
    ke_z: np.ndarray
    ex: np.ndarray | None = None
    ey: np.ndarray | None = None
    ez: np.ndarray | None = None
    bx: np.ndarray | None = None
    by: np.ndarray | None = None
    bz: np.ndarray | None = None

    @classmethod
    def from_npz(cls, path: Path, attrs: Dict[str, Any] | None = None) -> "GCSTrajectory":
        data = np.load(path)
        return cls(
            attrs=attrs or {},
            t=data["t"],
            x=data["x"],
            y=data["y"],
            z=data["z"],
            vx=data["vx"],
            vy=data["vy"],
            vz=data["vz"],
            phi=data["phi"],
            radius=data["radius"],
            ke_x=data["KEx"],
            ke_y=data["KEy"],
            ke_z=data["KEz"],
            ex=data["ex"] if "ex" in data else None,
            ey=data["ey"] if "ey" in data else None,
            ez=data["ez"] if "ez" in data else None,
            bx=data["bx"] if "bx" in data else None,
            by=data["by"] if "by" in data else None,
            bz=data["bz"] if "bz" in data else None,
        )


@dataclass(frozen=True)
class FieldGrid:
    attrs: Dict[str, Any]
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    phi: np.ndarray
    ex: np.ndarray
    ey: np.ndarray
    ez: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    bz: np.ndarray

    @classmethod
    def from_npz(cls, path: Path, resize: tuple[int, int, int], attrs: Dict[str, Any] | None = None) -> "FieldGrid":
        data = np.load(path)
        return cls(
            attrs=attrs or {},
            x=np.reshape(data["x"], resize),
            y=np.reshape(data["y"], resize),
            z=np.reshape(data["z"], resize),
            phi=np.reshape(data["phi"], resize),
            ex=np.reshape(data["ex"], resize),
            ey=np.reshape(data["ey"], resize),
            ez=np.reshape(data["ez"], resize),
            bx=np.reshape(data["bx"], resize),
            by=np.reshape(data["by"], resize),
            bz=np.reshape(data["bz"], resize),
        )
