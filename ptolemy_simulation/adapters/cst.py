"""CST adapter implementation."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from jinja2 import Environment, PackageLoader

from ptolemy_simulation.adapters.base import Adapter
from ptolemy_simulation.pipeline.models import CompileArtifact, RunVariant
from ptolemy_simulation.utils.naming import sanitize_vba_identifier


def mm_from_m(val_m: float) -> int:
    return int(round(float(val_m) * 1000.0))


def m_from_mm(val_mm: int | float) -> float:
    return float(val_mm) / 1000.0


def clamp_thickness_mm(val_mm: int) -> int:
    return max(1, val_mm)


def fmt_m(mm_value: int | float) -> str:
    return f"{m_from_mm(mm_value):.6f}"


def pitch_list_from_cfg(cfg: Dict[str, Any]) -> List[float]:
    if "pitches" in cfg and cfg["pitches"] is not None:
        return [float(p) for p in cfg["pitches"]]
    if "pitch_range" in cfg:
        pr = cfg["pitch_range"]
        start = float(pr["start_deg"])
        stop = float(pr["stop_deg"])
        step = float(pr["step_deg"])
        vals: List[float] = []
        v = start
        while v <= stop + 1e-9:
            vals.append(v)
            v += step
        return vals
    return []


def phi_list_from_cfg(cfg: Dict[str, Any]) -> List[float]:
    if "phis" in cfg and cfg["phis"] is not None:
        return [float(p) for p in cfg["phis"]]
    if "phi_deg" in cfg:
        return [float(cfg["phi_deg"])]
    return []


def load_bx_profile(profile_cfg: Dict[str, Any], base_dir: Path | None = None) -> Callable[[float], float]:
    if not profile_cfg:
        return lambda _z: 1.0

    file_path = Path(profile_cfg["file"])
    if not file_path.is_absolute() and base_dir is not None:
        file_path = (base_dir / file_path).resolve()
    normalize = bool(profile_cfg["normalize"])
    try:
        data = np.loadtxt(file_path)
    except OSError:
        return lambda _z: 1.0

    z = data[:, 0]
    bx = data[:, 1]
    if normalize:
        denom = np.max(np.abs(bx))
        if denom > 0:
            bx = bx / denom

    def interp(z_query: float) -> float:
        return float(np.interp(z_query, z, bx, left=bx[0], right=bx[-1]))

    return interp


def ramp(z: float, z0: float, z1: float, v0: float, v1: float) -> float:
    if z <= z0:
        return v0
    if z >= z1:
        return v1
    span = z1 - z0 if z1 != z0 else 1e-9
    return v0 + (v1 - v0) * (z - z0) / span


def hold(z: float, z0: float, z1: float, v: float) -> float:
    return v if z0 <= z <= z1 else 0.0


EVAL_FUNCS = {
    "ramp": ramp,
    "hold": hold,
    "abs": abs,
    "min": min,
    "max": max,
}


def eval_expr(expr: Optional[str], n: int, ctx: Dict[str, Any]) -> float:
    if not expr:
        return 0.0

    env: Dict[str, Any] = {}
    env.update(EVAL_FUNCS)
    env.update(ctx)
    env["n"] = n
    return float(eval(expr, {"__builtins__": {}}, env))


def distribute_widths(total_mm: int, n: int) -> List[int]:
    if n <= 0:
        raise ValueError("channels_x must be >= 1")
    base = total_mm // n
    rem = total_mm - base * n
    return [base + (1 if i < rem else 0) for i in range(n)]


def allocate_widths_by_ratio(total_mm: int, ratios: List[float]) -> List[int]:
    if not ratios:
        return []
    weights = [float(r) for r in ratios]
    total_weight = sum(weights)
    if total_weight <= 0 or any(w <= 0 for w in weights):
        raise ValueError("channel_width_ratios must be positive")
    raw = [total_mm * w / total_weight for w in weights]
    ints = [int(v) for v in raw]
    remainder = total_mm - sum(ints)
    fracs = sorted([(i, raw[i] - ints[i]) for i in range(len(raw))], key=lambda x: x[1], reverse=True)
    for i in range(remainder):
        ints[fracs[i % len(fracs)][0]] += 1
    return ints


def build_segments(z_start_mm: int, z_end_mm: int, len_nom_mm: int, gap_nom_mm: int) -> List[Tuple[int, int]]:
    if z_end_mm <= z_start_mm:
        raise ValueError("z_end must exceed z_start")

    length = max(1, len_nom_mm)
    gap = max(0, gap_nom_mm)
    span = z_end_mm - z_start_mm
    if span <= length:
        return [(z_start_mm, span)]

    segments: List[Tuple[int, int]] = []
    cursor = z_start_mm
    while cursor + length <= z_end_mm:
        segments.append((cursor, length))
        cursor = cursor + length + gap
    if not segments:
        segments.append((z_start_mm, span))
    return segments


@dataclass(frozen=True)
class BrickSpec:
    name: str
    material: str
    x1: str
    x2: str
    y1: str
    y2: str
    z1: str
    z2: str


@dataclass(frozen=True)
class PotentialSpec:
    name: str
    value: str
    solid: str
    face: str


@dataclass
class Voltages:
    bounce_xpos: Optional[str] = None
    bounce_xneg: Optional[str] = None
    channels_mode: str = "uniform"
    pos_y: Optional[str] = None
    neg_y: Optional[str] = None
    pos_y_channels: List[Optional[str]] = field(default_factory=list)
    neg_y_channels: List[Optional[str]] = field(default_factory=list)

    @classmethod
    def from_config(cls, data: Dict[str, Any], channels_x: int) -> "Voltages":
        channels_mode = str(data["channels_mode"])
        if channels_mode not in {"uniform", "asymmetric"}:
            raise ValueError("channels_mode must be 'uniform' or 'asymmetric'")

        pos_list = data.get("pos_y_channels", [])
        neg_list = data.get("neg_y_channels", [])
        if channels_mode == "asymmetric":
            if len(pos_list) not in (0, channels_x):
                raise ValueError("pos_y_channels length mismatch")
            if len(neg_list) not in (0, channels_x):
                raise ValueError("neg_y_channels length mismatch")

        return cls(
            bounce_xpos=data["bounce_xpos"],
            bounce_xneg=data["bounce_xneg"],
            channels_mode=channels_mode,
            pos_y=data.get("pos_y"),
            neg_y=data.get("neg_y"),
            pos_y_channels=[p for p in pos_list] if pos_list else [],
            neg_y_channels=[n for n in neg_list] if neg_list else [],
        )

    def pos_expr_for_channel(self, channel: int) -> Optional[str]:
        if self.channels_mode == "asymmetric" and self.pos_y_channels:
            return self.pos_y_channels[channel]
        return self.pos_y

    def neg_expr_for_channel(self, channel: int) -> Optional[str]:
        if self.channels_mode == "asymmetric" and self.neg_y_channels:
            return self.neg_y_channels[channel]
        return self.neg_y


@dataclass
class FilterSection:
    id: str
    component: str
    n_start: int
    z_start_mm: int
    z_end_mm: int
    include_termination_backplate: bool
    y_half_mm: int
    x_half_mm: int
    electrode_len_mm: int
    electrode_gap_mm: int
    electrode_thickness_mm: int
    pcb_thickness_mm: int
    channels_x: int
    channel_gap_mm: int
    corner_gap_mm: int
    channel_width_ratios: List[float]
    voltages: Voltages

    @classmethod
    def from_geometry(cls, element: Dict[str, Any], defaults: Dict[str, Any]) -> "FilterSection":
        def resolve(key: str) -> float:
            if key in element:
                return float(element[key])
            return float(defaults[key])

        n_span = element["n_electrodes_inclusive"]
        if not isinstance(n_span, (list, tuple)) or len(n_span) != 2:
            raise ValueError("n_electrodes_inclusive must be [n_start, n_end]")
        n_start, n_end = int(n_span[0]), int(n_span[1])
        if n_end < n_start:
            raise ValueError("n_electrodes_inclusive end must be >= start")

        filter_start_z_m = float(defaults["filter_start_z_m"])
        electrode_len_m = resolve("electrode_len_m")
        electrode_gap_m = resolve("electrode_gap_m")
        unit_size_m = electrode_len_m + electrode_gap_m

        z_start_m = filter_start_z_m + n_start * unit_size_m
        z_end_m = filter_start_z_m + (n_end + 1) * unit_size_m

        channels_x = int(element["channels_x"])
        ratios_raw = element["channel_width_ratios"]
        ratios = [float(v) for v in ratios_raw] if ratios_raw else []

        return cls(
            id=str(element["id"]),
            component=str(element["component"]),
            n_start=n_start,
            z_start_mm=mm_from_m(z_start_m),
            z_end_mm=mm_from_m(z_end_m),
            include_termination_backplate=bool(element["include_termination_backplate"]),
            y_half_mm=mm_from_m(float(element["y_half_m"])),
            x_half_mm=mm_from_m(float(element["x_half_m"])),
            electrode_len_mm=mm_from_m(electrode_len_m),
            electrode_gap_mm=mm_from_m(electrode_gap_m),
            electrode_thickness_mm=mm_from_m(resolve("electrode_thickness_m")),
            pcb_thickness_mm=mm_from_m(resolve("pcb_thickness_m")),
            channels_x=channels_x,
            channel_gap_mm=mm_from_m(resolve("channel_gap_m")),
            corner_gap_mm=mm_from_m(resolve("corner_gap_m")),
            channel_width_ratios=ratios,
            voltages=Voltages.from_config(element["voltages"], channels_x),
        )


class SymbolRegistry:
    """Fail-fast symbol collision registry."""

    def __init__(self) -> None:
        self._symbols: set[str] = set()

    def add(self, symbol: str, *, kind: str) -> None:
        key = symbol.lower()
        if key in self._symbols:
            raise ValueError(f"symbol collision for {kind}: {symbol}")
        self._symbols.add(key)


class CSTAdapter(Adapter):
    """Compile detector+simulator configs into monolithic CST macro"""

    name = "cst"

    def __init__(self) -> None:
        self.template_package = "ptolemy_simulation"
        self.template_path = "templates/cst"

    def compile(self, variant: RunVariant, output_dir: Path) -> CompileArtifact:
        output_dir.mkdir(parents=True, exist_ok=True)

        ctx = self._build_context(variant)
        env = Environment(
            loader=PackageLoader(self.template_package, self.template_path),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        build_template = env.get_template("main.mcs.j2")
        run_template = env.get_template("run.mcr.j2")

        build_macro_path = output_dir / f"{variant.run_id}.build.mcs"
        build_macro_path.write_text(build_template.render(**ctx))

        run_macro_path = output_dir / f"{variant.run_id}.run.mcr"
        run_macro_path.write_text(run_template.render(**ctx))

        compile_manifest = {
            "run_id": variant.run_id,
            "simulator": self.name,
            "build_macro": str(build_macro_path),
            "run_macro": str(run_macro_path),
            "components": ctx["components"],
            "build_calls": ctx["build_calls"],
            "voltage_calls": ctx["voltage_calls"],
            "mesh_box_count": len(ctx["mesh_boxes"]),
            "monitor_count": len(ctx["monitors"]),
            "postprocess": variant.sim["postprocess"],
        }
        manifest_path = output_dir / "compile_manifest.json"
        manifest_path.write_text(json.dumps(compile_manifest, indent=2))

        return CompileArtifact(
            run_id=variant.run_id,
            simulator=self.name,
            output_dir=output_dir,
            files={
                "macro": build_macro_path,
                "macro_build": build_macro_path,
                "macro_run": run_macro_path,
                "manifest": manifest_path,
            },
            metadata=compile_manifest,
        )

    def _build_context(self, variant: RunVariant) -> Dict[str, Any]:
        detector = variant.detector
        sim = variant.sim
        run_ns = sanitize_vba_identifier(variant.run_id)
        symbol_registry = SymbolRegistry()

        units = sim["units"]
        boundary = sim["boundary"]
        mesh_cfg = sim["mesh"]
        mesh_step_mm = float(mesh_cfg["step_mm"])
        mesh_buffer_mm = float(mesh_cfg["buffer_mm"])
        mesh_step_m = m_from_mm(mesh_step_mm)
        mesh_buffer_m = m_from_mm(mesh_buffer_mm)

        fields_cfg = sim["fields"]
        fields = {
            "magnetrun": bool(fields_cfg["magnetrun"]),
            "computeE": bool(fields_cfg["computeE"]),
            "computeB": bool(fields_cfg["computeB"]),
            "importE": bool(fields_cfg["importE"]),
            "importB": bool(fields_cfg["importB"]),
            "b_field_file": fields_cfg["b_field_file"],
            "e_field_file": fields_cfg["e_field_file"],
        }
        tracking_cfg = sim["tracking"]
        tracking = {
            "method": tracking_cfg["method"],
            "sim_time_ns": tracking_cfg["sim_time_ns"],
            "iterations": tracking_cfg["iterations"],
            "neglect_space_charge": bool(tracking_cfg["neglect_space_charge"]),
            "neglect_PEC_charging": bool(tracking_cfg["neglect_PEC_charging"]),
            "threads": int(tracking_cfg["threads"]),
            "distributed": bool(tracking_cfg["distributed"]),
            "traj_sampling": tracking_cfg["traj_sampling"],
        }
        solvers_cfg = sim["solvers"]
        solvers = {
            "e_static_accuracy": solvers_cfg["e_static_accuracy"],
            "m_static_accuracy": solvers_cfg["m_static_accuracy"],
            "max_threads": int(solvers_cfg["max_threads"]),
            "max_cpu_devices": int(solvers_cfg["max_cpu_devices"]),
        }
        toggles = sim["toggles"]

        exclusions = sim["exclusions"]
        monitors = list(sim["monitors"])
        postprocess = sim["postprocess"]

        background = sim["background"]
        background_xy = float(background["xy_mm"])
        background_zmin = float(background["zmin_mm"])
        background_zmax = float(background["zmax_mm"])

        profile_cfg = detector["profiles"]["bx"]
        config_dir = Path(detector["__config_dir"]) if "__config_dir" in detector else None
        bx_interp = load_bx_profile(profile_cfg, base_dir=config_dir)
        constants = {k: float(v) for k, v in detector["constants"].items()}

        context: Dict[str, Any] = {
            "run_id": variant.run_id,
            "run_ns": run_ns,
            "units": units,
            "boundary": boundary,
            "mesh": {"step_m": mesh_step_m, "buffer_m": mesh_buffer_m},
            "fields": fields,
            "tracking": tracking,
            "solvers": solvers,
            "toggles": toggles,
            "monitors": monitors,
            "exclusions": {
                "solids": list(exclusions["solids"]),
                "coils": list(exclusions["coils"]),
            },
            "postprocess": postprocess,
            "background_buffer_xy_m": m_from_mm(background_xy),
            "background_buffer_zmin_m": m_from_mm(background_zmin),
            "background_buffer_zmax_m": m_from_mm(background_zmax),
            "max_time_steps": int(sim["max_time_steps"]),
            "components": ["mesh"],
            "build_calls": [],
            "voltage_calls": [],
            "mesh_boxes": [],
            "filter_elements": [],
            "parallel_drains": [],
            "targets": [],
            "einzel_lenses": [],
            "analyzers": [],
            "magnets": [],
            "particles": {"singles": [], "rings": []},
        }

        filter_element_defaults = detector.get("filter_element_defaults", {})
        target_ref_for_sources: Optional[Dict[str, Any]] = None
        geometry = detector["geometry"]
        for element in geometry:
            etype = element["type"]
            element_id = str(element["id"])
            ns = sanitize_vba_identifier(f"{run_ns}_{element_id}")

            if etype == "lngs_magnet":
                build_sub = f"Build_{ns}"
                symbol_registry.add(build_sub, kind="subroutine")
                context["magnets"].append(
                    {
                        "id": element_id,
                        "ns": ns,
                        "component": element["component"],
                        "material": element["material"],
                        "build_sub": build_sub,
                    }
                )
                context["components"].append(element["component"])
                context["build_calls"].append(build_sub)
                continue

            if etype == "target":
                build_sub = f"Build_{ns}"
                apply_sub = f"Apply_{ns}"
                symbol_registry.add(build_sub, kind="subroutine")
                symbol_registry.add(apply_sub, kind="subroutine")

                center_x = float(element["center_x_m"])
                center_z = float(element["center_z_m"])
                rotation_deg = float(element["rotation_deg"])
                magnet_radius = float(element["magnet_radius_m"])
                magnet_height = float(element["magnet_height_m"])
                thickness = float(element["thickness_m"])
                remanence = float(element["remanence_T"])
                voltages = {
                    "inner": float(element["voltage_inner"]),
                    "catcher": float(element["voltage_catcher"]),
                }

                r_max = magnet_radius + 8 * thickness
                x_min = center_x - 2 * thickness
                x_max = center_x + 10 * magnet_height
                half_len = max(abs(x_min - center_x), abs(x_max - center_x))
                x_span = half_len + r_max
                z_span = half_len + r_max
                mesh_bounds = {
                    "x_min": f"{center_x - x_span - mesh_buffer_m:.6f}",
                    "x_max": f"{center_x + x_span + mesh_buffer_m:.6f}",
                    "y_min": f"{-r_max - mesh_buffer_m:.6f}",
                    "y_max": f"{r_max + mesh_buffer_m:.6f}",
                    "z_min": f"{center_z - z_span - mesh_buffer_m:.6f}",
                    "z_max": f"{center_z + z_span + mesh_buffer_m:.6f}",
                }

                target_ctx = {
                    "id": element_id,
                    "ns": ns,
                    "component": element["component"],
                    "build_sub": build_sub,
                    "apply_sub": apply_sub,
                    "center_x": center_x,
                    "center_z": center_z,
                    "rotation_deg": rotation_deg,
                    "magnet_radius": magnet_radius,
                    "magnet_height": magnet_height,
                    "thickness": thickness,
                    "remanence": remanence,
                    "voltages": voltages,
                    "mesh_bounds": mesh_bounds,
                }
                target_ref_for_sources = target_ctx
                context["targets"].append(target_ctx)
                context["components"].append(target_ctx["component"])
                context["build_calls"].append(build_sub)
                context["voltage_calls"].append(apply_sub)
                context["mesh_boxes"].append(
                    {
                        "name": f"mesh_box_{ns}",
                        "group": f"mesh_{ns}",
                        "bounds": mesh_bounds,
                    }
                )
                continue

            if etype == "einzel_lens":
                build_sub = f"Build_{ns}"
                apply_sub = f"Apply_{ns}"
                symbol_registry.add(build_sub, kind="subroutine")
                symbol_registry.add(apply_sub, kind="subroutine")

                lengths_m = [float(v) / 1000.0 for v in element["lengths_mm"]]
                potentials = [float(v) for v in element["potentials"]]
                if len(lengths_m) != len(potentials):
                    raise ValueError(f"einzel_lens '{element_id}' lengths_mm and potentials length mismatch")

                gap_m = float(element["gap_m"])
                inner_r = float(element["inner_radius_m"])
                wall = float(element["wall_thickness_m"])
                start_z = float(element["start_z_m"])

                cylinders = []
                z_curr = start_z
                for idx, (length, potential) in enumerate(zip(lengths_m, potentials)):
                    z1 = z_curr
                    z2 = z_curr + length
                    cname = f"{ns}_cyl_{idx}"
                    cylinders.append(
                        {
                            "name": cname,
                            "inner_r": inner_r,
                            "outer_r": inner_r + wall,
                            "z1": z1,
                            "z2": z2,
                            "potential": potential,
                        }
                    )
                    z_curr = z2 + gap_m
                exit_z = z_curr - gap_m
                outer_r = inner_r + wall
                mesh_bounds = {
                    "x_min": f"{-outer_r - mesh_buffer_m:.6f}",
                    "x_max": f"{outer_r + mesh_buffer_m:.6f}",
                    "y_min": f"{-outer_r - mesh_buffer_m:.6f}",
                    "y_max": f"{outer_r + mesh_buffer_m:.6f}",
                    "z_min": f"{start_z - mesh_buffer_m:.6f}",
                    "z_max": f"{exit_z + mesh_buffer_m:.6f}",
                }
                lens_ctx = {
                    "id": element_id,
                    "ns": ns,
                    "component": element["component"],
                    "build_sub": build_sub,
                    "apply_sub": apply_sub,
                    "cylinders": cylinders,
                    "exit_z": exit_z,
                    "mesh_bounds": mesh_bounds,
                }
                context["einzel_lenses"].append(lens_ctx)
                context["components"].append(lens_ctx["component"])
                context["build_calls"].append(build_sub)
                context["voltage_calls"].append(apply_sub)
                context["mesh_boxes"].append(
                    {
                        "name": f"mesh_box_{ns}",
                        "group": f"mesh_{ns}",
                        "bounds": mesh_bounds,
                    }
                )
                continue

            if etype == "electrostatic_analyzer":
                build_sub = f"Build_{ns}"
                apply_sub = f"Apply_{ns}"
                symbol_registry.add(build_sub, kind="subroutine")
                symbol_registry.add(apply_sub, kind="subroutine")

                inner_r = float(element["inner_radius_m"])
                center_r = float(element["center_radius_m"])
                outer_r = float(element["outer_radius_m"])
                pcb_thk = float(element["pcb_thickness_m"])
                shell_thk = float(element["shell_thickness_m"])
                outer_shell_r = outer_r + shell_thk
                x_span = element["x_span_m"]
                y_span = element["y_span_m"]
                slit_in = element["entrance_slit_mm"]
                slit_out = element["exit_slit_mm"]
                voltages = element["voltages"]

                center_y = center_r
                center_z = float(element["center_z_m"])

                plate_z1 = center_z - pcb_thk
                plate_z2 = center_z
                mesh_bounds = {
                    "x_min": f"{min(float(x_span[0]), -outer_shell_r) - mesh_buffer_m:.6f}",
                    "x_max": f"{max(float(x_span[1]), outer_shell_r) + mesh_buffer_m:.6f}",
                    "y_min": f"{min(float(y_span[0]), center_y - outer_shell_r) - mesh_buffer_m:.6f}",
                    "y_max": f"{max(float(y_span[1]), center_y + outer_shell_r) + mesh_buffer_m:.6f}",
                    "z_min": f"{min(plate_z1, center_z - outer_shell_r) - mesh_buffer_m:.6f}",
                    "z_max": f"{max(plate_z2, center_z + outer_shell_r) + mesh_buffer_m:.6f}",
                }

                analyzer_ctx = {
                    "id": element_id,
                    "ns": ns,
                    "component": element["component"],
                    "build_sub": build_sub,
                    "apply_sub": apply_sub,
                    "inner_r": inner_r,
                    "outer_r": outer_r,
                    "outer_shell_r": outer_shell_r,
                    "center_r": center_r,
                    "center_y": center_y,
                    "center_z": center_z,
                    "exit_y": 2 * center_r,
                    "voltages": {"inner": float(voltages["inner"]), "outer": float(voltages["outer"])},
                    "pcb": {
                        "x_min": float(x_span[0]),
                        "x_max": float(x_span[1]),
                        "y_min": float(y_span[0]),
                        "y_max": float(y_span[1]),
                        "z1": plate_z1,
                        "z2": plate_z2,
                        "entrance_slit": {"dy": float(slit_in[0]) / 1000.0, "dx": float(slit_in[1]) / 1000.0},
                        "exit_slit": {"dy": float(slit_out[0]) / 1000.0, "dx": float(slit_out[1]) / 1000.0},
                    },
                    "mesh_bounds": mesh_bounds,
                }
                context["analyzers"].append(analyzer_ctx)
                context["components"].append(analyzer_ctx["component"])
                context["build_calls"].append(build_sub)
                context["voltage_calls"].append(apply_sub)
                context["mesh_boxes"].append(
                    {
                        "name": f"mesh_box_{ns}",
                        "group": f"mesh_{ns}",
                        "bounds": mesh_bounds,
                    }
                )

                monitors.append(
                    {
                        "name": f"{element_id}_entrance",
                        "planes": 1,
                        "normal": "z",
                        "xlim": [
                            -analyzer_ctx["pcb"]["entrance_slit"]["dx"] / 2,
                            analyzer_ctx["pcb"]["entrance_slit"]["dx"] / 2,
                        ],
                        "ylim": [
                            -analyzer_ctx["pcb"]["entrance_slit"]["dy"] / 2,
                            analyzer_ctx["pcb"]["entrance_slit"]["dy"] / 2,
                        ],
                        "zlim": [plate_z1, plate_z1],
                    }
                )
                monitors.append(
                    {
                        "name": f"{element_id}_exit",
                        "planes": 1,
                        "normal": "z",
                        "xlim": [
                            -analyzer_ctx["pcb"]["exit_slit"]["dx"] / 2,
                            analyzer_ctx["pcb"]["exit_slit"]["dx"] / 2,
                        ],
                        "ylim": [
                            analyzer_ctx["exit_y"] - analyzer_ctx["pcb"]["exit_slit"]["dy"] / 2,
                            analyzer_ctx["exit_y"] + analyzer_ctx["pcb"]["exit_slit"]["dy"] / 2,
                        ],
                        "zlim": [plate_z1, plate_z1],
                    }
                )
                continue

            if etype == "parallel_drain":
                build_sub = f"Build_{ns}"
                apply_sub = f"Apply_{ns}"
                symbol_registry.add(build_sub, kind="subroutine")
                symbol_registry.add(apply_sub, kind="subroutine")

                pd = self._build_parallel_drain(element, mesh_buffer_m, ns)
                pd["id"] = element_id
                pd["build_sub"] = build_sub
                pd["apply_sub"] = apply_sub
                context["parallel_drains"].append(pd)
                context["components"].append(pd["component"])
                context["build_calls"].append(build_sub)
                context["voltage_calls"].append(apply_sub)
                context["mesh_boxes"].append(
                    {
                        "name": f"mesh_box_{ns}",
                        "group": f"mesh_{ns}",
                        "bounds": pd["mesh_bounds"],
                    }
                )
                continue

            if etype == "filter_element":
                build_sub = f"Build_{ns}"
                apply_sub = f"Apply_{ns}"
                symbol_registry.add(build_sub, kind="subroutine")
                symbol_registry.add(apply_sub, kind="subroutine")

                section = FilterSection.from_geometry(element, filter_element_defaults)
                filter_ctx = self._build_filter_element(
                    section=section,
                    mesh_buffer_mm=int(mesh_buffer_mm),
                    bx_interp=bx_interp,
                    constants=constants,
                    namespace=ns,
                )
                filter_ctx.update(
                    {
                        "id": element_id,
                        "component": element["component"],
                        "ns": ns,
                        "build_sub": build_sub,
                        "apply_sub": apply_sub,
                    }
                )

                context["filter_elements"].append(filter_ctx)
                context["components"].append(filter_ctx["component"])
                context["build_calls"].append(build_sub)
                context["voltage_calls"].append(apply_sub)
                context["mesh_boxes"].append(
                    {
                        "name": f"mesh_box_{ns}",
                        "group": f"mesh_{ns}",
                        "bounds": filter_ctx["mesh_bounds"],
                    }
                )
                continue

        context["particles"] = collect_particles(
            detector["sources"], target_ref_for_sources, mesh_buffer_m, run_ns
        )

        unique_components: list[str] = []
        seen_components: set[str] = set()
        for component in context["components"]:
            if component in seen_components:
                continue
            seen_components.add(component)
            unique_components.append(component)
        context["components"] = unique_components

        return context

    def _build_parallel_drain(self, cfg: Dict[str, Any], mesh_buffer_m: float, namespace: str) -> Dict[str, Any]:
        angles = cfg["angles_deg"]
        potentials = cfg["potentials"]
        if len(angles) < 2 or len(potentials) != len(angles) - 1:
            raise ValueError("parallel_drain requires angles_deg>=2 and matching potentials")

        angle_gap = float(cfg["angle_gap_deg"])
        center_x = float(cfg["ellipse_center"]["x_m"])
        center_z = float(cfg["ellipse_center"]["z_m"])
        axes = cfg["axes_m"]
        a = float(axes["a_m"])
        b = float(axes["b_m"])
        offsets = cfg["offsets_m"]
        inner = float(offsets["inner"])
        outer = float(offsets["outer"])
        y_at = float(cfg["y_at_plate_m"])
        thickness = float(cfg["electrode_thickness_m"])
        component = cfg["component"]
        face_number = str(cfg["face_number"])
        x_half = float(cfg["x_half_m"])
        y_half = float(cfg["y_half_m"])
        cutout_z_span = cfg["cutout_z_span_m"]

        def point_at(theta_deg: float, radial_delta: float) -> Tuple[float, float]:
            theta = math.radians(theta_deg)
            y_val = (a * b) / math.sqrt(b * b + (a * a) * math.tan(theta) ** 2)
            x_val = y_val * math.tan(theta)
            r = math.sqrt(x_val * x_val + y_val * y_val)
            x = center_x - (r + radial_delta) * math.cos(theta)
            z = center_z - (r + radial_delta) * math.sin(theta)
            return x, z

        arcs = []
        for idx in range(len(angles) - 1):
            th1 = float(angles[idx]) + angle_gap
            th2 = float(angles[idx + 1]) - angle_gap
            name = f"{namespace}_arc_{idx}"
            pot = float(potentials[idx])
            x1i, z1i = point_at(th1, -inner)
            x1o, z1o = point_at(th1, outer)
            x2o, z2o = point_at(th2, outer)
            x2i, z2i = point_at(th2, -inner)

            outline = [
                {"x": x1i, "y": y_at, "z": z1i},
                {"x": x1o, "y": y_at, "z": z1o},
                {"x": x2o, "y": y_at, "z": z2o},
                {"x": x2i, "y": y_at, "z": z2i},
                {"x": x1i, "y": y_at, "z": z1i},
            ]
            profile = [
                {"x": x1i, "y": y_at, "z": z1i},
                {"x": x1i, "y": y_at + thickness, "z": z1i},
            ]
            arcs.append({"name": name, "outline": outline, "profile": profile, "potential": pot, "face": face_number})

        points = [pt for arc in arcs for pt in (arc["outline"] + arc["profile"])]
        x_vals = [p["x"] for p in points]
        z_vals = [p["z"] for p in points]
        max_y = max([abs(p["y"]) for p in points] + [abs(y_at) + thickness])
        mesh_bounds = {
            "x_min": f"{min(x_vals) - mesh_buffer_m:.6f}",
            "x_max": f"{max(x_vals) + mesh_buffer_m:.6f}",
            "y_min": f"{-max_y - mesh_buffer_m:.6f}",
            "y_max": f"{max_y + mesh_buffer_m:.6f}",
            "z_min": f"{min(z_vals) - mesh_buffer_m:.6f}",
            "z_max": f"{max(z_vals) + mesh_buffer_m:.6f}",
        }

        return {
            "component": component,
            "curve": f"{namespace}_curve",
            "material": cfg["material"],
            "arcs": arcs,
            "x_half": x_half,
            "y_half": y_half,
            "thickness": thickness,
            "cutout_z_min": float(cutout_z_span[0]),
            "cutout_z_max": float(cutout_z_span[1]),
            "mesh_bounds": mesh_bounds,
        }

    def _build_filter_element(
        self,
        section: FilterSection,
        mesh_buffer_mm: int,
        bx_interp: Callable[[float], float],
        constants: Dict[str, float],
        namespace: str,
    ) -> Dict[str, Any]:
        pcb_thick_mm = clamp_thickness_mm(section.pcb_thickness_mm)
        electrode_offset_mm = max(0, section.electrode_thickness_mm)
        x_half_mm = section.x_half_mm
        y_half_mm = section.y_half_mm
        z_start_mm = section.z_start_mm
        z_end_mm = section.z_end_mm

        bricks: List[BrickSpec] = []
        potentials: List[PotentialSpec] = []

        bricks.extend(
            [
                BrickSpec(
                    name=f"{namespace}_pcb_bounce_xpos",
                    material="Epoxy resin",
                    x1=fmt_m(x_half_mm),
                    x2=fmt_m(x_half_mm + pcb_thick_mm),
                    y1=fmt_m(-y_half_mm),
                    y2=fmt_m(y_half_mm),
                    z1=fmt_m(z_start_mm),
                    z2=fmt_m(z_end_mm),
                ),
                BrickSpec(
                    name=f"{namespace}_pcb_bounce_xneg",
                    material="Epoxy resin",
                    x1=fmt_m(-x_half_mm - pcb_thick_mm),
                    x2=fmt_m(-x_half_mm),
                    y1=fmt_m(-y_half_mm),
                    y2=fmt_m(y_half_mm),
                    z1=fmt_m(z_start_mm),
                    z2=fmt_m(z_end_mm),
                ),
                BrickSpec(
                    name=f"{namespace}_pcb_channel_ypos",
                    material="Epoxy resin",
                    x1=fmt_m(-x_half_mm),
                    x2=fmt_m(x_half_mm),
                    y1=fmt_m(y_half_mm),
                    y2=fmt_m(y_half_mm + pcb_thick_mm),
                    z1=fmt_m(z_start_mm),
                    z2=fmt_m(z_end_mm),
                ),
                BrickSpec(
                    name=f"{namespace}_pcb_channel_yneg",
                    material="Epoxy resin",
                    x1=fmt_m(-x_half_mm),
                    x2=fmt_m(x_half_mm),
                    y1=fmt_m(-y_half_mm - pcb_thick_mm),
                    y2=fmt_m(-y_half_mm),
                    z1=fmt_m(z_start_mm),
                    z2=fmt_m(z_end_mm),
                ),
            ]
        )

        if section.include_termination_backplate:
            bricks.append(
                BrickSpec(
                    name=f"{namespace}_pcb_backplate",
                    material="Epoxy resin",
                    x1=fmt_m(-x_half_mm),
                    x2=fmt_m(x_half_mm),
                    y1=fmt_m(-y_half_mm),
                    y2=fmt_m(y_half_mm),
                    z1=fmt_m(z_end_mm),
                    z2=fmt_m(z_end_mm + pcb_thick_mm),
                )
            )

        corner_gap_mm = max(0, section.corner_gap_mm)
        y_min_mm = -y_half_mm + corner_gap_mm
        y_max_mm = y_half_mm - corner_gap_mm
        if y_max_mm <= y_min_mm:
            raise ValueError(f"corner_gap_m too large for {section.id}")

        x_pos_mm = x_half_mm - electrode_offset_mm
        x_neg_mm = -x_half_mm + electrode_offset_mm

        eval_ctx: Dict[str, Any] = dict(constants)
        eval_ctx["bx_norm"] = bx_interp

        unit_mm = section.electrode_len_mm + section.electrode_gap_mm
        filter_start_z_mm = z_start_mm - section.n_start * unit_mm

        def _z_lookup(electrode_n: int) -> float:
            z_mid_mm = filter_start_z_mm + electrode_n * unit_mm + section.electrode_len_mm / 2.0
            return z_mid_mm / 1000.0

        eval_ctx["z"] = _z_lookup

        seg_bxp = build_segments(z_start_mm, z_end_mm, section.electrode_len_mm, section.electrode_gap_mm)
        seg_bxn = build_segments(z_start_mm, z_end_mm, section.electrode_len_mm, section.electrode_gap_mm)

        for idx, (z0_mm, length_mm) in enumerate(seg_bxp):
            z1_mm, z2_mm = z0_mm, z0_mm + length_mm
            global_n = section.n_start + idx
            val = eval_expr(section.voltages.bounce_xpos, global_n, eval_ctx)
            shape = f"{namespace}_bounce_xpos_{idx}"
            bricks.append(
                BrickSpec(
                    name=shape,
                    material="PEC",
                    x1=fmt_m(x_pos_mm),
                    x2=fmt_m(x_pos_mm),
                    y1=fmt_m(y_min_mm),
                    y2=fmt_m(y_max_mm),
                    z1=fmt_m(z1_mm),
                    z2=fmt_m(z2_mm),
                )
            )
            potentials.append(
                PotentialSpec(
                    name=f"pot_{shape}",
                    value=str(val),
                    solid=f"{section.component}:{shape}",
                    face="4",
                )
            )

        for idx, (z0_mm, length_mm) in enumerate(seg_bxn):
            z1_mm, z2_mm = z0_mm, z0_mm + length_mm
            global_n = section.n_start + idx
            val = eval_expr(section.voltages.bounce_xneg, global_n, eval_ctx)
            shape = f"{namespace}_bounce_xneg_{idx}"
            bricks.append(
                BrickSpec(
                    name=shape,
                    material="PEC",
                    x1=fmt_m(x_neg_mm),
                    x2=fmt_m(x_neg_mm),
                    y1=fmt_m(y_min_mm),
                    y2=fmt_m(y_max_mm),
                    z1=fmt_m(z1_mm),
                    z2=fmt_m(z2_mm),
                )
            )
            potentials.append(
                PotentialSpec(
                    name=f"pot_{shape}",
                    value=str(val),
                    solid=f"{section.component}:{shape}",
                    face="4",
                )
            )

        channels_x = section.channels_x
        channel_gap_mm = max(0, section.channel_gap_mm)
        x_half_trim_mm = x_half_mm - corner_gap_mm
        available_width_mm = 2 * x_half_trim_mm - (channels_x - 1) * channel_gap_mm
        if available_width_mm <= 0:
            raise ValueError(f"channel_gap too large for {section.id}")

        if section.channel_width_ratios:
            if len(section.channel_width_ratios) != channels_x:
                raise ValueError("channel_width_ratios length mismatch")
            widths_mm = allocate_widths_by_ratio(available_width_mm, section.channel_width_ratios)
        else:
            widths_mm = distribute_widths(available_width_mm, channels_x)

        seg_cyp = build_segments(z_start_mm, z_end_mm, section.electrode_len_mm, section.electrode_gap_mm)
        seg_cyn = build_segments(z_start_mm, z_end_mm, section.electrode_len_mm, section.electrode_gap_mm)

        y_pos_mm = y_half_mm - electrode_offset_mm
        y_neg_mm = -y_half_mm + electrode_offset_mm
        widths_layout = list(reversed(widths_mm))
        x_cursor_mm = -x_half_trim_mm
        for idx_from_neg, width_mm in enumerate(widths_layout):
            chan_idx = channels_x - 1 - idx_from_neg
            x1_mm = x_cursor_mm
            x2_mm = x_cursor_mm + width_mm
            x_cursor_mm = x2_mm + (channel_gap_mm if idx_from_neg < channels_x - 1 else 0)

            pos_expr = section.voltages.pos_expr_for_channel(chan_idx)
            neg_expr = section.voltages.neg_expr_for_channel(chan_idx)

            for seg_idx, (z0_mm, length_mm) in enumerate(seg_cyp):
                z1_mm, z2_mm = z0_mm, z0_mm + length_mm
                global_n = section.n_start + seg_idx
                eval_ctx["chan"] = chan_idx
                value = eval_expr(pos_expr, global_n, eval_ctx)
                shape = f"{namespace}_channel_ypos_{chan_idx}_{seg_idx}"
                bricks.append(
                    BrickSpec(
                        name=shape,
                        material="PEC",
                        x1=fmt_m(x1_mm),
                        x2=fmt_m(x2_mm),
                        y1=fmt_m(y_pos_mm),
                        y2=fmt_m(y_pos_mm),
                        z1=fmt_m(z1_mm),
                        z2=fmt_m(z2_mm),
                    )
                )
                potentials.append(
                    PotentialSpec(
                        name=f"pot_{shape}",
                        value=str(value),
                        solid=f"{section.component}:{shape}",
                        face="3",
                    )
                )

            for seg_idx, (z0_mm, length_mm) in enumerate(seg_cyn):
                z1_mm, z2_mm = z0_mm, z0_mm + length_mm
                global_n = section.n_start + seg_idx
                eval_ctx["chan"] = chan_idx
                value = eval_expr(neg_expr, global_n, eval_ctx)
                shape = f"{namespace}_channel_yneg_{chan_idx}_{seg_idx}"
                bricks.append(
                    BrickSpec(
                        name=shape,
                        material="PEC",
                        x1=fmt_m(x1_mm),
                        x2=fmt_m(x2_mm),
                        y1=fmt_m(y_neg_mm),
                        y2=fmt_m(y_neg_mm),
                        z1=fmt_m(z1_mm),
                        z2=fmt_m(z2_mm),
                    )
                )
                potentials.append(
                    PotentialSpec(
                        name=f"pot_{shape}",
                        value=str(value),
                        solid=f"{section.component}:{shape}",
                        face="3",
                    )
                )

        z_max = z_end_mm + pcb_thick_mm if section.include_termination_backplate else z_end_mm

        mesh_bounds = {
            "x_min": fmt_m(-x_half_mm - pcb_thick_mm - mesh_buffer_mm),
            "x_max": fmt_m(x_half_mm + pcb_thick_mm + mesh_buffer_mm),
            "y_min": fmt_m(-y_half_mm - pcb_thick_mm - mesh_buffer_mm),
            "y_max": fmt_m(y_half_mm + pcb_thick_mm + mesh_buffer_mm),
            "z_min": fmt_m(z_start_mm - mesh_buffer_mm),
            "z_max": fmt_m(z_max + mesh_buffer_mm),
        }

        return {"bricks": bricks, "potentials": potentials, "mesh_bounds": mesh_bounds}


def collect_particles(
    pcfg: Dict[str, Any],
    target_ctx: Optional[Dict[str, Any]],
    _mesh_buffer_m: float,
    run_ns: str,
) -> Dict[str, Any]:
    particles: List[Dict[str, Any]] = []
    rings: List[Dict[str, Any]] = []

    if target_ctx:
        rotation = float(target_ctx["rotation_deg"])
        n = (math.cos(math.radians(rotation)), 0.0, -math.sin(math.radians(rotation)))
        target_center = (float(target_ctx["center_x"]), 0.0, float(target_ctx["center_z"]))
        target_offset = float(target_ctx["thickness"])
    else:
        n = (0.0, 0.0, -1.0)
        target_center = (0.0, 0.0, 0.0)
        target_offset = 0.0

    up = (0.0, 1.0, 0.0)
    t = (
        up[1] * n[2] - up[2] * n[1],
        up[2] * n[0] - up[0] * n[2],
        up[0] * n[1] - up[1] * n[0],
    )
    t_mag = math.sqrt(sum(v * v for v in t))
    if t_mag < 1e-9:
        t = (1.0, 0.0, 0.0)
        t_mag = 1.0
    t = tuple(v / t_mag for v in t)
    b = (
        n[1] * t[2] - n[2] * t[1],
        n[2] * t[0] - n[0] * t[2],
        n[0] * t[1] - n[1] * t[0],
    )

    default_origin = (
        target_center[0] + target_offset * n[0],
        target_center[1] + target_offset * n[1],
        target_center[2] + target_offset * n[2],
    )

    for idx, single in enumerate(pcfg["singles"]):
        pitches = pitch_list_from_cfg(single)
        phis = phi_list_from_cfg(single)
        energy = float(single["energy_eV"])
        on_target = bool(single["on_target_face"])
        if on_target:
            origin = default_origin
        else:
            origin_m = single["origin_m"]
            origin = (float(origin_m[0]), float(origin_m[1]), float(origin_m[2]))

        name_prefix = single["name_prefix"]
        for pitch_deg in pitches:
            for phi_deg in phis:
                p_rad = math.radians(pitch_deg)
                phi_rad = math.radians(phi_deg)
                direction = (
                    math.cos(p_rad) * n[0]
                    + math.sin(p_rad) * (math.cos(phi_rad) * t[0] + math.sin(phi_rad) * b[0]),
                    math.cos(p_rad) * n[1]
                    + math.sin(p_rad) * (math.cos(phi_rad) * t[1] + math.sin(phi_rad) * b[1]),
                    math.cos(p_rad) * n[2]
                    + math.sin(p_rad) * (math.cos(phi_rad) * t[2] + math.sin(phi_rad) * b[2]),
                )
                particles.append(
                    {
                        "name": f"{run_ns}_{name_prefix}_{idx}_p{int(pitch_deg)}_phi{int(phi_deg)}",
                        "pos": origin,
                        "dir": direction,
                        "energy": energy,
                    }
                )

    for idx, ring in enumerate(pcfg["rings"]):
        radii = ring["radii_m"]
        inner = float(ring["inner_radius_m"])
        lines = int(ring["lines"])
        pitch = float(ring["pitch_deg"])
        pitch_spread = float(ring["pitch_spread_deg"])
        phi = float(ring["phi_deg"])
        energy = float(ring["energy_eV"])
        on_target = bool(ring["on_target_face"])
        if on_target:
            center_point = default_origin
        else:
            origin_m = ring["origin_m"]
            center_point = (float(origin_m[0]), float(origin_m[1]), float(origin_m[2]))

        name_prefix = ring["name_prefix"]
        for radius_idx, radius in enumerate(radii):
            inner_use = inner
            if abs(pitch) > 1e-9 and inner_use <= 0.0:
                inner_use = max(1e-6, 0.01 * float(radius))
            rings.append(
                {
                    "name": f"{run_ns}_{name_prefix}_{idx}_{radius_idx}",
                    "center": center_point,
                    "normal": n,
                    "radius": float(radius),
                    "inner_radius": inner_use,
                    "lines": lines,
                    "pitch": pitch,
                    "pitch_spread": pitch_spread,
                    "phi": phi,
                    "energy": energy,
                }
            )

    return {"singles": particles, "rings": rings}
