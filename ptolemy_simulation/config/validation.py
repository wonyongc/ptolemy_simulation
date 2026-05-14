"""Schema validation for detector/simulator/run/study configs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Mapping

from .errors import ValidationError

ALLOWED_GEOMETRY_TYPES = {
    "lngs_magnet",
    "target",
    "filter_element",
    "parallel_drain",
    "einzel_lens",
    "electrostatic_analyzer",
}

ALLOWED_SIMULATORS = {"cst", "kassiopeia"}

_DEFAULT_TOGGLES = {
    "exclude_magnet_when_magnetrun_false": True,
    "run_solver_after_build": True,
    "export_transmission": True,
    "export_pitch_analysis": True,
    "export_trajectories": False,
    "export_field_maps": False,
}


def _require_mapping(data: Any, path: str) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValidationError("must be a mapping", path=path)
    return data


def _require_list(data: Any, path: str) -> list[Any]:
    if not isinstance(data, list):
        raise ValidationError("must be a list", path=path)
    return data


def _require_keys(payload: Mapping[str, Any], keys: Iterable[str], path: str) -> None:
    for key in keys:
        if key not in payload:
            raise ValidationError(f"missing required key '{key}'", path=path)


def validate_detector(data: Dict[str, Any]) -> Dict[str, Any]:
    payload = deepcopy(_require_mapping(data, "detector"))
    _require_keys(payload, ["schema_version", "experiment", "geometry", "sources"], "detector")

    geometry = _require_list(payload["geometry"], "detector.geometry")
    if not geometry:
        raise ValidationError("must contain at least one geometry element", path="detector.geometry")

    seen_ids: set[str] = set()
    for idx, element in enumerate(geometry):
        item = _require_mapping(element, f"detector.geometry[{idx}]")
        _require_keys(item, ["id", "type"], f"detector.geometry[{idx}]")
        if not isinstance(item["id"], str) or not item["id"].strip():
            raise ValidationError("id must be a non-empty string", path=f"detector.geometry[{idx}].id")
        if item["id"] in seen_ids:
            raise ValidationError("duplicate geometry id", path=f"detector.geometry[{idx}].id")
        seen_ids.add(item["id"])
        if item["type"] not in ALLOWED_GEOMETRY_TYPES:
            raise ValidationError(
                f"unsupported geometry type '{item['type']}'",
                path=f"detector.geometry[{idx}].type",
            )

        if item["type"] == "filter_element":
            _require_keys(
                item,
                [
                    "n_electrodes_inclusive",
                    "x_half_m",
                    "y_half_m",
                    "channels_x",
                    "voltages",
                ],
                f"detector.geometry[{idx}]",
            )

    filter_defaults = payload.get("filter_element_defaults", {})
    filter_defaults = _require_mapping(filter_defaults, "detector.filter_element_defaults")
    has_filter = any(e.get("type") == "filter_element" for e in geometry)
    if has_filter:
        _require_keys(
            filter_defaults,
            [
                "filter_start_z_m",
                "pcb_thickness_m",
                "electrode_thickness_m",
                "electrode_len_m",
                "electrode_gap_m",
                "channel_gap_m",
                "corner_gap_m",
            ],
            "detector.filter_element_defaults",
        )
    payload["filter_element_defaults"] = filter_defaults

    constants = payload.get("constants", {})
    constants_map = _require_mapping(constants, "detector.constants")
    payload["constants"] = {k: float(v) for k, v in constants_map.items()}

    profiles = payload.get("profiles", {})
    payload["profiles"] = _require_mapping(profiles, "detector.profiles")

    sources = _require_mapping(payload["sources"], "detector.sources")
    singles = _require_list(sources.get("singles", []), "detector.sources.singles")
    rings = _require_list(sources.get("rings", []), "detector.sources.rings")
    sources["singles"] = singles
    sources["rings"] = rings
    payload["sources"] = sources

    return payload


def validate_simulator(data: Dict[str, Any]) -> Dict[str, Any]:
    payload = deepcopy(_require_mapping(data, "simulator"))
    _require_keys(payload, ["schema_version", "simulator"], "simulator")

    simulator = payload["simulator"]
    if simulator not in ALLOWED_SIMULATORS:
        raise ValidationError(
            f"simulator must be one of {sorted(ALLOWED_SIMULATORS)}", path="simulator.simulator"
        )

    payload.setdefault("units", {"length": "m", "voltage": "V", "time": "ns"})
    payload.setdefault("boundary", "open")
    payload.setdefault("mesh", {"step_mm": 1.0, "buffer_mm": 0.0})
    payload.setdefault("fields", {})
    payload.setdefault("tracking", {})
    payload.setdefault("solvers", {})
    payload.setdefault("monitors", [])
    payload.setdefault("exclusions", {"solids": [], "coils": []})
    payload.setdefault("postprocess", {})

    toggles = payload.get("toggles", {})
    toggles_map = _require_mapping(toggles, "simulator.toggles")
    toggles_map.pop("include_default_lngs_magnet", None)
    merged_toggles = dict(_DEFAULT_TOGGLES)
    merged_toggles.update(toggles_map)
    for key, value in merged_toggles.items():
        if not isinstance(value, bool):
            raise ValidationError("toggle value must be boolean", path=f"simulator.toggles.{key}")
    payload["toggles"] = merged_toggles

    payload["monitors"] = _require_list(payload["monitors"], "simulator.monitors")
    payload["exclusions"] = _require_mapping(payload["exclusions"], "simulator.exclusions")
    payload["postprocess"] = _require_mapping(payload["postprocess"], "simulator.postprocess")

    if simulator == "cst":
        post = payload["postprocess"]
        post.setdefault("monitor_names", [])
        post.setdefault("output_dir", "results")
        post.setdefault("transmission_filename", "transmission.txt")
        post.setdefault("pitch_filename", "pitchanalysis.txt")
        post.setdefault("trajectory_filename", "trajectories.csv")
        field_exports = _require_mapping(post.get("field_exports", {}), "simulator.postprocess.field_exports")
        field_exports.setdefault("output_dir", "results/fields")
        field_exports.setdefault("e_field_result", "field_se_e")
        field_exports.setdefault("potential_result", "field_se_phi")
        field_exports.setdefault("b_field_result", "field_0_0_import_predefined_plot")
        field_exports["regions"] = _require_list(field_exports.get("regions", []), "simulator.postprocess.field_exports.regions")
        post["field_exports"] = field_exports
        post["monitor_names"] = _require_list(post["monitor_names"], "simulator.postprocess.monitor_names")

    return payload


def validate_run(data: Dict[str, Any]) -> Dict[str, Any]:
    payload = deepcopy(_require_mapping(data, "run"))
    _require_keys(payload, ["schema_version", "mode"], "run")

    mode = payload["mode"]
    if mode not in {"local", "della"}:
        raise ValidationError("mode must be 'local' or 'della'", path="run.mode")

    if mode == "local":
        _require_keys(payload, ["template_zip", "project_root", "unpack_dir"], "run")
        payload.setdefault("stage_only", True)
    else:
        _require_keys(payload, ["host", "project_root", "template_zip", "cst_bin", "slurm"], "run")
        slurm = _require_mapping(payload["slurm"], "run.slurm")
        _require_keys(slurm, ["cpus", "mem", "time", "mail_user"], "run.slurm")
        payload["slurm"] = slurm
        payload.setdefault("submit", False)
        payload.setdefault("auto_postprocess", True)

    return payload


def validate_study(data: Dict[str, Any]) -> Dict[str, Any]:
    payload = deepcopy(_require_mapping(data, "study"))
    _require_keys(
        payload,
        ["schema_version", "study_name", "detector_config", "sim_config", "run_config"],
        "study",
    )

    payload.setdefault("simulator", None)
    payload.setdefault("output_root", "generated/studies")
    payload.setdefault("run_name_template", None)
    payload.setdefault("matrix", {})
    payload.setdefault("overrides", {"detector": {}, "sim": {}, "run": {}})

    matrix = _require_mapping(payload["matrix"], "study.matrix")
    normalized_matrix: Dict[str, list[Any]] = {}
    for key, values in matrix.items():
        if not isinstance(key, str) or not key:
            raise ValidationError("matrix key must be non-empty string", path="study.matrix")
        normalized_matrix[key] = _require_list(values, f"study.matrix.{key}")
    payload["matrix"] = normalized_matrix

    overrides = _require_mapping(payload["overrides"], "study.overrides")
    for section in ("detector", "sim", "run"):
        section_overrides = _require_mapping(overrides.get(section, {}), f"study.overrides.{section}")
        for pointer in section_overrides.keys():
            if not isinstance(pointer, str) or not pointer.startswith("/"):
                raise ValidationError(
                    "override path must be a JSON pointer beginning with '/'",
                    path=f"study.overrides.{section}",
                )
        overrides[section] = section_overrides
    payload["overrides"] = overrides

    return payload
