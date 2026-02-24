"""Study matrix expansion and per-run override application."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any, Dict, Iterable

from ptolemy_simulation.config.models import StudyBundle
from ptolemy_simulation.pipeline.models import RunVariant
from ptolemy_simulation.utils.json_pointer import set_pointer
from ptolemy_simulation.utils.naming import slugify
from ptolemy_simulation.utils.templating import substitute


def _cartesian(matrix: Dict[str, list[Any]]) -> Iterable[Dict[str, Any]]:
    if not matrix:
        yield {}
        return

    items = list(matrix.items())
    keys = [k for k, _ in items]
    value_lists = [v for _, v in items]
    for values in product(*value_lists):
        yield dict(zip(keys, values))


def _make_run_id(index: int, variables: Dict[str, Any], template: str | None) -> str:
    if template:
        return slugify(template.format_map(variables))
    if not variables:
        return f"run-{index:04d}"
    joined = "-".join(f"{k}-{variables[k]}" for k in variables.keys())
    return slugify(joined)


def _apply_section_overrides(
    payload: Dict[str, Any],
    overrides: Dict[str, Any],
    variables: Dict[str, Any],
) -> Dict[str, Any]:
    out = deepcopy(payload)
    for pointer, raw_value in overrides.items():
        value = substitute(raw_value, variables)
        set_pointer(out, pointer, value)
    return out


def expand_study(bundle: StudyBundle) -> list[RunVariant]:
    study = bundle.study.raw
    matrix = study.get("matrix", {})
    overrides = study.get("overrides", {})
    detector_overrides = overrides.get("detector", {})
    sim_overrides = overrides.get("sim", {})
    run_overrides = overrides.get("run", {})
    template = study.get("run_name_template")

    variants: list[RunVariant] = []
    for index, variables in enumerate(_cartesian(matrix)):
        run_id = _make_run_id(index=index, variables=variables, template=template)
        detector = _apply_section_overrides(bundle.detector.raw, detector_overrides, variables)
        sim = _apply_section_overrides(bundle.simulator.raw, sim_overrides, variables)
        run = _apply_section_overrides(bundle.run.raw, run_overrides, variables)
        variants.append(
            RunVariant(
                index=index,
                run_id=run_id,
                variables=variables,
                simulator=sim["simulator"],
                detector=detector,
                sim=sim,
                run=run,
            )
        )
    return variants
