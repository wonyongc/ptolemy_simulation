"""Template and placeholder helpers."""

from __future__ import annotations

from typing import Any, Mapping


class SafeDict(dict):
    """format_map fallback that leaves unknown placeholders intact."""

    def __missing__(self, key: str) -> str:  # pragma: no cover
        return "{" + key + "}"


def substitute(value: Any, variables: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        if value.startswith("$") and value[1:] in variables:
            return variables[value[1:]]
        return value.format_map(SafeDict(**variables))
    if isinstance(value, list):
        return [substitute(item, variables) for item in value]
    if isinstance(value, dict):
        return {k: substitute(v, variables) for k, v in value.items()}
    return value
