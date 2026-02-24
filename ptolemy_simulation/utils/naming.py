"""Naming and sanitization helpers."""

from __future__ import annotations

import re


def slugify(value: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]", "-", value.strip())
    base = re.sub(r"-+", "-", base).strip("-")
    return base or "run"


def sanitize_vba_identifier(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not safe:
        safe = "id"
    if safe[0].isdigit():
        safe = f"r_{safe}"
    return safe
