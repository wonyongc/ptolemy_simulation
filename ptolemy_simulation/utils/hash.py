"""Stable hashing utilities."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_sha256(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
