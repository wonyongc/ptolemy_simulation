"""Configuration errors."""

from __future__ import annotations


class ValidationError(ValueError):
    """Raised when a config payload fails schema validation."""

    def __init__(self, message: str, *, path: str | None = None):
        self.path = path
        super().__init__(f"{path}: {message}" if path else message)
