"""Adapter registry."""

from __future__ import annotations

from typing import Dict

from ptolemy_simulation.adapters.base import Adapter


class AdapterRegistry:
    """Registry keyed by simulator name."""

    def __init__(self) -> None:
        self._adapters: Dict[str, Adapter] = {}

    def register(self, adapter: Adapter) -> None:
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> Adapter:
        if name not in self._adapters:
            raise KeyError(f"No adapter registered for simulator '{name}'")
        return self._adapters[name]


def build_default_registry() -> AdapterRegistry:
    from ptolemy_simulation.adapters.cst import CSTAdapter
    from ptolemy_simulation.adapters.kassiopeia import KassiopeiaAdapter

    registry = AdapterRegistry()
    registry.register(CSTAdapter())
    registry.register(KassiopeiaAdapter())
    return registry
