"""Compatibility imports for libraries provided by Home Assistant."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING


def _import_first(*module_names: str) -> ModuleType:
    """Return the first importable module from a compatibility list."""
    last_error: ModuleNotFoundError | None = None
    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as err:
            if err.name != module_name:
                raise
            last_error = err
    raise ModuleNotFoundError(
        f"None of the supported modules are installed: {', '.join(module_names)}"
    ) from last_error


if TYPE_CHECKING:
    import probatio as vol
else:
    vol = _import_first("probatio", "voluptuous")

Invalid = vol.Invalid

__all__ = ["Invalid", "vol"]
