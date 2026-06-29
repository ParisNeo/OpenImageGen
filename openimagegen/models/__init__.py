"""Model adapters package.

Importing this package triggers auto-registration of all built-in adapters.
Third-party adapters can register themselves via the `get_registry()` decorator.
"""

from __future__ import annotations

from .base import BaseModelAdapter
from .registry import ModelRegistry, get_registry

# Trigger auto-registration of built-in adapters.
from . import stable_diffusion  # noqa: F401
from . import stable_diffusion_xl  # noqa: F401
from . import kandinsky  # noqa: F401

__all__ = ["BaseModelAdapter", "ModelRegistry", "get_registry"]
