"""Backward-compatibility shim.

This file exists so existing deployments using
``uvicorn openimagegen.main:app`` continue to work.
New code should import from ``openimagegen.app``.
"""

from __future__ import annotations

from .app import app
from .service import ImageGenService

__all__ = ["app", "ImageGenService"]
