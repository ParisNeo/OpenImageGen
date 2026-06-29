"""Shared pytest fixtures for OpenImageGen tests.

This conftest is deliberately dependency-free: it only uses pytest, the
standard library, and OpenImageGen modules themselves. Tests that need
torch / diffusers / httpx must use ``pytest.importorskip`` at the top of
the test module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the project root is importable regardless of where pytest is invoked.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Snapshot and restore the model registry around each test.

    The registry is a singleton; without isolation, tests that register
    throwaway adapters would leak state to subsequent tests.
    """
    from openimagegen.models import get_registry

    registry = get_registry()
    snapshot = dict(registry._adapters)  # type: ignore[attr-defined]
    yield
    registry._adapters.clear()  # type: ignore[attr-defined]
    registry._adapters.update(snapshot)  # type: ignore[attr-defined]


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Provide an isolated output directory for a single test."""
    out = tmp_path / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture
def minimal_config(temp_output_dir: Path) -> dict:
    """A minimal in-memory config dict for service-level tests."""
    return {
        "models": {
            "sd15": {"name": "runwayml/stable-diffusion-v1-5", "type": "stable_diffusion"},
        },
        "settings": {
            "default_model": "sd15",
            "force_gpu": False,
            "use_gpu": False,
            "dtype": "float16",
            "output_folder": str(temp_output_dir),
            "model_cache_dir": str(temp_output_dir / "cache"),
            "port": 8089,
            "host": "127.0.0.1",
            "file_retention_time": 3600,
            "cleanup_interval_seconds": 900,
        },
        "generation": {
            "guidance_scale": 7.5,
            "num_inference_steps": 30,
            "num_images_per_prompt": 1,
        },
    }
