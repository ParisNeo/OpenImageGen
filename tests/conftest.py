"""
@file        conftest.py
@author      ParisNeo
@description Shared pytest fixtures for OpenImageGen tests.
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
def _isolate_registry():  # pyright: ignore[reportUnusedFunction]
    """Snapshot and restore the model registry around every test.

    Because the registry is a process-wide singleton shared across the
    entire test session, any test that registers a throwaway adapter
    (e.g. a mock backend) would permanently pollute the registry and
    cause cascading failures in subsequent tests.

    This fixture is ``autouse=True``, so pytest automatically wraps
    **every** test with it — no test needs to declare it as a
    dependency.  The lifecycle is:

    1. **Setup** — acquire the global registry via ``get_registry()``
       and take a shallow copy of its internal ``_adapters`` dict.
    2. **Yield** — the test runs and may freely register or remove
       adapters.
    3. **Teardown** — clear all adapters and restore the snapshot,
       guaranteeing the registry is in its pre-test state for the next
       test.

    The ``_adapters`` attribute is private (prefixed with an
    underscore) and therefore not exposed in the public type stubs,
    which is why each access carries a ``type: ignore`` comment.
    """
    from openimagegen.models import get_registry

    registry = get_registry()
    snapshot = dict(registry._adapters)  # type: ignore[attr-defined]
    yield
    registry._adapters.clear()  # type: ignore[attr-defined]
    registry._adapters.update(snapshot)  # type: ignore[attr-defined]


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Provide an isolated, per-test output directory.

    Wraps pytest's built-in ``tmp_path`` fixture to create a dedicated
    ``outputs`` subdirectory. This ensures each test gets a clean,
    filesystem-backed location for generated images and artifacts,
    which is automatically cleaned up by pytest at the end of the
    test session. Downstream fixtures (e.g. ``minimal_config``) depend
    on this for their ``output_folder`` and ``model_cache_dir`` paths.
    """
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