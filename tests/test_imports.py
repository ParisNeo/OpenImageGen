"""Smoke tests: package imports and built-in adapter auto-registration."""

from __future__ import annotations


def test_package_metadata_exposed():
    from openimagegen import __version__, __author__, __license__

    assert __version__ == "0.2.0"
    assert __author__ == "ParisNeo"
    assert __license__ == "Apache-2.0"


def test_models_package_imports_cleanly():
    from openimagegen.models import BaseModelAdapter, ModelRegistry, get_registry

    assert BaseModelAdapter is not None
    assert ModelRegistry is not None
    assert get_registry() is not None


def test_builtin_adapters_auto_registered():
    from openimagegen.models import get_registry

    types = get_registry().list_types()
    assert "stable_diffusion" in types
    assert "stable_diffusion_xl" in types
    assert "kandinsky" in types


def test_registry_is_singleton():
    from openimagegen.models import get_registry, ModelRegistry

    assert get_registry() is ModelRegistry()


def test_app_module_loads_without_lifespan():
    """Importing app.py must not crash even though `service` is None."""
    from openimagegen import app

    assert app.app is not None
    assert app.app.title == "OpenImageGen API"
    assert app.service is None  # Populated only during lifespan.


def test_service_class_is_importable():
    from openimagegen.service import ImageGenService
    from openimagegen.jobs import JobStore, InMemoryJobStore
    from openimagegen.schemas import (
        ImageGenerationRequest, JobStatus, JobSubmissionResponse,
    )

    assert ImageGenService is not None
    assert JobStore is not None
    assert InMemoryJobStore is not None
    assert ImageGenerationRequest is not None
    assert JobStatus is not None
    assert JobSubmissionResponse is not None
