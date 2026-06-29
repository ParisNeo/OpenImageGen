"""Tests for the model adapter registry."""

from __future__ import annotations

import pytest

from openimagegen.models import BaseModelAdapter, get_registry


class _StubAdapter(BaseModelAdapter):
    type_name = "stub_for_registry_tests"

    def load_pipeline(self, model_name, cache_dir, dtype, device):
        return None

    def generate(self, pipeline, prompt, **kwargs):
        return []


class _DuplicateAdapter(BaseModelAdapter):
    type_name = "stub_for_registry_tests"  # Same name to test override.

    def load_pipeline(self, model_name, cache_dir, dtype, device):
        return None

    def generate(self, pipeline, prompt, **kwargs):
        return []


class TestBasics:
    def test_register_returns_class(self):
        reg = get_registry()
        assert reg.register(_StubAdapter) is _StubAdapter

    def test_register_then_try_get(self):
        get_registry().register(_StubAdapter)
        assert get_registry().try_get("stub_for_registry_tests") is _StubAdapter

    def test_get_missing_raises_keyerror(self):
        with pytest.raises(KeyError) as exc:
            get_registry().get("definitely_not_registered_xyz")
        assert "definitely_not_registered_xyz" in str(exc.value)

    def test_try_get_missing_returns_none(self):
        assert get_registry().try_get("definitely_not_registered_xyz") is None

    def test_list_types_sorted(self):
        types = get_registry().list_types()
        assert types == sorted(types)

    def test_register_without_typename_raises(self):
        class BadAdapter(BaseModelAdapter):
            # type_name NOT overridden
            def load_pipeline(self, model_name, cache_dir, dtype, device):
                return None

            def generate(self, pipeline, prompt, **kwargs):
                return []

        with pytest.raises(ValueError, match="non-empty"):
            get_registry().register(BadAdapter)

    def test_register_override_logs_warning(self, caplog):
        get_registry().register(_StubAdapter)
        with caplog.at_level("WARNING", logger="OpenImageGen.models.registry"):
            get_registry().register(_DuplicateAdapter)
        assert any("Overriding" in r.message for r in caplog.records)


class TestSingleton:
    def test_singleton_returns_same_instance(self):
        from openimagegen.models import ModelRegistry

        a = get_registry()
        b = ModelRegistry()
        assert a is b
