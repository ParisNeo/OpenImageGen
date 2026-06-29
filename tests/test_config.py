"""Tests for openimagegen.config (no torch/diffusers dependency)."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from openimagegen import config as config_mod
from openimagegen.config import (
    DEFAULT_CONFIG, _merge_configs, get_config_search_paths, load_config,
)


class TestDefaults:
    def test_required_sections_present(self):
        assert "models" in DEFAULT_CONFIG
        assert "settings" in DEFAULT_CONFIG
        assert "generation" in DEFAULT_CONFIG

    def test_settings_has_core_keys(self):
        s = DEFAULT_CONFIG["settings"]
        for key in ("default_model", "force_gpu", "use_gpu", "dtype",
                    "output_folder", "model_cache_dir", "port", "host",
                    "file_retention_time"):
            assert key in s, f"Missing settings key: {key}"

    def test_default_models_have_name_and_type(self):
        for key, info in DEFAULT_CONFIG["models"].items():
            assert "name" in info, f"Model {key} missing 'name'"
            assert "type" in info, f"Model {key} missing 'type'"


class TestMergeConfigs:
    def test_loaded_overrides_defaults(self):
        merged = _merge_configs(DEFAULT_CONFIG, {"settings": {"port": 9999}})
        assert merged["settings"]["port"] == 9999
        assert merged["settings"]["host"] == "0.0.0.0"

    def test_new_section_added(self):
        merged = _merge_configs(DEFAULT_CONFIG, {"custom": {"foo": "bar"}})
        assert merged["custom"] == {"foo": "bar"}


class TestSearchPaths:
    def test_returns_list_of_paths(self):
        paths = get_config_search_paths()
        assert isinstance(paths, list)
        assert all(isinstance(p, Path) for p in paths)

    def test_cwd_always_last(self):
        assert get_config_search_paths()[-1] == Path.cwd() / "config.toml"


class TestLoadConfig:
    def test_explicit_path_loads_overrides(self, tmp_path: Path):
        cfg_file = tmp_path / "my.toml"
        cfg_file.write_text('[settings]\nport = 12345\n')
        result = load_config(explicit_path=str(cfg_file))
        assert result["settings"]["port"] == 12345
        assert "models" in result

    def test_invalid_toml_falls_back_to_defaults(self, tmp_path: Path):
        bad = tmp_path / "bad.toml"
        bad.write_text("this is = not valid toml [[[")
        result = load_config(explicit_path=str(bad))
        assert result["settings"]["port"] == 8089

    def test_suppress_logs_sets_error_level(self, tmp_path: Path):
        cfg_file = tmp_path / "ok.toml"
        cfg_file.write_text('[settings]\nport = 9999\n')
        try:
            load_config(explicit_path=str(cfg_file), suppress_logs=True)
            assert config_mod.logger.level == logging.ERROR
        finally:
            config_mod.logger.setLevel(logging.NOTSET)
