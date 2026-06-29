"""Configuration loading for OpenImageGen.

Handles TOML config discovery across OS-specific paths, environment variables,
CLI overrides, and fallback to a generated default file.
"""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

import toml

logger = logging.getLogger("OpenImageGen.config")

DEFAULT_CONFIG: Dict[str, Any] = {
    "models": {
        "stable_diffusion_1_5": {
            "name": "runwayml/stable-diffusion-v1-5",
            "type": "stable_diffusion",
        },
        "stable_diffusion_xl": {
            "name": "stabilityai/stable-diffusion-xl-base-1.0",
            "type": "stable_diffusion_xl",
        },
    },
    "settings": {
        "default_model": "stable_diffusion_1_5",
        "force_gpu": False,
        "use_gpu": True,
        "dtype": "float16",
        "output_folder": "./outputs",
        "model_cache_dir": "./models",
        "port": 8089,
        "host": "0.0.0.0",
        "file_retention_time": 3600,
        "cleanup_interval_seconds": 900,
    },
    "generation": {
        "guidance_scale": 7.5,
        "num_inference_steps": 50,
        "num_images_per_prompt": 1,
    },
}


def get_config_search_paths() -> List[Path]:
    """Return candidate config file locations based on the host OS."""
    system = platform.system()
    search_paths: List[Path] = []
    cwd = Path.cwd()
    home = Path.home()

    if system == "Linux":
        search_paths.extend([
            Path("/etc/openimagegen/config.toml"),
            Path("/usr/local/etc/openimagegen/config.toml"),
            home / ".config/openimagegen/config.toml",
        ])
    elif system == "Windows":
        appdata = os.getenv("APPDATA")
        if appdata:
            search_paths.append(Path(appdata) / "OpenImageGen/config.toml")
    elif system == "Darwin":
        search_paths.extend([
            home / "Library/Application Support/OpenImageGen/config.toml",
            Path("/usr/local/etc/openimagegen/config.toml"),
        ])

    search_paths.append(cwd / "config.toml")
    return search_paths


def _resolve_config_path(explicit: Optional[str] = None) -> Optional[Path]:
    """Resolve the config path with priority: explicit > env var > OS paths."""
    if explicit:
        path = Path(explicit).resolve()
        if path.exists():
            logger.info(f"Loading config from explicit path: {path}")
            return path
        logger.error(f"Config file specified but not found: {path}")

    env_path = os.getenv("OPENIMAGEGEN_CONFIG_OVERRIDE") or os.getenv("OPENIMAGEGEN_CONFIG")
    if env_path:
        path = Path(env_path).resolve()
        if path.exists():
            logger.info(f"Loading config from environment variable: {path}")
            return path
        logger.warning(f"Config file from env var not found: {path}")

    for path in get_config_search_paths():
        if path.exists():
            logger.info(f"Loading config from standard location: {path}")
            return path

    return None


def _merge_configs(default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `loaded` config on top of `default`."""
    merged: Dict[str, Any] = {}
    for key, value in default.items():
        merged[key] = value.copy() if isinstance(value, dict) else value
    for section, values in loaded.items():
        if section in merged and isinstance(merged[section], dict) and isinstance(values, dict):
            merged[section].update(values)
        else:
            merged[section] = values
    return merged


def load_config(explicit_path: Optional[str] = None, suppress_logs: bool = False) -> Dict[str, Any]:
    """Load configuration from file or fall back to defaults (and create one if missing)."""
    original_level: Optional[int] = None
    if suppress_logs:
        original_level = logger.level
        logger.setLevel(logging.ERROR)

    try:
        config_path = _resolve_config_path(explicit_path)

        if config_path:
            try:
                loaded = toml.load(config_path)
                logger.info(f"Successfully parsed config from {config_path}")
                return _merge_configs(DEFAULT_CONFIG, loaded)
            except Exception as e:
                logger.error(f"Failed to parse config at {config_path}: {e}. Using defaults.")
                return {k: (v.copy() if isinstance(v, dict) else v) for k, v in DEFAULT_CONFIG.items()}

        default_path = Path.cwd() / "config.toml"
        try:
            with open(default_path, "w", encoding="utf-8") as f:
                toml.dump(DEFAULT_CONFIG, f)
            logger.info(f"No config found. Created default at: {default_path}")
        except Exception as e:
            logger.error(f"Failed to write default config to {default_path}: {e}")

        return {k: (v.copy() if isinstance(v, dict) else v) for k, v in DEFAULT_CONFIG.items()}
    finally:
        if original_level is not None and original_level != logging.NOTSET:
            logger.setLevel(original_level)
