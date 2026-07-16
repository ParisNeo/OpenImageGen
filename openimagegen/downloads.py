"""
@file        downloads.py
@author      ParisNeo
@description Model downloading utilities for OpenImageGen.
             Provides a standalone utility to pre-fetch and cache diffusion model checkpoints
             without instantiating the full generation pipeline in memory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .models import get_registry

logger = logging.getLogger("OpenImageGen.downloads")


def download_model(
    model_name: str,
    model_type: str,
    cache_dir: str,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> bool:
    """Download a model checkpoint into the specified cache directory.
    
    Uses the registered adapter to determine the correct diffusers classes
    and fetches only the checkpoint files without loading them into memory.
    
    Args:
        model_name: The HuggingFace repository ID (e.g., 'runwayml/stable-diffusion-v1-5').
        model_type: The adapter type name (e.g., 'stable_diffusion').
        cache_dir: Local filesystem path to store the model weights.
        config_overrides: Optional dictionary of adapter-level configuration overrides.
        
    Returns:
        True if the download succeeded, False otherwise.
    """
    registry = get_registry()
    adapter_cls = registry.try_get(model_type)
    
    if adapter_cls is None:
        logger.error(f"No adapter registered for type {model_type!r}. Cannot download.")
        return False
        
    overrides = config_overrides or {}
    adapter = adapter_cls(**overrides)
    
    logger.info(f"Starting download for {model_name} (type: {model_type}) into {cache_dir}...")
    
    try:
        adapter.download_weights(
            model_name=model_name,
            cache_dir=cache_dir,
        )
        logger.info(f"Successfully downloaded {model_name}.")
        return True
    except Exception as e:
        logger.error(f"Failed to download {model_name}: {e}", exc_info=True)
        return False