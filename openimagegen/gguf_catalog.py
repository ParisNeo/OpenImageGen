
"""
@file        gguf_catalog.py
@author      ParisNeo
@description Curated, directly downloadable GGUF transformer presets.
"""

from __future__ import annotations

from typing import Any, Dict

GGUF_PRESETS: Dict[str, Dict[str, Any]] = {
    "flux-schnell-q2-k": {
        "id": "flux-schnell-q2-k",
        "model_key": "flux_schnell_gguf_q2_k",
        "title": "FLUX.1-schnell · Q2_K",
        "description": "Smallest practical GGUF test preset. Lower quality, four-step generation.",
        "base_repository": "Comfy-Org/FLUX.1-schnell",
        "base_repositories": ["black-forest-labs/FLUX.1-schnell"],
        "gguf_repository": "city96/FLUX.1-schnell-gguf",
        "gguf_filename": "flux1-schnell-Q2_K.gguf",
        "quantization": "Q2_K",
        "transformer_size_gb": 4.0,
        "recommended": True,
        "default_width": 512,
        "default_height": 512,
        "default_steps": 4,
        "default_guidance": 0.0,
    },
    "flux-schnell-q4-k-s": {
        "id": "flux-schnell-q4-k-s",
        "model_key": "flux_schnell_gguf_q4_k_s",
        "title": "FLUX.1-schnell · Q4_K_S",
        "description": "Balanced quality and size. A better default when disk space allows.",
        "base_repository": "black-forest-labs/FLUX.1-schnell",
        "gguf_repository": "city96/FLUX.1-schnell-gguf",
        "gguf_filename": "flux1-schnell-Q4_K_S.gguf",
        "quantization": "Q4_K_S",
        "transformer_size_gb": 6.8,
        "recommended": False,
        "default_width": 768,
        "default_height": 768,
        "default_steps": 4,
        "default_guidance": 0.0,
    },
    "flux-schnell-q8-0": {
        "id": "flux-schnell-q8-0",
        "model_key": "flux_schnell_gguf_q8_0",
        "title": "FLUX.1-schnell · Q8_0",
        "description": "Higher-quality quantization for quality comparisons and capable systems.",
        "base_repository": "black-forest-labs/FLUX.1-schnell",
        "gguf_repository": "city96/FLUX.1-schnell-gguf",
        "gguf_filename": "flux1-schnell-Q8_0.gguf",
        "quantization": "Q8_0",
        "transformer_size_gb": 12.7,
        "recommended": False,
        "default_width": 1024,
        "default_height": 1024,
        "default_steps": 4,
        "default_guidance": 0.0,
    },
}


def build_gguf_model_config(preset_id: str) -> Dict[str, Any]:
    """Translate a catalog preset into the adapter's persisted configuration."""
    preset = GGUF_PRESETS[preset_id]
    return {
        "name": preset["base_repository"],
        "type": "flux_gguf",
        "base_repositories": preset["base_repositories"],
        "gguf_source": preset["gguf_repository"],
        "gguf_filename": preset["gguf_filename"],
        "label": preset["title"],
        "quantization": preset["quantization"],
        "catalog_preset": preset_id,
    }
