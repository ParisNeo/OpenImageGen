"""
@file        base.py
@author      ParisNeo
@description Base adapter class for pluggable diffusion models.
Defines the abstract interface for model loading, generation, and weight downloading.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, cast

from PIL import Image

logger = logging.getLogger("OpenImageGen.models.base")


def create_progress_tqdm(callback):
    if callback is None:
        return None
    try:
        from tqdm import tqdm
    except ImportError:
        return None

    class _ReportingTqdm(tqdm):
        def update(self, n=1):
            super().update(n)
            if self.total:
                pct = min(100, int(100 * self.n / self.total))
                callback(pct)
    return _ReportingTqdm


class BaseModelAdapter(ABC):
    """Abstract base for diffusion-model adapters."""

    type_name: str = ""

    def __init__(self, **config_overrides: Any) -> None:
        self.config_overrides = config_overrides

    @abstractmethod
    def load_pipeline(self, model_name: str, cache_dir: str, dtype: Any, device: Any) -> Any:
        ...

    @abstractmethod
    def generate(
        self,
        pipeline: Any,
        prompt: str,
        negative_prompt: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        generator: Optional[Any] = None,
        callback: Optional[Any] = None,
        callback_steps: int = 1,
        **extra: Any,
    ) -> List[Image.Image]:
        ...

    def download_weights(self, model_name: str, cache_dir: str, progress_callback=None) -> None:
        """Pre-download model weights without loading the pipeline into memory.

        By default, this attempts to use `diffusers` internal snapshot download
        mechanisms. Adapters can override this to fetch specific sub-components.
        """
        from huggingface_hub import snapshot_download
        logger.info(f"[{self.type_name}] Fetching weights for {model_name} via huggingface_hub...")
        snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            tqdm_class=cast(Any, create_progress_tqdm(progress_callback)),
        )  # type: ignore[call-overload,arg-type]

    def apply_optimizations(self, pipeline: Any, device: Any) -> Any:
        if device.type == "cuda":
            if hasattr(pipeline, "enable_model_cpu_offload"):
                try:
                    pipeline.enable_model_cpu_offload()
                    logger.info(f"[{self.type_name}] Enabled model CPU offload.")
                    return pipeline
                except Exception as e:
                    logger.warning(f"[{self.type_name}] CPU offload failed: {e}")
            pipeline.to(device)
        else:
            pipeline.to(device)
        return pipeline

    def supports_param(self, param_name: str) -> bool:
        return param_name in {
            "prompt", "negative_prompt", "height", "width",
            "num_inference_steps", "guidance_scale",
            "num_images_per_prompt", "generator", "callback", "callback_steps",
        }

    def get_default_params(self) -> Dict[str, Any]:
        return {"guidance_scale": 7.5, "num_inference_steps": 50, "num_images_per_prompt": 1}

    def supports_negative_prompt(self) -> bool:
        """Return True if this adapter's pipeline natively supports negative prompts."""
        return True

    def supports_img2img(self) -> bool:
        """Return True if this adapter supports image-to-image generation."""
        return False

    def supports_lora(self) -> bool:
        """Whether the underlying diffusers pipeline implements LoRA hooks."""
        return True

    def generate_img2img(
        self,
        pipeline: Any,
        prompt: str,
        init_image: Image.Image,
        negative_prompt: Optional[str] = None,
        strength: float = 0.75,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        generator: Optional[Any] = None,
        callback: Optional[Any] = None,
        callback_steps: int = 1,
        **extra: Any,
    ) -> List[Image.Image]:
        """Generate images from an init image. Override in subclasses that support img2img."""
        raise NotImplementedError(f"{self.type_name} does not support img2img generation.")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} type={self.type_name!r}>"