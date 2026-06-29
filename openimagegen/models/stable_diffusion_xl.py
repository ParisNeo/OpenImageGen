"""Stable Diffusion XL adapter."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from PIL import Image

from .base import BaseModelAdapter
from .registry import get_registry

logger = logging.getLogger("OpenImageGen.models.stable_diffusion_xl")


@get_registry().register
class StableDiffusionXLAdapter(BaseModelAdapter):
    """Adapter for Stable Diffusion XL pipelines."""

    type_name = "stable_diffusion_xl"

    def load_pipeline(self, model_name: str, cache_dir: str, dtype: Any, device: Any) -> Any:
        from diffusers import StableDiffusionXLPipeline
        logger.info(f"Loading SDXL pipeline: {model_name}")
        load_kwargs: dict = {"torch_dtype": dtype, "cache_dir": cache_dir}
        if "variant" in self.config_overrides:
            load_kwargs["variant"] = self.config_overrides["variant"]
        pipeline = StableDiffusionXLPipeline.from_pretrained(model_name, **load_kwargs)
        return self.apply_optimizations(pipeline, device)

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
        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images_per_prompt,
            "generator": generator,
            "callback": callback,
            "callback_steps": callback_steps,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return pipeline(**kwargs).images

    def get_default_params(self) -> dict:
        return {"guidance_scale": 7.0, "num_inference_steps": 40, "num_images_per_prompt": 1}
