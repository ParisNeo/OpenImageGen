"""
@file        flux.py
@author      ParisNeo
@description Flux adapter (Black Forest Labs FLUX.1-schnell / "klein").
             FLUX.1-schnell is the distilled, fast variant of the Flux model family.
             It is optimised for ~4 inference steps and does not use classifier-free
             guidance (guidance_scale=0.0).  Negative prompts are not supported by the
             schnell pipeline and are silently dropped.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from PIL import Image

from .base import BaseModelAdapter
from .registry import get_registry

logger = logging.getLogger("OpenImageGen.models.flux")


@get_registry().register
class FluxAdapter(BaseModelAdapter):
    """Adapter for Black Forest Labs FLUX.1 text-to-image pipelines.

    Automatically registered into the global model registry via the
    @get_registry().register decorator at import time.
    """

    type_name = "flux"

    def load_pipeline(self, model_name: str, cache_dir: str, dtype: Any, device: Any) -> Any:
        from diffusers import FluxPipeline
        logger.info(f"Loading Flux pipeline: {model_name}")
        load_kwargs: dict = {"torch_dtype": dtype, "cache_dir": cache_dir}
        if "variant" in self.config_overrides:
            load_kwargs["variant"] = self.config_overrides["variant"]
        pipeline = FluxPipeline.from_pretrained(model_name, **load_kwargs)
        return self.apply_optimizations(pipeline, device)

    def download_weights(self, model_name: str, cache_dir: str) -> None:
        from diffusers import FluxPipeline
        logger.info(f"[{self.type_name}] Pre-fetching weights for {model_name}...")
        load_kwargs: dict = {"cache_dir": cache_dir}
        if "variant" in self.config_overrides:
            load_kwargs["variant"] = self.config_overrides["variant"]
        FluxPipeline.from_pretrained(model_name, **load_kwargs)

    def generate(
        self,
        pipeline: Any,
        prompt: str,
        negative_prompt: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Any] = None,
        callback: Optional[Any] = None,
        callback_steps: int = 1,
        **extra: Any,
    ) -> List[Image.Image]:
        kwargs = {
            "prompt": prompt,
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
        return {"guidance_scale": 0.0, "num_inference_steps": 4, "num_images_per_prompt": 1}