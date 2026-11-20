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




    def download_weights(self, model_name: str, cache_dir: str, progress_callback=None) -> None:
        from huggingface_hub import snapshot_download
        from .base import create_progress_tqdm
        logger.info(f"[{self.type_name}] Pre-fetching weights for {model_name}...")
        allow_patterns = [
            "*.json",
            "*.txt",
            "tokenizer/**",
            "text_encoder/**",
            "text_encoder_2/**",
            "transformer/**",
            "vae/**",
            "scheduler/**",
        ]
        if "variant" in self.config_overrides:
            variant = self.config_overrides["variant"]
            allow_patterns.extend([
                f"*{variant}*.safetensors",
                f"*{variant}*.bin",
            ])
        snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
            tqdm_class=create_progress_tqdm(progress_callback),
        )

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

    def supports_img2img(self) -> bool:
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
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Any] = None,
        callback: Optional[Any] = None,
        callback_steps: int = 1,
        **extra: Any,
    ) -> List[Image.Image]:
        from diffusers import FluxImg2ImgPipeline
        img2img_pipe = FluxImg2ImgPipeline(**pipeline.components)
        kwargs = {
            "prompt": prompt,
            "image": init_image,
            "strength": strength,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images_per_prompt,
            "generator": generator,
            "callback": callback,
            "callback_steps": callback_steps,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return img2img_pipe(**kwargs).images

    def get_default_params(self) -> dict:
        return {"guidance_scale": 0.0, "num_inference_steps": 4, "num_images_per_prompt": 1}

    def supports_negative_prompt(self) -> bool:
        return False