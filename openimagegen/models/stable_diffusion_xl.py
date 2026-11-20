"""
@file        stable_diffusion_xl.py
@author      ParisNeo
@description Stable Diffusion XL adapter.
             Implements the adapter interface for Stable Diffusion XL (SDXL) pipelines.
"""

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




    def download_weights(self, model_name: str, cache_dir: str, progress_callback=None) -> None:
            from huggingface_hub import snapshot_download
            from .base import create_progress_tqdm
            logger.info(f"[{self.type_name}] Pre-fetching weights for {model_name}...")
            allow_patterns = [
                "*.json",
                "*.txt",
                "tokenizer/**",
                "tokenizer_2/**",
                "text_encoder/**",
                "text_encoder_2/**",
                "unet/**",
                "vae/**",
                "scheduler/**",
                "feature_extractor/**",
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
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        generator: Optional[Any] = None,
        callback: Optional[Any] = None,
        callback_steps: int = 1,
        **extra: Any,
    ) -> List[Image.Image]:
        from diffusers import StableDiffusionXLImg2ImgPipeline
        img2img_pipe = StableDiffusionXLImg2ImgPipeline(**pipeline.components)
        kwargs = {
            "prompt": prompt,
            "image": init_image,
            "negative_prompt": negative_prompt,
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
        return {"guidance_scale": 7.0, "num_inference_steps": 40, "num_images_per_prompt": 1}