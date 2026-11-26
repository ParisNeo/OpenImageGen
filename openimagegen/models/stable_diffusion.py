"""
@file        stable_diffusion.py
@author      ParisNeo
@description Stable Diffusion 1.x adapter.
             Implements the adapter interface for Stable Diffusion 1.x text-to-image pipelines.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from PIL import Image

from .base import BaseModelAdapter
from .registry import get_registry

logger = logging.getLogger("OpenImageGen.models.stable_diffusion")


@get_registry().register
class StableDiffusionAdapter(BaseModelAdapter):
    """Adapter for Stable Diffusion 1.x text-to-image pipelines."""

    type_name = "stable_diffusion"

    def load_pipeline(self, model_name: str, cache_dir: str, dtype: Any, device: Any) -> Any:
        from pathlib import Path
        from diffusers import StableDiffusionPipeline

        logger.info(f"Loading Stable Diffusion pipeline: {model_name}")
        local_path = Path(model_name).expanduser()
        if local_path.is_file() and local_path.suffix.lower() in {".safetensors", ".ckpt"}:
            kwargs = dict(self.config_overrides)
            kwargs.pop("custom_pipeline", None)
            kwargs.pop("weight_name", None)
            pipeline = StableDiffusionPipeline.from_single_file(
                str(local_path),
                torch_dtype=dtype,
                cache_dir=cache_dir,
                local_files_only=True,
                **kwargs,
            )
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
                cache_dir=cache_dir,
                **self.config_overrides,
            )
        return self.apply_optimizations(pipeline, device)




    def download_weights(self, model_name: str, cache_dir: str, progress_callback=None) -> None:
            from huggingface_hub import snapshot_download
            from .base import create_progress_tqdm
            logger.info(f"[{self.type_name}] Pre-fetching weights for {model_name}...")
            snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                allow_patterns=[
                    "*.json",
                    "*.txt",
                    "tokenizer/**",
                    "text_encoder/**",
                    "unet/**",
                    "vae/**",
                    "scheduler/**",
                    "safety_checker/**",
                    "feature_extractor/**",
                ],
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

    def supports_inpainting(self) -> bool:
        return True

    def generate_inpainting(
        self,
        pipeline: Any,
        prompt: str,
        init_image: Image.Image,
        mask_image: Image.Image,
        negative_prompt: Optional[str] = None,
        strength: float = 0.85,
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
        from diffusers import StableDiffusionInpaintPipeline
        # SD 1.5 ships with a 9-channel inpainting UNet (sd-v1-5-inpainting) that
        # shares weights incompatibly with the standard UNet. When the loaded
        # checkpoint does not include inpaint UNet weights, Diffusers still lets
        # us construct the inpaint pipeline from the shared components, at the
        # cost of lower quality because the mask is only enforced through the
        # VAE-encoded latent space. Documented behaviour, not silent fallback.
        inpaint_pipe = StableDiffusionInpaintPipeline(**pipeline.components)
        kwargs = {
            "prompt": prompt,
            "image": init_image,
            "mask_image": mask_image,
            "negative_prompt": negative_prompt,
            "strength": strength,
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
        return inpaint_pipe(**kwargs).images

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
        from diffusers import StableDiffusionImg2ImgPipeline
        img2img_pipe = StableDiffusionImg2ImgPipeline(**pipeline.components)
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
        model_name = str(self.config_overrides.get("repository", "")).lower()
        if "turbo" in model_name:
            return {"guidance_scale": 0.0, "num_inference_steps": 2, "num_images_per_prompt": 1}
        return {"guidance_scale": 7.0, "num_inference_steps": 30, "num_images_per_prompt": 1}