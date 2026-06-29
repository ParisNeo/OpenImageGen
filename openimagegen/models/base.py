"""Base adapter class for pluggable diffusion models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger("OpenImageGen.models.base")


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

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} type={self.type_name!r}>"
