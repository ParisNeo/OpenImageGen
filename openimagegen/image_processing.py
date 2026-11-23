
"""
@file        image_processing.py
@author      ParisNeo
@description Dependency-light image transformations for workflow image chains.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

from PIL import Image, ImageFilter, ImageOps

MAX_IMAGE_PIXELS = 64_000_000

_RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def _validated_size(width: int, height: int) -> tuple[int, int]:
    width = max(1, int(width))
    height = max(1, int(height))
    if width > 8192 or height > 8192 or width * height > MAX_IMAGE_PIXELS:
        raise ValueError("Processed image exceeds the 8192px / 64-megapixel safety limit.")
    return width, height


def apply_image_operations(
    image: Image.Image,
    operations: Iterable[Dict[str, Any]],
) -> Image.Image:
    """Apply serializable workflow operations to a copy of an RGB image."""
    result = image.convert("RGB")
    for operation in operations:
        kind = operation.get("operation")
        if kind == "upscale":
            scale = float(operation.get("scale", 2))
            if scale not in {1.5, 2.0, 3.0, 4.0}:
                raise ValueError("Upscale factor must be 1.5, 2, 3, or 4.")
            method = str(operation.get("method", "lanczos"))
            if method not in _RESAMPLING:
                raise ValueError(f"Unknown resampling method '{method}'.")
            size = _validated_size(
                round(result.width * scale),
                round(result.height * scale),
            )
            result = result.resize(size, _RESAMPLING[method])
        elif kind == "resize":
            size = _validated_size(
                int(operation["width"]),
                int(operation["height"]),
            )
            method = str(operation.get("method", "lanczos"))
            if method not in _RESAMPLING:
                raise ValueError(f"Unknown resampling method '{method}'.")
            result = result.resize(size, _RESAMPLING[method])
        elif kind == "composite_rgba":
            layers = operation.get("layers", [])
            if not layers:
                raise ValueError("Layered Qwen output did not include RGBA layers.")
            from base64 import b64decode
            from io import BytesIO
            composite = Image.new("RGBA", result.size, (255, 255, 255, 255))
            for encoded in layers:
                with Image.open(BytesIO(b64decode(encoded, validate=True))) as layer:
                    composite.alpha_composite(layer.convert("RGBA"))
            result = composite.convert("RGB")
        elif kind == "filter":
            filter_name = str(operation.get("filter", "blur"))
            if filter_name == "blur":
                radius = max(0.0, min(50.0, float(operation.get("radius", 2))))
                result = result.filter(ImageFilter.GaussianBlur(radius))
            elif filter_name == "sharpen":
                result = result.filter(ImageFilter.UnsharpMask(
                    radius=max(0.1, min(50.0, float(operation.get("radius", 2)))),
                    percent=150,
                    threshold=3,
                ))
            elif filter_name == "invert":
                result = ImageOps.invert(result)
            elif filter_name == "grayscale":
                result = ImageOps.grayscale(result).convert("RGB")
            elif filter_name == "edges":
                result = ImageOps.grayscale(result).filter(ImageFilter.FIND_EDGES).convert("RGB")
            else:
                raise ValueError(f"Unknown image filter '{filter_name}'.")
        else:
            raise ValueError(f"Unknown image operation '{kind}'.")
    return result
