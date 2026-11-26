
"""
@file        packs.py
@author      ParisNeo
@description Additional built-in workflow nodes: prompt composition, seed utilities,
             numeric helpers, JSON path extraction, image transforms using the deferred
             image-reference pattern, and utility nodes for conditional/branching logic.

             Registered in ``openimagegen.nodes`` so the palette and API surface them
             automatically. All nodes return plain, JSON/PIL-safe values that match the
             declared output slots of their NodeSpec.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import random
import re
import uuid
from typing import Any, Dict, List, Optional

from .base import BaseWorkflowNode, NodeExecutionContext, NodeField, NodePort, NodeSpec
from .registry import get_node_registry


# ---------------------------------------------------------------------------
# Text & prompt composition
# ---------------------------------------------------------------------------

@get_node_registry().register
class TextConcatNode(BaseWorkflowNode):
    """Join multiple text values into a single string."""

    spec = NodeSpec(
        type="text_concat",
        title="Concatenate",
        category="Text",
        icon="🔗",
        color="#0891b2",
        description="Join up to four text inputs with a separator.",
        inputs=[
            NodePort(slot="a", label="text A", kind="text"),
            NodePort(slot="b", label="text B", kind="text"),
            NodePort(slot="c", label="text C", kind="text"),
            NodePort(slot="d", label="text D", kind="text"),
        ],
        outputs=[NodePort(slot="text", label="text", kind="text")],
        fields=[
            NodeField(key="separator", label="Separator", type="text", default=", "),
            NodeField(key="skip_empty", label="Skip empty parts", type="boolean", default=True),
        ],
    )

    def execute(self, params, inputs, context):
        separator = str(params.get("separator", ", "))
        skip_empty = bool(params.get("skip_empty", True))
        parts = []
        for slot in ("a", "b", "c", "d"):
            value = inputs.get(slot)
            if value is None:
                continue
            text = str(value)
            if skip_empty and not text.strip():
                continue
            parts.append(text)
        return {"text": separator.join(parts)}


@get_node_registry().register
class TextTemplateNode(BaseWorkflowNode):
    """Fill ``{placeholders}`` from inputs; unknown placeholders remain literal."""

    spec = NodeSpec(
        type="text_template",
        title="Template",
        category="Text",
        icon="📝",
        color="#0891b2",
        description="Fill ``{subject}`` style placeholders from connected text inputs.",
        inputs=[
            NodePort(slot="subject", label="subject", kind="text"),
            NodePort(slot="style", label="style", kind="text"),
            NodePort(slot="extra", label="extra", kind="text"),
        ],
        outputs=[NodePort(slot="text", label="text", kind="text")],
        fields=[
            NodeField(
                key="template",
                label="Template",
                type="textarea",
                default="{subject}, {style}, {extra}",
            ),
        ],
    )

    def execute(self, params, inputs, context):
        template = str(params.get("template", ""))
        mapping = {key: str(value) for key, value in inputs.items() if value is not None}
        return {"text": _SafeFormatDict(mapping).render(template)}


@get_node_registry().register
class TextReplaceNode(BaseWorkflowNode):
    """Literal or case-insensitive find-and-replace on incoming text."""

    spec = NodeSpec(
        type="text_replace",
        title="Find & Replace",
        category="Text",
        icon="🔁",
        color="#0891b2",
        description="Replace all occurrences of one substring with another.",
        inputs=[NodePort(slot="text", label="text", kind="text", required=True)],
        outputs=[NodePort(slot="text", label="text", kind="text")],
        fields=[
            NodeField(key="find", label="Find", type="text", required=True),
            NodeField(key="replace", label="Replace with", type="text", default=""),
            NodeField(key="case_sensitive", label="Case sensitive", type="boolean", default=True),
        ],
    )

    def execute(self, params, inputs, context):
        text = str(inputs["text"])
        needle = str(params.get("find", ""))
        if not needle:
            raise ValueError("Find text must not be empty.")
        replacement = str(params.get("replace", ""))
        if params.get("case_sensitive", True):
            return {"text": text.replace(needle, replacement)}
        return {"text": re.sub(re.escape(needle), replacement, text, flags=re.IGNORECASE)}


@get_node_registry().register
class TextTrimNode(BaseWorkflowNode):
    """Normalize whitespace and optionally collapse commas."""

    spec = NodeSpec(
        type="text_trim",
        title="Clean Text",
        category="Text",
        icon="✨",
        color="#0891b2",
        description="Trim whitespace and optionally collapse repeated separators.",
        inputs=[NodePort(slot="text", label="text", kind="text", required=True)],
        outputs=[NodePort(slot="text", label="text", kind="text")],
        fields=[
            NodeField(key="collapse_commas", label="Collapse repeated commas", type="boolean", default=True),
            NodeField(key="max_length", label="Max characters (0 = unlimited)", type="number", default=0, min=0, step=1),
        ],
    )

    def execute(self, params, inputs, context):
        text = str(inputs["text"]).strip()
        if params.get("collapse_commas", True):
            text = re.sub(r"\s*,\s*", ", ", text)
            text = re.sub(r"(?:,\s*)+", ", ", text).strip(", ")
        max_length = int(params.get("max_length", 0))
        if max_length > 0:
            text = text[:max_length].rstrip()
        return {"text": text}


# ---------------------------------------------------------------------------
# Value & numeric nodes
# ---------------------------------------------------------------------------

@get_node_registry().register
class NumberNode(BaseWorkflowNode):
    """Emit a fixed numeric value."""

    spec = NodeSpec(
        type="number",
        title="Number",
        category="Value",
        icon="🔢",
        color="#64748b",
        outputs=[NodePort(slot="number", label="number", kind="number")],
        fields=[NodeField(key="value", label="Value", type="number", default=0, step=0.01)],
    )

    def execute(self, params, inputs, context):
        return {"number": float(params.get("value", 0))}


@get_node_registry().register
class BooleanNode(BaseWorkflowNode):
    """Emit a fixed boolean value."""

    spec = NodeSpec(
        type="boolean",
        title="Boolean",
        category="Value",
        icon="☑",
        color="#64748b",
        outputs=[NodePort(slot="boolean", label="boolean", kind="boolean")],
        fields=[NodeField(key="value", label="Value", type="boolean", default=False)],
    )

    def execute(self, params, inputs, context):
        return {"boolean": bool(params.get("value", False))}


@get_node_registry().register
class RandomIntNode(BaseWorkflowNode):
    """Emit a uniformly random integer, useful as a seed source."""

    spec = NodeSpec(
        type="random_int",
        title="Random Integer",
        category="Value",
        icon="🎲",
        color="#64748b",
        description="Emit a random integer. Use the 0–2^31−1 range for valid seeds.",
        outputs=[NodePort(slot="number", label="number", kind="number")],
        fields=[
            NodeField(key="min", label="Minimum", type="number", default=0, step=1),
            NodeField(key="max", label="Maximum", type="number", default=2147483646, step=1),
        ],
    )

    def execute(self, params, inputs, context):
        low = int(params.get("min", 0))
        high = int(params.get("max", 2147483646))
        if high < low:
            raise ValueError("Maximum must be >= minimum.")
        return {"number": float(random.randint(low, high))}


@get_node_registry().register
class MathNode(BaseWorkflowNode):
    """Arithmetic over two numbers. Input ports override configured values."""

    spec = NodeSpec(
        type="math",
        title="Math",
        category="Value",
        icon="🧮",
        color="#64748b",
        description="Add, subtract, multiply, divide, min, max, or power two numbers.",
        inputs=[
            NodePort(slot="a", label="a (optional)", kind="number"),
            NodePort(slot="b", label="b (optional)", kind="number"),
        ],
        outputs=[NodePort(slot="number", label="result", kind="number")],
        fields=[
            NodeField(key="a", label="A", type="number", default=0, step=0.01),
            NodeField(key="b", label="B", type="number", default=0, step=0.01),
            NodeField(
                key="operation",
                label="Operation",
                type="select",
                default="add",
                options=[
                    {"value": "add", "label": "A + B"},
                    {"value": "subtract", "label": "A − B"},
                    {"value": "multiply", "label": "A × B"},
                    {"value": "divide", "label": "A ÷ B"},
                    {"value": "min", "label": "min(A, B)"},
                    {"value": "max", "label": "max(A, B)"},
                    {"value": "power", "label": "A ^ B"},
                ],
            ),
        ],
    )

    def execute(self, params, inputs, context):
        a = float(inputs.get("a", params.get("a", 0)))
        b = float(inputs.get("b", params.get("b", 0)))
        operation = params.get("operation", "add")
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Division by zero.")
            result = a / b
        elif operation == "min":
            result = min(a, b)
        elif operation == "max":
            result = max(a, b)
        elif operation == "power":
            result = a ** b
        else:
            raise ValueError(f"Unknown operation '{operation}'.")
        if not math.isfinite(result):
            raise ValueError("Result is not a finite number.")
        return {"number": float(result)}


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

@get_node_registry().register
class ParamsOverrideNode(BaseWorkflowNode):
    """Merge scalar overrides into generation params produced by a Sampler/Img2Img."""

    spec = NodeSpec(
        type="params_override",
        title="Override Params",
        category="Generation",
        icon="🧷",
        color="#db2777",
        description="Merge overrides into generation params, e.g. supply a seed from Random Integer.",
        inputs=[
            NodePort(slot="params", label="generation params", kind="params", required=True),
            NodePort(slot="seed", label="seed (optional)", kind="number"),
            NodePort(slot="steps", label="steps (optional)", kind="number"),
            NodePort(slot="guidance_scale", label="CFG (optional)", kind="number"),
        ],
        outputs=[NodePort(slot="params", label="generation params", kind="params")],
        fields=[
            NodeField(key="seed", label="Seed override", type="number", default=-1, step=1,
                      help="Leave -1 to preserve the incoming seed."),
            NodeField(key="width", label="Width override", type="number", default=0, min=0, max=2048, step=8),
            NodeField(key="height", label="Height override", type="number", default=0, min=0, max=2048, step=8),
            NodeField(key="num_inference_steps", label="Steps override", type="number", default=0, min=0, max=200, step=1),
        ],
    )

    def execute(self, params, inputs, context):
        merged = dict(inputs["params"])
        seed_input = inputs.get("seed")
        step_input = inputs.get("steps")
        cfg_input = inputs.get("guidance_scale")
        if seed_input is not None:
            merged["seed"] = int(seed_input)
        elif int(params.get("seed", -1)) >= 0:
            merged["seed"] = int(params["seed"])
        if step_input is not None:
            merged["num_inference_steps"] = int(step_input)
        elif int(params.get("num_inference_steps", 0)) > 0:
            merged["num_inference_steps"] = int(params["num_inference_steps"])
        if cfg_input is not None:
            merged["guidance_scale"] = float(cfg_input)
        if int(params.get("width", 0)) > 0:
            merged["width"] = int(params["width"])
        if int(params.get("height", 0)) > 0:
            merged["height"] = int(params["height"])
        return {"params": merged}


# ---------------------------------------------------------------------------
# JSON / text extraction
# ---------------------------------------------------------------------------

@get_node_registry().register
class JsonExtractorNode(BaseWorkflowNode):
    """Extract a primitive from JSON text using a dotted path."""

    spec = NodeSpec(
        type="json_extract",
        title="JSON Extract",
        category="Value",
        icon="🧾",
        color="#64748b",
        description="Parse JSON text and expose numeric/text/boolean values via a dotted path.",
        inputs=[NodePort(slot="json", label="json", kind="json", required=True)],
        outputs=[
            NodePort(slot="number", label="number", kind="number"),
            NodePort(slot="text", label="text", kind="text"),
            NodePort(slot="boolean", label="boolean", kind="boolean"),
        ],
        fields=[NodeField(key="path", label="Dot path", type="text", default="seed",
                          placeholder="data.0.seed", help="Use digit segments for array indices.")],
    )

    def execute(self, params, inputs, context):
        raw = inputs["json"]
        document = json.loads(raw) if isinstance(raw, str) else raw
        path = str(params.get("path", "")).strip()
        if not path:
            raise ValueError("JSON path cannot be empty.")
        current = document
        for segment in path.split("."):
            if isinstance(current, list):
                current = current[int(segment)]
            elif isinstance(current, dict):
                current = current[segment]
            else:
                raise ValueError(f"Cannot descend into {type(current).__name__} at '{segment}'.")
        result: Dict[str, Any] = {"text": str(current)}
        if isinstance(current, bool):
            result["boolean"] = current
        else:
            try:
                result["number"] = float(current)
            except (TypeError, ValueError):
                pass
        return result


# ---------------------------------------------------------------------------
# Deferred image transforms (PIL applied after diffusion / upload resolution)
# ---------------------------------------------------------------------------

class _ImageTransformNode(BaseWorkflowNode):
    """Creates a serializable image-reference consumed by later stages."""

    def operation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def execute(self, params, inputs, context):
        image = inputs["image"]
        if isinstance(image, dict):
            source = image.get("source")
            operations = list(image.get("operations", []))
        else:
            source = image
            operations = []
        operations.append(self.operation(params))
        return {"image": {"source": source, "operations": operations}}


@get_node_registry().register
class ImageUpscaleNode(_ImageTransformNode):
    spec = NodeSpec(
        type="image_upscale",
        title="Upscale",
        category="Image",
        icon="🔎",
        color="#8b5cf6",
        description="Deterministic resampling upscale between 1.5–4×. Not AI detail synthesis.",
        inputs=[NodePort(slot="image", label="image", kind="image", required=True)],
        outputs=[NodePort(slot="image", label="upscaled image", kind="image")],
        fields=[
            NodeField(key="scale", label="Scale", type="select", default="2", options=[
                {"value": "1.5", "label": "1.5×"},
                {"value": "2", "label": "2×"},
                {"value": "3", "label": "3×"},
                {"value": "4", "label": "4×"},
            ]),
            NodeField(key="method", label="Resampling", type="select", default="lanczos", options=[
                {"value": "lanczos", "label": "Lanczos"},
                {"value": "bicubic", "label": "Bicubic"},
                {"value": "bilinear", "label": "Bilinear"},
                {"value": "nearest", "label": "Nearest"},
            ]),
        ],
    )

    def operation(self, params):
        return {
            "operation": "upscale",
            "scale": float(params.get("scale", 2)),
            "method": params.get("method", "lanczos"),
        }


@get_node_registry().register
class ImageResizeNode(_ImageTransformNode):
    spec = NodeSpec(
        type="image_resize",
        title="Resize Image",
        category="Image",
        icon="↔",
        color="#8b5cf6",
        description="Resize to exact dimensions.",
        inputs=[NodePort(slot="image", label="image", kind="image", required=True)],
        outputs=[NodePort(slot="image", label="resized image", kind="image")],
        fields=[
            NodeField(key="width", label="Width", type="number", default=1024, min=1, max=8192, step=1),
            NodeField(key="height", label="Height", type="number", default=1024, min=1, max=8192, step=1),
            NodeField(key="method", label="Resampling", type="select", default="lanczos", options=[
                {"value": "lanczos", "label": "Lanczos"},
                {"value": "bicubic", "label": "Bicubic"},
                {"value": "bilinear", "label": "Bilinear"},
                {"value": "nearest", "label": "Nearest"},
            ]),
        ],
    )

    def operation(self, params):
        return {
            "operation": "resize",
            "width": int(params["width"]),
            "height": int(params["height"]),
            "method": params.get("method", "lanczos"),
        }


@get_node_registry().register
class ImageFilterNode(_ImageTransformNode):
    spec = NodeSpec(
        type="image_filter",
        title="Image Filter",
        category="Image",
        icon="✨",
        color="#8b5cf6",
        description="Blur, sharpen, invert, grayscale, or extract edges.",
        inputs=[NodePort(slot="image", label="image", kind="image", required=True)],
        outputs=[NodePort(slot="image", label="filtered image", kind="image")],
        fields=[
            NodeField(key="filter", label="Filter", type="select", default="sharpen", options=[
                {"value": "sharpen", "label": "Sharpen"},
                {"value": "blur", "label": "Gaussian blur"},
                {"value": "edges", "label": "Edges"},
                {"value": "grayscale", "label": "Grayscale"},
                {"value": "invert", "label": "Invert colors"},
            ]),
            NodeField(key="radius", label="Blur/sharpen radius", type="number", default=2, min=0, max=50, step=.5),
        ],
    )

    def operation(self, params):
        return {
            "operation": "filter",
            "filter": params.get("filter", "sharpen"),
            "radius": float(params.get("radius", 2)),
        }


@get_node_registry().register
class ImageCropNode(_ImageTransformNode):
    spec = NodeSpec(
        type="image_crop",
        title="Crop Image",
        category="Image",
        icon="✂",
        color="#8b5cf6",
        description="Crop a rectangle anchored at (x, y).",
        inputs=[NodePort(slot="image", label="image", kind="image", required=True)],
        outputs=[NodePort(slot="image", label="cropped image", kind="image")],
        fields=[
            NodeField(key="x", label="Left", type="number", default=0, min=0, step=1),
            NodeField(key="y", label="Top", type="number", default=0, min=0, step=1),
            NodeField(key="width", label="Width", type="number", default=512, min=1, max=8192, step=1),
            NodeField(key="height", label="Height", type="number", default=512, min=1, max=8192, step=1),
        ],
    )

    def operation(self, params):
        return {
            "operation": "crop",
            "x": int(params.get("x", 0)),
            "y": int(params.get("y", 0)),
            "width": int(params["width"]),
            "height": int(params["height"]),
        }


@get_node_registry().register
class ImageRotateNode(_ImageTransformNode):
    spec = NodeSpec(
        type="image_rotate",
        title="Rotate / Flip",
        category="Image",
        icon="🔄",
        color="#8b5cf6",
        description="Rotate by a right-angle multiple and/or mirror the image.",
        inputs=[NodePort(slot="image", label="image", kind="image", required=True)],
        outputs=[NodePort(slot="image", label="rotated image", kind="image")],
        fields=[
            NodeField(key="angle", label="Right-angle rotation", type="select", default="0", options=[
                {"value": "0", "label": "0°"},
                {"value": "90", "label": "90° clockwise"},
                {"value": "180", "label": "180°"},
                {"value": "270", "label": "270° clockwise"},
            ]),
            NodeField(key="flip", label="Mirror", type="select", default="none", options=[
                {"value": "none", "label": "None"},
                {"value": "horizontal", "label": "Horizontal"},
                {"value": "vertical", "label": "Vertical"},
            ]),
        ],
    )

    def operation(self, params):
        return {
            "operation": "rotate",
            "angle": int(params.get("angle", 0)),
            "flip": params.get("flip", "none"),
        }


@get_node_registry().register
class ImageAdjustNode(_ImageTransformNode):
    spec = NodeSpec(
        type="image_adjust",
        title="Adjust Colors",
        category="Image",
        icon="🎛",
        color="#8b5cf6",
        description="Brightness, contrast, saturation, and sharpness tuned on a 0.0–2.0 scale.",
        inputs=[NodePort(slot="image", label="image", kind="image", required=True)],
        outputs=[NodePort(slot="image", label="adjusted image", kind="image")],
        fields=[
            NodeField(key="brightness", label="Brightness", type="number", default=1.0, min=0, max=2, step=.05),
            NodeField(key="contrast", label="Contrast", type="number", default=1.0, min=0, max=2, step=.05),
            NodeField(key="saturation", label="Saturation", type="number", default=1.0, min=0, max=2, step=.05),
            NodeField(key="sharpness", label="Sharpness", type="number", default=1.0, min=0, max=2, step=.05),
        ],
    )

    def operation(self, params):
        return {
            "operation": "adjust",
            "brightness": float(params.get("brightness", 1.0)),
            "contrast": float(params.get("contrast", 1.0)),
            "saturation": float(params.get("saturation", 1.0)),
            "sharpness": float(params.get("sharpness", 1.0)),
        }


@get_node_registry().register
class ImagePadNode(_ImageTransformNode):
    spec = NodeSpec(
        type="image_pad",
        title="Pad / Border",
        category="Image",
        icon="⬚",
        color="#8b5cf6",
        description="Add a solid-color frame around an image.",
        inputs=[NodePort(slot="image", label="image", kind="image", required=True)],
        outputs=[NodePort(slot="image", label="padded image", kind="image")],
        fields=[
            NodeField(key="padding", label="Padding (px)", type="number", default=16, min=0, max=512, step=1),
            NodeField(key="color", label="Color (#RRGGBB)", type="text", default="#000000"),
        ],
    )

    def operation(self, params):
        return {
            "operation": "pad",
            "padding": int(params.get("padding", 16)),
            "color": str(params.get("color", "#000000")),
        }


# ---------------------------------------------------------------------------
# Utility & control-flow helpers
# ---------------------------------------------------------------------------

@get_node_registry().register
class TextHashNode(BaseWorkflowNode):
    """Stable hexadecimal digest of text."""

    spec = NodeSpec(
        type="text_hash",
        title="Text Hash",
        category="Utility",
        icon="#️⃣",
        color="#64748b",
        description="Stable hexadecimal digest for caching or filename generation.",
        inputs=[NodePort(slot="text", label="text", kind="text", required=True)],
        outputs=[NodePort(slot="hash", label="hash", kind="text")],
        fields=[
            NodeField(key="algorithm", label="Algorithm", type="select", default="sha256", options=[
                {"value": "sha256", "label": "SHA-256"},
                {"value": "sha1", "label": "SHA-1"},
                {"value": "md5", "label": "MD5"},
            ]),
            NodeField(key="length", label="Maximum length (0 = unlimited)", type="number", default=0, min=0, max=128, step=1),
        ],
    )

    def execute(self, params, inputs, context):
        payload = str(inputs["text"]).encode("utf-8")
        algorithm = str(params.get("algorithm", "sha256"))
        if algorithm not in {"sha256", "sha1", "md5"}:
            raise ValueError(f"Unsupported hashing algorithm '{algorithm}'.")
        digest = hashlib.new(algorithm, payload).hexdigest()
        limit = int(params.get("length", 0))
        if limit > 0:
            digest = digest[:limit]
        return {"hash": digest}


@get_node_registry().register
class TextLengthNode(BaseWorkflowNode):
    """Report the length of a text value in characters and tokens (approx)."""

    spec = NodeSpec(
        type="text_length",
        title="Text Length",
        category="Utility",
        icon="📏",
        color="#64748b",
        description="Count characters and approximate tokens in a text value.",
        inputs=[NodePort(slot="text", label="text", kind="text", required=True)],
        outputs=[
            NodePort(slot="characters", label="characters", kind="number"),
            NodePort(slot="words", label="words", kind="number"),
        ],
    )

    def execute(self, params, inputs, context):
        text = str(inputs["text"])
        return {
            "characters": float(len(text)),
            "words": float(len(text.split())),
        }


@get_node_registry().register
class UuidNode(BaseWorkflowNode):
    """Emit a fresh UUID4 as text."""

    spec = NodeSpec(
        type="uuid",
        title="UUID",
        category="Utility",
        icon="🆔",
        color="#64748b",
        description="Generate a random UUID4 string for filenames or run identifiers.",
        outputs=[NodePort(slot="uuid", label="uuid", kind="text")],
    )

    def execute(self, params, inputs, context):
        return {"uuid": str(uuid.uuid4())}


@get_node_registry().register
class Base64EncodeNode(BaseWorkflowNode):
    """Base64-encode UTF-8 text."""

    spec = NodeSpec(
        type="base64_encode",
        title="Base64 Encode",
        category="Utility",
        icon="🔐",
        color="#64748b",
        inputs=[NodePort(slot="text", label="text", kind="text", required=True)],
        outputs=[NodePort(slot="encoded", label="encoded", kind="text")],
    )

    def execute(self, params, inputs, context):
        return {"encoded": base64.b64encode(str(inputs["text"]).encode("utf-8")).decode("ascii")}


def _SafeFormatDict(mapping):
    """Return an object whose ``render`` leaves unknown placeholders intact."""

    class _Renderer:
        def __init__(self, values):
            self.values = values

        def render(self, template: str) -> str:
            class _Fmt(dict):
                def __missing__(inner_self, key):
                    return "{" + key + "}"

            try:
                return template.format_map(_Fmt(self.values))
            except (IndexError, ValueError) as error:
                raise ValueError(f"Template rendering failed: {error}") from error

    return _Renderer(mapping)
