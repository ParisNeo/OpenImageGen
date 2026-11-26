"""
@file        community_pack.py
@author      ParisNeo
@description Additional built-in workflow nodes: text utilities, JSON, math, seed,
             prompt builders, image blending, and metadata. These complement the
             core nodes in builtin.py without modifying that module.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, List

from .base import BaseWorkflowNode, NodeExecutionContext, NodeField, NodePort, NodeSpec
from .registry import get_node_registry


@get_node_registry().register
class TextConcatNode(BaseWorkflowNode):
    """Join up to four text inputs with a configurable separator."""

    spec = NodeSpec(
        type="text_concat",
        title="Concatenate Text",
        category="Text",
        icon="🔗",
        color="#0ea5e9",
        description="Combine multiple text values with an optional separator.",
        inputs=[
            NodePort(slot="a", label="text A", kind="text"),
            NodePort(slot="b", label="text B", kind="text"),
            NodePort(slot="c", label="text C", kind="text"),
            NodePort(slot="d", label="text D", kind="text"),
        ],
        outputs=[NodePort(slot="text", label="text", kind="text")],
        fields=[
            NodeField(key="text", label="Fallback / inline text", type="textarea", default=""),
            NodeField(key="separator", label="Separator", type="text", default=", "),
            NodeField(key="skip_empty", label="Skip empty parts", type="boolean", default=True),
        ],
    )

    def execute(self, params: Dict[str, Any], inputs: Dict[str, Any], context: NodeExecutionContext) -> Dict[str, Any]:
        parts: List[str] = []
        for slot in ("a", "b", "c", "d"):
            value = inputs.get(slot)
            if value is None:
                continue
            text = str(value)
            if params.get("skip_empty", True) and not text.strip():
                continue
            parts.append(text)
        inline = str(params.get("text", "") or "")
        if inline:
            parts.append(inline)
        separator = str(params.get("separator", ", "))
        return {"text": separator.join(parts)}


@get_node_registry().register
class TextReplaceNode(BaseWorkflowNode):
    """Case-insensitive or literal string replacement on incoming text."""

    spec = NodeSpec(
        type="text_replace",
        title="Replace Text",
        category="Text",
        icon="🔁",
        color="#0ea5e9",
        description="Find and replace substrings. Useful for removing forbidden terms.",
        inputs=[NodePort(slot="text", label="text", kind="text")],
        outputs=[NodePort(slot="text", label="text", kind="text")],
        fields=[
            NodeField(key="find", label="Find", type="text", required=True, default=""),
            NodeField(key="replace", label="Replace with", type="text", default=""),
            NodeField(key="case_sensitive", label="Case sensitive", type="boolean", default=False),
        ],
    )

    def execute(self, params: Dict[str, Any], inputs: Dict[str, Any], context: NodeExecutionContext) -> Dict[str, Any]:
        text = str(inputs.get("text") or "")
        find = str(params.get("find", ""))
        if not find:
            raise ValueError("The 'find' parameter must not be empty.")
        replace = str(params.get("replace", ""))
        if params.get("case_sensitive", False):
            return {"text": text.replace(find, replace)}
        lowered = text.lower()
        target = find.lower()
        if target not in lowered:
            return {"text": text}
        # Rebuild without re-matching the insertion (avoids infinite expansion).
        out: List[str] = []
        cursor = 0
        while True:
            idx = lowered.find(target, cursor)
            if idx < 0:
                out.append(text[cursor:])
                break
            out.append(text[cursor:idx])
            out.append(replace)
            cursor = idx + len(find)
        return {"text": "".join(out)}


@get_node_registry().register
class TextTrimNode(BaseWorkflowNode):
    """Trim whitespace and optionally collapse runs of spaces."""

    spec = NodeSpec(
        type="text_trim",
        title="Trim Text",
        category="Text",
        icon="✂️",
        color="#0ea5e9",
        description="Strip leading/trailing whitespace and optionally normalize spacing.",
        inputs=[NodePort(slot="text", label="text", kind="text")],
        outputs=[NodePort(slot="text", label="text", kind="text")],
        fields=[NodeField(key="collapse", label="Collapse whitespace", type="boolean", default=True)],
    )

    def execute(self, params: Dict[str, Any], inputs: Dict[str, Any], context: NodeExecutionContext) -> Dict[str, Any]:
        text = str(inputs.get("text") or "").strip()
        if params.get("collapse", True):
            text = " ".join(part for part in text.split() if part)
        return {"text": text}


@get_node_registry().register
class TextTemplateNode(BaseWorkflowNode):
    """Render a prompt template with ``{key}`` placeholders filled from inputs."""

    spec = NodeSpec(
        type="text_template",
        title="Text Template",
        category="Text",
        icon="📝",
        color="#0ea5e9",
        description="Fill placeholders such as {subject} and {style} from input ports.",
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
                default="a {subject}, {style}, {extra}",
            ),
        ],
    )

    def execute(self, params: Dict[str, Any], inputs: Dict[str, Any], context: NodeExecutionContext) -> Dict[str, Any]:
        template = str(params.get("template", ""))
        if not template:
            return {"text": ""}
        mapping = {key: str(value or "") for key, value in inputs.items()}
        try:
            rendered = template.format_map(_SafeFormatDict(mapping))
        except (KeyError, ValueError) as error:
            raise ValueError(f"Template rendering failed: {error}") from error
        return {"text": rendered}


class _SafeFormatDict(dict):
    """``format_map`` helper that leaves unknown placeholders untouched."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@get_node_registry().register
class SeedNode(BaseWorkflowNode):
    """Produce a deterministic or random integer seed for the downstream sampler."""

    spec = NodeSpec(
        type="seed",
        title="Seed",
        category="Parameters",
        icon="🎲",
        color="#a855f7",
        description="Generate a random seed or pin a fixed one for reproducible runs.",
        outputs=[NodePort(slot="seed", label="seed", kind="number")],
        fields=[
            NodeField(
                key="mode",
                label="Mode",
                type="select",
                default="random",
                options=[
                    {"value": "random", "label": "Random each run"},
                    {"value": "fixed", "label": "Fixed value"},
                ],
            ),
            NodeField(key="value", label="Fixed seed", type="number", default=42, min=0, max=2147483647, step=1),
        ],
    )

    def execute(self, params: Dict[str, Any], inputs: Dict[str, Any], context: NodeExecutionContext) -> Dict[str, Any]:
        if params.get("mode", "random") == "fixed":
            return {"seed": int(params.get("value", 42))}
        return {"seed": random.randint(0, 2147483647)}


@get_node_registry().register
class JsonParseNode(BaseWorkflowNode):
    """Parse a JSON string into a text field or validate a response."""

    spec = NodeSpec(
        type="json_parse",
        title="Parse JSON",
        category="Utility",
        icon="{}",
        color="#64748b",
        description="Parse JSON text and expose a field by dotted path.",
        inputs=[NodePort(slot="text", label="json text", kind="text")],
        outputs=[
            NodePort(slot="value", label="value", kind="any"),
            NodePort(slot="raw", label="raw text", kind="text"),
        ],
        fields=[
            NodeField(key="path", label="Dot path", type="text", default="", placeholder="choices.0.message.content"),
        ],
    )

    def execute(self, params: Dict[str, Any], inputs: Dict[str, Any], context: NodeExecutionContext) -> Dict[str, Any]:
        text = str(inputs.get("text") or "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON: {error.msg} at position {error.pos}.") from error
        path = str(params.get("path", "")).strip()
        value: Any = data
        if path:
            for part in path.split("."):
                if isinstance(value, list):
                    try:
                        value = value[int(part)]
                    except (ValueError, IndexError) as error:
                        raise ValueError(f"JSON path '{path}' failed at '{part}': {error}") from error
                elif isinstance(value, dict):
                    if part not in value:
                        raise ValueError(f"JSON path '{path}' has no key '{part}'.")
                    value = value[part]
                else:
                    raise ValueError(f"JSON path '{path}' cannot descend into {type(value).__name__}.")
        return {"value": value, "raw": text}


@get_node_registry().register
class MathNode(BaseWorkflowNode):
    """Evaluate a small arithmetic expression over numeric inputs (no eval)."""

    spec = NodeSpec(
        type="math",
        title="Math",
        category="Utility",
        icon="🧮",
        color="#64748b",
        description="Add, subtract, multiply, divide, or clamp numeric values.",
        inputs=[
            NodePort(slot="a", label="a", kind="number"),
            NodePort(slot="b", label="b", kind="number"),
        ],
        outputs=[NodePort(slot="value", label="value", kind="number")],
        fields=[
            NodeField(
                key="operation",
                label="Operation",
                type="select",
                default="add",
                options=[
                    {"value": "add", "label": "a + b"},
                    {"value": "subtract", "label": "a - b"},
                    {"value": "multiply", "label": "a × b"},
                    {"value": "divide", "label": "a ÷ b"},
                    {"value": "min", "label": "min(a, b)"},
                    {"value": "max", "label": "max(a, b)"},
                ],
            ),
            NodeField(key="a", label="Fallback a", type="number", default=0),
            NodeField(key="b", label="Fallback b", type="number", default=0),
        ],
    )

    def execute(self, params: Dict[str, Any], inputs: Dict[str, Any], context: NodeExecutionContext) -> Dict[str, Any]:
        a = float(inputs.get("a", params.get("a", 0)))
        b = float(inputs.get("b", params.get("b", 0)))
        operation = str(params.get("operation", "add"))
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
        else:
            raise ValueError(f"Unsupported operation '{operation}'.")
        return {"value": int(result) if result == int(result) else result}


@get_node_registry().register
class ImageBlendNode(BaseWorkflowNode):
    """Blend two existing images using one of the standard composite modes."""

    spec = NodeSpec(
        type="image_blend",
        title="Blend Images",
        category="Image processing",
        icon="🌗",
        color="#ec4899",
        description="Combine two images at a chosen alpha or using a blend mode.",
        inputs=[
            NodePort(slot="base", label="base", kind="image"),
            NodePort(slot="overlay", label="overlay", kind="image"),
        ],
        outputs=[NodePort(slot="image", label="image", kind="image")],
        fields=[
            NodeField(
                key="mode",
                label="Mode",
                type="select",
                default="normal",
                options=[
                    {"value": "normal", "label": "Normal (alpha)"},
                    {"value": "multiply", "label": "Multiply"},
                    {"value": "screen", "label": "Screen"},
                    {"value": "overlay", "label": "Overlay"},
                ],
            ),
            NodeField(key="alpha", label="Alpha", type="number", default=0.5, min=0.0, max=1.0, step=0.05),
        ],
    )

    def execute(self, params: Dict[str, Any], inputs: Dict[str, Any], context: NodeExecutionContext) -> Dict[str, Any]:
        base = inputs.get("base")
        overlay = inputs.get("overlay")
        if base is None or overlay is None:
            raise ValueError("Blend Images requires both base and overlay images.")
        from PIL import Image as PILImage, ImageChops

        if not isinstance(base, PILImage.Image) or not isinstance(overlay, PILImage.Image):
            raise ValueError("Blend Images inputs must be decoded PIL images.")
        # The service resolves deferred references before this node runs.
        if overlay.size != base.size:
            overlay = overlay.resize(base.size, PILImage.Resampling.LANCZOS)
        base_rgb = base.convert("RGBA")
        overlay_rgb = overlay.convert("RGBA")
        mode = str(params.get("mode", "normal"))
        alpha = max(0.0, min(1.0, float(params.get("alpha", 0.5))))

        if mode == "multiply":
            blended = ImageChops.multiply(base_rgb, overlay_rgb)
        elif mode == "screen":
            blended = ImageChops.screen(base_rgb, overlay_rgb)
        elif mode == "overlay":
            blended = ImageChops.overlay(base_rgb, overlay_rgb)
        elif mode == "normal":
            blended = overlay_rgb
        else:
            raise ValueError(f"Unsupported blend mode '{mode}'.")
        result = PILImage.blend(base_rgb, blended, alpha)
        return {"image": result}


@get_node_registry().register
class PromptFromMetadataNode(BaseWorkflowNode):
    """Expose a previous generation's prompt so it can seed another run."""

    spec = NodeSpec(
        type="image_metadata",
        title="Image Metadata",
        category="Utility",
        icon="🧾",
        color="#64748b",
        description="Read embedded PNG metadata (prompt, seed, model) from an existing image.",
        inputs=[NodePort(slot="image", label="image", kind="image")],
        outputs=[
            NodePort(slot="prompt", label="prompt", kind="text"),
            NodePort(slot="negative_prompt", label="negative prompt", kind="text"),
            NodePort(slot="seed", label="seed", kind="text"),
            NodePort(slot="model", label="model", kind="text"),
        ],
    )

    def execute(self, params: Dict[str, Any], inputs: Dict[str, Any], context: NodeExecutionContext) -> Dict[str, Any]:
        image = inputs.get("image")
        if image is None:
            raise ValueError("Image Metadata requires an image input.")
        text = getattr(image, "text", None) or {}
        info = dict(getattr(image, "info", {}) or {})
        merged = {**text, **{k: v for k, v in info.items() if isinstance(v, str)}}
        return {
            "prompt": merged.get("prompt", ""),
            "negative_prompt": merged.get("negative_prompt", ""),
            "seed": merged.get("seed", ""),
            "model": merged.get("model", ""),
        }


@get_node_registry().register
class HashingNode(BaseWorkflowNode):
    """Stable hash of a prompt, useful for caching or filename generation."""

    spec = NodeSpec(
        type="text_hash",
        title="Text Hash",
        category="Utility",
        icon="#️⃣",
        color="#64748b",
        description="Generate a stable hexadecimal digest from text.",
        inputs=[NodePort(slot="text", label="text", kind="text")],
        outputs=[NodePort(slot="hash", label="hash", kind="text")],
        fields=[
            NodeField(
                key="algorithm",
                label="Algorithm",
                type="select",
                default="sha256",
                options=[
                    {"value": "sha256", "label": "SHA-256"},
                    {"value": "sha1", "label": "SHA-1"},
                    {"value": "md5", "label": "MD5"},
                ],
            ),
            NodeField(key="length", label="Maximum length", type="number", default=0, min=0, max=128, step=1),
        ],
    )

    def execute(self, params: Dict[str, Any], inputs: Dict[str, Any], context: NodeExecutionContext) -> Dict[str, Any]:
        text = str(inputs.get("text") or "").encode("utf-8")
        algorithm = str(params.get("algorithm", "sha256"))
        if algorithm not in {"sha256", "sha1", "md5"}:
            raise ValueError(f"Unsupported hashing algorithm '{algorithm}'.")
        digest = hashlib.new(algorithm, text).hexdigest()
        limit = int(params.get("length", 0))
        if limit > 0:
            digest = digest[:limit]
        return {"hash": digest}