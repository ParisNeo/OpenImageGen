"""
@file        extra.py
@author      ParisNeo
@description Additional built-in workflow nodes for prompt enrichment,
             dimension calculation, seeding, text formatting, and numeric clamping.
             Every node here either takes or returns only serializable primitives
             (text/number/boolean), which keeps them compatible with the deferred
             image model used by the workflow engine.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict

from .base import BaseWorkflowNode, NodeExecutionContext, NodeField, NodePort, NodeSpec
from .registry import get_node_registry

_STYLE_PRESETS: Dict[str, str] = {
    "photoreal": "professional photograph, natural lighting, sharp focus, detailed textures, 85mm lens",
    "cinematic": "cinematic lighting, dramatic composition, shallow depth of field, 35mm film",
    "illustration": "detailed illustration, clean lineart, vibrant colors, hand-painted",
    "minimalist": "minimalist design, clean composition, generous negative space",
    "concept_art": "concept art, intricate detail, atmospheric depth, muted palette",
}

_RATIO_PRESETS: Dict[str, tuple[int, int]] = {
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "3:2": (3, 2),
    "2:3": (2, 3),
}


@get_node_registry().register
class StyleSuffixNode(BaseWorkflowNode):
    """Append curated quality/style tokens to a prompt with per-family presets."""

    spec = NodeSpec(
        type="style_suffix",
        title="Style Suffix",
        category="Text",
        icon="🎨",
        color="#059669",
        description="Append a curated style boilerplate to a prompt. Custom text overrides the preset.",
        inputs=[NodePort(slot="text", label="text", kind="text", required=True)],
        outputs=[NodePort(slot="text", label="text", kind="text")],
        fields=[
            NodeField(
                key="preset",
                label="Style preset",
                type="select",
                default="photoreal",
                options=[{"value": p, "label": p.replace("_", " ").title()} for p in _STYLE_PRESETS],
            ),
            NodeField(key="custom", label="Custom suffix", type="textarea", default=""),
            NodeField(key="separator", label="Separator", type="text", default=", "),
        ],
    )

    def execute(self, params: Dict[str, Any], inputs: Dict[str, Any], context: NodeExecutionContext) -> Dict[str, Any]:
        base = str(inputs.get("text") or "").strip()
        if not base:
            raise ValueError("Style Suffix requires a non-empty text input.")
        custom = str(params.get("custom", "")).strip()
        suffix = custom or _STYLE_PRESETS.get(str(params.get("preset", "")), "")
        if not suffix:
            return {"text": base}
        return {"text": f"{base}{params.get('separator', ', ')}{suffix}"}


@get_node_registry().register
class AspectRatioNode(BaseWorkflowNode):
    """Compute 8-aligned generation dimensions from an aspect ratio and long edge."""

    spec = NodeSpec(
        type="aspect_ratio",
        title="Aspect Ratio",
        category="Value",
        icon="📐",
        color="#64748b",
        description="Convert an aspect ratio preset into 8-aligned width/height values.",
        outputs=[
            NodePort(slot="width", label="width", kind="number"),
            NodePort(slot="height", label="height", kind="number"),
        ],
        fields=[
            NodeField(
                key="ratio",
                label="Ratio",
                type="select",
                default="1:1",
                options=[{"value": r, "label": r} for r in _RATIO_PRESETS],
            ),
            NodeField(
                key="long_edge",
                label="Long edge (px)",
                type="number",
                default=1024,
                min=64,
                max=4096,
                step=8,
            ),
        ],
    )

    def execute(self, params: Dict[str, Any], inputs: Dict[str, Any], context: NodeExecutionContext) -> Dict[str, Any]:
        ratio = _RATIO_PRESETS.get(str(params.get("ratio", "1:1")))
        if ratio is None:
            raise ValueError("Unknown aspect ratio preset.")
        long_edge = max(64, int(params.get("long_edge", 1024)))
        # Keep 8-pixel alignment required by diffusion pipelines.
        long_eight = max(64, (long_edge // 8) * 8)
        numerator, denominator = ratio
        if numerator >= denominator:
            width = long_eight
            height = max(64, round(long_edge * denominator / numerator / 8) * 8)
        else:
            width = max(64, round(long_edge * numerator / denominator / 8) * 8)
            height = long_eight
        return {"width": width, "height": height}


@get_node_registry().register
class SeedNode(BaseWorkflowNode):
    """Emit a reproducible fixed seed or a fresh random one for every run."""

    spec = NodeSpec(
        type="seed",
        title="Seed",
        category="Value",
        icon="🎲",
        color="#a855f7",
        description="Feed a controlled seed into Params Override or an external node.",
        outputs=[NodePort(slot="number", label="seed", kind="number")],
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
            NodeField(
                key="value",
                label="Fixed seed",
                type="number",
                default=42,
                min=0,
                max=2147483647,
                step=1,
            ),
        ],
    )

    def execute(self, params: Dict[str, Any], inputs: Dict[str, Any], context: NodeExecutionContext) -> Dict[str, Any]:
        if params.get("mode") == "fixed":
            return {"number": int(params.get("value", 42))}
        return {"number": random.randint(0, 2147483647)}


@get_node_registry().register
class TextHashNode(BaseWorkflowNode):
    """Compute a stable hash of prompt text for caching or filenames."""

    spec = NodeSpec(
        type="text_hash",
        title="Text Hash",
        category="Text",
        icon="#️⃣",
        color="#64748b",
        description="Stable hexadecimal digest of a text input.",
        inputs=[NodePort(slot="text", label="text", kind="text", required=True)],
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
            NodeField(
                key="length",
                label="Max length (0 = full)",
                type="number",
                default=0,
                min=0,
                max=128,
                step=1,
            ),
        ],
    )

    def execute(self, params: Dict[str, Any], inputs: Dict[str, Any], context: NodeExecutionContext) -> Dict[str, Any]:
        text = str(inputs.get("text") or "").encode("utf-8")
        algorithm = str(params.get("algorithm", "sha256"))
        if algorithm not in {"sha256", "sha1", "md5"}:
            raise ValueError(f"Unsupported algorithm '{algorithm}'.")
        digest = hashlib.new(algorithm, text).hexdigest()
        limit = int(params.get("length", 0))
        if limit > 0:
            digest = digest[:limit]
        return {"hash": digest}


@get_node_registry().register
class TextCaseNode(BaseWorkflowNode):
    """Normalize the case of a text input."""

    spec = NodeSpec(
        type="text_case",
        title="Text Case",
        category="Text",
        icon="Aa",
        color="#0ea5e9",
        description="Upper-case, lower-case, title-case, or sentence-case the input.",
        inputs=[NodePort(slot="text", label="text", kind="text", required=True)],
        outputs=[NodePort(slot="text", label="text", kind="text")],
        fields=[
            NodeField(
                key="mode",
                label="Mode",
                type="select",
                default="lower",
                options=[
                    {"value": "lower", "label": "lowercase"},
                    {"value": "upper", "label": "UPPERCASE"},
                    {"value": "title", "label": "Title Case"},
                    {"value": "sentence", "label": "Sentence case"},
                ],
            ),
        ],
    )

    def execute(self, params: Dict[str, Any], inputs: Dict[str, Any], context: NodeExecutionContext) -> Dict[str, Any]:
        text = str(inputs.get("text") or "")
        mode = str(params.get("mode", "lower"))
        if mode == "lower":
            return {"text": text.lower()}
        if mode == "upper":
            return {"text": text.upper()}
        if mode == "title":
            return {"text": text.title()}
        if mode == "sentence":
            return {"text": text[:1].upper() + text[1:].lower()}
        raise ValueError(f"Unknown text case mode '{mode}'.")


@get_node_registry().register
class TextCountNode(BaseWorkflowNode):
    """Expose character and word counts from a text input."""

    spec = NodeSpec(
        type="text_count",
        title="Text Count",
        category="Text",
        icon="🔢",
        color="#64748b",
        description="Count characters and words; useful for prompt-length validation.",
        inputs=[NodePort(slot="text", label="text", kind="text", required=True)],
        outputs=[
            NodePort(slot="chars", label="characters", kind="number"),
            NodePort(slot="words", label="words", kind="number"),
        ],
    )

    def execute(self, params: Dict[str, Any], inputs: Dict[str, Any], context: NodeExecutionContext) -> Dict[str, Any]:
        text = str(inputs.get("text") or "")
        return {
            "chars": len(text),
            "words": len(text.split()),
        }


@get_node_registry().register
class NumberClampNode(BaseWorkflowNode):
    """Constrain a numeric value between a minimum and maximum."""

    spec = NodeSpec(
        type="number_clamp",
        title="Clamp Number",
        category="Value",
        icon="🔒",
        color="#64748b",
        description="Constrain a value to [min, max]; useful for bounding seeds or dimensions.",
        inputs=[
            NodePort(slot="value", label="value (optional)", kind="number"),
            NodePort(slot="min", label="min (optional)", kind="number"),
            NodePort(slot="max", label="max (optional)", kind="number"),
        ],
        outputs=[NodePort(slot="value", label="clamped number", kind="number")],
        fields=[
            NodeField(key="value", label="Fallback value", type="number", default=0, step=0.01),
            NodeField(key="min", label="Fallback min", type="number", default=0, step=0.01),
            NodeField(key="max", label="Fallback max", type="number", default=100, step=0.01),
        ],
    )

    def execute(self, params: Dict[str, Any], inputs: Dict[str, Any], context: NodeExecutionContext) -> Dict[str, Any]:
        value = float(inputs.get("value", params.get("value", 0)))
        low = float(inputs.get("min", params.get("min", 0)))
        high = float(inputs.get("max", params.get("max", 100)))
        if high < low:
            low, high = high, low
        return {"value": max(low, min(high, value))}