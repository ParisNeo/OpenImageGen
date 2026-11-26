
"""Built-in generation, text, and preview nodes using the public plugin API."""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseWorkflowNode, NodeExecutionContext, NodeField, NodePort, NodeSpec
from .registry import get_node_registry

registry = get_node_registry()

class StaticNode(BaseWorkflowNode):
    output_slot = "value"
    def execute(self, params: Dict[str, Any], inputs: Dict[str, Any], context: NodeExecutionContext) -> Dict[str, Any]:
        return {self.output_slot: params.get(self.spec.fields[0].key, "")}

@registry.register
class TextPromptNode(StaticNode):
    spec = NodeSpec(
        type="text_prompt", title="Prompt", category="Text", icon="📝", color="#059669",
        outputs=[NodePort(slot="text", label="text", kind="text")],
        fields=[NodeField(key="text", label="Prompt", type="textarea", required=True,
                          default="A serene mountain landscape at golden hour")],
    )
    output_slot = "text"

@registry.register
class ModelNode(StaticNode):
    spec = NodeSpec(
        type="model", title="Model", category="Generation", icon="🤖", color="#d97706",
        outputs=[NodePort(slot="model", label="model", kind="model")],
        fields=[NodeField(key="model_name", label="Checkpoint", type="model", required=True, default="")],
    )
    output_slot = "model"

@registry.register
class ImageLoaderNode(StaticNode):
    spec = NodeSpec(
        type="image_loader", title="Load Image", category="Image", icon="🖼", color="#9333ea",
        outputs=[NodePort(slot="image", label="image", kind="image")],
        fields=[NodeField(key="image_id", label="Source image", type="upload", required=True, default="")],
    )
    output_slot = "image"

@registry.register
class ConcatenateTextNode(BaseWorkflowNode):
    spec = NodeSpec(
        type="text_concatenate", title="Join Text", category="Text", icon="🔗", color="#0891b2",
        description="Join up to four text inputs with a configurable separator.",
        inputs=[NodePort(slot=f"text_{index}", label=f"text {index}", kind="text") for index in range(1, 5)],
        outputs=[NodePort(slot="text", label="text", kind="text")],
        fields=[NodeField(key="separator", label="Separator", type="text", default="\n")],
    )
    def execute(self, params, inputs, context):
        values = [str(value).strip() for value in inputs.values() if str(value).strip()]
        return {"text": str(params.get("separator", "\n")).join(values)}

GENERATION_FIELDS = [
    NodeField(key="quality_preset", label="Quality profile", type="select", default="recommended", options=[
        {"value": "draft", "label": "Draft · fastest"},
        {"value": "recommended", "label": "Recommended · balanced"},
        {"value": "quality", "label": "Quality · slower"},
    ], help="Applies model-specific native resolution, steps, CFG, and sampler settings."),
    NodeField(key="width", label="Width", type="number", default=512, min=64, max=2048, step=8),
    NodeField(key="height", label="Height", type="number", default=512, min=64, max=2048, step=8),
    NodeField(key="num_inference_steps", label="Steps", type="number", default=30, min=1, max=200, step=1),
    NodeField(key="guidance_scale", label="CFG", type="number", default=7, min=0, max=30, step=.5),
    NodeField(key="sampler", label="Sampler", type="select", default="dpmpp_2m_karras", options=[
        {"value": "dpmpp_2m_karras", "label": "DPM++ 2M Karras · balanced"},
        {"value": "dpmpp_2m_sde_karras", "label": "DPM++ 2M SDE Karras · quality"},
        {"value": "dpmpp_2m", "label": "DPM++ 2M"},
        {"value": "dpmpp_sde_karras", "label": "DPM++ SDE Karras"},
        {"value": "unipc", "label": "UniPC · fast at low steps"},
        {"value": "flowmatch_euler", "label": "FlowMatch Euler · FLUX / SD3 / Qwen"},
        {"value": "euler_a", "label": "Euler Ancestral · Turbo"},
        {"value": "euler", "label": "Euler"},
        {"value": "ddim", "label": "DDIM"},
    ]),
    NodeField(key="seed", label="Seed (−1 = random)", type="number", default=-1, step=1),
    NodeField(key="num_images_per_prompt", label="Batch", type="number", default=1, min=1, max=8, step=1),
]
SLOT_TO_PARAM = {"prompt": "prompt", "negative_prompt": "negative_prompt", "model": "model_name", "image": "init_image_id"}

@registry.register
class LoRANode(BaseWorkflowNode):
    spec = NodeSpec(
        type="lora", title="LoRA", category="Generation", icon="🧬", color="#c026d3",
        description="Apply a Hugging Face or local LoRA to the connected checkpoint. Chain nodes to combine adapters.",
        inputs=[NodePort(slot="model", label="base model or LoRA", kind="model", required=True)],
        outputs=[NodePort(slot="model", label="model + LoRA", kind="model")],
        fields=[
            NodeField(key="source", label="Repository or local asset", type="text", required=True,
                      placeholder="owner/lora-repository"),
            NodeField(key="weight_name", label="Weight filename", type="text", required=True,
                      placeholder="pytorch_lora_weights.safetensors"),
            NodeField(key="weight", label="Strength", type="number", default=1.0, min=0, max=4, step=.05),
        ],
    )

    def execute(self, params, inputs, context):
        model = inputs["model"]
        if isinstance(model, str):
            model = {"model_name": model, "loras": []}
        else:
            model = {"model_name": model["model_name"], "loras": list(model.get("loras", []))}
        model["loras"].append({
            "source": str(params["source"]).strip(),
            "weight_name": str(params["weight_name"]).strip(),
            "weight": float(params.get("weight", 1.0)),
        })
        return {"model": model}

class GenerationNode(BaseWorkflowNode):
    def execute(self, params, inputs, context):
        result = {SLOT_TO_PARAM[slot]: value for slot, value in inputs.items()
                  if slot in SLOT_TO_PARAM and slot != "model"}
        model = inputs.get("model")
        if isinstance(model, dict):
            result["model_name"] = model["model_name"]
            result["loras"] = model.get("loras", [])
        elif model:
            result["model_name"] = model
        for key, value in params.items():
            result.setdefault(key, value)
        return {"params": result, "image": None}

@registry.register
class SamplerNode(GenerationNode):
    spec = NodeSpec(
        type="sampler", title="Sampler", category="Generation", icon="⚙", color="#db2777",
        inputs=[NodePort(slot="prompt", label="prompt", kind="text", required=True),
                NodePort(slot="negative_prompt", label="negative", kind="text"),
                NodePort(slot="model", label="model", kind="model")],
        description="Build a text-to-image request. The output is generation params, not a rendered image.",
        outputs=[
            NodePort(slot="params", label="generation params", kind="params"),
            NodePort(slot="image", label="generated image (after run)", kind="image"),
        ], fields=GENERATION_FIELDS,
    )

@registry.register
class Img2ImgNode(GenerationNode):
    spec = NodeSpec(
        type="img2img", title="Img2Img", category="Generation", icon="🎨", color="#e11d48",
        inputs=[NodePort(slot="image", label="image", kind="image", required=True),
                NodePort(slot="prompt", label="prompt", kind="text", required=True),
                NodePort(slot="negative_prompt", label="negative", kind="text"),
                NodePort(slot="model", label="model", kind="model")],
        description="Build an image-to-image request. The output is generation params, not a rendered image.",
        outputs=[
            NodePort(slot="params", label="generation params", kind="params"),
            NodePort(slot="image", label="generated image (after run)", kind="image"),
        ],
        fields=[NodeField(key="strength", label="Strength", type="number", default=.75, min=.1, max=1, step=.05), *GENERATION_FIELDS],
    )

@registry.register
class TextPrependAppendNode(BaseWorkflowNode):
    spec = NodeSpec(
        type="text_prepend_append", title="Wrap Text", category="Text", icon="🔤", color="#0891b2",
        description="Add a prefix and/or suffix to a text input. Useful for consistent style and quality tokens.",
        inputs=[NodePort(slot="text", label="text", kind="text", required=True)],
        outputs=[NodePort(slot="text", label="text", kind="text")],
        fields=[
            NodeField(key="prefix", label="Prefix", type="textarea", default="", placeholder="photorealistic, "),
            NodeField(key="suffix", label="Suffix", type="textarea", default="", placeholder=", 8k, highly detailed"),
            NodeField(key="separator", label="Joiner", type="text", default=""),
        ],
    )

    def execute(self, params, inputs, context):
        prefix = str(params.get("prefix", "")).strip()
        suffix = str(params.get("suffix", "")).strip()
        joiner = params.get("separator", "")
        body = str(inputs["text"]).strip()
        parts = [part for part in (prefix, body, suffix) if part]
        return {"text": joiner.join(parts)}


@registry.register
class TextReplaceNode(BaseWorkflowNode):
    spec = NodeSpec(
        type="text_replace", title="Find & Replace", category="Text", icon="🔍", color="#0891b2",
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
        import re
        return {"text": re.sub(re.escape(needle), replacement, text, flags=re.IGNORECASE)}


@registry.register
class TextTrimNode(BaseWorkflowNode):
    spec = NodeSpec(
        type="text_trim", title="Clean Text", category="Text", icon="✨", color="#0891b2",
        description="Trim whitespace and optionally collapse consecutive separators.",
        inputs=[NodePort(slot="text", label="text", kind="text", required=True)],
        outputs=[NodePort(slot="text", label="text", kind="text")],
        fields=[
            NodeField(key="collapse_commas", label="Collapse repeated commas", type="boolean", default=True),
            NodeField(key="max_length", label="Max characters (0 = unlimited)", type="number", default=0, min=0, step=1),
        ],
    )

    def execute(self, params, inputs, context):
        import re
        text = str(inputs["text"]).strip()
        if params.get("collapse_commas", True):
            text = re.sub(r"\s*,\s*", ", ", text)
            text = re.sub(r"(?:,\s*)+", ", ", text).strip(", ")
        max_length = int(params.get("max_length", 0))
        if max_length > 0:
            text = text[:max_length].rstrip()
        return {"text": text}


@registry.register
class NumberNode(BaseWorkflowNode):
    spec = NodeSpec(
        type="number", title="Number", category="Value", icon="🔢", color="#64748b",
        outputs=[NodePort(slot="number", label="number", kind="number")],
        fields=[NodeField(key="value", label="Value", type="number", default=0, step=0.01)],
    )

    def execute(self, params, inputs, context):
        return {"number": float(params.get("value", 0))}


@registry.register
class BooleanNode(BaseWorkflowNode):
    spec = NodeSpec(
        type="boolean", title="Boolean", category="Value", icon="☑", color="#64748b",
        outputs=[NodePort(slot="boolean", label="boolean", kind="boolean")],
        fields=[NodeField(key="value", label="Value", type="boolean", default=False)],
    )

    def execute(self, params, inputs, context):
        return {"boolean": bool(params.get("value", False))}


@registry.register
class RandomIntNode(BaseWorkflowNode):
    spec = NodeSpec(
        type="random_int", title="Random Integer", category="Value", icon="🎲", color="#64748b",
        description="Non-deterministic by design. Use -1 range to emulate an unseeded sampler.",
        outputs=[NodePort(slot="number", label="number", kind="number")],
        fields=[
            NodeField(key="min", label="Minimum", type="number", default=0, step=1),
            NodeField(key="max", label="Maximum", type="number", default=2147483646, step=1),
        ],
    )

    def execute(self, params, inputs, context):
        import random
        low = int(params.get("min", 0))
        high = int(params.get("max", 2147483646))
        if high < low:
            raise ValueError("Maximum must be >= minimum.")
        return {"number": float(random.randint(low, high))}


@registry.register
class MathNode(BaseWorkflowNode):
    spec = NodeSpec(
        type="math", title="Math", category="Value", icon="🧮", color="#64748b",
        description="Apply an arithmetic operation to two numbers. Values arriving via inputs override configured fields.",
        inputs=[NodePort(slot="a", label="a (optional)", kind="number"),
                NodePort(slot="b", label="b (optional)", kind="number")],
        outputs=[NodePort(slot="number", label="result", kind="number")],
        fields=[
            NodeField(key="a", label="A", type="number", default=0, step=0.01),
            NodeField(key="b", label="B", type="number", default=0, step=0.01),
            NodeField(key="operation", label="Operation", type="select", default="add", options=[
                {"value": "add", "label": "A + B"}, {"value": "subtract", "label": "A − B"},
                {"value": "multiply", "label": "A × B"}, {"value": "divide", "label": "A ÷ B"},
                {"value": "min", "label": "min(A, B)"}, {"value": "max", "label": "max(A, B)"},
                {"value": "power", "label": "A ^ B"},
            ]),
        ],
    )

    def execute(self, params, inputs, context):
        a = float(inputs.get("a", params.get("a", 0)))
        b = float(inputs.get("b", params.get("b", 0)))
        op = params.get("operation", "add")
        if op == "add":
            result = a + b
        elif op == "subtract":
            result = a - b
        elif op == "multiply":
            result = a * b
        elif op == "divide":
            if b == 0:
                raise ValueError("Division by zero.")
            result = a / b
        elif op == "min":
            result = min(a, b)
        elif op == "max":
            result = max(a, b)
        elif op == "power":
            result = a ** b
        else:
            raise ValueError(f"Unknown operation '{op}'.")
        return {"number": float(result)}


@registry.register
class ParamsOverrideNode(BaseWorkflowNode):
    spec = NodeSpec(
        type="params_override", title="Override Params", category="Generation", icon="🧷", color="#db2777",
        description="Merge scalar overrides into generation params, e.g. a random seed produced by Random Integer.",
        inputs=[
            NodePort(slot="params", label="generation params", kind="params", required=True),
            NodePort(slot="seed", label="seed (optional)", kind="number"),
            NodePort(slot="steps", label="steps (optional)", kind="number"),
            NodePort(slot="guidance_scale", label="CFG (optional)", kind="number"),
        ],
        outputs=[NodePort(slot="params", label="generation params", kind="params")],
        fields=[
            NodeField(key="seed", label="Seed override", type="number", default=-1, step=1,
                      help="Leave -1 to keep the incoming params' seed."),
            NodeField(key="width", label="Width override", type="number", default=0, min=0, max=2048, step=8),
            NodeField(key="height", label="Height override", type="number", default=0, min=0, max=2048, step=8),
            NodeField(key="num_inference_steps", label="Steps override", type="number", default=0, min=0, max=200, step=1),
        ],
    )

    def execute(self, params, inputs, context):
        merged = dict(inputs["params"])
        # Input-port values take precedence over configured fields.
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


@registry.register
class JsonExtractorNode(BaseWorkflowNode):
    spec = NodeSpec(
        type="json_extract", title="JSON Extract", category="Value", icon="🧾", color="#64748b",
        description="Parse a JSON document and expose numeric/boolean values, e.g. a seed emitted by an external API.",
        inputs=[NodePort(slot="json", label="json", kind="json", required=True)],
        outputs=[NodePort(slot="number", label="number", kind="number"),
                NodePort(slot="text", label="text", kind="text"),
                NodePort(slot="boolean", label="boolean", kind="boolean")],
        fields=[NodeField(key="path", label="Dot path", type="text", default="seed",
                          placeholder="data.0.seed", help="Use digits for array indices.")],
    )

    def execute(self, params, inputs, context):
        import json
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
        result = {"text": str(current)}
        if isinstance(current, bool):
            result["boolean"] = current
        else:
            try:
                result["number"] = float(current)
            except (TypeError, ValueError):
                pass
        return result


class ImageTransformNode(BaseWorkflowNode):
    """Create a serializable image reference consumed after image materialization."""

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


@registry.register
class ImageUpscaleNode(ImageTransformNode):
    spec = NodeSpec(
        type="image_upscale", title="Upscale", category="Image", icon="🔎", color="#8b5cf6",
        description="Resize an image by 1.5–4× using high-quality resampling. This is not an AI detail hallucinator.",
        inputs=[NodePort(slot="image", label="image", kind="image", required=True)],
        outputs=[NodePort(slot="image", label="upscaled image", kind="image")],
        fields=[
            NodeField(key="scale", label="Scale", type="select", default="2", options=[
                {"value": "1.5", "label": "1.5×"}, {"value": "2", "label": "2×"},
                {"value": "3", "label": "3×"}, {"value": "4", "label": "4×"},
            ]),
            NodeField(key="method", label="Resampling", type="select", default="lanczos", options=[
                {"value": "lanczos", "label": "Lanczos · best general quality"},
                {"value": "bicubic", "label": "Bicubic"},
                {"value": "bilinear", "label": "Bilinear"},
                {"value": "nearest", "label": "Nearest · pixel art"},
            ]),
        ],
    )

    def operation(self, params):
        return {
            "operation": "upscale",
            "scale": float(params.get("scale", 2)),
            "method": params.get("method", "lanczos"),
        }


@registry.register
class ImageResizeNode(ImageTransformNode):
    spec = NodeSpec(
        type="image_resize", title="Resize Image", category="Image", icon="↔", color="#8b5cf6",
        description="Resize to exact dimensions. Chain with Preview Image or another processor.",
        inputs=[NodePort(slot="image", label="image", kind="image", required=True)],
        outputs=[NodePort(slot="image", label="resized image", kind="image")],
        fields=[
            NodeField(key="width", label="Width", type="number", default=1024, min=1, max=8192, step=1),
            NodeField(key="height", label="Height", type="number", default=1024, min=1, max=8192, step=1),
            NodeField(key="method", label="Resampling", type="select", default="lanczos", options=[
                {"value": "lanczos", "label": "Lanczos"}, {"value": "bicubic", "label": "Bicubic"},
                {"value": "bilinear", "label": "Bilinear"}, {"value": "nearest", "label": "Nearest"},
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


@registry.register
class ImageFilterNode(ImageTransformNode):
    spec = NodeSpec(
        type="image_filter", title="Image Filter", category="Image", icon="✨", color="#8b5cf6",
        description="Blur, sharpen, invert, grayscale, or extract edges from an image.",
        inputs=[NodePort(slot="image", label="image", kind="image", required=True)],
        outputs=[NodePort(slot="image", label="filtered image", kind="image")],
        fields=[
            NodeField(key="filter", label="Filter", type="select", default="sharpen", options=[
                {"value": "sharpen", "label": "Sharpen"}, {"value": "blur", "label": "Gaussian blur"},
                {"value": "edges", "label": "Edges"}, {"value": "grayscale", "label": "Grayscale"},
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


@registry.register
class ImageCropNode(ImageTransformNode):
    spec = NodeSpec(
        type="image_crop", title="Crop Image", category="Image", icon="✂", color="#8b5cf6",
        description="Crop a rectangle from an image. Dimensions are anchored at the configured position.",
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


@registry.register
class ImageRotateNode(ImageTransformNode):
    spec = NodeSpec(
        type="image_rotate", title="Rotate / Flip", category="Image", icon="🔄", color="#8b5cf6",
        description="Rotate by a right-angle multiple and/or mirror the image.",
        inputs=[NodePort(slot="image", label="image", kind="image", required=True)],
        outputs=[NodePort(slot="image", label="rotated image", kind="image")],
        fields=[
            NodeField(key="angle", label="Right-angle rotation", type="select", default="0", options=[
                {"value": "0", "label": "0°"}, {"value": "90", "label": "90° clockwise"},
                {"value": "180", "label": "180°"}, {"value": "270", "label": "270° clockwise"},
            ]),
            NodeField(key="flip", label="Mirror", type="select", default="none", options=[
                {"value": "none", "label": "None"}, {"value": "horizontal", "label": "Horizontal"},
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


@registry.register
class ImageAdjustNode(ImageTransformNode):
    spec = NodeSpec(
        type="image_adjust", title="Adjust Colors", category="Image", icon="🎛", color="#8b5cf6",
        description="Brightness, contrast, saturation, and sharpness tuning on a 0.0–2.0 scale.",
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


@registry.register
class ImagePadNode(ImageTransformNode):
    spec = NodeSpec(
        type="image_pad", title="Pad / Border", category="Image", icon="⬚", color="#8b5cf6",
        description="Add a solid-color frame around an image, useful before compositing.",
        inputs=[NodePort(slot="image", label="image", kind="image", required=True)],
        outputs=[NodePort(slot="image", label="padded image", kind="image")],
        fields=[
            NodeField(key="padding", label="Padding (px)", type="number", default=16, min=0, max=512, step=1),
            NodeField(key="color", label="Color (#RRGGBB or name)", type="text", default="#000000"),
        ],
    )

    def operation(self, params):
        return {
            "operation": "pad",
            "padding": int(params.get("padding", 16)),
            "color": str(params.get("color", "#000000")),
        }


@registry.register
class ImageTextNode(ImageTransformNode):
    spec = NodeSpec(
        type="image_text", title="Add Text", category="Image", icon="🔤", color="#8b5cf6",
        description="Overlay text using the bundled PIL default font; large sizes use a scalable fallback when available.",
        inputs=[NodePort(slot="image", label="image", kind="image", required=True),
                NodePort(slot="text", label="text (optional)", kind="text")],
        outputs=[NodePort(slot="image", label="annotated image", kind="image")],
        fields=[
            NodeField(key="text", label="Text", type="textarea", default=""),
            NodeField(key="position", label="Anchor", type="select", default="bottom-right", options=[
                {"value": "top-left", "label": "Top left"}, {"value": "top-right", "label": "Top right"},
                {"value": "bottom-left", "label": "Bottom left"}, {"value": "bottom-right", "label": "Bottom right"},
                {"value": "center", "label": "Center"},
            ]),
            NodeField(key="size", label="Font size", type="number", default=32, min=8, max=256, step=1),
            NodeField(key="color", label="Color (#RRGGBB)", type="text", default="#ffffff"),
            NodeField(key="shadow", label="Add text shadow", type="boolean", default=True),
        ],
    )

    def operation(self, params):
        text = str(params.get("text", ""))
        if not text:
            raise ValueError("Text cannot be empty.")
        return {
            "operation": "text",
            "text": text,
            "position": params.get("position", "bottom-right"),
            "size": int(params.get("size", 32)),
            "color": str(params.get("color", "#ffffff")),
            "shadow": bool(params.get("shadow", True)),
        }


@registry.register
class OutputNode(BaseWorkflowNode):
    spec = NodeSpec(
        type="output", title="Output", category="Generation", icon="🎯", color="#7c3aed",
        description="Run generation, then expose the first rendered image to connected visualizers.",
        inputs=[NodePort(slot="params", label="generation params", kind="params", required=True)],
        outputs=[NodePort(slot="image", label="generated image (after run)", kind="image")],
    )
    def execute(self, params, inputs, context):
        # Image resolution is deferred until the service finishes diffusion.
        return {"result": inputs["params"], "image": None}

@registry.register
class TextOutputNode(BaseWorkflowNode):
    """Display intermediate or terminal text without running a diffusion model."""
    spec = NodeSpec(
        type="text_output", title="Preview Text", category="Output", icon="📄", color="#0891b2",
        inputs=[NodePort(slot="text", label="text", kind="text", required=True)],
    )
    def execute(self, params, inputs, context):
        return {"result": {"text": str(inputs["text"])}}

@registry.register
class ImagePreviewNode(BaseWorkflowNode):
    spec = NodeSpec(
        type="image_preview", title="Preview Image", category="Output", icon="👁", color="#7c3aed",
        description="Preview Load Image immediately, or display Output's generated image after the workflow completes.",
        inputs=[NodePort(slot="image", label="existing image", kind="image", required=True)],
    )
    def execute(self, params, inputs, context):
        if inputs["image"] is None:
            return {"result": {}}
        if isinstance(inputs["image"], dict):
            return {"result": {"image_ref": inputs["image"]}}
        image = str(inputs["image"])
        if image.startswith(("/download/", "/uploads/")):
            return {"result": {"image_url": image}}
        return {"result": {"image_id": image}}
