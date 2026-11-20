
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
    NodeField(key="width", label="Width", type="number", default=512, min=64, max=2048, step=8),
    NodeField(key="height", label="Height", type="number", default=512, min=64, max=2048, step=8),
    NodeField(key="num_inference_steps", label="Steps", type="number", default=30, min=1, max=200, step=1),
    NodeField(key="guidance_scale", label="CFG", type="number", default=7, min=0, max=30, step=.5),
    NodeField(key="sampler", label="Sampler", type="select", default="dpmpp_2m_karras", options=[
        {"value": "dpmpp_2m_karras", "label": "DPM++ 2M Karras · recommended"},
        {"value": "euler_a", "label": "Euler Ancestral"},
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
