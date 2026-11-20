
"""
@file        workflow.py
@author      ParisNeo
@description Registry-driven DAG validation and multi-output workflow execution.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from .nodes import NodeExecutionContext, NodeSpec, get_node_registry

logger = logging.getLogger("OpenImageGen.workflow")

_ALLOWED_PARAM_KEYS = {
    "prompt", "negative_prompt", "model_name", "width", "height",
    "num_inference_steps", "guidance_scale", "sampler", "seed", "num_images_per_prompt",
    "init_image_id", "strength", "loras",
}

class WorkflowValidationError(ValueError):
    pass

class WorkflowNode(BaseModel):
    id: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    type: str = Field(..., min_length=1, max_length=64)
    params: Dict[str, Any] = Field(default_factory=dict)
    # "source_id" selects its first output; "source_id.slot" selects explicitly.
    inputs: Dict[str, str] = Field(default_factory=dict)

class WorkflowDefinition(BaseModel):
    name: str = Field("untitled_workflow", max_length=128)
    description: Optional[str] = Field(None, max_length=512)
    nodes: List[WorkflowNode] = Field(..., min_length=1)

def node_specs() -> List[Dict[str, Any]]:
    return [spec.model_dump() for spec in get_node_registry().specs()]

def _parse_reference(reference: str, source: WorkflowNode) -> Tuple[str, str]:
    """Parse an output reference, disambiguating dotted ids by longest prefix."""
    if reference == source.id:
        candidates = [source.id]
    else:
        candidates = [
            reference[:index] for index, char in enumerate(reference) if char == "."
        ]
        candidates.append(reference)
    for node_id in sorted(candidates, key=len, reverse=True):
        if node_id != source.id:
            continue
        remainder = reference[len(node_id):]
        if not remainder:
            return node_id, ""
        if remainder.startswith(".") and remainder[1:] and "." not in remainder[1:]:
            return node_id, remainder[1:]
    if "." not in reference:
        return reference, ""
    node_id, slot = reference.rsplit(".", 1)
    if not node_id or not slot:
        raise WorkflowValidationError(f"Invalid output reference '{reference}'.")
    return node_id, slot

def _index_and_validate(workflow: WorkflowDefinition) -> Tuple[Dict[str, WorkflowNode], Dict[str, NodeSpec], Dict[str, str]]:
    registry = get_node_registry()
    index: Dict[str, WorkflowNode] = {}
    specs: Dict[str, NodeSpec] = {}
    references: Dict[str, str] = {}
    for node in workflow.nodes:
        if node.id in index:
            raise WorkflowValidationError(f"Duplicate node id '{node.id}'.")
        try:
            specs[node.id] = registry.get(node.type).spec
        except KeyError:
            raise WorkflowValidationError(f"Unknown workflow node type '{node.type}'.")
        index[node.id] = node

    for node in workflow.nodes:
        spec = specs[node.id]
        input_specs = {port.slot: port for port in spec.inputs}
        field_keys = {field.key for field in spec.fields}
        unknown_params = set(node.params) - field_keys
        if unknown_params:
            raise WorkflowValidationError(
                f"Node '{node.id}' has unknown parameter(s): {', '.join(sorted(unknown_params))}."
            )
        for field in spec.fields:
            if field.required and not field.secret and node.params.get(field.key) in (None, ""):
                raise WorkflowValidationError(f"Node '{node.id}' requires field '{field.key}'.")
        for port in spec.inputs:
            if port.required and port.slot not in node.inputs:
                raise WorkflowValidationError(f"Node '{node.id}' requires input '{port.slot}'.")
        for slot, reference in node.inputs.items():
            if slot not in input_specs:
                raise WorkflowValidationError(f"Node '{node.id}' has unknown input '{slot}'.")
            source_id = reference.split(".", 1)[0]
            if source_id not in index:
                # Dotted IDs are not accepted by the id regex; split remains safe.
                raise WorkflowValidationError(f"Input '{node.id}.{slot}' references missing node '{source_id}'.")
            if source_id == node.id:
                raise WorkflowValidationError(f"Node '{node.id}' cannot reference itself.")
            source_spec = specs[source_id]
            output_slot = reference[len(source_id) + 1:] if reference != source_id else source_spec.outputs[0].slot if source_spec.outputs else ""
            output = next((port for port in source_spec.outputs if port.slot == output_slot), None)
            if output is None:
                raise WorkflowValidationError(f"Node '{source_id}' has no output '{output_slot}'.")
            if output.kind != input_specs[slot].kind and "any" not in {output.kind, input_specs[slot].kind}:
                raise WorkflowValidationError(
                    f"Cannot connect {output.kind} to {input_specs[slot].kind} at '{node.id}.{slot}'."
                )
            references[f"{node.id}.{slot}"] = output_slot
    return index, specs, references

def _detect_cycles(index: Dict[str, WorkflowNode]) -> None:
    visited: Set[str] = set()
    visiting: Set[str] = set()
    def visit(node_id: str) -> None:
        if node_id in visited:
            return
        if node_id in visiting:
            raise WorkflowValidationError(f"Cycle detected at node '{node_id}'.")
        visiting.add(node_id)
        for reference in index[node_id].inputs.values():
            visit(reference.split(".", 1)[0])
        visiting.remove(node_id)
        visited.add(node_id)
    for node_id in index:
        visit(node_id)

def validate_workflow(workflow: WorkflowDefinition) -> None:
    index, _, _ = _index_and_validate(workflow)
    _detect_cycles(index)
    if sum(node.type in {"output", "text_output", "image_preview"} for node in workflow.nodes) < 1:
        raise WorkflowValidationError("Workflow requires at least one output or preview node.")

def execute_workflow(
    workflow: WorkflowDefinition,
    report: Optional[Callable[[int, str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Tuple[str, Dict[str, Any]]:
    validate_workflow(workflow)
    index, specs, references = _index_and_validate(workflow)
    registry = get_node_registry()
    terminals = [
        node for node in workflow.nodes
        if node.type in {"output", "text_output", "image_preview"}
    ]
    terminal = next((node for node in terminals if node.type == "output"), terminals[0])
    ordered: List[str] = []
    visited: Set[str] = set()
    def visit(node_id: str) -> None:
        if node_id in visited:
            return
        visited.add(node_id)
        for reference in index[node_id].inputs.values():
            visit(reference.split(".", 1)[0])
        ordered.append(node_id)
    # Execute every terminal branch. A graph may generate an image while also
    # displaying the expanded prompt or previewing an independent input image.
    for terminal_node in terminals:
        visit(terminal_node.id)

    values: Dict[str, Dict[str, Any]] = {}
    context = NodeExecutionContext(
        report=report or (lambda progress, message: None),
        is_cancelled=is_cancelled or (lambda: False),
    )
    for position, node_id in enumerate(ordered):
        context.checkpoint()
        node = index[node_id]
        spec = specs[node_id]
        inputs = {
            slot: values[reference.split(".", 1)[0]][references[f"{node_id}.{slot}"]]
            for slot, reference in node.inputs.items()
        }
        context.report(int(position / max(1, len(ordered)) * 100), f"Executing {spec.title} ({node_id})…")
        try:
            output = registry.get(node.type).execute(dict(node.params), inputs, context)
        except Exception as error:
            raise WorkflowValidationError(f"Node '{node_id}' ({spec.title}) failed: {error}") from error
        if not isinstance(output, dict):
            raise WorkflowValidationError(f"Node '{node_id}' did not return an output dictionary.")
        for port in spec.outputs:
            if port.slot not in output:
                raise WorkflowValidationError(f"Node '{node_id}' omitted output '{port.slot}'.")
        values[node_id] = output
        for field in spec.fields:
            if field.secret:
                node.params.pop(field.key, None)

    request = values[terminal.id]["result"]
    node_results = {
        node.id: values[node.id].get("result", {})
        for node in terminals
    }
    postprocess: List[Dict[str, Any]] = []
    if terminal.type == "image_preview":
        image_reference = request.get("image_ref", {}) if isinstance(request, dict) else {}
        postprocess = list(image_reference.get("operations", []))
        # A generator's image port is deferred until diffusion. Compile its
        # params so Sampler/Img2Img can feed Preview Image without Output.
        source_reference = terminal.inputs["image"]
        source_node = index[source_reference.split(".", 1)[0]]
        while source_node.type not in {"sampler", "img2img"}:
            if source_node.type == "output":
                source_reference = source_node.inputs["params"]
            elif "image" in source_node.inputs:
                source_reference = source_node.inputs["image"]
            else:
                break
            source_node = index[source_reference.split(".", 1)[0]]
        if source_node.type in {"sampler", "img2img"}:
            request = values[source_node.id]["params"]
            terminal = source_node
    if terminal.type not in {"output", "sampler", "img2img"}:
        if isinstance(request, dict):
            request = {**request, "_node_results": node_results}
        return "result", request
    if not isinstance(request, dict):
        raise WorkflowValidationError("Output must resolve to a generation request.")
    compiled = {key: value for key, value in request.items() if key in _ALLOWED_PARAM_KEYS and value is not None}
    if isinstance(compiled.get("seed"), (int, float)) and compiled["seed"] < 0:
        compiled.pop("seed")
    if not str(compiled.get("prompt", "")).strip():
        raise WorkflowValidationError("Workflow must resolve a non-empty image prompt.")
    job_type = "img2img" if compiled.get("init_image_id") else "text2img"
    if job_type == "img2img":
        compiled.setdefault("strength", .75)
    if postprocess:
        compiled["_postprocess"] = postprocess
    compiled["_node_results"] = node_results
    return job_type, compiled

def compile_workflow(workflow: WorkflowDefinition) -> Tuple[str, Dict[str, Any]]:
    return execute_workflow(workflow)

def redact_workflow(workflow: WorkflowDefinition) -> Dict[str, Any]:
    registry = get_node_registry()
    result = workflow.model_dump()
    for serialized, node in zip(result["nodes"], workflow.nodes):
        try:
            secret_fields = {field.key for field in registry.get(node.type).spec.fields if field.secret}
        except KeyError:
            continue
        for key in secret_fields:
            if serialized["params"].get(key):
                serialized["params"][key] = "[redacted]"
    return result
