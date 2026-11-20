
"""Tests for registry discovery, remote text execution, and workflow validation."""

import json
from unittest.mock import MagicMock, patch

from openimagegen.nodes import NodeExecutionContext, get_node_registry
from openimagegen.nodes.openai_compatible import OpenAICompatibleNode, _endpoint
from openimagegen.workflow import WorkflowDefinition, execute_workflow, redact_workflow, validate_workflow


def _remote_graph():
    return WorkflowDefinition.model_validate({
        "nodes": [
            {"id": "remote", "type": "openai_compatible", "params": {
                "base_url": "http://localhost:1234/v1", "model": "test", "prompt": "a fox",
                "auth_mode": "none",
            }},
            {"id": "model", "type": "model", "params": {"model_name": "stable_diffusion_1_5"}},
            {"id": "sample", "type": "sampler", "inputs": {"prompt": "remote.text", "model": "model.model"}},
            {"id": "out", "type": "output", "inputs": {"params": "sample.params"}},
        ]
    })


def test_openai_node_is_discoverable_and_typed():
    spec = get_node_registry().get("openai_compatible").spec
    assert spec.outputs[0].kind == "text"
    assert next(field for field in spec.fields if field.key == "api_key").secret


def test_openai_graph_can_feed_sampler():
    validate_workflow(_remote_graph())


def test_openai_graph_executes_with_mocked_transport():
    payload = {"choices": [{"message": {"content": "a red fox in a forest"}}]}
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode()
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    opener = MagicMock()
    opener.open.return_value = response
    with patch("openimagegen.nodes.openai_compatible.build_opener", return_value=opener):
        job_type, request = execute_workflow(_remote_graph())
    assert job_type == "text2img"
    assert request["prompt"] == "a red fox in a forest"
    request_body = json.loads(opener.open.call_args.args[0].data)
    assert request_body["model"] == "test"


def test_output_can_feed_generated_image_visualizer():
    graph = WorkflowDefinition.model_validate({"nodes": [
        {"id": "prompt", "type": "text_prompt", "params": {"text": "a rabbit at a gala"}},
        {"id": "sample", "type": "sampler", "inputs": {"prompt": "prompt.text"}},
        {"id": "out", "type": "output", "inputs": {"params": "sample.params"}},
        {"id": "preview", "type": "image_preview", "inputs": {"image": "out.image"}},
    ]})
    job_type, request = execute_workflow(graph)
    assert job_type == "text2img"
    assert request["prompt"] == "a rabbit at a gala"
    assert get_node_registry().get("output").spec.outputs[0].kind == "image"


def test_sampler_image_output_can_feed_visualizer_directly():
    graph = WorkflowDefinition.model_validate({"nodes": [
        {"id": "prompt", "type": "text_prompt", "params": {"text": "a rabbit at a gala"}},
        {"id": "sample", "type": "sampler", "inputs": {"prompt": "prompt.text"}},
        {"id": "preview", "type": "image_preview", "inputs": {"image": "sample.image"}},
    ]})
    job_type, request = execute_workflow(graph)
    assert job_type == "text2img"
    assert request["prompt"] == "a rabbit at a gala"


def test_lora_chain_compiles_into_generation_request():
    graph = WorkflowDefinition.model_validate({"nodes": [
        {"id": "prompt", "type": "text_prompt", "params": {"text": "a rabbit at a gala"}},
        {"id": "base", "type": "model", "params": {"model_name": "stable_diffusion_1_5"}},
        {"id": "style", "type": "lora", "params": {
            "source": "owner/style-lora", "weight_name": "style.safetensors", "weight": 0.65,
        }, "inputs": {"model": "base.model"}},
        {"id": "sample", "type": "sampler", "inputs": {
            "prompt": "prompt.text", "model": "style.model",
        }},
        {"id": "preview", "type": "image_preview", "inputs": {"image": "sample.image"}},
    ]})
    job_type, request = execute_workflow(graph)
    assert job_type == "text2img"
    assert request["model_name"] == "stable_diffusion_1_5"
    assert request["loras"] == [{
        "source": "owner/style-lora", "weight_name": "style.safetensors", "weight": 0.65,
    }]


def test_generated_image_can_flow_through_upscale_and_filter():
    graph = WorkflowDefinition.model_validate({"nodes": [
        {"id": "prompt", "type": "text_prompt", "params": {"text": "a rabbit at a gala"}},
        {"id": "sample", "type": "sampler", "inputs": {"prompt": "prompt.text"}},
        {"id": "upscale", "type": "image_upscale", "params": {
            "scale": "2", "method": "lanczos",
        }, "inputs": {"image": "sample.image"}},
        {"id": "filter", "type": "image_filter", "params": {
            "filter": "sharpen", "radius": 2,
        }, "inputs": {"image": "upscale.image"}},
        {"id": "preview", "type": "image_preview", "inputs": {"image": "filter.image"}},
    ]})
    job_type, request = execute_workflow(graph)
    assert job_type == "text2img"
    assert request["_postprocess"] == [
        {"operation": "upscale", "scale": 2.0, "method": "lanczos"},
        {"operation": "filter", "filter": "sharpen", "radius": 2.0},
    ]


def test_uploaded_image_processing_returns_materializable_reference():
    graph = WorkflowDefinition.model_validate({"nodes": [
        {"id": "source", "type": "image_loader", "params": {"image_id": "upload-id"}},
        {"id": "resize", "type": "image_resize", "params": {
            "width": 640, "height": 480, "method": "bicubic",
        }, "inputs": {"image": "source.image"}},
        {"id": "preview", "type": "image_preview", "inputs": {"image": "resize.image"}},
    ]})
    job_type, result = execute_workflow(graph)
    assert job_type == "result"
    assert result["image_ref"]["source"] == "upload-id"
    assert result["image_ref"]["operations"][0]["operation"] == "resize"


def test_text_and_generation_previews_execute_together():
    graph = WorkflowDefinition.model_validate({"nodes": [
        {"id": "source", "type": "text_prompt", "params": {"text": "a gala rabbit"}},
        {"id": "sample", "type": "sampler", "inputs": {"prompt": "source.text"}},
        {"id": "out", "type": "output", "inputs": {"params": "sample.params"}},
        {"id": "text", "type": "text_output", "inputs": {"text": "source.text"}},
    ]})
    job_type, request = execute_workflow(graph)
    assert job_type == "text2img"
    assert request["prompt"] == "a gala rabbit"
    assert request["_node_results"]["text"] == {"text": "a gala rabbit"}

def test_text_preview_workflow_without_gpu():
    graph = WorkflowDefinition.model_validate({"nodes": [
        {"id": "source", "type": "text_prompt", "params": {"text": "hello"}},
        {"id": "preview", "type": "text_output", "inputs": {"text": "source.text"}},
    ]})
    job_type, result = execute_workflow(graph)
    assert job_type == "result"
    assert result["text"] == "hello"
    assert result["_node_results"] == {"preview": {"text": "hello"}}


def test_endpoint_normalization():
    assert _endpoint("http://localhost:1234/v1")[0] == "http://localhost:1234/v1/chat/completions"
    assert _endpoint("https://example.test/v1/chat/completions")[1] is True


def test_openai_node_supports_opt_in_browser_persistence_metadata():
    fields = {field.key: field for field in get_node_registry().get("openai_compatible").spec.fields}
    assert fields["api_key"].secret is True
    assert fields["remember_api_key"].default is False
    assert fields["remember_api_key"].secret is False


def test_secret_is_redacted_from_job_details():
    workflow = WorkflowDefinition.model_validate({
        "nodes": [{"id": "remote", "type": "openai_compatible", "params": {
            "base_url": "http://localhost:1234/v1", "model": "test", "api_key": "secret"
        }}]
    })
    details = redact_workflow(workflow)
    assert details["nodes"][0]["params"]["api_key"] == "[redacted]"
    assert "secret" not in json.dumps(details)
