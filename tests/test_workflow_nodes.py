
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


def test_text_preview_workflow_without_gpu():
    graph = WorkflowDefinition.model_validate({"nodes": [
        {"id": "source", "type": "text_prompt", "params": {"text": "hello"}},
        {"id": "preview", "type": "text_output", "inputs": {"text": "source.text"}},
    ]})
    assert execute_workflow(graph) == ("result", {"text": "hello"})


def test_endpoint_normalization():
    assert _endpoint("http://localhost:1234/v1")[0] == "http://localhost:1234/v1/chat/completions"
    assert _endpoint("https://example.test/v1/chat/completions")[1] is True


def test_secret_is_redacted_from_job_details():
    workflow = WorkflowDefinition.model_validate({
        "nodes": [{"id": "remote", "type": "openai_compatible", "params": {
            "base_url": "http://localhost:1234/v1", "model": "test", "api_key": "secret"
        }}]
    })
    details = redact_workflow(workflow)
    assert details["nodes"][0]["params"]["api_key"] == "[redacted]"
    assert "secret" not in json.dumps(details)
