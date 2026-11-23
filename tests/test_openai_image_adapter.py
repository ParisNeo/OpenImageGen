
"""Tests for the OpenAI-compatible remote image adapter."""

import base64
import io
import json
import os
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from openimagegen.models import get_registry
from openimagegen.models.openai_image import OpenAIImageAdapter, RemoteImagePipeline


def _image_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), "navy").save(stream, format="PNG")
    return stream.getvalue()


def test_remote_adapter_is_registered_and_has_no_downloads():
    adapter = OpenAIImageAdapter()
    assert "openai_image" in get_registry().list_types()
    assert not adapter.supports_negative_prompt()
    assert not adapter.supports_lora()
    progress = MagicMock()
    adapter.download_weights("openai/gpt-image-1", "./models", progress)
    progress.assert_called_once_with(100)


@patch("openimagegen.models.openai_image.build_opener")
def test_remote_generation_decodes_base64_image(build_opener):
    response = MagicMock()
    response.read.return_value = json.dumps({
        "data": [{"b64_json": base64.b64encode(_image_bytes()).decode()}]
    }).encode()
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    opener = MagicMock()
    opener.open.return_value = response
    build_opener.return_value = opener

    pipeline = RemoteImagePipeline(
        model="gpt-image-1",
        endpoint="https://api.openai.com/v1/images/generations",
        api_key="test-key",
        timeout_seconds=180,
        verify_ssl=True,
    )
    images = OpenAIImageAdapter().generate(
        pipeline=pipeline,
        prompt="a small observatory",
        width=1024,
        height=1024,
    )

    assert len(images) == 1
    assert images[0].size == (8, 8)
    request = opener.open.call_args.args[0]
    assert json.loads(request.data) == {
        "model": "gpt-image-1",
        "prompt": "a small observatory",
        "n": 1,
        "size": "1024x1024",
    }
    assert request.get_header("Authorization") == "Bearer test-key"


@patch.dict(os.environ, {}, clear=True)
def test_missing_remote_credentials_fail_explicitly():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIImageAdapter().load_pipeline(
            "openai/gpt-image-1",
            "./models",
            None,
            None,
        )
