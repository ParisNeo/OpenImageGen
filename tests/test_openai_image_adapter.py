
"""Tests for the OpenAI-compatible remote image adapter."""

import base64
import io
import os
from unittest.mock import MagicMock, patch

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


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
@patch("openimagegen.models.openai_image.requests.post")
def test_remote_generation_decodes_base64_image(post):
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "data": [{"b64_json": base64.b64encode(_image_bytes()).decode()}]
    }
    post.return_value = response
    pipeline = RemoteImagePipeline({"remote_model": "gpt-image-1", "quality": "high"})
    images = OpenAIImageAdapter().generate(
        pipeline=pipeline, prompt="a small observatory", width=1024, height=1024
    )
    assert len(images) == 1
    assert images[0].size == (8, 8)
    payload = post.call_args.kwargs["json"]
    assert payload == {
        "model": "gpt-image-1",
        "prompt": "a small observatory",
        "n": 1,
        "size": "1024x1024",
        "quality": "high",
    }


@patch.dict(os.environ, {}, clear=True)
def test_missing_remote_credentials_fail_explicitly():
    adapter = OpenAIImageAdapter()
    pipeline = adapter.load_pipeline("openai/gpt-image-1", "./models", None, None)
    try:
        adapter.generate(pipeline=pipeline, prompt="test")
    except ValueError as error:
        assert "OPENAI_API_KEY" in str(error)
    else:
        raise AssertionError("Missing credentials were not rejected.")
