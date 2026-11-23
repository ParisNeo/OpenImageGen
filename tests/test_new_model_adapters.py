
"""Focused tests for the additional Diffusers model adapters."""

from unittest.mock import MagicMock

from openimagegen.models import get_registry
from openimagegen.models.pixart import PixArtSigmaAdapter
from openimagegen.models.sana import SanaAdapter
from openimagegen.models.stable_diffusion_3 import StableDiffusion3Adapter
from openimagegen.models.qwen_image import QwenImageEditAdapter, QwenImageLayeredAdapter


def test_service_metadata_is_not_forwarded_to_pipeline_constructors():
    for adapter in (
        StableDiffusion3Adapter(repository="example/model", variant="fp16"),
        PixArtSigmaAdapter(repository="example/model", weight_name=None),
        SanaAdapter(repository="example/model", variant="bf16"),
    ):
        assert adapter._load_kwargs() == {}


def test_additional_adapters_are_registered():
    registered = set(get_registry().list_types())
    assert {"stable_diffusion_3", "pixart_sigma", "sana"} <= registered


def test_sd35_recommended_defaults():
    adapter = StableDiffusion3Adapter()
    assert adapter.get_default_params() == {
        "guidance_scale": 4.5,
        "num_inference_steps": 28,
        "num_images_per_prompt": 1,
    }


def test_pixart_forwards_generation_parameters():
    pipeline = MagicMock()
    pipeline.return_value.images = ["image"]
    adapter = PixArtSigmaAdapter()

    result = adapter.generate(
        pipeline=pipeline,
        prompt="a mountain observatory",
        negative_prompt="blurred",
        width=1024,
        height=1024,
        generator="generator",
        callback="callback",
    )

    assert result == ["image"]
    kwargs = pipeline.call_args.kwargs
    assert kwargs["guidance_scale"] == 4.5
    assert kwargs["num_inference_steps"] == 20
    assert kwargs["negative_prompt"] == "blurred"
    assert kwargs["callback"] == "callback"


def test_sana_forwards_generation_parameters():
    pipeline = MagicMock()
    pipeline.return_value.images = ["image"]
    adapter = SanaAdapter()

    result = adapter.generate(pipeline=pipeline, prompt="a paper lantern")

    assert result == ["image"]
    kwargs = pipeline.call_args.kwargs
    assert kwargs["guidance_scale"] == 4.5
    assert kwargs["num_inference_steps"] == 20
    assert "negative_prompt" not in kwargs
    assert "height" not in kwargs
    assert "width" not in kwargs
