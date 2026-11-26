"""
@file        test_inpainting_schemas.py
@author      ParisNeo
@description Tests for inpaint/outpaint request validation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from openimagegen.schemas import InpaintingRequest, OutpaintingRequest


def test_inpainting_requires_init_and_mask_ids():
    req = InpaintingRequest(
        prompt="fix the sky",
        init_image_id="abc",
        mask_image_id="xyz",
    )
    assert req.init_image_id == "abc"
    assert req.mask_image_id == "xyz"
    assert req.strength == 0.85


def test_inpainting_requires_mask_id():
    with pytest.raises(ValidationError):
        InpaintingRequest(prompt="x", init_image_id="abc")  # type: ignore[call-arg]


def test_outpainting_requires_at_least_one_direction():
    with pytest.raises(ValidationError):
        OutpaintingRequest(prompt="x", init_image_id="abc")


def test_outpainting_accepts_single_direction():
    req = OutpaintingRequest(prompt="x", init_image_id="abc", expand_left=64)
    assert req.expand_left == 64
    assert req.expand_right == 0


def test_outpainting_rejects_oversized_total():
    with pytest.raises(ValidationError):
        OutpaintingRequest(
            prompt="x",
            init_image_id="abc",
            expand_left=1500,
            expand_right=1000,
        )