"""
@file        test_schemas.py
@author      ParisNeo
@description Tests for Pydantic v2 schemas.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from openimagegen.schemas import (
    ImageGenerationRequest, JobStatus, JobSubmissionResponse,
)


class TestImageGenerationRequest:
    def test_minimal_only_prompt(self):
        """Verify that a request with only a prompt sets all other fields to None."""
        req = ImageGenerationRequest(prompt="a cat")
        assert req.prompt == "a cat"
        assert req.negative_prompt is None
        assert req.model_name is None
        assert req.height is None
        assert req.width is None
        assert req.num_inference_steps is None
        assert req.guidance_scale is None
        assert req.seed is None
        assert req.num_images_per_prompt is None

    def test_full_request(self):
        """Verify that all fields are correctly stored when every parameter is provided."""
        req = ImageGenerationRequest(
            prompt="dog", negative_prompt="blurry", model_name="sdxl",
            height=1024, width=1024, num_inference_steps=40,
            guidance_scale=8.0, seed=42, num_images_per_prompt=2,
        )
        assert req.prompt == "dog"
        assert req.model_name == "sdxl"
        assert req.num_inference_steps == 40
        assert req.seed == 42

    def test_steps_alias_populates_field(self):
        """Verify that the ``steps`` alias correctly maps to ``num_inference_steps``."""
        req = ImageGenerationRequest(prompt="x", steps=30)
        assert req.num_inference_steps == 30

    def test_field_name_also_works(self):
        """Verify that the canonical field name ``num_inference_steps`` is accepted directly."""
        req = ImageGenerationRequest(prompt="x", num_inference_steps=25)
        assert req.num_inference_steps == 25

    def test_missing_prompt_rejected(self):
        """Verify that omitting the required ``prompt`` field raises a ``ValidationError``."""
        with pytest.raises(ValidationError):
            ImageGenerationRequest()  # type: ignore[call-arg]

    def test_empty_prompt_allowed_at_schema(self):
        """Verify that an empty string prompt passes schema validation.

        The service layer is responsible for guarding against empty prompts;
        the Pydantic schema itself does not enforce non-emptiness.
        """
        req = ImageGenerationRequest(prompt="")
        assert req.prompt == ""

    def test_zero_height_rejected(self):
        """Verify that a height of zero is rejected by the schema validator."""
        with pytest.raises(ValidationError):
            ImageGenerationRequest(prompt="x", height=0)

    def test_negative_height_rejected(self):
        """Verify that a negative height value is rejected by the schema validator."""
        with pytest.raises(ValidationError):
            ImageGenerationRequest(prompt="x", height=-100)

    def test_zero_steps_rejected(self):
        """Verify that zero inference steps is rejected by the schema validator."""
        with pytest.raises(ValidationError):
            ImageGenerationRequest(prompt="x", num_inference_steps=0)

    def test_zero_guidance_rejected(self):
        """Verify that a guidance scale of zero is rejected by the schema validator."""
        with pytest.raises(ValidationError):
            ImageGenerationRequest(prompt="x", guidance_scale=0.0)

    def test_too_many_images_rejected(self):
        """Verify that requesting more images than the maximum allowed raises a ``ValidationError``."""
        with pytest.raises(ValidationError):
            ImageGenerationRequest(prompt="x", num_images_per_prompt=100)

    def test_zero_images_rejected(self):
        """Verify that requesting zero images is rejected by the schema validator."""
        with pytest.raises(ValidationError):
            ImageGenerationRequest(prompt="x", num_images_per_prompt=0)
class TestJobStatus:
    def _make(self, **overrides) -> JobStatus:
        """Build a ``JobStatus`` instance with overridable default fields.

        Provides a compact factory for the ``TestJobStatus`` test class,
        centralising the required fields (``job_id``, ``status``,
        ``created_at``, ``expires_at``) so that individual tests only
        declare the keyword arguments relevant to their assertions.

        Parameters
        ----------
        **overrides:
            Keyword arguments forwarded to the ``JobStatus`` constructor,
            overriding any of the default values.

        Returns
        -------
        JobStatus
            A validated Pydantic model instance ready for assertion.
        """
        defaults = dict(
            job_id="abc", status="pending",
            created_at=1000.0, expires_at=2000.0,
        )
        defaults.update(overrides)
        return JobStatus(**defaults)

    def test_minimal(self):
        s = self._make()
        assert s.job_id == "abc"
        assert s.status == "pending"
        assert s.progress == 0
        assert s.message is None
        assert s.image_urls is None

    def test_progress_bounds(self):
        with pytest.raises(ValidationError):
            self._make(progress=-1)
        with pytest.raises(ValidationError):
            self._make(progress=101)

    def test_with_image_urls(self):
        s = self._make(status="completed", image_urls=["/download/abc/0"])
        assert s.image_urls == ["/download/abc/0"]


class TestJobSubmissionResponse:
    def test_default_message(self):
        r = JobSubmissionResponse(job_id="xyz")
        assert r.message == "Job submitted successfully"

    def test_custom_message(self):
        r = JobSubmissionResponse(job_id="xyz", message="custom")
        assert r.message == "custom"