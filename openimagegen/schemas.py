"""Pydantic v2 schemas for the OpenImageGen API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ImageGenerationRequest(BaseModel):
    """Request body for POST /submit."""
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt (model-dependent support)")
    model_name: Optional[str] = Field(None, description="Model key from config; uses default if omitted")
    height: Optional[int] = Field(None, gt=0, description="Image height in pixels")
    width: Optional[int] = Field(None, gt=0, description="Image width in pixels")
    num_inference_steps: Optional[int] = Field(
        None, alias="steps", gt=0, description="Denoising steps (alias: steps)"
    )
    guidance_scale: Optional[float] = Field(None, gt=0, description="Classifier-free guidance scale")
    seed: Optional[int] = Field(None, description="Seed for reproducibility; None = random")
    num_images_per_prompt: Optional[int] = Field(
        None, gt=0, le=8, description="Number of images per prompt"
    )


class JobStatus(BaseModel):
    """Status of a generation job, polled via GET /status/{job_id}."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="pending | processing | completed | failed")
    progress: int = Field(0, ge=0, le=100, description="Generation progress percentage")
    message: Optional[str] = Field(None, description="Human-readable status message")
    image_urls: Optional[List[str]] = Field(None, description="Download URLs (populated on completion)")
    created_at: float = Field(..., description="Unix timestamp of job creation")
    expires_at: float = Field(..., description="Unix timestamp when artifacts expire")
    request_details: Optional[ImageGenerationRequest] = Field(
        None, description="Echo of the original request"
    )


class JobSubmissionResponse(BaseModel):
    """Response from POST /submit."""
    job_id: str
    message: str = "Job submitted successfully"
