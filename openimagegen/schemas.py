
"""
@file        schemas.py
@author      ParisNeo
@description Pydantic v2 schemas for generation, model assets, and administration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

class LoRARequest(BaseModel):
    """A LoRA weight file or Hugging Face repository and its generation strength."""

    source: str = Field(..., min_length=1, max_length=512)
    weight_name: Optional[str] = Field(None, max_length=512)
    weight: float = Field(1.0, ge=0.0, le=4.0)
    adapter_name: Optional[str] = Field(None, max_length=64)

class ImageGenerationRequest(BaseModel):
    """Request body for POST /submit."""

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt (model-dependent support)")
    model_name: Optional[str] = Field(None, description="Configured model key")
    height: Optional[int] = Field(None, gt=0, le=4096)
    width: Optional[int] = Field(None, gt=0, le=4096)
    num_inference_steps: Optional[int] = Field(None, alias="steps", gt=0, le=500)
    guidance_scale: Optional[float] = Field(None, ge=0, le=50)
    sampler: Optional[Literal["dpmpp_2m_karras", "euler_a", "euler", "ddim"]] = None
    seed: Optional[int] = Field(None, ge=0)
    num_images_per_prompt: Optional[int] = Field(None, gt=0, le=8)
    loras: List[LoRARequest] = Field(default_factory=list, max_length=8)

    @field_validator("width", "height")
    @classmethod
    def validate_dimension_divisible_by_8(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value % 8 != 0:
            raise ValueError("Image dimensions must be divisible by 8.")
        return value

class ImageToImageRequest(ImageGenerationRequest):
    """Request body for POST /submit_img2img."""

    init_image_id: str = Field(..., description="ID returned by POST /upload")
    strength: float = Field(0.75, gt=0, le=1.0)

class JobStatus(BaseModel):
    """Status of a generation job."""

    job_id: str
    status: Literal["pending", "processing", "completed", "failed", "cancelled"]
    job_type: Literal["text2img", "img2img"] = "text2img"
    progress: int = Field(0, ge=0, le=100)
    message: Optional[str] = None
    image_urls: Optional[List[str]] = None
    created_at: float
    expires_at: float
    seed_used: Optional[int] = None
    owner: Optional[str] = None
    request_details: Optional[Any] = None
    node_results: Dict[str, Any] = Field(default_factory=dict)
    active_node_ids: List[str] = Field(default_factory=list)

class PreloadRequest(BaseModel):
    """Request body for POST /preload."""

    model_keys: Optional[List[str]] = None

class JobSubmissionResponse(BaseModel):
    job_id: str
    message: str = "Job submitted successfully"

class AssetDefinition(BaseModel):
    """An uploaded, downloaded, or registered model asset."""

    name: str = Field(..., min_length=1, max_length=128)
    source: str = Field(..., min_length=1, max_length=2048)
    kind: Literal["lora", "gguf", "custom"] = "custom"
    weight_name: Optional[str] = Field(None, max_length=512)

class AssetUploadRequest(BaseModel):
    """Optional metadata for an uploaded model asset."""

    kind: Literal["lora", "gguf", "custom"] = "custom"

class RuntimeSettingsUpdate(BaseModel):
    """Settings that can safely change without reconstructing model pipelines."""

    default_model: Optional[str] = Field(None, max_length=128)
    auto_download_models: Optional[bool] = None
    file_retention_time: Optional[int] = Field(None, ge=60)
    cleanup_interval_seconds: Optional[int] = Field(None, ge=30)
    generation: Optional[Dict[str, Any]] = None
    execution: Optional[Dict[str, Any]] = None

class APIUserCreate(BaseModel):
    """Create an isolated API identity."""

    user_id: str = Field(..., pattern=r"^[A-Za-z0-9_-]{2,64}$")
    is_admin: bool = False
    max_concurrent_jobs: int = Field(2, ge=1, le=64)
    quota_per_hour: int = Field(200, ge=1, le=100000)

class APIKeyRotationResponse(BaseModel):
    """The only response that contains a user's API key."""

    user_id: str
    api_key: str

class WorkflowSubmissionResponse(JobSubmissionResponse):
    pass
