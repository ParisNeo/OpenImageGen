
"""
@file        schemas.py
@author      ParisNeo
@description Pydantic v2 schemas for generation, model assets, and administration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    sampler: Optional[Literal[
        "dpmpp_2m", "dpmpp_2m_karras", "dpmpp_2m_sde_karras",
        "dpmpp_sde_karras", "unipc", "flowmatch_euler", "euler_a", "euler", "ddim",
    ]] = None
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


class InpaintingRequest(ImageGenerationRequest):
    """Request body for POST /submit_inpainting.

    ``init_image_id`` and ``mask_image_id`` reference images previously uploaded
    via POST /upload. The mask's white pixels mark regions that should be
    regenerated; black pixels are preserved from the init image.
    """

    init_image_id: str = Field(..., description="ID of the base image to inpaint")
    mask_image_id: str = Field(..., description="ID of the greyscale mask (white = edit)")
    strength: float = Field(0.85, gt=0, le=1.0)


class OutpaintingRequest(ImageGenerationRequest):
    """Request body for POST /submit_outpainting.

    Expands an image outward by generating brand-new pixels around the original.
    The service composes an enlarged canvas, pastes the original into the
    requested position, and builds a mask covering only the new area.
    """

    init_image_id: str = Field(..., description="ID of the source image")
    expand_left: int = Field(0, ge=0, le=2048)
    expand_right: int = Field(0, ge=0, le=2048)
    expand_top: int = Field(0, ge=0, le=2048)
    expand_bottom: int = Field(0, ge=0, le=2048)
    strength: float = Field(0.9, gt=0, le=1.0)

    @model_validator(mode="after")
    def validate_expansion(self) -> "OutpaintingRequest":
        total = self.expand_left + self.expand_right
        vertical = self.expand_top + self.expand_bottom
        if total == 0 and vertical == 0:
            raise ValueError("At least one expansion direction must be greater than 0.")
        if total > 2048 or vertical > 2048:
            raise ValueError("Horizontal or vertical expansion cannot exceed 2048px combined.")
        return self

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

class RemoteImageModelConfig(BaseModel):
    """Persistent, non-secret configuration for a hosted image-generation model."""

    name: str = Field(..., min_length=1, max_length=256)
    base_url: str = Field("https://api.openai.com/v1", min_length=1, max_length=2048)
    api_key_env: str = Field("OPENAI_API_KEY", max_length=128)
    timeout_seconds: int = Field(180, ge=1, le=600)
    verify_ssl: bool = True
    quality: Optional[str] = Field(None, max_length=64)

    @field_validator("name", "base_url")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("This value cannot be empty.")
        return value

    @field_validator("api_key_env")
    @classmethod
    def validate_environment_variable(cls, value: str) -> str:
        value = value.strip()
        if value and (
            not value.replace("_", "").isalnum() or value[0].isdigit()
        ):
            raise ValueError("API key environment variable name is invalid.")
        return value

    def to_model_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            **self.model_dump(exclude_none=True),
            "type": "openai_image",
        }
        return config


class PreloadRequest(BaseModel):
    """Request body for POST /preload."""

    model_keys: Optional[List[str]] = None


class PromptEnhanceRequest(BaseModel):
    """Request body for POST /v1/prompt/enhance."""

    prompt: str = Field(..., min_length=1, max_length=8000)
    negative_prompt: Optional[str] = Field(None, max_length=4000)
    model_name: Optional[str] = Field(None, max_length=128)

    @field_validator("prompt")
    @classmethod
    def require_visible_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt must contain visible text.")
        return value


class PromptEnhanceResponse(BaseModel):
    """LLM-produced prompt rewrite, optionally with a matching negative prompt."""

    prompt: str
    negative_prompt: Optional[str] = None

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
