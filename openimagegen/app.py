
"""
@file        app.py
@author      ParisNeo
@description FastAPI application with multi-user API-key auth, bounded worker
pool execution, ComfyUI-style workflow compilation, WebSocket progress
streaming, static UI, image upload, and legacy routes.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image as PILImage

from .auth import AuthManager, UserContext
from .config import save_config
from .gguf_catalog import GGUF_PRESETS, build_gguf_model_config
from .schemas import (
    ImageGenerationRequest,
    ImageToImageRequest,
    InpaintingRequest,
    JobStatus,
    JobSubmissionResponse,
    OutpaintingRequest,
    PreloadRequest,
    PromptEnhanceRequest,
    PromptEnhanceResponse,
    RemoteImageModelConfig,
)
from .models import get_registry
from .nodes import get_node_registry
from .models.presets import generation_profiles, generation_templates, recommended_generation_params
from .nodes.openai_compatible import chat_completion, list_remote_models
from .service import ImageGenService
from .workflow import (
    WorkflowDefinition,
    WorkflowValidationError,
    node_specs,
    redact_workflow,
    validate_workflow,
)

logger = logging.getLogger("OpenImageGen.app")

service: Optional[ImageGenService] = None
auth_manager: Optional[AuthManager] = None

MAX_UPLOAD_SIZE = 10 * 1024 * 1024

_PROMPT_ENHANCE_SYSTEM = (
    "You are an expert prompt engineer for image-generation diffusion models "
    "(Stable Diffusion, SDXL, SD3, FLUX, Qwen Image). Rewrite the user's prompt "
    "so that it becomes a richer, more visually concrete generation prompt: keep "
    "the original subject and intent, add specific detail on composition, lighting, "
    "materials, camera or rendering style, and negative tokens where helpful. "
    "Do NOT invent a different subject, add people, or add brand names. "
    "Reply with a SINGLE JSON object only, no markdown, no comments, no other text. "
    "The JSON object MUST be an object with exactly these keys: "
    '"prompt" (string, the enhanced positive prompt) and '
    '"negative_prompt" (string, comma-separated negative tokens, or "" if not applicable).'
)

_PROMPT_ENHANCE_NEGATIVE_RULE = (
    " The target model family does NOT support negative prompts: set "
    '"negative_prompt" to an empty string in your JSON reply.'
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup/shutdown: service + auth bootstrap, cleanup task."""
    global service, auth_manager
    logger.info("OpenImageGen starting up...")
    try:
        service = ImageGenService()
        service.ws_manager.set_loop(asyncio.get_event_loop())
        auth_manager = AuthManager(service.config)
        if auth_manager.enabled:
            logger.info(f"Auth enabled: {len(auth_manager.list_users())} API key(s) registered.")
        else:
            logger.info("Auth disabled — all requests resolve to admin/anonymous.")
    except Exception as e:
        logger.critical(f"Service initialization failed: {e}", exc_info=True)
        raise

    cleanup_task = asyncio.create_task(_periodic_cleanup())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        if service is not None:
            try:
                service.shutdown()
            except Exception as e:
                logger.debug(f"Executor shutdown warning: {e}")
        logger.info("OpenImageGen shut down.")


async def _periodic_cleanup() -> None:
    """Background task that runs cleanup_expired_files on a fixed interval."""
    while True:
        try:
            await asyncio.sleep(service.cleanup_interval_seconds if service else 900)
        except asyncio.CancelledError:
            raise
        try:
            if service is not None:
                cleaned = service.cleanup_expired_files()
                if cleaned:
                    logger.info(f"Periodic cleanup removed {cleaned} expired job(s).")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Periodic cleanup error: {e}", exc_info=True)


app = FastAPI(
    title="OpenImageGen API",
    description="Open source image generation API with multi-user auth and workflow graphs.",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_service() -> ImageGenService:
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")
    return service


def _model_supports_negative_prompt(svc: ImageGenService, model_name: Optional[str]) -> bool:
    """Return whether the selected model family accepts negative prompts."""
    key = model_name or svc.default_model
    info = svc.models_config.get(key) if key else None
    if not info:
        return True
    adapter_cls = svc.registry.try_get(info.get("type", ""))
    if adapter_cls is None:
        return True
    try:
        return bool(adapter_cls().supports_negative_prompt())
    except Exception:
        return True


def _build_enhance_user_message(request: PromptEnhanceRequest) -> str:
    """Compose the user turn sent to the enhancement LLM."""
    parts = [f"Original prompt:\n{request.prompt.strip()}"]
    negative = (request.negative_prompt or "").strip()
    if negative:
        parts.append(f"Original negative prompt:\n{negative}")
    parts.append("Return only the JSON object described in the system message.")
    return "\n\n".join(parts)


def _first_balanced_object(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` substring, respecting strings/escapes."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            character = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
        start = text.find("{", start + 1)
    return None


def _parse_enhancement(text: str) -> tuple[str, Optional[str]]:
    """Parse an LLM reply into (prompt, negative_prompt-or-None)."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("The LLM returned an empty prompt.")

    candidate = raw
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        # Drop a leading language label such as "json" from a fenced block.
        newline = candidate.find("\n")
        if newline != -1 and not candidate[:newline].strip().startswith("{"):
            candidate = candidate[newline + 1:]
        candidate = candidate.strip()

    payload = None
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        block = _first_balanced_object(candidate)
        if block is not None:
            try:
                payload = json.loads(block)
            except json.JSONDecodeError:
                payload = None

    if not isinstance(payload, dict):
        return raw, None

    prompt_value = payload.get("prompt")
    if not isinstance(prompt_value, str) or not prompt_value.strip():
        raise ValueError("The LLM reply did not contain an enhanced prompt.")
    negative_value = payload.get("negative_prompt")
    if isinstance(negative_value, str) and negative_value.strip():
        negative: Optional[str] = negative_value.strip()
    else:
        negative = None
    return prompt_value.strip(), negative


def get_current_user(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
) -> UserContext:
    """Resolve X-API-Key or Authorization: Bearer credentials."""
    token = x_api_key
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if auth_manager is None:
        return UserContext(user_id="anonymous", is_admin=True)
    return auth_manager.resolve(token)


def _require_admin(user: UserContext) -> None:
    if auth_manager is not None and auth_manager.enabled and not user.is_admin:
        raise HTTPException(status_code=403, detail="Administrator access required.")


def _check_job_access(user: UserContext, job: JobStatus) -> None:
    if auth_manager is None:
        return
    if not auth_manager.can_access(user, job.owner):
        raise HTTPException(status_code=403, detail="You do not have access to this job.")


def _submit_generation(
    svc: ImageGenService,
    user: UserContext,
    request: ImageGenerationRequest | ImageToImageRequest,
    job_type: str,
) -> JobSubmissionResponse:
    """Shared job submission: validation, quota, ownership, bounded dispatch."""
    if auth_manager is not None and auth_manager.enabled:
        auth_manager.enforce_quota(user.user_id)

    requested_model = request.model_name or svc.default_model
    if not svc.is_model_available(requested_model):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{requested_model}' is not downloaded. "
                f"Download it first via POST /preload or the CLI: "
                f"openimagegen download --models {requested_model}"
            ),
        )

    job_id = str(uuid.uuid4())
    created_at = time.time()
    expires_at = created_at + svc.file_retention_time

    owner_id = user.user_id if (auth_manager is not None and auth_manager.enabled) else None

    svc.jobs.add(JobStatus(
        job_id=job_id,
        status="pending",
        created_at=created_at,
        expires_at=expires_at,
        request_details=request,
        job_type=job_type,
        owner=owner_id,
    ))

    accepted = svc.submit_generation(job_id, request, job_type, owner=owner_id)
    if not accepted:
        svc.jobs.delete(job_id)
        raise HTTPException(
            status_code=429,
            detail=(
                f"User '{user.user_id}' already has the maximum number of concurrent "
                f"jobs running "
                f"({svc.max_concurrent_jobs_per_user}). Try again shortly."
            ),
        )

    preview = getattr(request, "prompt", "") or ""
    logger.info(
        f"Submitted {job_type} job {job_id} for user '{user.user_id}' "
        f"(prompt: '{preview[:50]}...')"
    )
    return JobSubmissionResponse(job_id=job_id)


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/webui")


@app.get("/health")
async def health_check() -> Dict:
    """Service health and configuration snapshot."""
    svc = _require_service()
    torch = __import__("torch")
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "default_model": svc.default_model,
        "loaded_models": list(svc.pipelines.keys()),
        "available_model_types": svc.registry.list_types(),
        "device": str(svc.device),
        "dtype": str(svc.dtype).split(".")[-1],
        "force_gpu": svc.force_gpu,
        "use_gpu": svc.use_gpu,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "is_loading_models": svc.is_loading,
        "loading_model_key": svc.loading_model_key,
        "is_downloading_models": svc.is_downloading,
        "download_status": svc.download_status,
        "download_progress": svc.download_progress,
        "download_jobs": svc.download_jobs.list(),
        "active_jobs": svc.jobs.count(),
        "output_folder": str(svc.output_folder.resolve()),
        "model_cache_dir": str(svc.model_cache_dir.resolve()),
        "memory_info": svc.get_memory_info(),
        "hardware": svc.hardware,
        "auth_enabled": bool(auth_manager and auth_manager.enabled),
        "max_concurrent_jobs_per_user": svc.max_concurrent_jobs_per_user,
        "prompt_enhancement": {
            "enabled": bool(svc.config.get("llm", {}).get("enabled", False)),
            "model": str(svc.config.get("llm", {}).get("model") or ""),
        },
    }


@app.get("/models")
async def get_models() -> Dict:
    """List configured model keys with their adapter types and capabilities."""
    svc = _require_service()
    models = []
    for key, info in svc.models_config.items():
        model_type = info.get("type", "unknown")
        adapter_cls = svc.registry.try_get(model_type)
        supports_neg = True
        supports_i2i = False
        supports_inpaint = False
        if adapter_cls is not None:
            try:
                tmp_adapter = adapter_cls()
                supports_neg = tmp_adapter.supports_negative_prompt()
                supports_i2i = tmp_adapter.supports_img2img()
                supports_inpaint = tmp_adapter.supports_inpainting()
            except Exception:
                pass
        models.append({
            "key": key,
            "type": model_type,
            "name": info.get("name", ""),
            "loaded": key in svc.pipelines,
            "downloaded": svc._is_model_downloaded(key),
            "supports_negative_prompt": supports_neg,
            "supports_img2img": supports_i2i,
            "supports_inpainting": supports_inpaint,
            "generation_defaults": recommended_generation_params(key, info),
            "generation_profiles": generation_profiles(key, info),
            "execution_hint": svc.execution_hint(key, info),
        })
    return {"available_models": models}


@app.get("/v1/api-models")
async def list_api_models(user: UserContext = Depends(get_current_user)) -> Dict:
    """List hosted image-provider configurations without exposing credentials."""
    _require_admin(user)
    svc = _require_service()
    models = [
        {"key": key, **info}
        for key, info in svc.models_config.items()
        if info.get("type") == "openai_image"
    ]
    return {"models": models}


@app.put("/v1/api-models/{model_key}")
async def configure_api_model(
    model_key: str,
    request: RemoteImageModelConfig,
    user: UserContext = Depends(get_current_user),
) -> Dict:
    """Create or replace a hosted image model and persist it immediately."""
    _require_admin(user)
    svc = _require_service()
    if not 2 <= len(model_key) <= 64 or not model_key.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(
            status_code=400,
            detail="Model key must be 2–64 letters, digits, underscores, or hyphens.",
        )

    model_config = request.to_model_config()
    previous = svc.models_config.get(model_key)
    svc.models_config[model_key] = model_config
    svc.config.setdefault("models", {})[model_key] = model_config
    try:
        save_config(svc.config)
    except Exception as error:
        if previous is None:
            svc.models_config.pop(model_key, None)
            svc.config.get("models", {}).pop(model_key, None)
        else:
            svc.models_config[model_key] = previous
            svc.config["models"][model_key] = previous
        raise HTTPException(status_code=500, detail=f"Could not save configuration: {error}") from error

    if previous != model_config and model_key in svc.pipelines:
        svc.unload_model(model_key)
    return {"key": model_key, **model_config}


@app.delete("/v1/api-models/{model_key}")
async def delete_api_model(
    model_key: str,
    user: UserContext = Depends(get_current_user),
) -> Dict:
    """Remove a hosted model configuration and unload any stale pipeline."""
    _require_admin(user)
    svc = _require_service()
    previous = svc.models_config.get(model_key)
    if previous is None or previous.get("type") != "openai_image":
        raise HTTPException(status_code=404, detail="Hosted image model not found.")

    del svc.models_config[model_key]
    svc.config.get("models", {}).pop(model_key, None)
    try:
        save_config(svc.config)
    except Exception as error:
        svc.models_config[model_key] = previous
        svc.config.setdefault("models", {})[model_key] = previous
        raise HTTPException(status_code=500, detail=f"Could not save configuration: {error}") from error
    svc.unload_model(model_key)
    return {"status": "deleted", "key": model_key}


@app.get("/v1/generation/templates")
async def get_generation_templates() -> Dict:
    """Return shared prompt ideas; family-aware settings are resolved by each client."""
    return {"templates": generation_templates()}


@app.post("/v1/prompt/enhance", response_model=PromptEnhanceResponse)
async def enhance_prompt(
    request: PromptEnhanceRequest,
    user: UserContext = Depends(get_current_user),
) -> PromptEnhanceResponse:
    """Rewrite a prompt using the configured OpenAI-compatible LLM."""
    svc = _require_service()
    llm_cfg = svc.config.get("llm") or {}
    if not llm_cfg.get("enabled", False):
        raise HTTPException(
            status_code=400,
            detail=(
                "Prompt enhancement is disabled. Enable [llm].enabled in the configuration "
                "and set the matching API key environment variable."
            ),
        )

    model_id = str(llm_cfg.get("model") or "").strip()
    if not model_id:
        raise HTTPException(
            status_code=400,
            detail="Prompt enhancement requires [llm].model to be configured.",
        )

    if auth_manager is not None and auth_manager.enabled:
        auth_manager.enforce_quota(user.user_id)

    supports_negative = _model_supports_negative_prompt(svc, request.model_name)
    system_prompt = _PROMPT_ENHANCE_SYSTEM
    custom_system = str(llm_cfg.get("system_prompt") or "").strip()
    if custom_system:
        system_prompt = f"{system_prompt}\n\nAdditional instructions:\n{custom_system}"
    if not supports_negative:
        system_prompt = f"{system_prompt}{_PROMPT_ENHANCE_NEGATIVE_RULE}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_enhance_user_message(request)},
    ]

    try:
        reply = await asyncio.to_thread(
            chat_completion,
            str(llm_cfg.get("base_url") or "http://127.0.0.1:11434/v1"),
            model_id,
            messages,
            api_key_env=str(llm_cfg.get("api_key_env") or "OPENAI_API_KEY"),
            timeout_seconds=int(llm_cfg.get("timeout_seconds", 60)),
            verify_ssl=bool(llm_cfg.get("verify_ssl", True)),
            temperature=float(llm_cfg.get("temperature", 0.7)),
            max_tokens=int(llm_cfg.get("max_tokens", 512)),
        )
        enhanced_prompt, enhanced_negative = _parse_enhancement(reply)
    except ValueError as error:
        logger.warning(f"Prompt enhancement produced an unusable reply: {error}")
        raise HTTPException(status_code=502, detail=f"Prompt enhancement failed: {error}") from error
    except RuntimeError as error:
        logger.warning(f"Prompt enhancement transport error: {error}")
        raise HTTPException(status_code=502, detail=f"Prompt enhancement failed: {error}") from error

    if not supports_negative:
        enhanced_negative = None

    return PromptEnhanceResponse(prompt=enhanced_prompt, negative_prompt=enhanced_negative)


@app.post("/preload")
async def preload_models(
    request: PreloadRequest,
    user: UserContext = Depends(get_current_user),
):
    """Trigger background download of model weights. Admin-only when auth is enabled."""
    svc = _require_service()
    if auth_manager is not None and auth_manager.enabled and not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required for model preloading.")
    if svc.is_downloading:
        raise HTTPException(status_code=409, detail="A download operation is already in progress.")

    target_keys = request.model_keys or list(svc.models_config.keys())
    invalid_keys = [k for k in target_keys if k not in svc.models_config]
    if invalid_keys:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model key(s): {', '.join(invalid_keys)}. "
                   f"Available: {', '.join(svc.models_config.keys())}",
        )

    download_job = svc.start_download(target_keys)
    return {"status": "preloading_started", "models": target_keys, "download_job": download_job}


@app.post("/models/{model_key}/load")
async def load_model_endpoint(
    model_key: str,
    user: UserContext = Depends(get_current_user),
):
    """Explicitly load a model pipeline into memory."""
    svc = _require_service()
    if auth_manager is not None and auth_manager.enabled and not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required.")
    if model_key not in svc.models_config:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' is not configured.")
    if model_key in svc.pipelines:
        return {"status": "already_loaded", "model": model_key}
    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(None, svc.load_model, model_key)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to load model '{model_key}'.")
    return {"status": "loaded", "model": model_key}


@app.post("/models/{model_key}/unload")
async def unload_model_endpoint(
    model_key: str,
    user: UserContext = Depends(get_current_user),
):
    """Unload a model pipeline from memory to free resources."""
    svc = _require_service()
    if auth_manager is not None and auth_manager.enabled and not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required.")
    loop = asyncio.get_event_loop()
    unloaded = await loop.run_in_executor(None, svc.unload_model, model_key)
    if not unloaded:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' is not loaded.")
    return {"status": "unloaded", "model": model_key}


@app.get("/gguf/presets")
async def list_gguf_presets() -> Dict:
    """Return curated GGUF presets and their complete installation state."""
    svc = _require_service()
    presets = []
    for preset in GGUF_PRESETS.values():
        model_key = preset["model_key"]
        presets.append({
            **preset,
            "installed": model_key in svc.models_config,
            "downloaded": svc._is_model_downloaded(model_key),
            "loaded": model_key in svc.pipelines,
            "execution_hint": svc.execution_hint(
                model_key,
                svc.models_config.get(model_key) or build_gguf_model_config(preset["id"]),
            ),
            "total_size_note": (
                "Transformer size only. FLUX text encoders, VAE, tokenizer, and scheduler "
                "are downloaded separately."
            ),
        })
    return {"presets": presets}


@app.post("/gguf/presets/{preset_id}/install")
async def install_gguf_preset(
    preset_id: str,
    user: UserContext = Depends(get_current_user),
) -> Dict:
    """Persist a preset and start downloading its base components and GGUF file."""
    _require_admin(user)
    svc = _require_service()
    if preset_id not in GGUF_PRESETS:
        raise HTTPException(status_code=404, detail="Unknown GGUF preset.")

    preset = GGUF_PRESETS[preset_id]
    model_key = preset["model_key"]
    newly_installed = model_key not in svc.models_config
    if newly_installed:
        model_config = build_gguf_model_config(preset_id)
        svc.models_config[model_key] = model_config
        svc.config.setdefault("models", {})[model_key] = model_config
        try:
            save_config(svc.config)
        except Exception as error:
            svc.models_config.pop(model_key, None)
            svc.config.get("models", {}).pop(model_key, None)
            raise HTTPException(status_code=500, detail=f"Could not save configuration: {error}")

    download_job = None
    if not svc._is_model_downloaded(model_key):
        download_job = svc.start_download([model_key])

    return {
        "model_key": model_key,
        "installed": True,
        "newly_installed": newly_installed,
        "download_started": bool(download_job and download_job.get("status") in {"pending", "running"}),
        "download_job": download_job,
        "downloaded": svc._is_model_downloaded(model_key),
    }


@app.get("/v1/downloads")
async def list_downloads(user: UserContext = Depends(get_current_user)) -> Dict:
    """List observable model downloads, including queued and completed jobs."""
    _require_admin(user)
    jobs = _require_service().download_jobs.list()
    return {"jobs": jobs, "count": len(jobs)}


@app.get("/v1/downloads/{download_id}")
async def get_download(download_id: str, user: UserContext = Depends(get_current_user)) -> Dict:
    _require_admin(user)
    job = _require_service().download_jobs.get(download_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Download job not found.")
    return job


@app.get("/memory")
async def get_memory_info():
    svc = _require_service()
    return svc.get_memory_info()


# ---------------------------------------------------------------------------
# Job endpoints
# ---------------------------------------------------------------------------

@app.post("/cancel/{job_id}")
async def cancel_job(job_id: str, user: UserContext = Depends(get_current_user)):
    """Cancel a pending or processing job (owner or admin only)."""
    svc = _require_service()
    job = svc.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    _check_job_access(user, job)
    if not svc.cancel_job(job_id):
        raise HTTPException(status_code=409, detail=f"Job '{job_id}' cannot be cancelled.")
    return {"status": "cancelled", "job_id": job_id}


@app.get("/v1/jobs")
async def list_my_jobs(user: UserContext = Depends(get_current_user)):
    """List the current user's jobs, newest first (admins see everything)."""
    svc = _require_service()
    result = []
    for jid in svc.jobs.list_ids():
        job = svc.jobs.get(jid)
        if job is None:
            continue
        if auth_manager is not None and auth_manager.enabled and not auth_manager.can_access(user, job.owner):
            continue
        result.append({
            "job_id": job.job_id,
            "status": job.status,
            "job_type": job.job_type,
            "owner": job.owner,
            "progress": job.progress,
            "created_at": job.created_at,
            "image_urls": job.image_urls or [],
        })
    result.sort(key=lambda x: x["created_at"], reverse=True)
    return {"jobs": result, "count": len(result)}


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str, user: UserContext = Depends(get_current_user)):
    """Retrieve the current status of a job."""
    svc = _require_service()
    job = svc.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    _check_job_access(user, job)
    return job


@app.post("/submit", response_model=JobSubmissionResponse)
async def submit_job(
    request: ImageGenerationRequest,
    user: UserContext = Depends(get_current_user),
):
    """Submit a text-to-image generation job."""
    return _submit_generation(_require_service(), user, request, "text2img")


@app.post("/submit_img2img", response_model=JobSubmissionResponse)
async def submit_img2img_job(
    request: ImageToImageRequest,
    user: UserContext = Depends(get_current_user),
):
    """Submit an image-to-image generation job."""
    svc = _require_service()
    if svc.get_upload_path(request.init_image_id) is None:
        raise HTTPException(
            status_code=400,
            detail=f"Init image '{request.init_image_id}' not found. Upload first.",
        )
    return _submit_generation(svc, user, request, "img2img")


@app.post("/submit_inpainting", response_model=JobSubmissionResponse)
async def submit_inpainting_job(
    request: InpaintingRequest,
    user: UserContext = Depends(get_current_user),
):
    """Submit a mask-guided inpainting job."""
    svc = _require_service()
    if svc.get_upload_path(request.init_image_id) is None:
        raise HTTPException(
            status_code=400,
            detail=f"Init image '{request.init_image_id}' not found. Upload first.",
        )
    if svc.get_mask_path(request.mask_image_id) is None:
        raise HTTPException(
            status_code=400,
            detail=f"Mask '{request.mask_image_id}' not found. Upload first.",
        )
    return _submit_generation(svc, user, request, "inpainting")


@app.post("/submit_outpainting", response_model=JobSubmissionResponse)
async def submit_outpainting_job(
    request: OutpaintingRequest,
    user: UserContext = Depends(get_current_user),
):
    """Submit an outpainting job that expands an existing image outward."""
    svc = _require_service()
    if svc.get_upload_path(request.init_image_id) is None:
        raise HTTPException(
            status_code=400,
            detail=f"Init image '{request.init_image_id}' not found. Upload first.",
        )
    return _submit_generation(svc, user, request, "outpainting")


@app.post("/upload_mask")
async def upload_mask(
    file: UploadFile = File(...),
    user: UserContext = Depends(get_current_user),
):
    """Upload a mask image (any type). Returns ``mask_id`` for inpainting requests."""
    svc = _require_service()
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Mask must be an image.")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum 10 MB.")

    try:
        img = PILImage.open(io.BytesIO(contents))
        mask = img.convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid mask image.")

    mask_id = str(uuid.uuid4())
    svc.save_mask(mask, mask_id)
    return {"mask_id": mask_id, "url": f"/uploads/mask_{mask_id}"}


@app.get("/v1/nodes")
async def list_workflow_nodes() -> Dict:
    """Declarative schemas used by workflow UIs to render built-in and community nodes."""
    return {"nodes": node_specs(), "api_version": 1}


@app.post("/v1/nodes/openai_compatible/models")
async def discover_openai_compatible_models(
    request: Dict[str, Any],
    user: UserContext = Depends(get_current_user),
) -> Dict:
    """Discover remote models without persisting credentials in the workflow."""
    try:
        return {"models": list_remote_models(request)}
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/v1/nodes/reload")
async def reload_workflow_nodes(user: UserContext = Depends(get_current_user)) -> Dict:
    """Reload trusted Python community plugins after an administrator changes them."""
    _require_admin(user)
    svc = _require_service()
    plugin_config = svc.config.get("plugins", {})
    if not plugin_config.get("enabled", False):
        raise HTTPException(status_code=400, detail="Enable plugins.enabled before reloading plugins.")
    errors = get_node_registry().reload_plugins(plugin_config.get("paths", ["./nodes"]))
    return {"nodes": node_specs(), "errors": errors}


@app.post("/v1/workflows/submit", response_model=JobSubmissionResponse)
async def submit_workflow(
    workflow: WorkflowDefinition,
    user: UserContext = Depends(get_current_user),
):
    """Validate, enqueue, execute, and generate a registry-driven workflow."""
    svc = _require_service()
    try:
        validate_workflow(workflow)
    except WorkflowValidationError as error:
        raise HTTPException(status_code=400, detail=f"Workflow invalid: {error}")

    if auth_manager is not None and auth_manager.enabled:
        auth_manager.enforce_quota(user.user_id)
    job_id = str(uuid.uuid4())
    created_at = time.time()
    owner_id = user.user_id if (auth_manager is not None and auth_manager.enabled) else None
    svc.jobs.add(JobStatus(
        job_id=job_id, status="pending", created_at=created_at,
        expires_at=created_at + svc.file_retention_time,
        request_details=redact_workflow(workflow), owner=owner_id,
    ))
    if not svc.submit_workflow(job_id, workflow, owner=owner_id):
        svc.jobs.delete(job_id)
        raise HTTPException(status_code=429, detail="Per-user concurrent job limit reached.")
    return JobSubmissionResponse(job_id=job_id)


# ---------------------------------------------------------------------------
# Websocket progress stream
# ---------------------------------------------------------------------------

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """Real-time job progress stream. Clients may send "cancel" to abort."""
    if service is None:
        await websocket.close(code=1011, reason="Service not initialized")
        return
    await service.ws_manager.connect(job_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "cancel":
                service.cancel_job(job_id)
    except WebSocketDisconnect:
        pass
    finally:
        await service.ws_manager.disconnect(job_id, websocket)


# ---------------------------------------------------------------------------
# Upload / Download
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    user: UserContext = Depends(get_current_user),
):
    """Upload an image for img2img. Returns an ``image_id`` to reference in requests."""
    svc = _require_service()
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum 10 MB.")

    try:
        img = PILImage.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    image_id = str(uuid.uuid4())
    svc.save_upload(img, image_id)
    return {"image_id": image_id, "url": f"/uploads/{image_id}"}


@app.get("/uploads/{image_id}")
async def serve_uploaded_image(image_id: str):
    """Serve an uploaded image (public, keyed by unguessable UUID)."""
    svc = _require_service()
    # Mask IDs are prefixed at storage time; the route accepts both plain image
    # IDs (init images) and ``mask_`` IDs (masks) so the frontend only needs one
    # URL pattern.
    if image_id.startswith("mask_"):
        filepath = svc.get_mask_path(image_id[len("mask_"):])
    else:
        filepath = svc.get_upload_path(image_id)
    if filepath is None:
        raise HTTPException(status_code=404, detail="Upload not found.")
    return FileResponse(path=filepath, media_type="image/png")


@app.get("/download/{job_id}/{image_index}")
async def download_image(job_id: str, image_index: int):
    """Download a generated image (public URL; job_id is unguessable)."""
    svc = _require_service()
    job = svc.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job '{job_id}' is {job.status}, not completed.")
    if not job.image_urls or image_index < 0 or image_index >= len(job.image_urls):
        raise HTTPException(status_code=404, detail=f"Invalid image index {image_index}.")

    filename = f"image_{job_id}_{image_index}.png"
    filepath = svc.output_folder / filename
    if not filepath.exists():
        logger.error(f"File missing for job {job_id}, index {image_index}: {filepath}")
        raise HTTPException(status_code=404, detail="Image file missing on disk.")

    return FileResponse(path=filepath, media_type="image/png", filename=filename)


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"Serving static files from: {static_dir}")

    def _serve_html(filename: str, title: str):
        filepath = static_dir / filename
        if not filepath.exists():
            logger.warning(f"{filename} not found in {static_dir}")
            return None

        async def _handler():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return HTMLResponse(content=f.read(), status_code=200)
            except Exception as e:
                logger.error(f"Failed to serve {filename}: {e}")
                raise HTTPException(status_code=500, detail=f"Could not load {title}.")
        return _handler

    _webui_handler = _serve_html("webui.html", "Web UI")
    if _webui_handler is not None:
        app.get("/webui", response_class=HTMLResponse)(_webui_handler)

    _workflow_handler = _serve_html("workflow.html", "Workflow editor")
    if _workflow_handler is not None:
        app.get("/workflow", response_class=HTMLResponse)(_workflow_handler)

# Serve the generated Sphinx site separately from FastAPI's interactive /docs.
# OPENIMAGEGEN_DOCS_DIR supports installed deployments where documentation is
# hosted from a shared directory rather than the source checkout.
default_help_dir = Path(__file__).resolve().parents[1] / "docs" / "_build" / "html"
help_dir = Path(os.getenv("OPENIMAGEGEN_DOCS_DIR", str(default_help_dir))).expanduser()
if (help_dir / "index.html").is_file():
    app.mount("/help", StaticFiles(directory=help_dir, html=True), name="sphinx-help")
    logger.info(f"Serving Sphinx documentation from: {help_dir}")
else:
    async def sphinx_help_unavailable() -> HTMLResponse:
        return HTMLResponse(
            """<!doctype html><html lang="en"><head><meta charset="utf-8">
            <meta name="viewport" content="width=device-width,initial-scale=1">
            <title>OpenImageGen help</title></head><body style="max-width:760px;margin:48px auto;
            padding:0 24px;font:16px/1.6 system-ui;background:#10151b;color:#edf2f7">
            <h1>OpenImageGen documentation</h1>
            <p>The Sphinx HTML documentation has not been built for this installation.</p>
            <p>From the source directory, run:</p>
            <pre style="padding:16px;background:#171e26;border-radius:8px;overflow:auto"><code>python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html</code></pre>
            <p>Restart the server afterward. Alternatively, set <code>OPENIMAGEGEN_DOCS_DIR</code>
            to a directory containing the generated <code>index.html</code>.</p>
            <p><a style="color:#34d399" href="/docs">Open the interactive API reference</a> ·
            <a style="color:#34d399" href="/webui">Return to Studio</a></p></body></html>""",
            status_code=503,
        )

    app.get("/help", include_in_schema=False)(sphinx_help_unavailable)
    app.get("/help/", include_in_schema=False)(sphinx_help_unavailable)
