
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
import logging
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
    JobStatus,
    JobSubmissionResponse,
    PreloadRequest,
)
from .models import get_registry
from .nodes import get_node_registry
from .models.presets import generation_profiles, generation_templates, recommended_generation_params
from .nodes.openai_compatible import list_remote_models
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
        "torch_version": __import__("torch").__version__,
        "torch_cuda_build": __import__("torch").version.cuda,
        "cuda_available": __import__("torch").cuda.is_available(),
        "gpu_name": (
            __import__("torch").cuda.get_device_name(0)
            if __import__("torch").cuda.is_available()
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
        if adapter_cls is not None:
            try:
                tmp_adapter = adapter_cls()
                supports_neg = tmp_adapter.supports_negative_prompt()
                supports_i2i = tmp_adapter.supports_img2img()
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
            "generation_defaults": recommended_generation_params(key, info),
            "generation_profiles": generation_profiles(key, info),
            "execution_hint": svc.execution_hint(key, info),
        })
    return {"available_models": models}


@app.get("/v1/generation/templates")
async def get_generation_templates() -> Dict:
    """Return shared prompt ideas; family-aware settings are resolved by each client."""
    return {"templates": generation_templates()}


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
