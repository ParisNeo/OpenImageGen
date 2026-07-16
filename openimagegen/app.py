"""
@file        app.py
@author      ParisNeo
@description FastAPI application: HTTP routes, lifespan, static UI, and route handlers.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .schemas import ImageGenerationRequest, JobStatus, JobSubmissionResponse
from .service import ImageGenService

logger = logging.getLogger("OpenImageGen.app")

service: Optional[ImageGenService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the application's startup and shutdown lifecycle.

    On startup:
        Initializes the global `ImageGenService` instance. If initialization
        fails, the exception is re-raised, aborting the application boot sequence.
        Schedules a background asyncio task to periodically invoke
        `service.cleanup_expired_files` at a configurable interval.

    On shutdown:
        Gracefully cancels the periodic cleanup background task and awaits
        its termination, ensuring no lingering file I/O operations are interrupted
    abruptly.

    Args:
        app: The FastAPI application instance associated with this lifespan.

    Yields:
        Passes control to the application's request handling phase.
    """
    global service
    logger.info("OpenImageGen starting up...")
    try:
        service = ImageGenService()
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
    description="Open source image generation API using diffusion models.",
    version="0.2.0",
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
    """Return the initialized service or raise 503."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")
    return service


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root to the Web UI."""
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
        "is_loading_models": svc.is_loading,
        "is_downloading_models": svc.is_downloading,
        "download_status": svc.download_status,
        "active_jobs": svc.jobs.count(),
        "output_folder": str(svc.output_folder.resolve()),
        "model_cache_dir": str(svc.model_cache_dir.resolve()),
        "memory_info": svc.get_memory_info(),
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
        if adapter_cls is not None:
            try:
                tmp_adapter = adapter_cls()
                supports_neg = tmp_adapter.supports_negative_prompt()
            except Exception:
                pass
        models.append({
            "key": key,
            "type": model_type,
            "name": info.get("name", ""),
            "loaded": key in svc.pipelines,
            "downloaded": svc.download_status.get(key) == "completed",
            "supports_negative_prompt": supports_neg,
        })
    return {"available_models": models}


@app.post("/preload")
async def preload_models(
    background_tasks: BackgroundTasks,
    model_keys: Optional[List[str]] = None,
):
    """Trigger background download of model weights without loading them into memory."""
    svc = _require_service()
    if svc.is_downloading:
        raise HTTPException(status_code=409, detail="A download operation is already in progress.")

    target_keys = model_keys or list(svc.models_config.keys())
    invalid_keys = [k for k in target_keys if k not in svc.models_config]
    if invalid_keys:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model key(s): {', '.join(invalid_keys)}. "
                   f"Available: {', '.join(svc.models_config.keys())}",
        )

    background_tasks.add_task(svc.preload_models, target_keys)
    return {"status": "preloading_started", "models": target_keys}


@app.post("/models/{model_key}/load")
async def load_model_endpoint(model_key: str):
    """Explicitly load a model pipeline into memory."""
    svc = _require_service()
    if model_key not in svc.models_config:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' is not configured.")
    if model_key in svc.pipelines:
        return {"status": "already_loaded", "model": model_key}
    success = svc.load_model(model_key)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to load model '{model_key}'.")
    return {"status": "loaded", "model": model_key}


@app.post("/models/{model_key}/unload")
async def unload_model_endpoint(model_key: str):
    """Unload a model pipeline from memory to free resources."""
    svc = _require_service()
    if not svc.unload_model(model_key):
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' is not loaded.")
    return {"status": "unloaded", "model": model_key}


@app.get("/memory")
async def get_memory_info():
    """Return current memory and resource usage information."""
    svc = _require_service()
    return svc.get_memory_info()


@app.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a pending or processing job."""
    svc = _require_service()
    if not svc.cancel_job(job_id):
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found or cannot be cancelled.",
        )
    return {"status": "cancelled", "job_id": job_id}


@app.post("/submit", response_model=JobSubmissionResponse)
async def submit_job(
    request: ImageGenerationRequest,
    background_tasks: BackgroundTasks,
):
    """Submit an image generation job and return its job_id."""
    svc = _require_service()
    job_id = str(uuid.uuid4())
    created_at = time.time()
    expires_at = created_at + svc.file_retention_time

    requested_model = request.model_name or svc.default_model
    if not svc.is_model_available(requested_model):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{requested_model}' is not loaded or configured.",
        )

    svc.jobs.add(JobStatus(
        job_id=job_id,
        status="pending",
        created_at=created_at,
        expires_at=expires_at,
        request_details=request,
    ))
    logger.info(f"Submitted job {job_id} for prompt: '{request.prompt[:50]}...'")
    background_tasks.add_task(svc.generate_images, job_id, request)
    return JobSubmissionResponse(job_id=job_id)


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Retrieve the current status of a job."""
    svc = _require_service()
    job = svc.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@app.get("/download/{job_id}/{image_index}")
async def download_image(job_id: str, image_index: int):
    """Download a specific generated image for a completed job."""
    svc = _require_service()
    job = svc.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job '{job_id}' is {job.status}, not completed.",
        )
    if not job.image_urls or image_index < 0 or image_index >= len(job.image_urls):
        raise HTTPException(
            status_code=404,
            detail=f"Invalid image index {image_index} for job '{job_id}'.",
        )

    filename = f"image_{job_id}_{image_index}.png"
    filepath = svc.output_folder / filename
    if not filepath.exists():
        logger.error(f"File missing for job {job_id}, index {image_index}: {filepath}")
        raise HTTPException(status_code=404, detail="Image file missing on disk.")

    return FileResponse(path=filepath, media_type="image/png", filename=filename)


static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"Serving static files from: {static_dir}")

    webui_file = static_dir / "webui.html"
    if webui_file.exists():
        @app.get("/webui", response_class=HTMLResponse)
        async def serve_webui():
            try:
                with open(webui_file, "r", encoding="utf-8") as f:
                    return HTMLResponse(content=f.read(), status_code=200)
            except Exception as e:
                logger.error(f"Failed to serve webui.html: {e}")
                raise HTTPException(status_code=500, detail="Could not load Web UI.")
    else:
        logger.warning(f"webui.html not found in {static_dir}")