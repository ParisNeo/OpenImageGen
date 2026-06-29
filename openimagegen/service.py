"""Image generation service: orchestrates adapters, jobs, and artifact cleanup."""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .config import load_config
from .jobs import InMemoryJobStore, JobStore
from .models import get_registry
from .schemas import ImageGenerationRequest, JobStatus

logger = logging.getLogger("OpenImageGen.service")


class ImageGenService:
    """Coordinates configuration, model adapters, job execution, and cleanup."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        job_store: Optional[JobStore] = None,
    ) -> None:
        self.config = config or load_config()
        self.models_config: Dict[str, Any] = self.config.get("models", {})
        self.settings: Dict[str, Any] = self.config.get("settings", {})
        self.generation_defaults: Dict[str, Any] = self.config.get("generation", {})

        self.default_model: Optional[str] = self.settings.get(
            "default_model",
            next(iter(self.models_config), None) if self.models_config else None,
        )
        self.force_gpu: bool = bool(self.settings.get("force_gpu", False))
        self.use_gpu: bool = bool(self.settings.get("use_gpu", True))
        self.device = self._get_device()
        dtype_name = self.settings.get("dtype", "float16")
        self.dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16

        self.output_folder = Path(self.settings.get("output_folder", "./outputs"))
        self.model_cache_dir = Path(self.settings.get("model_cache_dir", "./models"))
        self.file_retention_time: int = int(self.settings.get("file_retention_time", 3600))
        self.cleanup_interval_seconds: int = int(self.settings.get("cleanup_interval_seconds", 900))

        self.output_folder.mkdir(exist_ok=True, parents=True)
        self.model_cache_dir.mkdir(exist_ok=True, parents=True)

        self.pipelines: Dict[str, Any] = {}
        self.adapters: Dict[str, Any] = {}
        self.jobs: JobStore = job_store or InMemoryJobStore()
        self.registry = get_registry()

        self.load_pipelines()

    def _get_device(self) -> torch.device:
        if self.force_gpu:
            if torch.cuda.is_available():
                logger.info("force_gpu=True and CUDA available; using GPU.")
                return torch.device("cuda")
            raise RuntimeError("force_gpu is True but CUDA is not available.")
        if self.use_gpu and torch.cuda.is_available():
            logger.info("use_gpu=True and CUDA available; using GPU.")
            return torch.device("cuda")
        if (
            self.use_gpu
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            logger.info("use_gpu=True and MPS available; using Apple Silicon GPU.")
            return torch.device("mps")
        logger.info("Using CPU.")
        return torch.device("cpu")

    def load_pipelines(self) -> None:
        """Load all configured models via their registered adapters."""
        if not self.models_config:
            logger.warning("No models defined in configuration.")
            return

        for model_key, model_info in self.models_config.items():
            model_name = model_info.get("name")
            model_type = model_info.get("type")
            if not model_name or not model_type:
                logger.warning(f"Skipping '{model_key}': missing 'name' or 'type'.")
                continue

            adapter_cls = self.registry.try_get(model_type)
            if adapter_cls is None:
                logger.error(
                    f"No adapter registered for type {model_type!r}. "
                    f"Available types: {self.registry.list_types()}"
                )
                continue

            logger.info(f"Loading '{model_key}' ({model_name}) via {adapter_cls.__name__}...")
            try:
                overrides = {k: v for k, v in model_info.items() if k not in ("name", "type")}
                adapter = adapter_cls(**overrides)
                pipeline = adapter.load_pipeline(
                    model_name=model_name,
                    cache_dir=str(self.model_cache_dir),
                    dtype=self.dtype,
                    device=self.device,
                )
                self.pipelines[model_key] = pipeline
                self.adapters[model_key] = adapter
                logger.info(f"Loaded '{model_key}' on {self.device} with dtype {self.dtype}.")
            except Exception as e:
                logger.error(f"Failed to load '{model_key}': {e}", exc_info=True)

    def list_models(self) -> List[str]:
        """Return the list of configured model keys."""
        return list(self.models_config.keys())

    def is_model_available(self, model_key: str) -> bool:
        """Return True if the model was successfully loaded."""
        return model_key in self.pipelines

    def _build_progress_callback(self, job_id: str, total_steps: int):
        """Factory for the diffusers `callback` that updates job progress."""
        last_update = [0.0]

        def callback(step: int, timestep: int, latents: Any) -> None:
            now = time.time()
            if now - last_update[0] < 1.0 and step != total_steps - 1:
                return
            last_update[0] = now
            progress = min(int(((step + 1) / total_steps) * 100), 100)
            job = self.jobs.get(job_id)
            if job is not None and job.progress != progress:
                job.progress = progress
                self.jobs.update(job)
                logger.debug(f"Job {job_id}: {progress}% (step {step + 1}/{total_steps})")

        return callback

    def generate_images(self, job_id: str, request: ImageGenerationRequest) -> None:
        """Execute image generation in a background task."""
        job = self.jobs.get(job_id)
        if job is None:
            logger.error(f"Job {job_id} missing from store at generation start.")
            return

        job.status = "processing"
        job.progress = 0
        job.message = "Starting generation..."
        job.request_details = request
        self.jobs.update(job)

        model_key = request.model_name or self.default_model
        if not model_key or model_key not in self.pipelines:
            msg = f"Model '{model_key}' is not loaded."
            logger.error(f"Job {job_id}: {msg}")
            job.status = "failed"
            job.message = msg
            self.jobs.update(job)
            return

        pipeline = self.pipelines[model_key]
        adapter = self.adapters[model_key]
        adapter_defaults = adapter.get_default_params()
        merged_defaults = {**self.generation_defaults, **adapter_defaults}

        num_inference_steps = (
            request.num_inference_steps
            or merged_defaults.get("num_inference_steps", 50)
        )
        guidance_scale = (
            request.guidance_scale
            or merged_defaults.get("guidance_scale", 7.5)
        )
        num_images = (
            request.num_images_per_prompt
            or merged_defaults.get("num_images_per_prompt", 1)
        )

        callback = self._build_progress_callback(job_id, num_inference_steps)

        try:
            generator: Optional[torch.Generator] = None
            if request.seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(request.seed)
                logger.info(f"Job {job_id}: using seed {request.seed}")
            else:
                logger.info(f"Job {job_id}: using random seed")

            start = time.time()
            job.message = "Generating images..."
            self.jobs.update(job)

            autocast_ctx = (
                torch.autocast(self.device.type, dtype=self.dtype)
                if self.device.type != "cpu"
                else nullcontext()
            )
            with torch.no_grad(), autocast_ctx:
                images: List[Any] = adapter.generate(
                    pipeline=pipeline,
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    height=request.height,
                    width=request.width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images,
                    generator=generator,
                    callback=callback,
                    callback_steps=1,
                )

            elapsed = time.time() - start
            logger.info(f"Job {job_id}: generated {len(images)} image(s) in {elapsed:.2f}s")

            image_urls: List[str] = []
            save_errors: List[str] = []
            for i, img in enumerate(images):
                filename = f"image_{job_id}_{i}.png"
                filepath = self.output_folder / filename
                try:
                    img.save(filepath, "PNG")
                    image_urls.append(f"/download/{job_id}/{i}")
                    logger.info(f"Job {job_id}: saved {filename}")
                except Exception as save_err:
                    logger.error(f"Job {job_id}: failed to save image {i}: {save_err}")
                    save_errors.append(str(save_err))

            if not image_urls:
                job.status = "failed"
                job.progress = 0
                job.message = f"All image saves failed: {'; '.join(save_errors)}"
            else:
                job.status = "completed"
                job.progress = 100
                job.image_urls = image_urls
                job.message = f"Generated {len(image_urls)} image(s) in {elapsed:.2f}s."
                if save_errors:
                    job.message += f" Some saves failed: {'; '.join(save_errors)}"
            self.jobs.update(job)

        except Exception as e:
            logger.error(f"Job {job_id}: generation failed: {e}", exc_info=True)
            job.status = "failed"
            job.progress = 0
            job.message = f"Generation failed: {e}"
            self.jobs.update(job)
        finally:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                logger.debug(f"Job {job_id}: cleared CUDA cache.")

    def cleanup_expired_files(self) -> int:
        """Delete expired jobs and their associated files. Returns number cleaned."""
        expired_ids = self.jobs.list_expired()
        if not expired_ids:
            return 0

        logger.info(f"Cleaning {len(expired_ids)} expired job(s)...")
        cleaned = 0
        for job_id in expired_ids:
            try:
                for file_path in self.output_folder.glob(f"image_{job_id}_*.png"):
                    try:
                        file_path.unlink()
                        logger.debug(f"Deleted {file_path.name}")
                    except OSError as e:
                        logger.error(f"Failed to delete {file_path.name}: {e}")
            except Exception as e:
                logger.error(f"Cleanup glob error for {job_id}: {e}")
            self.jobs.delete(job_id)
            cleaned += 1
            logger.info(f"Removed expired job record {job_id}.")
        return cleaned
