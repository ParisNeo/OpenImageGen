"""
@file        service.py
@author      ParisNeo
@description Image generation service: orchestrates adapters, jobs, and artifact cleanup.
Coordinates configuration, model adapters, job execution, periodic cleanup,
WebSocket progress streaming, image-to-image generation, and PNG metadata embedding.
"""

from __future__ import annotations

import importlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from PIL import Image as PILImage
from PIL.PngImagePlugin import PngInfo

from .config import load_config
from .download_jobs import DownloadJobManager
from .downloads import download_model
from .hardware import hardware_profile, model_execution_hint
from .image_processing import apply_image_operations
from .jobs import InMemoryJobStore, JobStore
from .gguf_catalog import GGUF_PRESETS, build_gguf_model_config
from .models import get_registry
from .schemas import ImageGenerationRequest, ImageToImageRequest, JobStatus
from .websocket_manager import WebSocketManager

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
        if self.device.type == "cpu":
            self.dtype = torch.float32
            if dtype_name != "float32":
                logger.warning("CPU generation requires float32; configured dtype is ignored.")
        elif dtype_name == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16

        self.output_folder = Path(self.settings.get("output_folder", "./outputs"))
        self.model_cache_dir = Path(self.settings.get("model_cache_dir", "./models"))
        self.file_retention_time: int = int(self.settings.get("file_retention_time", 3600))
        self.cleanup_interval_seconds: int = int(self.settings.get("cleanup_interval_seconds", 900))

        self.output_folder.mkdir(exist_ok=True, parents=True)
        self.model_cache_dir.mkdir(exist_ok=True, parents=True)

        self.uploads_folder = self.output_folder / "uploads"
        self.uploads_folder.mkdir(exist_ok=True, parents=True)
        from .assets import AssetManager
        self.assets = AssetManager(
            Path(self.settings.get("asset_dir", "./model_assets")),
            max_bytes=int(float(self.settings.get("max_asset_upload_gb", 32)) * 1024**3),
        )

        self.pipelines: Dict[str, Any] = {}
        self.adapters: Dict[str, Any] = {}
        self.jobs: JobStore = job_store or InMemoryJobStore()
        self._cancelled_jobs: set[str] = set()
        self.registry = get_registry()
        from .nodes import get_node_registry

        plugin_config = self.config.get("plugins", {})
        if plugin_config.get("enabled", False):
            errors = get_node_registry().load_plugins(plugin_config.get("paths", ["./nodes"]))
            if errors:
                logger.warning(f"Some workflow plugins failed to load: {errors}")
        self.is_loading: bool = False
        self.is_downloading: bool = False
        self.download_status: Dict[str, str] = {}
        self.download_progress: Dict[str, Any] = {}
        self._load_lock = threading.Lock()
        self._download_lock = threading.Lock()
        self.download_jobs = DownloadJobManager()

        self.ws_manager = WebSocketManager()
        self.model_last_used: Dict[str, float] = {}
        self.loading_model_key: Optional[str] = None
        self.hardware = hardware_profile(self.device)

        execution_cfg = self.config.get("execution", {}) if isinstance(self.config, dict) else {}
        max_workers = max(1, int(execution_cfg.get("max_workers", 2)))
        self.max_concurrent_jobs_per_user: int = max(
            1, int(execution_cfg.get("max_concurrent_jobs_per_user", 1))
        )
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="oig-gen"
        )
        self._active_per_user: Dict[str, int] = {}
        self._user_lock = threading.Lock()


    def _get_device(self) -> torch.device:
        """Select an accelerator or explain precisely why CUDA is unavailable."""
        cuda_available = torch.cuda.is_available()
        if self.use_gpu and cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(
                f"Using NVIDIA GPU '{gpu_name}' with PyTorch {torch.__version__} "
                f"(CUDA {torch.version.cuda})."
            )
            return torch.device("cuda")

        mps_available = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if self.use_gpu and mps_available:
            logger.info("Using Apple Silicon GPU through MPS.")
            return torch.device("mps")

        if self.force_gpu:
            raise RuntimeError(
                "GPU execution is required, but PyTorch cannot access a GPU. "
                f"Installed PyTorch: {torch.__version__}; "
                f"PyTorch CUDA build: {torch.version.cuda or 'none (CPU-only)'}; "
                f"torch.cuda.is_available(): {cuda_available}. "
                "Install a CUDA-enabled PyTorch build in the same Python environment "
                "used to launch OpenImageGen, then restart the server."
            )

        if self.use_gpu:
            logger.warning(
                "GPU requested but unavailable: PyTorch=%s, CUDA build=%s. "
                "Falling back to CPU because force_gpu=false.",
                torch.__version__,
                torch.version.cuda or "none",
            )
        return torch.device("cpu")


    def ensure_model_loaded(self, model_key: str) -> Any:
        """Lazy-load a model pipeline if not already loaded. Thread-safe."""
        if model_key in self.pipelines:
            return self.pipelines[model_key]

        if model_key not in self.models_config:
            raise ValueError(f"Model '{model_key}' is not configured.")

        model_info = self.models_config[model_key]
        model_name = model_info.get("name")
        model_type = model_info.get("type")
        if not model_name or not model_type:
            raise ValueError(f"Model '{model_key}' is missing 'name' or 'type'.")

        adapter_cls = self.registry.try_get(model_type)
        if adapter_cls is None:
            raise ValueError(
                f"No adapter registered for type {model_type!r}. "
                f"Available types: {self.registry.list_types()}"
            )

        with self._load_lock:
            if model_key in self.pipelines:
                return self.pipelines[model_key]

            self.is_loading = True
            self.loading_model_key = model_key
            try:
                logger.info(f"Lazy-loading '{model_key}' ({model_name}) via {adapter_cls.__name__}...")
                overrides = {k: v for k, v in model_info.items() if k not in ("name", "type")}
                overrides["repository"] = model_name
                adapter = adapter_cls(**overrides)
                if not self._is_model_downloaded(model_key):
                    logger.info(f"Model '{model_key}' is not cached; downloading before loading...")
                    download_job = self.start_download([model_key])
                    download_id = download_job["job_id"]
                    while True:
                        current_download = self.download_jobs.get(download_id)
                        if current_download is None:
                            raise RuntimeError(f"Download job '{download_id}' disappeared.")
                        if current_download["status"] not in {"pending", "running"}:
                            break
                        time.sleep(0.25)
                    downloaded = bool(
                        current_download.get("results", {}).get(model_key, False)
                    )
                    if not downloaded:
                        raise RuntimeError(
                            f"Automatic download failed for model '{model_key}' ({model_name}). "
                            + str(current_download.get("message") or "Unknown download error.")
                        )
                pipeline = adapter.load_pipeline(
                    model_name=model_name,
                    cache_dir=str(self.model_cache_dir),
                    dtype=self.dtype,
                    device=self.device,
                )
                self.pipelines[model_key] = pipeline
                self.adapters[model_key] = adapter
                self.model_last_used[model_key] = time.time()
                logger.info(f"Loaded '{model_key}' on {self.device} with dtype {self.dtype}.")
                return pipeline
            finally:
                self.is_loading = False
                self.loading_model_key = None

    def load_model(self, model_key: str) -> bool:
        """Explicitly load a model pipeline into memory. Returns True on success."""
        try:
            self.ensure_model_loaded(model_key)
            return True
        except Exception as e:
            logger.error(f"Failed to load model '{model_key}': {e}", exc_info=True)
            return False

    def unload_model(self, model_key: str) -> bool:
        """Unload a model pipeline from memory to free GPU/CPU resources."""
        if model_key not in self.pipelines:
            return False

        pipeline = self.pipelines.pop(model_key)
        self.adapters.pop(model_key, None)
        self.model_last_used.pop(model_key, None)

        del pipeline
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info(f"Unloaded model '{model_key}' from memory.")
        return True

    def execution_hint(self, model_key: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate the best execution mode for this model on the server PC."""
        return model_execution_hint(
            model_key,
            model_info,
            self.hardware,
            force_gpu=self.force_gpu,
        )

    def get_memory_info(self) -> Dict[str, Any]:
        """Return current memory and resource usage information."""
        info: Dict[str, Any] = {
            "loaded_models": list(self.pipelines.keys()),
            "loaded_model_count": len(self.pipelines),
            "device": str(self.device),
        }
        if self.device.type == "cuda" and torch.cuda.is_available():
            info["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
            info["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 2)
            if torch.cuda.device_count() > 0:
                info["gpu_memory_total_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
                )
        return info


    def submit_generation(
        self,
        job_id: str,
        request: Any,
        job_type: str,
        owner: Optional[str] = None,
    ) -> bool:
        """Schedule generation on the bounded worker pool with per-user fairness."""
        worker = self.generate_img2img if job_type == "img2img" else self.generate_images
        return self._submit_worker(job_id, worker, request, owner)

    def _submit_worker(
        self,
        job_id: str,
        worker: Callable[[str, Any], None],
        payload: Any,
        owner: Optional[str],
    ) -> bool:
        if owner:
            with self._user_lock:
                active = self._active_per_user.get(owner, 0)
                if active >= self.max_concurrent_jobs_per_user:
                    return False
                self._active_per_user[owner] = active + 1
        self.executor.submit(self._run_and_release, worker, job_id, payload, owner)
        return True

    def submit_workflow(self, job_id: str, workflow: Any, owner: Optional[str] = None) -> bool:
        """Execute all graph nodes, then feed the terminal request to an adapter."""
        return self._submit_worker(job_id, self.execute_workflow_job, workflow, owner)


    def execute_workflow_job(self, job_id: str, workflow: Any) -> None:
        """Execute graph nodes, save non-image results, or dispatch terminal generation."""
        from .schemas import ImageGenerationRequest, ImageToImageRequest
        from .workflow import execute_workflow

        job = self.jobs.get(job_id)
        if job is None or job.status == "cancelled":
            return
        job.status = "processing"
        job.message = "Executing workflow nodes…"
        self.jobs.update(job)
        self._notify_job_update(job_id)

        def report(progress: int, message: str) -> None:
            current = self.jobs.get(job_id)
            if current is not None:
                current.progress = max(0, min(95, progress))
                current.message = message
                # execute_workflow reports "Executing <title> (<node id>)…".
                # Keep node activity structured for any frontend, not only the editor.
                if message.startswith("Executing ") and message.endswith("…"):
                    active_id = message.rsplit("(", 1)[-1][:-2]
                    current.active_node_ids = [active_id] if active_id else []
                self.jobs.update(current)
                self._notify_job_update(job_id)

        def cancelled() -> bool:
            current = self.jobs.get(job_id)
            return job_id in self._cancelled_jobs or current is None or current.status == "cancelled"

        try:
            job_type, values = execute_workflow(workflow, report=report, is_cancelled=cancelled)
            postprocess = values.pop("_postprocess", []) if isinstance(values, dict) else []
            node_results = values.pop("_node_results", {}) if isinstance(values, dict) else {}
            if cancelled():
                return
            if job_type == "result":
                current = self.jobs.get(job_id)
                if current is not None:
                    current.status = "completed"
                    current.active_node_ids = []
                    current.progress = 100
                    current.message = "Workflow completed."
                    current.request_details = {"result": values}
                    current.node_results = node_results
                    if values.get("image_id"):
                        current.image_urls = [f"/uploads/{values['image_id']}"]
                    elif values.get("image_url"):
                        current.image_urls = [values["image_url"]]
                    elif values.get("image_ref"):
                        current.image_urls = [self._save_processed_reference(job_id, values["image_ref"])]
                    self.jobs.update(current)
                    self._notify_job_update(job_id)
                return
            request = ImageToImageRequest(**values) if job_type == "img2img" else ImageGenerationRequest(**values)
            current = self.jobs.get(job_id)
            if current is not None:
                current.job_type = "img2img" if job_type == "img2img" else "text2img"
                current.request_details = request
                current.node_results = node_results
                current.active_node_ids = [
                    node.id for node in workflow.nodes
                    if node.type == ("img2img" if job_type == "img2img" else "sampler")
                ]
                self.jobs.update(current)
            if job_type == "img2img":
                assert isinstance(request, ImageToImageRequest)
                self.generate_img2img(job_id, request, postprocess=postprocess)
            else:
                self.generate_images(job_id, request, postprocess=postprocess)
        except Exception as error:
            logger.error(f"Workflow job {job_id} failed: {error}", exc_info=True)
            current = self.jobs.get(job_id)
            if current is not None and current.status != "cancelled":
                current.status = "failed"
                current.active_node_ids = []
                current.progress = 0
                current.message = str(error)
                self.jobs.update(current)
                self._notify_job_update(job_id)



    def _run_and_release(
        self,
        worker: Callable[[str, Any], None],
        job_id: str,
        request: Any,
        owner: Optional[str],
    ) -> None:
        try:
            worker(job_id, request)
        except Exception as e:
            logger.error(
                f"Unhandled error in generation worker for job {job_id}: {e}",
                exc_info=True,
            )
            job = self.jobs.get(job_id)
            if job is not None and job.status not in {"completed", "failed", "cancelled"}:
                job.status = "failed"
                job.progress = 0
                job.message = str(e)
                self.jobs.update(job)
                self._notify_job_update(job_id)
        finally:
            if owner:
                with self._user_lock:
                    remaining = self._active_per_user.get(owner, 0) - 1
                    if remaining <= 0:
                        self._active_per_user.pop(owner, None)
                    else:
                        self._active_per_user[owner] = remaining

    def active_jobs_for(self, owner: str) -> int:
        with self._user_lock:
            return self._active_per_user.get(owner, 0)

    def shutdown(self) -> None:
        """Stop accepting generation and download work."""
        self.executor.shutdown(wait=False, cancel_futures=False)
        self.download_jobs.shutdown()

    def start_download(self, model_keys: List[str]) -> Dict[str, Any]:
        """Create a persistent download job, deduplicating active identical requests."""
        def worker(report: Callable[[Optional[int], str], None]) -> Dict[str, bool]:
            total = len(model_keys)

            def report_model(model_key: str, percentage: Optional[int], message: str) -> None:
                index = model_keys.index(model_key)
                overall = None
                if percentage is not None:
                    overall = int(((index + percentage / 100) / total) * 100)
                report(overall, f"{model_key}: {message}")

            return self.preload_models(model_keys, progress_callback=report_model)

        label = ", ".join(model_keys)
        return self.download_jobs.submit(label, model_keys, worker)

    def cancel_job(self, job_id: str) -> bool:
        """Mark a job as cancelled. Returns True if the job was found and is cancellable."""
        job = self.jobs.get(job_id)
        if job is None:
            return False
        if job.status in ("completed", "failed", "cancelled"):
            return False
        self._cancelled_jobs.add(job_id)
        job.status = "cancelled"
        job.message = "Job cancelled by user."
        self.jobs.update(job)
        self._notify_job_update(job_id)
        logger.info(f"Job {job_id} cancelled.")
        return True

    def preload_models(
        self,
        model_keys: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str, Optional[int], str], None]] = None,
    ) -> Dict[str, bool]:
        """Pre-download model weights without loading pipelines into memory.

        Args:
            model_keys: Optional list of specific model keys to download. 
                        If None, downloads all configured models.

        Returns:
            A dictionary mapping model keys to download success status.
        """
        target_keys = model_keys or list(self.models_config.keys())
        results = {}

        def _make_callback(model_key: str):
            def callback(pct: int) -> None:
                self.download_progress[model_key] = {"status": "downloading", "percentage": pct}
                if progress_callback:
                    progress_callback(model_key, pct, f"Downloading files ({pct}%)")
            return callback

        with self._download_lock:
            self.is_downloading = True
            try:
                for key in target_keys:
                    if key not in self.models_config:
                        logger.warning(f"Cannot download '{key}': not in config.")
                        results[key] = False
                        continue

                    info = self.models_config[key]
                    name = info.get("name")
                    mtype = info.get("type")
                    if not name or not mtype:
                        logger.error(f"Cannot download '{key}': missing name or type.")
                        results[key] = False
                        continue

                    self.download_status[key] = "downloading"
                    self.download_progress[key] = {"status": "starting", "percentage": 0}
                    if progress_callback:
                        progress_callback(key, 0, "Preparing download…")
                    logger.info(f"Pre-downloading weights for '{key}' ({name})...")

                    overrides = {k: v for k, v in info.items() if k not in ("name", "type")}
                    download_callback = _make_callback(key)
                    success = download_model(
                        model_name=name,
                        model_type=mtype,
                        cache_dir=str(self.model_cache_dir),
                        config_overrides=overrides,
                        progress_callback=download_callback,
                    )

                    self.download_status[key] = "completed" if success else "failed"
                    self.download_progress[key] = {
                        "status": "completed" if success else "failed",
                        "percentage": 100 if success else 0,
                    }
                    results[key] = success
                    if progress_callback:
                        error_detail = "Download failed; check server logs."
                        if not success:
                            # The adapter's own error is logged above; preserve a useful
                            # repository-specific message in the persistent job.
                            error_detail = (
                                f"Could not download {name}. If it is gated, accept its license "
                                "and configure HF_TOKEN. See the server log for the exact error."
                            )
                        progress_callback(
                            key,
                            100 if success else None,
                            "Files are ready." if success else error_detail,
                        )

                return results
            finally:
                self.is_downloading = False

    def list_models(self) -> List[str]:
        """Return the list of configured model keys."""
        return list(self.models_config.keys())

    def is_model_available(self, model_key: str) -> bool:
        """Return whether a configured model can be loaded now or downloaded on demand."""
        if model_key not in self.models_config:
            return False
        if model_key in self.pipelines:
            return True
        if bool(self.settings.get("auto_download_models", True)):
            return True
        return self._is_model_downloaded(model_key)


    def _is_model_downloaded(self, model_key: str) -> bool:
        """Check every repository required by a model, including GGUF base fallbacks."""
        info = self.models_config.get(model_key, {})
        if info.get("type") == "openai_image":
            return True
        model_name = info.get("name")
        if not model_name:
            return False
        if Path(model_name).expanduser().is_file():
            return True
        try:
            from huggingface_hub import try_to_load_from_cache

            repositories = [model_name]
            if info.get("type") == "flux_gguf":
                fallback = info.get("base_repositories", [])
                if isinstance(fallback, str):
                    fallback = [fallback]
                repositories.extend(fallback)

                gguf_path = info.get("gguf_path") or info.get("transformer_gguf")
                if gguf_path:
                    if not Path(gguf_path).expanduser().is_file():
                        return False
                else:
                    source = info.get("gguf_source")
                    filename = info.get("gguf_filename")
                    if not source or not filename:
                        return False
                    cached_gguf = try_to_load_from_cache(
                        repo_id=source,
                        filename=filename,
                        cache_dir=str(self.model_cache_dir),
                    )
                    if not cached_gguf or not Path(str(cached_gguf)).is_file():
                        return False

            for repository in repositories:
                cached_index = try_to_load_from_cache(
                    repo_id=repository,
                    filename="model_index.json",
                    cache_dir=str(self.model_cache_dir),
                )
                if cached_index and Path(str(cached_index)).is_file():
                    return True
            return False
        except Exception:
            return False


    @contextmanager
    def _active_loras(self, pipeline: Any, adapter: Any, request: Any):
        """Load request-scoped adapters and always restore the cached pipeline."""
        loras = list(getattr(request, "loras", []) or [])
        if not loras:
            yield
            return
        if not adapter.supports_lora():
            raise ValueError(f"Model adapter '{adapter.type_name}' does not support LoRAs.")
        if not hasattr(pipeline, "load_lora_weights") or not hasattr(pipeline, "set_adapters"):
            raise ValueError("The loaded pipeline does not expose Diffusers LoRA hooks.")

        names: List[str] = []
        weights: List[float] = []
        try:
            for index, lora in enumerate(loras):
                name = f"oig_lora_{index}"
                source = self.assets.resolve_source(
                    lora.source,
                    self.model_cache_dir,
                    weight_name=lora.weight_name,
                )
                pipeline.load_lora_weights(source, adapter_name=name)
                names.append(name)
                weights.append(float(lora.weight))
            pipeline.set_adapters(names, adapter_weights=weights)
            yield
        finally:
            if names and hasattr(pipeline, "unload_lora_weights"):
                try:
                    pipeline.unload_lora_weights()
                except Exception as error:
                    logger.warning("Could not unload request LoRAs cleanly: %s", error)


    def _configure_scheduler(self, pipeline: Any, sampler: Optional[str]) -> None:
        """Apply the selected scheduler, importing only the class actually used.

        Schedulers are resolved lazily so that optional dependencies (notably
        ``torchsde`` for the DPMSolver SDE family) are only required when the
        caller selects a sampler that needs them. Importing the entire scheduler
        set up-front would otherwise break every generation on a minimal install.
        """
        if not sampler:
            return
        scheduler_targets: Dict[str, tuple[str, str]] = {
            "dpmpp_2m": ("scheduling_dpmsolver_multistep", "DPMSolverMultistepScheduler"),
            "dpmpp_2m_karras": ("scheduling_dpmsolver_multistep", "DPMSolverMultistepScheduler"),
            "dpmpp_2m_sde_karras": ("scheduling_dpmsolver_multistep", "DPMSolverMultistepScheduler"),
            "dpmpp_sde_karras": ("scheduling_dpmsolver_sde", "DPMSolverSDEScheduler"),
            "unipc": ("scheduling_unipc_multistep", "UniPCMultistepScheduler"),
            "flowmatch_euler": ("scheduling_flow_match_euler_discrete", "FlowMatchEulerDiscreteScheduler"),
            "euler_a": ("scheduling_euler_ancestral_discrete", "EulerAncestralDiscreteScheduler"),
            "euler": ("scheduling_euler_discrete", "EulerDiscreteScheduler"),
            "ddim": ("scheduling_ddim", "DDIMScheduler"),
        }
        target = scheduler_targets.get(sampler)
        if target is None:
            raise ValueError(f"Unsupported sampler '{sampler}'.")
        module_name, class_name = target
        try:
            module = importlib.import_module(f"diffusers.schedulers.{module_name}")
            scheduler_class = getattr(module, class_name)
        except ModuleNotFoundError as error:
            missing = getattr(error, "name", None) or "an optional dependency"
            raise ValueError(
                f"Sampler '{sampler}' requires the optional '{missing}' package. "
                f"Install it in the server environment with `pip install {missing}` "
                f"or choose a different sampler."
            ) from error

        options: Dict[str, Any] = {}
        if sampler.startswith("dpmpp_2m"):
            options = {
                "algorithm_type": "dpmsolver++",
                "final_sigmas_type": "sigma_min",
                "use_karras_sigmas": sampler.endswith("_karras"),
            }
            if sampler == "dpmpp_2m_sde_karras":
                options["solver_type"] = "midpoint"
        elif sampler == "dpmpp_sde_karras":
            options = {"use_karras_sigmas": True}
        elif sampler == "unipc":
            options = {"final_sigmas_type": "sigma_min"}

        if not hasattr(pipeline, "scheduler"):
            raise ValueError(f"Model pipeline does not support local sampler '{sampler}'.")
        pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config, **options)


    def _build_progress_callback(self, job_id: str, total_steps: int):
        """Factory for the diffusers `callback` that updates job progress and notifies WebSocket clients."""
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
                job.message = f"Denoising… step {step + 1}/{total_steps}"
                self.jobs.update(job)
                logger.debug(f"Job {job_id}: {progress}% (step {step + 1}/{total_steps})")
                self.ws_manager.notify_sync(job_id, {
                    "status": "processing",
                    "progress": progress,
                    "message": job.message,
                    "job_id": job_id,
                })

        return callback

    def _notify_job_update(self, job_id: str) -> None:
        """Send current job status to connected WebSocket clients."""
        job = self.jobs.get(job_id)
        if job is not None:
            self.ws_manager.notify_sync(job_id, job.model_dump())

    def save_upload(self, image: Any, image_id: str) -> Path:
        """Save an uploaded image to the uploads folder."""
        filepath = self.uploads_folder / f"{image_id}.png"
        image.save(filepath, "PNG")
        return filepath

    def get_upload_path(self, image_id: str) -> Optional[Path]:
        """Return the path to an uploaded image, or None if not found."""
        filepath = self.uploads_folder / f"{image_id}.png"
        return filepath if filepath.exists() else None

    def _embed_png_metadata(self, img: Any, filepath: Path, request: Any, model_key: str, seed: Any, steps: int, cfg: float) -> None:
        """Save a PIL image with embedded generation parameters as PNG text chunks."""
        png_meta = PngInfo()
        png_meta.add_text("prompt", request.prompt)
        if getattr(request, "negative_prompt", None):
            png_meta.add_text("negative_prompt", request.negative_prompt)
        png_meta.add_text("model", model_key or "")
        if seed is not None:
            png_meta.add_text("seed", str(seed))
        png_meta.add_text("steps", str(steps))
        png_meta.add_text("guidance_scale", str(cfg))
        if getattr(request, "sampler", None):
            png_meta.add_text("sampler", request.sampler)
        png_meta.add_text("width", str(img.width))
        png_meta.add_text("height", str(img.height))
        if hasattr(request, "strength"):
            png_meta.add_text("strength", str(request.strength))
        png_meta.add_text("software", "OpenImageGen v0.2.0")
        img.save(filepath, "PNG", pnginfo=png_meta)

    def _save_processed_reference(self, job_id: str, reference: Dict[str, Any]) -> str:
        """Materialize a transformed uploaded image as a normal job artifact."""
        source = reference.get("source")
        if not source:
            raise ValueError("The processed image has no uploaded source.")
        source_path = self.get_upload_path(str(source))
        if source_path is None:
            raise FileNotFoundError(f"Source image '{source}' was not found or has expired.")
        with PILImage.open(source_path) as source_image:
            image = apply_image_operations(source_image, reference.get("operations", []))
        filename = f"image_{job_id}_0.png"
        image.save(self.output_folder / filename, "PNG")
        return f"/download/{job_id}/0"

    def generate_images(
        self,
        job_id: str,
        request: ImageGenerationRequest,
        postprocess: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Execute text-to-image generation in a background task."""
        job = self.jobs.get(job_id)
        if job is None:
            logger.error(f"Job {job_id} missing from store at generation start.")
            return

        self._cancelled_jobs.discard(job_id)
        if job.status == "cancelled":
            logger.info(f"Job {job_id} was cancelled before generation started.")
            return

        job.active_node_ids = []
        job.status = "processing"
        job.progress = 0
        job.message = "Starting generation..."
        job.request_details = request
        self.jobs.update(job)
        self._notify_job_update(job_id)

        model_key = request.model_name or self.default_model
        if not model_key or not self.is_model_available(model_key):
            if model_key and model_key in self.models_config:
                msg = f"Model '{model_key}' is not downloaded. Please download it first."
            else:
                msg = f"Model '{model_key}' is not configured."
            logger.error(f"Job {job_id}: {msg}")
            job.status = "failed"
            job.message = msg
            self.jobs.update(job)
            self._notify_job_update(job_id)
            return

        job.message = f"Loading model '{model_key}' into memory..."
        self.jobs.update(job)
        self._notify_job_update(job_id)

        try:
            pipeline = self.ensure_model_loaded(model_key)
        except Exception as e:
            msg = f"Failed to load model '{model_key}': {e}"
            logger.error(f"Job {job_id}: {msg}", exc_info=True)
            job.status = "failed"
            job.message = msg
            self.jobs.update(job)
            self._notify_job_update(job_id)
            return

        current_job = self.jobs.get(job_id)
        if current_job is not None and current_job.status == "cancelled":
            logger.info(f"Job {job_id} was cancelled during model loading.")
            return

        adapter = self.adapters[model_key]
        self.model_last_used[model_key] = time.time()
        self._configure_scheduler(pipeline, getattr(request, "sampler", None))
        adapter_defaults = adapter.get_default_params()
        merged_defaults = {**self.generation_defaults, **adapter_defaults}

        num_inference_steps = (
            request.num_inference_steps
            if request.num_inference_steps is not None
            else merged_defaults.get("num_inference_steps", 50)
        )
        guidance_scale = (
            request.guidance_scale
            if request.guidance_scale is not None
            else merged_defaults.get("guidance_scale", 7.5)
        )
        num_images = (
            request.num_images_per_prompt
            if request.num_images_per_prompt is not None
            else merged_defaults.get("num_images_per_prompt", 1)
        )

        callback = self._build_progress_callback(job_id, num_inference_steps)

        try:
            generator: Optional[torch.Generator] = None
            actual_seed = request.seed
            if actual_seed is not None:
                actual_seed = int(actual_seed)
                generator = torch.Generator(device=self.device).manual_seed(actual_seed)
                logger.info(f"Job {job_id}: using seed {actual_seed}")
            else:
                actual_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
                generator = torch.Generator(device=self.device).manual_seed(actual_seed)
                logger.info(f"Job {job_id}: using random seed {actual_seed}")

            start = time.time()
            job.message = "Generating images..."
            self.jobs.update(job)

            autocast_ctx = (
                torch.autocast(self.device.type, dtype=self.dtype)
                if self.device.type != "cpu"
                else nullcontext()
            )
            with torch.no_grad(), autocast_ctx, self._active_loras(pipeline, adapter, request):
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

            if postprocess:
                job.message = "Applying workflow image processing…"
                images = [apply_image_operations(image, postprocess) for image in images]

            elapsed = time.time() - start

            image_urls: List[str] = []
            save_errors: List[str] = []
            for i, img in enumerate(images):
                filename = f"image_{job_id}_{i}.png"
                filepath = self.output_folder / filename
                try:
                    self._embed_png_metadata(img, filepath, request, model_key, actual_seed, num_inference_steps, guidance_scale)
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
                job.seed_used = actual_seed
                job.message = f"Generated {len(image_urls)} image(s) in {elapsed:.2f}s."
                if save_errors:
                    job.message += f" Some saves failed: {'; '.join(save_errors)}"
            self.jobs.update(job)
            self._notify_job_update(job_id)

        except Exception as e:
            logger.error(f"Job {job_id}: generation failed: {e}", exc_info=True)
            job.status = "failed"
            job.progress = 0
            job.message = f"Generation failed: {e}"
            self.jobs.update(job)
            self._notify_job_update(job_id)
        finally:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                logger.debug(f"Job {job_id}: cleared CUDA cache.")

    def generate_img2img(
        self,
        job_id: str,
        request: ImageToImageRequest,
        postprocess: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Execute image-to-image generation in a background task."""
        job = self.jobs.get(job_id)
        if job is None:
            logger.error(f"Job {job_id} missing from store at img2img start.")
            return

        self._cancelled_jobs.discard(job_id)
        if job.status == "cancelled":
            logger.info(f"Job {job_id} was cancelled before img2img started.")
            return

        init_image_path = self.get_upload_path(request.init_image_id)
        if init_image_path is None:
            msg = f"Init image '{request.init_image_id}' not found."
            logger.error(f"Job {job_id}: {msg}")
            job.status = "failed"
            job.message = msg
            self.jobs.update(job)
            self._notify_job_update(job_id)
            return

        try:
            init_image = PILImage.open(init_image_path).convert("RGB")
        except Exception as e:
            msg = f"Failed to load init image: {e}"
            logger.error(f"Job {job_id}: {msg}")
            job.status = "failed"
            job.message = msg
            self.jobs.update(job)
            self._notify_job_update(job_id)
            return

        if request.width and request.height:
            init_image = init_image.resize((request.width, request.height), PILImage.Resampling.LANCZOS)

        job.active_node_ids = []
        job.status = "processing"
        job.progress = 0
        job.message = "Starting img2img generation..."
        job.request_details = request
        self.jobs.update(job)
        self._notify_job_update(job_id)

        model_key = request.model_name or self.default_model
        if not model_key or not self.is_model_available(model_key):
            if model_key and model_key in self.models_config:
                msg = f"Model '{model_key}' is not downloaded. Please download it first."
            else:
                msg = f"Model '{model_key}' is not configured."
            logger.error(f"Job {job_id}: {msg}")
            job.status = "failed"
            job.message = msg
            self.jobs.update(job)
            self._notify_job_update(job_id)
            return

        job.message = f"Loading model '{model_key}' into memory..."
        self.jobs.update(job)
        self._notify_job_update(job_id)

        try:
            pipeline = self.ensure_model_loaded(model_key)
        except Exception as e:
            msg = f"Failed to load model '{model_key}': {e}"
            logger.error(f"Job {job_id}: {msg}", exc_info=True)
            job.status = "failed"
            job.message = msg
            self.jobs.update(job)
            self._notify_job_update(job_id)
            return

        current_job = self.jobs.get(job_id)
        if current_job is not None and current_job.status == "cancelled":
            logger.info(f"Job {job_id} was cancelled during model loading.")
            return

        adapter = self.adapters[model_key]
        self._configure_scheduler(pipeline, getattr(request, "sampler", None))
        if adapter.type_name == "qwen_image_edit":
            request.width = None
            request.height = None
        if not adapter.supports_img2img():
            msg = f"Model '{model_key}' does not support img2img generation."
            logger.error(f"Job {job_id}: {msg}")
            job.status = "failed"
            job.message = msg
            self.jobs.update(job)
            self._notify_job_update(job_id)
            return

        self.model_last_used[model_key] = time.time()

        adapter_defaults = adapter.get_default_params()
        merged_defaults = {**self.generation_defaults, **adapter_defaults}

        num_inference_steps = (
            request.num_inference_steps
            if request.num_inference_steps is not None
            else merged_defaults.get("num_inference_steps", 50)
        )
        guidance_scale = (
            request.guidance_scale
            if request.guidance_scale is not None
            else merged_defaults.get("guidance_scale", 7.5)
        )
        num_images = (
            request.num_images_per_prompt
            if request.num_images_per_prompt is not None
            else merged_defaults.get("num_images_per_prompt", 1)
        )
        strength = request.strength if request.strength is not None else 0.75

        effective_steps = max(int(num_inference_steps * strength), 1)
        callback = self._build_progress_callback(job_id, effective_steps)

        try:
            generator: Optional[torch.Generator] = None
            actual_seed: Optional[int] = request.seed
            if actual_seed is not None:
                actual_seed = int(actual_seed)
                generator = torch.Generator(device=self.device).manual_seed(actual_seed)
                logger.info(f"Job {job_id}: using seed {actual_seed}")
            else:
                actual_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
                generator = torch.Generator(device=self.device).manual_seed(actual_seed)
                logger.info(f"Job {job_id}: using random seed {actual_seed}")

            start = time.time()
            job.message = "Transforming image..."
            self.jobs.update(job)

            autocast_ctx = (
                torch.autocast(self.device.type, dtype=self.dtype)
                if self.device.type != "cpu"
                else nullcontext()
            )
            with torch.no_grad(), autocast_ctx, self._active_loras(pipeline, adapter, request):
                images: List[Any] = adapter.generate_img2img(
                    pipeline=pipeline,
                    prompt=request.prompt,
                    init_image=init_image,
                    negative_prompt=request.negative_prompt,
                    strength=strength,
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
            logger.info(f"Job {job_id}: img2img generated {len(images)} image(s) in {elapsed:.2f}s")

            image_urls: List[str] = []
            save_errors: List[str] = []
            for i, img in enumerate(images):
                filename = f"image_{job_id}_{i}.png"
                filepath = self.output_folder / filename
                try:
                    self._embed_png_metadata(img, filepath, request, model_key, actual_seed, num_inference_steps, guidance_scale)
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
                job.seed_used = actual_seed
                job.message = f"Generated {len(image_urls)} image(s) in {elapsed:.2f}s."
                if save_errors:
                    job.message += f" Some saves failed: {'; '.join(save_errors)}"
            self.jobs.update(job)
            self._notify_job_update(job_id)

        except Exception as e:
            logger.error(f"Job {job_id}: img2img failed: {e}", exc_info=True)
            job.status = "failed"
            job.progress = 0
            job.message = f"Generation failed: {e}"
            self.jobs.update(job)
            self._notify_job_update(job_id)
        finally:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                logger.debug(f"Job {job_id}: cleared CUDA cache.")

    def cleanup_expired_files(self) -> int:
        """Delete expired jobs and their associated files. Returns number cleaned."""
        expired_ids = self.jobs.list_expired()
        if not expired_ids:
            self._cleanup_old_uploads()
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

        self._cleanup_old_uploads()
        return cleaned

    def _cleanup_old_uploads(self) -> int:
        """Remove uploaded images older than file_retention_time."""
        now = time.time()
        cleaned = 0
        try:
            for file_path in self.uploads_folder.glob("*.png"):
                try:
                    if now - file_path.stat().st_mtime > self.file_retention_time:
                        file_path.unlink()
                        cleaned += 1
                        logger.debug(f"Deleted expired upload: {file_path.name}")
                except OSError as e:
                    logger.error(f"Failed to delete upload {file_path.name}: {e}")
        except Exception as e:
            logger.error(f"Upload cleanup error: {e}")
        return cleaned