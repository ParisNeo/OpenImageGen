import logging
import uuid
import time
import os
import shutil
import platform
import sys
from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import torch
from diffusers import (
    DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline,
    KandinskyPipeline, KandinskyV22Pipeline, # Add more as needed
    AutoPipelineForText2Image, # Generic loader
    DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler # Example Schedulers
)
from diffusers.utils.loading_utils import load_image # Utility for potential img2img later
from PIL import Image
import toml
import pipmaster as pm # Optional: for dependency checks

# --- Dependency Check (Optional but Recommended) ---
def check_and_install_dependencies():
    logger.info("Checking dependencies...")
    # Core dependencies are usually handled by setup.py/requirements.txt
    # This is more for optional or specific hardware needs if any.
    # Example: Ensure correct torch version for CUDA
    required_packages = ["torch", "diffusers", "transformers", "accelerate", "Pillow", "toml", "fastapi", "uvicorn"]
    installed = True
    for package in required_packages:
        if not pm.is_installed(package):
            logger.warning(f"{package} not found. Please install requirements: pip install -r requirements.txt")
            installed = False
            # Optionally attempt installation:
            # try:
            #     logger.info(f"Attempting to install {package}...")
            #     pm.install(package)
            # except Exception as e:
            #     logger.error(f"Failed to install {package}: {e}")
            #     installed = False

    if not installed:
        logger.error("Missing core dependencies. Please install them.")
        # sys.exit(1) # Or raise an error depending on desired behavior

    # Example: Check accelerate version
    if not pm.is_version_higher("accelerate", "0.25.0"):
        logger.warning("Accelerate version might be too old. Consider upgrading: pip install --upgrade accelerate")

    logger.info("Dependency check completed.")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("openimagegen.log"), # Log to file
        logging.StreamHandler(sys.stdout)       # Log to console
    ]
)
logger = logging.getLogger("OpenImageGen")

# --- Run Dependency Check ---
# check_and_install_dependencies() # Uncomment to enable check on startup

# --- FastAPI App Initialization ---
app = FastAPI(
    title="OpenImageGen API",
    description="Open source image generation API using diffusion models.",
    version="0.1.0"
)

# --- Configuration Loading Logic ---
def get_config_search_paths() -> List[Path]:
    """Determines potential config file locations based on OS."""
    system = platform.system()
    search_paths = []
    cwd = Path.cwd()
    home = Path.home()

    # OS-specific paths (higher priority)
    if system == "Linux":
        search_paths.extend([
            Path("/etc/openimagegen/config.toml"),
            Path("/usr/local/etc/openimagegen/config.toml"),
            home / ".config/openimagegen/config.toml",
        ])
    elif system == "Windows":
        appdata = os.getenv("APPDATA")
        if appdata:
            search_paths.append(Path(appdata) / "OpenImageGen/config.toml") # Changed folder name slightly
    elif system == "Darwin": # macOS
        search_paths.extend([
            home / "Library/Application Support/OpenImageGen/config.toml", # Changed folder name slightly
            Path("/usr/local/etc/openimagegen/config.toml"),
        ])

    # Current working directory (lowest priority before default)
    search_paths.append(cwd / "config.toml")

    return search_paths

def load_config() -> Dict[str, Any]:
    """Loads configuration from file or creates a default one."""
    DEFAULT_CONFIG = {
        "models": {
            "stable_diffusion_1_5": {"name": "runwayml/stable-diffusion-v1-5", "type": "stable_diffusion"},
            "stable_diffusion_xl": {"name": "stabilityai/stable-diffusion-xl-base-1.0", "type": "stable_diffusion_xl"}
        },
        "settings": {
            "default_model": "stable_diffusion_1_5",
            "force_gpu": False,
            "use_gpu": True,
            "dtype": "float16",
            "output_folder": "./outputs",
            "model_cache_dir": "./models",
            "port": 8089,
            "host": "0.0.0.0",
            "file_retention_time": 3600 # 1 hour in seconds
        },
        "generation": {
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "num_images_per_prompt": 1
        }
    }

    config_path_override = os.getenv("OPENIMAGEGEN_CONFIG_OVERRIDE")
    env_config_path = os.getenv("OPENIMAGEGEN_CONFIG") # Original env var name

    # Priority: CLI override > Env Var > Standard Paths > Default
    config_path_to_load = None

    if config_path_override:
        path = Path(config_path_override)
        if path.exists():
            logger.info(f"Loading config from CLI override: {path}")
            config_path_to_load = path
        else:
            logger.error(f"Config file specified via override not found: {path}")
            # Fall through to try other methods or default

    if not config_path_to_load and env_config_path:
        path = Path(env_config_path)
        if path.exists():
            logger.info(f"Loading config from environment variable OPENIMAGEGEN_CONFIG: {path}")
            config_path_to_load = path
        else:
            logger.warning(f"Config file specified via OPENIMAGEGEN_CONFIG not found: {path}")
            # Fall through

    if not config_path_to_load:
        search_paths = get_config_search_paths()
        for path in search_paths:
            if path.exists():
                logger.info(f"Loading config from standard location: {path}")
                config_path_to_load = path
                break

    if config_path_to_load:
        try:
            loaded_config = toml.load(config_path_to_load)
            # Simple merge strategy: Update default config with loaded values
            # This ensures all keys exist, even if the user's config is minimal
            merged_config = DEFAULT_CONFIG.copy()
            for section, values in loaded_config.items():
                if section in merged_config and isinstance(merged_config[section], dict):
                    merged_config[section].update(values)
                else:
                     merged_config[section] = values # Add new sections if any
            return merged_config
        except Exception as e:
            logger.error(f"Failed to load config file {config_path_to_load}: {e}. Using default config.")
            return DEFAULT_CONFIG
    else:
        # If no config file found anywhere, create default in CWD
        default_path = Path.cwd() / "config.toml"
        try:
            with open(default_path, "w", encoding="utf-8") as f:
                toml.dump(DEFAULT_CONFIG, f)
            logger.info(f"No config file found. Created default config.toml at: {default_path}")
        except Exception as e:
            logger.error(f"Failed to create default config file at {default_path}: {e}")
        return DEFAULT_CONFIG

def load_config_for_cli() -> Dict[str, Any]:
    """Helper to load config just for CLI host/port reading, suppresses logging noise."""
    original_level = logger.level
    logger.setLevel(logging.ERROR) # Temporarily reduce logging
    config_data = load_config()
    logger.setLevel(original_level) # Restore logging level
    return config_data


config = load_config()

# --- Pydantic Models for API ---
class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    model_name: Optional[str] = None # If None, uses default from config
    height: Optional[int] = None # Pipeline default if None
    width: Optional[int] = None  # Pipeline default if None
    num_inference_steps: Optional[int] = Field(None, alias="steps") # Allow 'steps' as alias
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None # Use None for random seed
    num_images_per_prompt: Optional[int] = None # Overrides config default
    # Add other potential parameters like eta, scheduler choice, etc. later

class JobStatus(BaseModel):
    job_id: str
    status: str = Field(..., description="Current status: pending, processing, completed, failed")
    progress: int = Field(0, ge=0, le=100, description="Generation progress percentage")
    message: Optional[str] = None
    image_urls: Optional[List[str]] = Field(None, description="List of URLs to download generated images upon completion")
    created_at: float = Field(..., description="Timestamp when the job was created")
    expires_at: float = Field(..., description="Timestamp when the job artifacts expire")
    request_details: Optional[ImageGenerationRequest] = None # Store the request for info

# --- In-Memory Job Store ---
# Warning: This is not persistent. Use Redis or a DB for production.
jobs: Dict[str, JobStatus] = {}

# --- Image Generation Service ---
class ImageGenService:
    def __init__(self):
        self.models_config = config.get("models", {})
        self.settings = config.get("settings", {})
        self.generation_defaults = config.get("generation", {})

        self.default_model = self.settings.get("default_model", next(iter(self.models_config.keys())) if self.models_config else None)
        self.force_gpu = self.settings.get("force_gpu", False)
        self.use_gpu = self.settings.get("use_gpu", True)
        self.device = self._get_device()
        self.dtype = torch.float16 if self.settings.get("dtype", "float16") == "float16" else torch.bfloat16
        self.output_folder = Path(self.settings.get("output_folder", "./outputs"))
        self.model_cache_dir = Path(self.settings.get("model_cache_dir", "./models"))
        self.file_retention_time = self.settings.get("file_retention_time", 3600)

        self.output_folder.mkdir(exist_ok=True, parents=True)
        self.model_cache_dir.mkdir(exist_ok=True, parents=True)

        self.pipelines: Dict[str, DiffusionPipeline] = {}
        self.load_pipelines()

    def _get_device(self) -> torch.device:
        """Determines the torch device based on config and availability."""
        if self.force_gpu:
            if torch.cuda.is_available():
                logger.info("Forcing GPU usage.")
                return torch.device("cuda")
            else:
                logger.error("force_gpu is True, but CUDA is not available!")
                raise RuntimeError("CUDA not available, but force_gpu is set.")
        elif self.use_gpu and torch.cuda.is_available():
            logger.info("GPU usage enabled and CUDA is available. Using GPU.")
            return torch.device("cuda")
        # Add MPS (Apple Silicon) support if desired
        # elif self.use_gpu and torch.backends.mps.is_available():
        #     logger.info("GPU usage enabled and MPS is available. Using MPS.")
        #     return torch.device("mps")
        else:
            logger.info("GPU usage disabled or CUDA/MPS not available. Using CPU.")
            return torch.device("cpu")

    def load_pipelines(self):
        """Loads models defined in the config file."""
        if not self.models_config:
            logger.warning("No models defined in the configuration file.")
            return

        for model_key, model_info in self.models_config.items():
            model_name = model_info.get("name")
            model_type = model_info.get("type")
            if not model_name or not model_type:
                logger.warning(f"Skipping model '{model_key}': Missing 'name' or 'type' in config.")
                continue

            logger.info(f"Loading model '{model_key}' ({model_name}, type: {model_type})...")
            try:
                pipeline_options = {
                    "torch_dtype": self.dtype,
                    "cache_dir": self.model_cache_dir,
                    # Add variant if specified in config, e.g., variant="fp16" for SDXL fp16 weights
                    # "variant": model_info.get("variant", None)
                }
                # Use AutoPipeline for simplicity, or specific pipelines for more control
                pipeline = AutoPipelineForText2Image.from_pretrained(model_name, **pipeline_options)

                # --- Scheduler Optimization (Optional) ---
                # Use a potentially faster scheduler
                # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                # pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

                # --- Memory Optimizations ---
                if self.device == torch.device("cuda"):
                    try:
                        # Offload parts of the model to CPU RAM to save VRAM
                        pipeline.enable_model_cpu_offload()
                        logger.info(f"Enabled model CPU offload for '{model_key}'.")
                    except AttributeError:
                         logger.warning(f"Model CPU offload not available for pipeline type of '{model_key}'. Loading directly to GPU.")
                         pipeline.to(self.device)
                    # Tiling can save VRAM for VAE at the cost of speed
                    # try:
                    #     pipeline.enable_vae_tiling()
                    #     logger.info(f"Enabled VAE tiling for '{model_key}'.")
                    # except AttributeError:
                    #     pass # Not all pipelines support it
                else:
                    pipeline.to(self.device) # Move to CPU

                self.pipelines[model_key] = pipeline
                logger.info(f"Successfully loaded model '{model_key}' to {self.device} with dtype {self.dtype}.")

            except Exception as e:
                logger.error(f"Failed to load model '{model_key}' ({model_name}): {e}", exc_info=True)
                # Optionally remove the failed model key or keep it to indicate loading failure


    def _progress_callback(self, job_id: str, total_steps: int):
        """Creates a callback function for diffusers pipeline progress."""
        last_update_time = time.time()
        update_interval = 1.0 # Update status at most every second

        def callback(step: int, timestep: int, latents: torch.FloatTensor):
            nonlocal last_update_time
            current_time = time.time()
            if job_id in jobs and (current_time - last_update_time > update_interval or step == total_steps - 1) :
                progress = min(int(((step + 1) / total_steps) * 100), 100)
                if jobs[job_id].progress != progress: # Only update if progress changed
                    jobs[job_id].progress = progress
                    logger.debug(f"Job {job_id} progress: {progress}% (Step {step + 1}/{total_steps})")
                last_update_time = current_time

        return callback

    def generate_images(self, job_id: str, request: ImageGenerationRequest) -> None:
        """The background task function for generating images."""
        if job_id not in jobs:
            logger.error(f"Job {job_id} not found in job store at start of generation.")
            return

        jobs[job_id].status = "processing"
        jobs[job_id].progress = 0
        jobs[job_id].message = "Starting image generation..."
        jobs[job_id].request_details = request # Store request details

        model_key = request.model_name or self.default_model
        if not model_key or model_key not in self.pipelines:
            error_msg = f"Model '{model_key}' not found or failed to load."
            logger.error(f"Job {job_id}: {error_msg}")
            jobs[job_id].status = "failed"
            jobs[job_id].message = error_msg
            return

        pipeline = self.pipelines[model_key]
        logger.info(f"Job {job_id}: Starting generation with model '{model_key}' on device {self.device}.")

        # Prepare generation parameters, merging request, config defaults, and handling None
        num_inference_steps = request.num_inference_steps or self.generation_defaults.get("num_inference_steps", 50)
        guidance_scale = request.guidance_scale or self.generation_defaults.get("guidance_scale", 7.5)
        num_images = request.num_images_per_prompt or self.generation_defaults.get("num_images_per_prompt", 1)

        gen_params = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "height": request.height, # Let pipeline handle default if None
            "width": request.width,   # Let pipeline handle default if None
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images,
            "callback": self._progress_callback(job_id, num_inference_steps),
            "callback_steps": 1 # Call callback at every step
        }

        # Handle seed: None means random, otherwise set generator
        if request.seed is not None:
            # Ensure generator is on the correct device
            generator = torch.Generator(device=self.device).manual_seed(request.seed)
            gen_params["generator"] = generator
            logger.info(f"Job {job_id}: Using seed {request.seed}")
        else:
             logger.info(f"Job {job_id}: Using random seed.")


        # Filter out None values for height/width if pipeline handles defaults well
        if gen_params["height"] is None: del gen_params["height"]
        if gen_params["width"] is None: del gen_params["width"]
        if gen_params["negative_prompt"] is None: del gen_params["negative_prompt"]

        try:
            start_time = time.time()
            jobs[job_id].message = "Generating images..."

            # Run the diffusion pipeline
            with torch.no_grad(): # Ensure no gradients are computed
                # Use torch.autocast for mixed precision if on GPU
                # Note: autocast might be implicitly handled by pipeline.to(dtype) or offloading
                # context = torch.autocast(self.device.type, dtype=self.dtype) if self.device.type != 'cpu' else nullcontext()
                # with context:
                results = pipeline(**gen_params)

            images: List[Image.Image] = results.images
            elapsed_time = time.time() - start_time
            logger.info(f"Job {job_id}: Generated {len(images)} image(s) in {elapsed_time:.2f}s.")

            # Save images and generate URLs
            image_urls = []
            image_filenames = []
            save_errors = []
            for i, img in enumerate(images):
                filename = f"image_{job_id}_{i}.png"
                filepath = self.output_folder / filename
                try:
                    img.save(filepath, "PNG")
                    image_urls.append(f"/download/{job_id}/{i}") # URL path for download endpoint
                    image_filenames.append(filename)
                    logger.info(f"Job {job_id}: Saved image {i+1}/{len(images)} to {filepath}")
                except Exception as save_e:
                    logger.error(f"Job {job_id}: Failed to save image {i}: {save_e}")
                    save_errors.append(str(save_e))

            # Update job status
            if not image_urls: # No images were successfully saved
                 jobs[job_id].status = "failed"
                 jobs[job_id].progress = 0 # Reset progress on failure
                 jobs[job_id].message = f"Image generation succeeded but saving failed. Errors: {'; '.join(save_errors)}"
            else:
                jobs[job_id].status = "completed"
                jobs[job_id].progress = 100 # Ensure 100% on completion
                jobs[job_id].image_urls = image_urls
                jobs[job_id].message = f"Images generated successfully ({len(image_urls)} saved)."
                if save_errors:
                    jobs[job_id].message += f" Some images failed to save: {'; '.join(save_errors)}"


        except Exception as e:
            logger.error(f"Job {job_id}: Image generation failed: {e}", exc_info=True)
            jobs[job_id].status = "failed"
            jobs[job_id].progress = 0 # Reset progress
            jobs[job_id].message = f"Image generation failed: {str(e)}"

        finally:
            # --- Clean up GPU memory (Important!) ---
            if self.device == torch.device("cuda"):
                torch.cuda.empty_cache()
                logger.debug(f"Job {job_id}: Cleared CUDA cache.")
            # -----------------------------------------


    def cleanup_expired_files(self):
        """Removes expired job records and their associated image files."""
        current_time = time.time()
        expired_job_ids = [job_id for job_id, job in jobs.items() if current_time > job.expires_at]

        if not expired_job_ids:
            return # Nothing to clean

        logger.info(f"Running cleanup for {len(expired_job_ids)} expired job(s)...")
        for job_id in expired_job_ids:
            job = jobs.get(job_id)
            if not job: continue # Should not happen, but safety check

            # Delete associated image files based on job status/urls
            # Assume filenames follow the pattern image_{job_id}_{index}.png
            try:
                # Glob pattern to find all images for this job_id
                files_to_delete = list(self.output_folder.glob(f"image_{job_id}_*.png"))
                if files_to_delete:
                    logger.info(f"Job {job_id}: Found {len(files_to_delete)} files to delete.")
                    for file_path in files_to_delete:
                        try:
                            file_path.unlink()
                            logger.info(f"Job {job_id}: Deleted expired file: {file_path.name}")
                        except OSError as delete_err:
                            logger.error(f"Job {job_id}: Failed to delete file {file_path.name}: {delete_err}")
                else:
                     logger.info(f"Job {job_id}: No image files found matching pattern for deletion.")

            except Exception as glob_err:
                logger.error(f"Job {job_id}: Error finding files for deletion: {glob_err}")

            # Remove job record from memory
            del jobs[job_id]
            logger.info(f"Job {job_id}: Removed expired job record.")
        logger.info("Cleanup finished.")


# --- Create Service Instance ---
try:
    service = ImageGenService()
except Exception as service_init_error:
     logger.critical(f"Failed to initialize ImageGenService: {service_init_error}", exc_info=True)
     # Exit or prevent FastAPI from starting if service fails critically
     sys.exit("Critical error during service initialization. Check logs.")


# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    logger.info("OpenImageGen API starting up...")
    # You could add a periodic background task for cleanup here if needed
    # from fastapi_utils.tasks import repeat_every
    # @repeat_every(seconds=60 * 60) # Run every hour
    # async def run_cleanup():
    #     service.cleanup_expired_files()

@app.get("/health")
async def health_check():
    """Provides health status and basic configuration info."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "default_model": service.default_model,
        "loaded_models": list(service.pipelines.keys()),
        "device": str(service.device),
        "dtype": str(service.dtype).split('.')[-1], # e.g., "float16"
        "force_gpu": service.force_gpu,
        "use_gpu": service.use_gpu,
        "active_jobs": len(jobs),
        "output_folder": str(service.output_folder.resolve()),
        "model_cache_dir": str(service.model_cache_dir.resolve()),
    }

@app.get("/models")
async def get_models():
    """Lists the models configured in the system."""
    return {"available_models": list(service.models_config.keys())}

@app.post("/submit", response_model=Dict[str, str])
async def submit_job(request: ImageGenerationRequest, background_tasks: BackgroundTasks):
    """Submits an image generation job."""
    job_id = str(uuid.uuid4())
    created_at = time.time()
    expires_at = created_at + service.file_retention_time

    # Validate model name if provided
    requested_model = request.model_name or service.default_model
    if requested_model not in service.pipelines:
         raise HTTPException(status_code=400, detail=f"Model '{requested_model}' is not loaded or configured.")

    # Store job info
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        created_at=created_at,
        expires_at=expires_at,
        request_details=request # Keep track of what was requested
    )
    logger.info(f"Submitted job {job_id} for prompt: '{request.prompt[:50]}...'")

    # Add generation task to background
    background_tasks.add_task(service.generate_images, job_id, request)
    # Add cleanup task (runs after the generation task)
    background_tasks.add_task(service.cleanup_expired_files)

    return {"job_id": job_id, "message": "Job submitted successfully"}

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Retrieves the current status of a generation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job with ID '{job_id}' not found.")
    # Trigger cleanup check opportunistically on status request
    # service.cleanup_expired_files() # Maybe too aggressive, run periodically instead
    return jobs[job_id]

@app.get("/download/{job_id}/{image_index}", response_class=Response)
async def download_image(job_id: str, image_index: int):
    """Downloads a specific generated image file for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job with ID '{job_id}' not found.")

    job = jobs[job_id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job '{job_id}' is not completed (status: {job.status}).")
    if not job.image_urls or image_index < 0 or image_index >= len(job.image_urls):
        raise HTTPException(status_code=404, detail=f"Image index {image_index} is invalid for job '{job_id}'.")

    # Construct filename based on expected pattern
    filename = f"image_{job_id}_{image_index}.png"
    filepath = service.output_folder / filename

    if not filepath.exists():
        logger.error(f"File not found for job {job_id}, index {image_index} at path: {filepath}")
        raise HTTPException(status_code=404, detail=f"Image file not found on server for job '{job_id}', index {image_index}.")

    return FileResponse(
        path=filepath,
        media_type="image/png",
        filename=filename # Suggest filename to browser
    )

# --- Static Files & Basic Web UI (Optional) ---
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
         logger.warning(f"webui.html not found in static directory: {static_dir}")

# --- Main Execution Guard (for direct run, though CLI is preferred) ---
if __name__ == "__main__":
    # This block is mainly for debugging. Use the CLI (`openimagegen`) or uvicorn directly for running.
    host = config["settings"].get("host", "127.0.0.1") # Default to localhost for direct run
    port = config["settings"].get("port", 8089)
    print(f"Running directly via main.py (DEBUG MODE) on http://{host}:{port}")
    print("Use 'openimagegen' command or 'uvicorn openimagegen.main:app' for production.")
    import uvicorn
    uvicorn.run(app, host=host, port=port)

