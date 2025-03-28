# OpenImageGen

<div align="center">
  <!-- Add a relevant logo/icon later -->
  <!-- <img src="https://github.com/ParisNeo/OpenImageGen/blob/main/assets/icon.png" alt="Logo" width="200" height="200"> -->
  <p>🎨</p>
</div>

[![License](https://img.shields.io/github/license/ParisNeo/OpenImageGen)](https://github.com/ParisNeo/OpenImageGen/blob/main/LICENSE) <!-- Update URL -->
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-311/)
<!-- Add other badges as needed -->

Open source image generation API using various diffusion models via the `diffusers` library.

## Features

- 🎨 Generate images from text prompts using state-of-the-art diffusion models.
- 🔄 Support for multiple models (Stable Diffusion, SDXL, Kandinsky, etc.).
- ⚙️ Configurable via `config.toml` with flexible search paths.
- 🚀 FastAPI-based RESTful API with asynchronous job processing.
- 📊 Job status checking and image downloading.
- 🧹 Automatic file purging after a configurable time.
- 🔧 Control over GPU usage, data type, and generation parameters.
- 📝 Logging for debugging and monitoring.
- 📦 Easy installation via pip.
- 🐧 Ubuntu systemd service support (example provided).
- 🐳 Docker integration (example provided).

## Installation

### Prerequisites

- Python 3.11 or higher
- CUDA-enabled GPU (highly recommended for performance)
- Git (for installing from source)

### Install via pip (Once published)

```bash
# pip install openimagegen # (Coming soon)
```

### Install from source

Clone the repository:

```bash
git clone https://github.com/ParisNeo/OpenImageGen.git # Update URL
cd OpenImageGen
```

Install dependencies (preferably in a virtual environment):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
pip install . # Installs OpenImageGen itself
```

## Usage

### Run the API

Use the command-line interface:

```bash
openimagegen --host 0.0.0.0 --port 8089
```

Or directly with Uvicorn:

```bash
uvicorn openimagegen.main:app --host 0.0.0.0 --port 8089
```

You can specify a custom config file:

```bash
openimagegen --host 0.0.0.0 --port 8089 --config /path/to/custom_config.toml
```

Alternatively, set the `OPENIMAGEGEN_CONFIG` environment variable:

```bash
export OPENIMAGEGEN_CONFIG=/path/to/custom_config.toml
openimagegen --host 0.0.0.0 --port 8089
```

### Config File Search Paths

The API searches for `config.toml` in the following locations (priority order):

1.  Path specified via the `--config` command-line argument.
2.  Path specified via the `OPENIMAGEGEN_CONFIG` environment variable.
3.  System-specific locations:
    - **Linux:** `/etc/openimagegen/config.toml`, `/usr/local/etc/openimagegen/config.toml`, `~/.config/openimagegen/config.toml`, `./config.toml`
    - **Windows:** `%APPDATA%/openimagegen/config.toml`, `./config.toml`
    - **macOS:** `~/Library/Application Support/openimagegen/config.toml`, `/usr/local/etc/openimagegen/config.toml`, `./config.toml`
4.  If no config file is found, a default `config.toml` is created in the current directory.

## API Endpoints

- `GET /health`: Check service status and configuration.
- `GET /models`: List available models defined in the config.
- `POST /submit`: Submit an image generation job and get a job ID.
- `GET /status/{job_id}`: Check the status and progress of a job.
- `GET /download/{job_id}/{image_index}`: Download a specific generated image for a job.
- `GET /webui`: (Optional) Access a basic web interface.

## Example Usage (using curl)

### Submit a Job

```bash
curl -X POST "http://localhost:8089/submit" \
-H "Content-Type: application/json" \
-d '{
    "prompt": "A photorealistic astronaut riding a horse on the moon",
    "negative_prompt": "low quality, blurry, cartoon, drawing",
    "model_name": "stable_diffusion_xl",
    "height": 1024,
    "width": 1024,
    "steps": 30,
    "guidance_scale": 7.0,
    "num_images_per_prompt": 2,
    "seed": 12345
}'
```

Response:

```json
{
    "job_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "message": "Job submitted successfully"
}
```

### Check Job Status

```bash
curl "http://localhost:8089/status/a1b2c3d4-e5f6-7890-1234-567890abcdef"
```

Response (during processing):

```json
{
    "job_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "status": "processing",
    "progress": 50,
    "message": "Generating images...",
    "image_urls": null,
    "created_at": 1678886400.123,
    "expires_at": 1678890000.123
}
```

Response (when completed):

```json
{
    "job_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "status": "completed",
    "progress": 100,
    "message": "Images generated successfully",
    "image_urls": [
        "/download/a1b2c3d4-e5f6-7890-1234-567890abcdef/0",
        "/download/a1b2c3d4-e5f6-7890-1234-567890abcdef/1"
    ],
    "created_at": 1678886400.123,
    "expires_at": 1678890000.123
}
```

### Download an Image

Download the first generated image (index 0):

```bash
curl "http://localhost:8089/download/a1b2c3d4-e5f6-7890-1234-567890abcdef/0" --output image_0.png
```

Download the second generated image (index 1):

```bash
curl "http://localhost:8089/download/a1b2c3d4-e5f6-7890-1234-567890abcdef/1" --output image_1.png
```

## Configuration

Edit `config.toml` to customize models, settings, and generation defaults.

```toml
[models]
# Add models from Hugging Face Hub
stable_diffusion_1_5 = {name = "runwayml/stable-diffusion-v1-5", type = "stable_diffusion"}
stable_diffusion_xl = {name = "stabilityai/stable-diffusion-xl-base-1.0", type = "stable_diffusion_xl"}
# Example with Refiner for SDXL
# stable_diffusion_xl_refiner = {name = "stabilityai/stable-diffusion-xl-refiner-1.0", type = "stable_diffusion_xl_refiner"}

[settings]
default_model = "stable_diffusion_1_5"
force_gpu = false
use_gpu = true
dtype = "float16" # "float16" or "bfloat16"
output_folder = "./outputs"
model_cache_dir = "./models"
port = 8089
host = "0.0.0.0"
file_retention_time = 3600 # 1 hour

[generation]
guidance_scale = 7.5
num_inference_steps = 50
num_images_per_prompt = 1
```

- **Model Types:** `stable_diffusion`, `stable_diffusion_xl`, `kandinsky`, `deepfloyd_if`. The code needs specific loading logic for each type. SDXL might need handling for base + refiner models.
- **`force_gpu`**: Requires a GPU or raises an error.
- **`use_gpu`**: Uses GPU if available.
- **`dtype`**: `float16` is generally faster and uses less VRAM; `bfloat16` can offer better stability/quality on compatible hardware (Ampere+).

## Setting Up as an Ubuntu Service

(Adapt the instructions from `OpenVideoGen/README.md`, replacing `openvideogen` with `openimagegen`, updating paths, username, and the `ExecStart` command).

## Docker Integration

(Adapt the instructions from `OpenVideoGen/README.md`, replacing `openvideogen` with `openimagegen`, updating paths, port mappings, and image names).

## Supported Models (Examples)

- `stable_diffusion_1_5`: Stable Diffusion v1.5
- `stable_diffusion_xl`: Stable Diffusion XL Base 1.0
- `kandinsky`: Kandinsky 2.1/2.2 (requires specific pipeline handling)
- `deepfloyd_if`: DeepFloyd IF (requires specific pipeline handling and potentially multiple stages)

Add models to `config.toml`. You may need to extend `main.py` to handle the loading and generation logic for different pipeline types.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
