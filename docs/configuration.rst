Configuration
=============

Configuration discovery
-----------------------

OpenImageGen loads TOML configuration using this priority:

1. ``--config PATH``
2. ``OPENIMAGEGEN_CONFIG_OVERRIDE`` or ``OPENIMAGEGEN_CONFIG``
3. Operating-system-specific configuration paths
4. ``config.toml`` in the current working directory

If no configuration exists, a default file is created in the working directory.
Loaded values are recursively merged with built-in defaults.

In-app documentation
--------------------

Studio's **?** button opens the generated Sphinx site at ``/help/``; FastAPI's
interactive OpenAPI reference remains at ``/docs``. Build the site before
starting the server:

.. code-block:: bash

   python -m pip install -e ".[docs]"
   python -m sphinx -W --keep-going -b html docs docs/_build/html

For an installed or separately deployed documentation build, set
``OPENIMAGEGEN_DOCS_DIR`` to the directory containing ``index.html``. If no
build is found, ``/help/`` displays build instructions instead of failing
silently. Studio's **Guide** button opens the interactive onboarding, which is
shown automatically on a browser's first visit.

Example
-------

.. code-block:: toml

   [settings]
   default_model = "stable_diffusion_1_5"
   host = "127.0.0.1"
   port = 8089
   force_gpu = false
   use_gpu = true
   dtype = "float16"
   auto_download_models = true
   output_folder = "./outputs"
   model_cache_dir = "./models"
   asset_dir = "./model_assets"
   file_retention_time = 3600
   cleanup_interval_seconds = 900

   [generation]
   guidance_scale = 7.0
   num_inference_steps = 30
   num_images_per_prompt = 1

   [execution]
   max_workers = 2
   max_concurrent_jobs_per_user = 1

   [models.stable_diffusion_1_5]
   name = "runwayml/stable-diffusion-v1-5"
   type = "stable_diffusion"

Important settings
------------------

``default_model``
   Used when a request or workflow does not specify a model.

``force_gpu``
   Fail rather than silently using CPU when GPU execution is unavailable.

``dtype``
   ``float16`` or ``bfloat16`` on supported accelerators. CPU generation always
   uses ``float32``.

``output_folder``
   Generated PNG files and uploaded source images.

``model_cache_dir``
   Hugging Face cache used by model adapters.

``file_retention_time``
   Lifetime, in seconds, of jobs, generated files, and uploaded images.

``execution.max_workers``
   Size of the generation thread pool. Multiple workers do not make a single
   GPU workflow parallel or reduce its VRAM requirements.

Hosted image APIs
-----------------

.. code-block:: toml

   [models.gpt_image_1]
   name = "gpt-image-1"
   type = "openai_image"
   base_url = "https://api.openai.com/v1"
   api_key_env = "OPENAI_API_KEY"
   timeout_seconds = 180

Only the environment variable's **name** is configured; its secret value remains
outside TOML. Compatible services can use another base URL or an empty
``api_key_env`` when authentication is intentionally disabled.

Administrators can also use **API** in Studio to add, edit, or delete hosted
models without manually editing TOML. The form supports the provider model ID,
base URL, credential environment-variable name, timeout, optional quality, and
TLS verification. Changes take effect immediately and are persisted to the
active configuration file. API key values are never accepted or displayed by
this interface.

Remote image providers
----------------------

.. code-block:: toml

   [models.openai_gpt_image_1]
   name = "openai/gpt-image-1"
   type = "openai_image"
   remote_model = "gpt-image-1"
   api_key_env = "OPENAI_API_KEY"
   base_url = "https://api.openai.com/v1"
   quality = "high"
   timeout_seconds = 180

The remote adapter does not support negative prompts, LoRAs, or image-to-image.
API keys are read from the server environment and are never stored in workflows.

Custom pipeline execution
-------------------------

Runtime configuration rejects ``custom_pipeline`` overrides. Diffusers custom
pipelines can execute repository code and therefore must be installed and
reviewed explicitly rather than enabled through an untrusted configuration
change.