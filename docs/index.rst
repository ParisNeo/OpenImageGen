OpenImageGen
============

OpenImageGen is a FastAPI-based image-generation service built around pluggable
Diffusers model adapters. It provides:

* A conventional **Studio** interface at ``/webui``.
* A typed, visual **workflow editor** at ``/workflow``.
* Text-to-image and image-to-image generation.
* Automatic Hugging Face model downloads and an observable download center.
* Stable Diffusion, SDXL, SD3/3.5, PixArt-Sigma, Sana, FLUX, FLUX GGUF,
  Qwen Image, Kandinsky, and OpenAI-compatible remote image API adapters.
* Reusable workflow nodes for prompts, remote OpenAI-compatible text models,
  LoRAs, image transforms, and previews.
* Asynchronous jobs, WebSocket progress, cancellation, retention-based cleanup,
  and optional multi-user API-key authentication.
* Trusted Python community-node plugins.

.. warning::
   Generation jobs and model downloads are held in process memory. Restarting
   the service discards their status. Generated files remain subject to the
   configured retention policy.

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   onboarding
   quickstart
   configuration
   models
   workflows
   security

.. toctree::
   :maxdepth: 2
   :caption: Developers

   api
   plugins
   development

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`