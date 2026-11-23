Models and adapters
===================

Configured model keys are independent of Hugging Face repository names. This
allows several variants of one family to coexist:

.. code-block:: toml

   [models.dreamshaper_8]
   name = "Lykon/dreamshaper-8"
   type = "stable_diffusion"

   [models.sdxl_turbo]
   name = "stabilityai/sdxl-turbo"
   type = "stable_diffusion_xl"

Recommended starting points
---------------------------

The built-in defaults were checked against the official model cards rather than
treating every adapter as a conventional SD 1.5 model. Notable sources include
`SD 1.5 <https://huggingface.co/runwayml/stable-diffusion-v1-5>`_,
`DreamShaper 8 <https://huggingface.co/Lykon/dreamshaper-8>`_,
`SDXL <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>`_,
`SD Turbo <https://huggingface.co/stabilityai/sd-turbo>`_,
`SDXL Turbo <https://huggingface.co/stabilityai/sdxl-turbo>`_,
`Qwen Image <https://huggingface.co/Qwen/Qwen-Image>`_, and
`Sana <https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers>`_.
There is no universally optimal seed, prompt, or sampler: these are starting
points for evaluation, not a guarantee that every prompt will render correctly.

.. list-table::
   :header-rows: 1
   :widths: 22 24 18 18 18

   * - Family
     - Resolution
     - Steps
     - CFG
     - Notes
   * - SD 1.5 / DreamShaper
     - 512×512
     - 28–35
     - 6–7
     - Supports negative prompts and Img2Img.
   * - SDXL
     - 1024×1024
     - 25–40
     - 5–7
     - Substantially heavier than SD 1.5.
   * - SD / SDXL Turbo
     - 512×512
     - 1–4
     - 0
     - Distilled; standard SD settings and negative prompts are unsuitable.
   * - SDXL Turbo
     - 1024×1024
     - 1–4
     - 0
     - Distilled; use Auto 1111 sampler naming in workflows.
   * - FLUX.1-schnell
     - 512–1024 square
     - 4
     - 0
     - Negative prompts are not supported.
   * - Stable Diffusion 3.5 Medium
     - 1024×1024
     - 28
     - 4.5
     - Gated Hugging Face repository; set ``HF_TOKEN`` after accepting its license.
   * - PixArt-Sigma
     - 1024×1024
     - 20
     - 4.5
     - Efficient text-to-image pipeline; text-to-image only in this adapter.
   * - Sana 1.6B
     - 1024×1024
     - 20
     - 4.5
     - Efficient model; the preset requests its bfloat16 weights.
   * - OpenAI-compatible remote image
     - 1024×1024 by default
     - 1
     - 1
     - Calls ``/images/generations``; requires an API-key environment variable. No local weights are downloaded.
   * - Qwen Image
     - 1328×1328
     - 50
     - 4
     - Very large model; text-to-image only in this adapter.

Hosted OpenAI-compatible models
-------------------------------

The ``openai_image`` adapter calls ``POST /images/generations`` and supports
both base64 and image-URL responses. The included ``gpt_image_1`` and
``dall_e_3`` presets require ``OPENAI_API_KEY`` in the server environment. API
keys are never stored in model configuration or workflow files.

These models have no local weights: preload is a no-op and they appear as
available without occupying the Hugging Face cache. They do not support negative
prompts, LoRAs, local samplers, or Diffusers step/CFG controls. Resolution and
batch count are sent to the provider. For a compatible third-party service, set
``base_url`` and optionally set ``api_key_env = ""`` for an unauthenticated
local endpoint. TLS verification is enabled by default.

The workflow editor applies family-specific defaults when a checkpoint changes.
Use **Reset settings for selected model** to restore them explicitly. Studio and
the workflow editor also expose **Draft**, **Recommended**, and **Quality**
profiles. These preserve native training resolutions instead of upscaling SD 1.5
to resolutions that can produce duplicated subjects or distorted anatomy.

Distilled models intentionally use low step counts, zero CFG, and Euler
Ancestral. FLUX, SD3, and Qwen profiles use FlowMatch Euler. Increasing CFG on a
distilled checkpoint generally makes rendering worse rather than better.

Schedulers
----------

Requests accept DPM++ 2M, DPM++ 2M Karras, DPM++ 2M SDE Karras, DPM++ SDE
Karras, UniPC, FlowMatch Euler, Euler Ancestral, Euler, and DDIM. DPM++ 2M Karras
is the recommended general SD/SDXL starting point; the SDE variant trades speed
for additional sampling detail. FlowMatch Euler is intended for flow-matching
families. The sampler and generation parameters are also embedded in PNG
metadata. Sampler controls are omitted for hosted image APIs.

FLUX GGUF
---------

A GGUF file contains the quantized FLUX transformer, not the complete model.
The adapter also needs compatible text encoders, tokenizer, VAE, and scheduler
components:

.. code-block:: toml

   [models.flux_schnell_gguf_q4_k_s]
   name = "black-forest-labs/FLUX.1-schnell"
   type = "flux_gguf"
   base_repositories = ["Comfy-Org/FLUX.1-schnell"]
   gguf_source = "city96/FLUX.1-schnell-gguf"
   gguf_filename = "flux1-schnell-Q4_K_S.gguf"

The **GGUF** catalog in Studio lists supported presets and installation status.
Displayed GGUF sizes refer to the transformer alone.

LoRAs
-----

A LoRA node accepts a Hugging Face repository and weight filename, or a local
model asset. Multiple LoRA nodes can be chained. Adapters are loaded for the
current request and unloaded afterward.

.. code-block:: text

   Model.model → LoRA.model → Sampler.model

GGUF FLUX intentionally reports LoRAs as unsupported. Unsupported combinations
fail explicitly instead of silently ignoring the adapter.

Adapter lifecycle
-----------------

Pipelines are lazily loaded and cached. ``/models/{model_key}/load`` and
``/models/{model_key}/unload`` provide explicit lifecycle control. Downloads are
tracked separately from loaded pipelines.