Installation
============

Requirements
------------

OpenImageGen requires **Python 3.11 or newer**. PyTorch, Diffusers, and their
transitive dependencies require substantial disk space. Model downloads require
additional storage; large models can require tens of gigabytes.

For GPU generation, install a PyTorch build compatible with the installed
graphics driver **in the same Python environment used to run OpenImageGen**.
Installing a CPU-only PyTorch wheel does not provide CUDA support.

Windows
-------

From the repository root:

.. code-block:: powershell

   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -e .

Start the server:

.. code-block:: powershell

   .\run.bat

The launcher activates ``.venv`` or ``venv`` when present and uses the installed
``openimagegen`` command when available.

Linux and macOS
---------------

.. code-block:: bash

   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e .
   openimagegen serve --host 127.0.0.1 --port 8089

Verify the runtime
------------------

Open ``http://127.0.0.1:8089/health`` and inspect:

* ``device``
* ``dtype``
* ``torch_version``
* ``torch_cuda_build``
* ``cuda_available``
* ``gpu_name``

``force_gpu = true`` makes startup fail when no supported GPU is available.
Set it to ``false`` to allow the much slower CPU fallback.

Download models
---------------

Models are downloaded automatically before first use when
``auto_download_models = true``. They can also be downloaded explicitly:

.. code-block:: bash

   openimagegen download --models stable_diffusion_1_5 dreamshaper_8

Omit ``--models`` to download every configured model. Large families such as
Qwen Image and FLUX should generally be downloaded individually.

Some Hugging Face repositories require accepting a license and supplying an
access token. Set ``HF_TOKEN`` in the **server process environment**; do not put
the token into a workflow or commit it to configuration files.

Remote image models use their provider credentials instead of ``HF_TOKEN``.
For the included OpenAI-compatible presets, set ``OPENAI_API_KEY`` in the server
environment. The adapter sends requests to ``/v1/images/generations`` by default;
use the model's ``base_url`` and ``api_key_env`` settings for compatible providers.

Hosted image models use ``OPENAI_API_KEY`` by default instead of ``HF_TOKEN``.
They do not require a model download or a local GPU, but generation still runs
through the same asynchronous job API.