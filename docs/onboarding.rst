
Onboarding
==========

OpenImageGen provides two creation surfaces:

* **Studio** at ``/webui`` for prompt-driven image generation.
* **Workflow editor** at ``/workflow`` for reusable, typed node graphs.

Studio includes a **Guide** button that opens the interactive onboarding. It
appears automatically on a browser's first visit and can be reopened at any
time. **?** opens this Sphinx documentation in a new tab.

First local generation
----------------------

1. Start OpenImageGen and open ``http://127.0.0.1:8089/webui``.
2. Select a local model such as ``stable_diffusion_1_5``.
3. Enter a prompt describing the subject, environment, lighting, and framing.
4. Keep the **Recommended** quality profile initially.
5. Select **Generate**. If necessary, model weights download automatically.
6. Follow progress in the header and the **Downloads** panel.

Local generation requires enough disk space and memory for the selected model.
GPU availability and execution estimates appear in the model information and
service information panels. A configured ``force_gpu = true`` deliberately
prevents a CPU fallback.

Hosted image APIs
-----------------

Hosted image models do not download local weights and do not require a local
GPU. They use the provider's OpenAI-compatible ``/images/generations`` API.

For OpenAI, expose the credential to the **server process**, for example in the
current PowerShell session:

.. code-block:: powershell

   $env:OPENAI_API_KEY = "your-provider-secret"
   .\run.bat

Then select **API** in Studio and configure:

``Configuration key``
   A unique local identifier such as ``my_gpt_image``.

``Provider model ID``
   The identifier sent to the provider, such as ``gpt-image-1`` or
   ``dall-e-3``.

``Base URL``
   ``https://api.openai.com/v1`` for OpenAI, or the equivalent compatible
   provider URL.

``API key environment variable``
   ``OPENAI_API_KEY`` by default. Enter only the variable name, never its
   value. Leave it empty only for an intentionally unauthenticated provider.

The configuration is persisted in the active TOML file. Provider credentials
remain in the environment and are not returned by the configuration API.

.. note::
   The API-key field in Studio's header authenticates a browser to OpenImageGen
   when multi-user authentication is enabled. It is not an OpenAI provider key.

First workflow
--------------

Open **Workflow editor** and create this graph:

.. code-block:: text

   Prompt.text ───────────────→ Sampler.prompt
   Model.model ───────────────→ Sampler.model
   Sampler.generation params ─→ Output.generation params

A **Prompt starter graph** in the palette creates these nodes automatically.
Connect Sampler's **generated image** output to Preview Image when an immediate
result preview is desired. Select **Run workflow** to execute the graph.

The OpenAI Compatible text node can expand a prompt through a compatible
``/chat/completions`` endpoint. Its user-entered secrets are excluded from
workflow exports and templates; server environment credentials are preferred
for shared deployments.

Help and operations
-------------------

The complete documentation is served at ``/help/`` when built. FastAPI's
interactive OpenAPI reference remains available at ``/docs``. Jobs and download
history are process-local; generated images and uploads expire according to
``file_retention_time``.
