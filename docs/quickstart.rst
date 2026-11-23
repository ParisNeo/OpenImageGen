Quick start
===========

Studio
------

1. Start OpenImageGen.
2. Visit ``http://127.0.0.1:8089/webui``.
3. Enter a prompt and select a model.
4. Choose dimensions, steps, guidance, and a seed, or choose a shared prompt
   template. Templates are filtered by model family and use its native profile.
5. Click **Generate**.

Generated images include PNG metadata such as the prompt, model, seed, steps,
guidance, and final output dimensions.

For SD 1.5-family checkpoints, start with 512×512, 28–35 steps, and CFG 6–7.
For SDXL, start at 1024×1024. Turbo and distilled checkpoints use different
settings; see :doc:`models`.

Workflow editor
---------------

Visit ``http://127.0.0.1:8089/workflow`` and build:

.. code-block:: text

   Prompt.text ───────────────→ Sampler.prompt
   Model.model ───────────────→ Sampler.model
   Sampler.generation params ─→ Output.generation params

Click **Run workflow**. The active node is highlighted while it executes. The
palette's **Prompt starter graphs** add a complete prompt/model/sampler/output
branch, and the Sampler's quality profile can be changed independently.

A connected visualizer can be added directly:

.. code-block:: text

   Sampler.generated image ───→ Preview Image.existing image

The image output is **deferred**: it represents an image only after diffusion
has completed. Generation parameters and rendered images are intentionally
different port types.

Image processing
----------------

Generated or uploaded images can pass through processing nodes:

.. code-block:: text

   Sampler.generated image
       → Upscale.image
       → Image Filter.image
       → Preview Image.existing image

An uploaded-image workflow does not require a diffusion model:

.. code-block:: text

   Load Image.image
       → Resize Image.image
       → Preview Image.existing image

Remote prompt expansion
-----------------------

The OpenAI Compatible node supports ``/models`` discovery and
``/chat/completions`` on compatible local or remote servers:

.. code-block:: text

   OpenAI Compatible.text ─→ Preview Text.text
   OpenAI Compatible.text ─→ Sampler.prompt

Preview Text displays a returned value only after its upstream node succeeds.
All terminal preview branches execute even when the same graph also contains an
Output node.

Select **No authentication** only for servers that do not require credentials.
For a user-supplied key, optional browser persistence is available. Keys remain
masked by default and are excluded from workflow exports and node templates.