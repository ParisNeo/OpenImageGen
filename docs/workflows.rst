Workflow engine
===============

Graph format
------------

A workflow is a directed acyclic graph of declarative nodes:

.. code-block:: json

   {
     "name": "rabbit_gala",
     "nodes": [
       {
         "id": "prompt",
         "type": "text_prompt",
         "params": {"text": "An elegant rabbit at a gala"},
         "inputs": {}
       },
       {
         "id": "sample",
         "type": "sampler",
         "params": {
           "width": 512,
           "height": 512,
           "num_inference_steps": 30,
           "guidance_scale": 7,
           "sampler": "dpmpp_2m_karras"
         },
         "inputs": {"prompt": "prompt.text"}
       },
       {
         "id": "out",
         "type": "output",
         "params": {},
         "inputs": {"params": "sample.params"}
       }
     ]
   }

References use ``source_node_id.output_slot``. Omitting the slot selects the
source's first output and is supported for compatibility. Explicit slots are
recommended when a node has multiple outputs.

Validation
----------

Before execution, OpenImageGen validates:

* Unique node identifiers and registered node types.
* Known parameters, required fields, and required inputs.
* Existing source nodes and output slots.
* Port-type compatibility.
* Self-references and graph cycles.
* Presence of at least one Output or Preview node.
* A non-empty final image prompt for generation workflows.

Only nodes reachable from terminal Output or Preview nodes execute. All terminal
branches are included, allowing image generation and text previews in one run.

Deferred images
---------------

Sampler and Img2Img expose:

``params``
   A serializable generation request consumed by Output.

``image``
   A deferred reference resolved by the service after generation.

An image chain can therefore include resize, upscale, and filter nodes. These
operations run on actual PIL images after diffusion. Uploaded image references
can be processed without running a model.

The built-in Upscale node uses deterministic resampling, not an AI detail
synthesis model. Processing is constrained by output-dimension and pixel-count
limits.

Results and progress
--------------------

Jobs expose ``active_node_ids`` for the running graph node and ``node_results``
for terminal preview values. Updates are available through polling and
WebSockets.

Only one primary generation request is selected per run. The current engine is
not a general multi-generation scheduler or batch-image execution engine.

Inpainting and outpainting (Studio)
-----------------------------------

Studio exposes four generation modes:

``Text → Image``
   Standard text-to-image generation (``POST /submit``).

``Image → Image``
   Denoising-based transformation using a full image prompt
   (``POST /submit_img2img``).

``Inpaint``
   Mask-guided editing (``POST /submit_inpainting``). Upload an init image,
   then paint white where you want changes on the mask canvas. The mask is
   uploaded via ``POST /upload_mask`` and stored as a greyscale PNG
   (``mask_<id>.png``) alongside the init image.

``Outpaint``
   Expands the canvas outward (``POST /submit_outpainting``). Provide
   ``expand_top``, ``expand_bottom``, ``expand_left``, and ``expand_right``
   (each 0–2048, combined ≤ 2048 per axis). The service composes an enlarged
   canvas, pastes the original into position, and builds a mask covering only
   the newly added area.

The service routes both flows through ``BaseModelAdapter.generate_inpainting``.
Adapters that do not implement this hook report ``supports_inpainting == False``
in ``GET /models`` and the UI warns before submission. Current support:
Stable Diffusion 1.x adapters. Dedicated SDXL inpaint, FLUX edit, and Qwen Edit
flows require their specialised pipelines and are not enabled here.

Mask editor semantics:

* White pixels regenerate. Black pixels are preserved.
* Painting uses a brush (white) or eraser (black) with a configurable size.
* The mask canvas resizes to the uploaded image and invalidates its cached
  ``mask_image_id`` whenever paint or invert changes the pixels.

Reusable node templates
-----------------------

The editor can save a node's ordinary settings as a browser-local reusable
template. Templates contain no graph connections, secret fields, or API-key
persistence preferences. Workflow export similarly excludes secret values.