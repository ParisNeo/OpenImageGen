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

Reusable node templates
-----------------------

The editor can save a node's ordinary settings as a browser-local reusable
template. Templates contain no graph connections, secret fields, or API-key
persistence preferences. Workflow export similarly excludes secret values.