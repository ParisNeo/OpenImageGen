HTTP and Python API
===================

Interactive HTTP reference
--------------------------

FastAPI serves OpenAPI documentation at ``/docs`` and the schema at
``/openapi.json`` while the application is running.

Generation
----------

``POST /submit``
   Submit an :class:`openimagegen.schemas.ImageGenerationRequest`.

``POST /submit_img2img``
   Submit an :class:`openimagegen.schemas.ImageToImageRequest`; its
   ``init_image_id`` must reference an existing upload.

``POST /v1/workflows/submit``
   Validate and submit a workflow graph.

``GET /status/{job_id}``
   Retrieve status, progress, active nodes, preview results, seed, and image
   URLs.

``POST /cancel/{job_id}``
   Request cancellation. Cancellation is cooperative at workflow and supported
   generation checkpoints.

``GET /v1/jobs``
   List jobs visible to the authenticated caller.

Images and models
-----------------

``POST /upload``
   Upload an image and receive an unguessable ``image_id``.

``GET /uploads/{image_id}``
   Retrieve an uploaded image.

``GET /download/{job_id}/{image_index}``
   Retrieve a completed generated image.

``GET /models``
   List configured models, download state, loaded state, and capabilities.

``POST /preload``
   Start downloading selected model keys.

``POST /models/{model_key}/load`` and ``POST /models/{model_key}/unload``
   Explicitly manage a model pipeline.

``GET /gguf/presets`` and ``POST /gguf/presets/{preset_id}/install``
   Inspect and install curated FLUX GGUF presets.

``GET /v1/downloads`` and ``GET /v1/downloads/{download_id}``
   Inspect persistent-for-process model download jobs.

Workflow discovery
------------------

``GET /v1/nodes``
   Return declarative schemas for built-in and loaded community nodes.

``POST /v1/nodes/openai_compatible/models``
   Discover models from the supplied OpenAI-compatible server configuration.

``POST /v1/nodes/reload``
   Reload trusted community plugins when plugins are enabled.

``WS /ws/{job_id}``
   Receive job updates. Sending the text ``cancel`` requests cancellation.

Python request example
----------------------

.. code-block:: python

   import requests

   base = "http://127.0.0.1:8089"
   response = requests.post(
       f"{base}/submit",
       json={
           "prompt": "A watercolor fox in a forest",
           "model_name": "dreamshaper_8",
           "width": 512,
           "height": 512,
           "num_inference_steps": 30,
           "guidance_scale": 7,
           "sampler": "dpmpp_2m_karras",
       },
       timeout=30,
   )
   response.raise_for_status()
   job_id = response.json()["job_id"]

Python modules
--------------

Application and service
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: openimagegen.app
   :members:

.. automodule:: openimagegen.service
   :members:

Workflows and schemas
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: openimagegen.workflow
   :members:

.. automodule:: openimagegen.schemas
   :members:

Nodes
~~~~~

.. automodule:: openimagegen.nodes.base
   :members:

.. automodule:: openimagegen.nodes.registry
   :members:

Models
~~~~~~

.. automodule:: openimagegen.models.base
   :members:

.. automodule:: openimagegen.models.registry
   :members: