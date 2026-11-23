Community nodes
===============

OpenImageGen discovers trusted Python workflow plugins from configured folders:

.. code-block:: toml

   [plugins]
   enabled = true
   paths = ["./community_nodes"]

.. danger::
   Plugins execute with the server's privileges. Never enable uploads,
   third-party downloads, or unreviewed files as plugin directories.

Minimal node
------------

.. code-block:: python

   from openimagegen.nodes import (
       BaseWorkflowNode,
       NodeField,
       NodePort,
       NodeSpec,
       get_node_registry,
   )

   @get_node_registry().register
   class UppercaseNode(BaseWorkflowNode):
       spec = NodeSpec(
           type="community_uppercase",
           title="Uppercase Text",
           category="Community examples",
           icon="🔠",
           inputs=[NodePort(slot="text", label="text", kind="text")],
           outputs=[NodePort(slot="text", label="text", kind="text")],
           fields=[
               NodeField(
                   key="text",
                   label="Fallback text",
                   type="textarea",
                   default="",
               )
           ],
       )

       def execute(self, params, inputs, context):
           context.checkpoint()
           value = inputs.get("text") or params.get("text") or ""
           return {"text": str(value).upper()}

A node must return a dictionary containing every declared output slot. Port
kinds include ``text``, ``model``, ``image``, ``params``, ``number``,
``boolean``, ``json``, and ``any``.

Secrets and cancellation
------------------------

Set ``secret=True`` on credential fields so workflow history redacts them.
Do not log credentials or include them in returned values. Call
``context.checkpoint()`` before and after potentially long operations where
possible.

Reloading
---------

Use **Nodes** in the editor or ``POST /v1/nodes/reload`` after changing trusted
plugin files. Reload removes previous community registrations and imports the
configured files again. Existing saved workflows using a removed node type will
no longer validate.