
"""
Example trusted community node. Copy to your configured plugin directory and enable plugins.
Community plugins execute with server privileges; review third-party code before installing.
"""

from openimagegen.nodes import BaseWorkflowNode, NodeField, NodePort, NodeSpec, get_node_registry

@get_node_registry().register
class UppercaseNode(BaseWorkflowNode):
    spec = NodeSpec(
        type="community_uppercase",
        title="Uppercase Text",
        category="Community examples",
        icon="🔠",
        color="#0ea5e9",
        description="Uppercase incoming text or the configured value.",
        inputs=[NodePort(slot="text", label="text", kind="text")],
        outputs=[NodePort(slot="text", label="text", kind="text")],
        fields=[NodeField(key="text", label="Fallback text", type="textarea", default="")],
    )

    def execute(self, params, inputs, context):
        return {"text": str(inputs.get("text") or params.get("text") or "").upper()}
