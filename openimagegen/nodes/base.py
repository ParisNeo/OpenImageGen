
"""
@file        base.py
@author      ParisNeo
@description Public API for built-in and community workflow nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

PortKind = Literal["text", "model", "image", "params", "number", "boolean", "json", "any"]
FieldKind = Literal[
    "text", "textarea", "password", "number", "boolean", "select", "model", "upload"
]

class NodePort(BaseModel):
    """A named, typed connection point. ``slot`` is the serialized input key."""

    slot: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
    label: str
    kind: PortKind = "any"
    required: bool = False

class NodeField(BaseModel):
    """Declarative editor control and validation metadata."""

    key: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
    label: str
    type: FieldKind = "text"
    default: Any = None
    required: bool = False
    secret: bool = False
    placeholder: Optional[str] = None
    help: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: List[Dict[str, str]] = Field(default_factory=list)

class NodeSpec(BaseModel):
    """Serializable node contract consumed by any workflow frontend."""

    type: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    title: str
    category: str = "Community"
    description: str = ""
    color: str = "#64748b"
    icon: str = "🧩"
    inputs: List[NodePort] = Field(default_factory=list)
    outputs: List[NodePort] = Field(default_factory=list)
    fields: List[NodeField] = Field(default_factory=list)
    origin: str = "builtin"

@dataclass
class NodeExecutionContext:
    """Services exposed to nodes. Community code must not receive API secrets globally."""

    report: Callable[[int, str], None] = lambda progress, message: None
    is_cancelled: Callable[[], bool] = lambda: False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def checkpoint(self, message: str = "Node cancelled.") -> None:
        if self.is_cancelled():
            raise RuntimeError(message)

class BaseWorkflowNode:
    """Subclass this and decorate with ``@get_node_registry().register``."""

    spec: NodeSpec

    def execute(
        self,
        params: Dict[str, Any],
        inputs: Dict[str, Any],
        context: NodeExecutionContext,
    ) -> Dict[str, Any]:
        """Return values keyed by output slot. Must not mutate the input graph."""
        raise NotImplementedError
