
"""
@file        registry.py
@author      ParisNeo
@description Thread-safe workflow node registry and trusted Python plugin discovery.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Type

from .base import BaseWorkflowNode, NodeSpec

logger = logging.getLogger("OpenImageGen.nodes")

class NodeRegistry:
    """Registers executable node classes and exposes frontend-neutral schemas."""

    def __init__(self) -> None:
        self._handlers: Dict[str, BaseWorkflowNode] = {}
        self._lock = threading.RLock()
        self._plugin_modules: List[str] = []
        self._loading_origin = "builtin"

    def register(self, handler_class: Type[BaseWorkflowNode]) -> Type[BaseWorkflowNode]:
        if not isinstance(getattr(handler_class, "spec", None), NodeSpec):
            raise TypeError(f"{handler_class.__name__} must define a NodeSpec.")
        handler = handler_class()
        with self._lock:
            node_type = handler.spec.type
            if node_type in self._handlers:
                raise ValueError(f"Workflow node type '{node_type}' is already registered.")
            handler.spec.origin = self._loading_origin
            self._handlers[node_type] = handler
            logger.info(f"Registered {handler.spec.origin} workflow node '{node_type}'.")
        return handler_class

    def get(self, node_type: str) -> BaseWorkflowNode:
        with self._lock:
            handler = self._handlers.get(node_type)
        if handler is None:
            raise KeyError(node_type)
        return handler

    def specs(self) -> List[NodeSpec]:
        with self._lock:
            specs = [handler.spec.model_copy(deep=True) for handler in self._handlers.values()]
        return sorted(specs, key=lambda spec: (spec.category.lower(), spec.title.lower()))

    def load_plugins(self, paths: Iterable[str]) -> List[str]:
        """Import trusted Python plugins. Never enable this for untrusted uploads."""
        errors: List[str] = []
        for configured_path in paths:
            directory = Path(configured_path).expanduser().resolve()
            if not directory.is_dir():
                continue
            for file in sorted(directory.glob("*.py")):
                if file.name.startswith("_"):
                    continue
                module_name = f"openimagegen_community_{abs(hash(str(file)))}_{file.stem}"
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file)
                    if spec is None or spec.loader is None:
                        raise ImportError("Could not create a Python module loader.")
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    self._loading_origin = "community"
                    spec.loader.exec_module(module)
                    self._plugin_modules.append(module_name)
                except Exception as error:
                    sys.modules.pop(module_name, None)
                    errors.append(f"{file}: {error}")
                    logger.error(f"Failed to load workflow plugin '{file}': {error}", exc_info=True)
                finally:
                    self._loading_origin = "builtin"
        return errors

    def reload_plugins(self, paths: Iterable[str]) -> List[str]:
        """Replace community registrations after an administrator edits plugin files."""
        with self._lock:
            for node_type in [
                key for key, handler in self._handlers.items()
                if handler.spec.origin == "community"
            ]:
                del self._handlers[node_type]
            for module_name in self._plugin_modules:
                sys.modules.pop(module_name, None)
            self._plugin_modules.clear()
        return self.load_plugins(paths)

_registry = NodeRegistry()

def get_node_registry() -> NodeRegistry:
    return _registry
