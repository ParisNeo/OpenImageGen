"""Model adapter registry with thread-safe singleton."""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Type

from .base import BaseModelAdapter

logger = logging.getLogger("OpenImageGen.models.registry")


class ModelRegistry:
    """Thread-safe singleton mapping `type_name` -> adapter class."""

    _instance: Optional["ModelRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._adapters: Dict[str, Type[BaseModelAdapter]] = {}
                    cls._instance = instance
        return cls._instance

    def register(self, adapter_cls: Type[BaseModelAdapter]) -> Type[BaseModelAdapter]:
        if not adapter_cls.type_name:
            raise ValueError(f"{adapter_cls.__name__} must define a non-empty `type_name`.")
        existing = self._adapters.get(adapter_cls.type_name)
        if existing is not None and existing is not adapter_cls:
            logger.warning(
                f"Overriding existing adapter for type {adapter_cls.type_name!r}: "
                f"{existing.__name__} -> {adapter_cls.__name__}"
            )
        self._adapters[adapter_cls.type_name] = adapter_cls
        logger.debug(f"Registered adapter: {adapter_cls.__name__} as {adapter_cls.type_name!r}")
        return adapter_cls

    def get(self, type_name: str) -> Type[BaseModelAdapter]:
        if type_name not in self._adapters:
            raise KeyError(
                f"No adapter registered for type {type_name!r}. "
                f"Available: {sorted(self._adapters.keys())}"
            )
        return self._adapters[type_name]

    def try_get(self, type_name: str) -> Optional[Type[BaseModelAdapter]]:
        return self._adapters.get(type_name)

    def list_types(self) -> List[str]:
        return sorted(self._adapters.keys())

    def clear(self) -> None:
        self._adapters.clear()


def get_registry() -> ModelRegistry:
    return ModelRegistry()
