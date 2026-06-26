from __future__ import annotations

from typing import Any, Type

from .base import BaseWatermark


WATERMARK_REGISTRY: dict[str, Type[BaseWatermark]] = {}


def register_watermark(cls: Type[BaseWatermark]) -> Type[BaseWatermark]:
    if not cls.name or cls.name == "base":
        raise ValueError("Watermark class must define a non-empty unique name")
    key = cls.name.lower()
    if key in WATERMARK_REGISTRY:
        raise ValueError(f"Watermark method '{key}' is already registered")
    WATERMARK_REGISTRY[key] = cls
    return cls


def build_watermark(name: str, **params: Any) -> BaseWatermark:
    key = name.lower()
    if key not in WATERMARK_REGISTRY:
        known = ", ".join(sorted(WATERMARK_REGISTRY))
        raise KeyError(f"Unknown watermark method '{name}'. Available methods: {known}")
    return WATERMARK_REGISTRY[key](**params)


def list_watermarks() -> list[dict[str, str]]:
    return [
        {
            "name": cls.name,
            "description": cls.description,
        }
        for cls in WATERMARK_REGISTRY.values()
    ]
