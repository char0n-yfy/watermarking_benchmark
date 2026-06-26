from __future__ import annotations

from typing import Any, Type

from .base import BaseAttack


ATTACK_REGISTRY: dict[str, Type[BaseAttack]] = {}


def register_attack(cls: Type[BaseAttack]) -> Type[BaseAttack]:
    if not cls.name or cls.name == "base":
        raise ValueError("Attack class must define a non-empty unique name")
    key = cls.name.lower()
    if key in ATTACK_REGISTRY:
        raise ValueError(f"Attack '{key}' is already registered")
    ATTACK_REGISTRY[key] = cls
    return cls


def build_attack(name: str, **params: Any) -> BaseAttack:
    key = name.lower()
    if key not in ATTACK_REGISTRY:
        known = ", ".join(sorted(ATTACK_REGISTRY))
        raise KeyError(f"Unknown attack '{name}'. Available attacks: {known}")
    return ATTACK_REGISTRY[key](**params)


def list_attacks() -> list[dict[str, str]]:
    return [
        {
            "name": cls.name,
            "description": cls.description,
        }
        for cls in ATTACK_REGISTRY.values()
    ]
