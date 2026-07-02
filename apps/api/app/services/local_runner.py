from __future__ import annotations

"""Compatibility facade for local experiment planning and execution."""

from app.services.local_executor import (
    LocalRunRequest,
    run_local_experiment,
)
from app.services.local_plan import (
    _attack_params,
    _attack_variants_for_attack,
    _strengths_for_attack,
    estimate_selection,
    normalize_selection,
)

__all__ = [
    "LocalRunRequest",
    "_attack_params",
    "_attack_variants_for_attack",
    "_strengths_for_attack",
    "estimate_selection",
    "normalize_selection",
    "run_local_experiment",
]
