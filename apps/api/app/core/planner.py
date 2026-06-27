from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class AttackSelection:
    preset_id: str
    method: str
    strengths: tuple[float, ...]


@dataclass(frozen=True)
class ExperimentSpec:
    spec_id: str
    dataset_version_ids: tuple[str, ...]
    algorithm_version_ids: tuple[str, ...]
    attack_presets: tuple[AttackSelection, ...]
    seeds: tuple[int, ...]
    max_samples_per_dataset: int | None = None
    params: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExperimentCell:
    cell_key: str
    dataset_version_id: str
    algorithm_version_id: str
    attack_preset_id: str
    attack_method: str
    attack_strength: float
    seed: int


def _non_empty(values: Iterable[T], field_name: str) -> tuple[T, ...]:
    items = tuple(values)
    if not items:
        raise ValueError(f"{field_name} must not be empty")
    return items


def materialize_cells(spec: ExperimentSpec) -> list[ExperimentCell]:
    datasets = _non_empty(spec.dataset_version_ids, "dataset_version_ids")
    algorithms = _non_empty(spec.algorithm_version_ids, "algorithm_version_ids")
    attacks = _non_empty(spec.attack_presets, "attack_presets")
    seeds = _non_empty(spec.seeds, "seeds")

    expanded_attacks = []
    for attack in attacks:
        strengths = attack.strengths or (0.0,)
        for strength in strengths:
            expanded_attacks.append((attack.preset_id, attack.method, float(strength)))

    cells: list[ExperimentCell] = []
    for index, (dataset_id, algorithm_id, attack_info, seed) in enumerate(
        product(datasets, algorithms, expanded_attacks, seeds),
        start=1,
    ):
        attack_preset_id, attack_method, attack_strength = attack_info
        cell_key = (
            f"{spec.spec_id}:{index:06d}:"
            f"{dataset_id}:{algorithm_id}:{attack_preset_id}:{attack_strength:g}:{seed}"
        )
        cells.append(
            ExperimentCell(
                cell_key=cell_key,
                dataset_version_id=dataset_id,
                algorithm_version_id=algorithm_id,
                attack_preset_id=attack_preset_id,
                attack_method=attack_method,
                attack_strength=attack_strength,
                seed=seed,
            )
        )
    return cells


def estimate_cell_count(spec: ExperimentSpec) -> int:
    attack_strength_count = sum(max(1, len(attack.strengths)) for attack in spec.attack_presets)
    return (
        len(spec.dataset_version_ids)
        * len(spec.algorithm_version_ids)
        * max(1, attack_strength_count)
        * len(spec.seeds)
    )
