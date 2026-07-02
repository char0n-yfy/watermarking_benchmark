from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluator.attacks.runner import AttackJob, run_attack_dir


def run_attack_dry_run(
    run_id: str,
    attack_name: str,
    params: dict[str, Any],
    input_dir: str,
    output_dir: str,
    device: str = "cpu",
    seed: int | None = 42,
) -> list[dict[str, Any]]:
    job = AttackJob(
        run_id=run_id,
        attack_name=attack_name,
        params=params,
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        device=device,
        seed=seed,
    )
    return [result.to_json() for result in run_attack_dir(job)]
