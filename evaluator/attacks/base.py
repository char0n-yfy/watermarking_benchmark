from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class AttackContext:
    """Runtime metadata passed to attacks.

    Image content is exchanged through files. The context only carries metadata
    that an attack may need for reproducibility or optional model loading.
    """

    run_id: str
    sample_id: str
    attack_name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    workspace_dir: Path | None = None
    device: str = "cpu"
    seed: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AttackResult:
    """Result metadata for one attacked image."""

    input_path: Path
    output_path: Path
    attack_name: str
    params: Mapping[str, Any]
    elapsed_ms: float
    ok: bool = True
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> JsonDict:
        data = asdict(self)
        data["input_path"] = str(self.input_path)
        data["output_path"] = str(self.output_path)
        return data


class BaseAttack:
    """Base class for all image-to-image attacks.

    Contract:
    - input_path must point to a readable RGB-like image file.
    - output_path is where the attacked image must be written.
    - attack() returns only metadata; the image result is the output file.
    - implementations should be deterministic when context.seed is fixed.
    """

    name = "base"
    description = ""
    output_ext = ".png"

    def __init__(self, **params: Any) -> None:
        self.params: JsonDict = dict(params)

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        """Write the attacked image to output_path and return extra metadata."""
        raise NotImplementedError

    def attack(self, input_path: str | Path, output_path: str | Path, context: AttackContext) -> AttackResult:
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        try:
            metadata = self.apply(input_path, output_path, context)
            ok = True
            error = None
        except Exception as exc:
            metadata = {}
            ok = False
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - started) * 1000

        return AttackResult(
            input_path=input_path,
            output_path=output_path,
            attack_name=self.name,
            params=self.params,
            elapsed_ms=elapsed_ms,
            ok=ok,
            error=error,
            metadata=metadata,
        )

    @staticmethod
    def write_manifest(path: str | Path, results: list[AttackResult]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump([result.to_json() for result in results], f, ensure_ascii=False, indent=2)
