from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from evaluator.image_io import save_png_image
from evaluator.image_protocol import image_size, semantic_size_change_attack


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class AttackBatchCapability:
    """Declared batch execution capability for an attack implementation."""

    supported: bool
    stage: str = "attack"
    source: str = "declared"
    reason: str | None = None

    def to_json(self) -> JsonDict:
        return {
            "supported": self.supported,
            "stage": self.stage,
            "source": self.source,
            "reason": self.reason,
        }


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
    thread_safe_parallel = False
    batch_stage = "attack"
    batch_capability: AttackBatchCapability | Mapping[str, Any] | str | bool | None = "auto"

    def __init__(self, **params: Any) -> None:
        self.params: JsonDict = dict(params)

    def release(self) -> None:
        """Release heavyweight runtime state held by this attack instance."""
        return None

    def apply(self, input_path: Path, output_path: Path, context: AttackContext) -> Mapping[str, Any]:
        """Write the attacked image to output_path and return extra metadata."""
        raise NotImplementedError

    def apply_batch_impl(
        self,
        jobs: list[tuple[Path, Path, AttackContext]],
    ) -> list[Mapping[str, Any]]:
        raise NotImplementedError

    def supports_batch(self) -> bool:
        return self.batch_capability_info().supported

    def batch_capability_info(self) -> AttackBatchCapability:
        has_batch_impl = type(self).apply_batch_impl is not BaseAttack.apply_batch_impl
        raw = self.batch_capability
        stage = str(getattr(self, "batch_stage", "attack") or "attack")
        if isinstance(raw, AttackBatchCapability):
            return raw
        if isinstance(raw, Mapping):
            supported = raw.get("supported")
            if supported is None:
                supported = has_batch_impl
            return AttackBatchCapability(
                supported=bool(supported),
                stage=str(raw.get("stage") or stage),
                source=str(raw.get("source") or "declared"),
                reason=str(raw.get("reason")) if raw.get("reason") is not None else None,
            )
        if raw in (False, "none", "disabled"):
            return AttackBatchCapability(
                supported=False,
                stage=stage,
                source="declared",
                reason="batch disabled by attack capability declaration",
            )
        if raw in (True, "native"):
            return AttackBatchCapability(supported=has_batch_impl, stage=stage, source="declared")
        return AttackBatchCapability(supported=has_batch_impl, stage=stage, source="auto")

    def _protocol_metadata(
        self,
        input_path: Path,
        output_path: Path,
        metadata: Mapping[str, Any],
    ) -> JsonDict:
        enriched = dict(metadata)
        input_size = image_size(input_path)
        output_size = image_size(output_path)
        if input_size is None and output_size is None:
            return enriched

        semantic_size_change = semantic_size_change_attack(
            self.name,
            self.params,
            enriched,
            input_size,
            output_size,
        )
        pre_protocol_output_size = output_size
        protocol_resized_output = False
        if input_size is not None and output_size is not None and input_size != output_size and not semantic_size_change:
            with Image.open(output_path) as opened:
                restored = opened.convert("RGB").resize(tuple(input_size), Image.Resampling.BICUBIC)
            save_png_image(restored, output_path)
            output_size = image_size(output_path)
            protocol_resized_output = True

        size_preserving = input_size == output_size if input_size is not None and output_size is not None else None
        if size_preserving is True:
            size_policy = "resized_back_to_input" if protocol_resized_output else "preserve_input_size"
        elif semantic_size_change:
            size_policy = "semantic_size_change"
        else:
            size_policy = "changed_size"

        enriched.update(
            {
                "inputSize": input_size,
                "preProtocolOutputSize": pre_protocol_output_size,
                "outputSize": output_size,
                "protocolResizedOutput": protocol_resized_output,
                "sizePreserving": size_preserving,
                "sizeChangeSemantic": semantic_size_change,
                "sizePolicy": size_policy,
            }
        )
        return enriched

    def attack(self, input_path: str | Path, output_path: str | Path, context: AttackContext) -> AttackResult:
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        try:
            metadata = self._protocol_metadata(input_path, output_path, self.apply(input_path, output_path, context))
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
