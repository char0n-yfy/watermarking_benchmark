from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from evaluator.watermarking.base import BaseWatermark, WatermarkContext
from evaluator.watermarking.registry import register_watermark
from evaluator.watermarking.utils import (
    bit_accuracy,
    bits_from_message,
    bits_to_string,
    packaged_algorithm_dir,
    packaged_weights_dir,
    prepend_sys_path,
    require_path,
)


class _TrustMarkBase(BaseWatermark):
    default_model_type = "C"

    def __init__(
        self,
        repo_dir: str | Path | None = None,
        weights_dir: str | Path | None = None,
        model_type: str | None = None,
        wm_strength: float | None = None,
        **params: Any,
    ) -> None:
        model_type = (model_type or self.default_model_type).upper()
        strength = float(wm_strength if wm_strength is not None else (1.25 if model_type == "C" else 1.0))
        super().__init__(
            repo_dir=str(repo_dir) if repo_dir is not None else None,
            weights_dir=str(weights_dir) if weights_dir is not None else None,
            model_type=model_type,
            wm_strength=strength,
            **params,
        )
        self.repo_dir = require_path(repo_dir or packaged_algorithm_dir("trustmark"), "TrustMark repo_dir")
        self.weights_dir = require_path(weights_dir or packaged_weights_dir("trustmark"), "TrustMark weights_dir")
        self.model_type = model_type
        self.wm_strength = strength
        if self.model_type not in {"C", "Q"}:
            raise ValueError("Only TrustMark-C and TrustMark-Q are packaged")
        self._loaded = False
        self._tm = None

    def _sync_model_files(self) -> None:
        model_dir = self.repo_dir / "trustmark" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            f"trustmark_{self.model_type}.yaml",
            f"encoder_{self.model_type}.ckpt",
            f"decoder_{self.model_type}.ckpt",
        ):
            src = require_path(self.weights_dir / name, f"TrustMark weight {name}")
            dst = model_dir / name
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dst)

    def _load(self, device_name: str) -> None:
        if self._loaded:
            return
        self._sync_model_files()
        with prepend_sys_path(self.repo_dir, ["trustmark"]):
            from trustmark import TrustMark

            self._tm = TrustMark(verbose=False, model_type=self.model_type, device=device_name, loadRemover=False)
        self._loaded = True

    def _bits(self, context: WatermarkContext) -> list[int]:
        assert self._tm is not None
        capacity = int(self._tm.schemaCapacity())
        return bits_from_message(context.message, capacity, seed=context.seed)

    def embed_impl(self, input_path: Path, output_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._tm is not None

        bits = self._bits(context)
        message = bits_to_string(bits)
        image = Image.open(input_path).convert("RGB")
        encoded = self._tm.encode(image, message, MODE="binary", WM_STRENGTH=self.wm_strength)
        encoded.save(output_path)
        return {
            "message": message,
            "model_type": self.model_type,
            "wm_strength": self.wm_strength,
            "payload_bits": len(bits),
            "encoding_mode": "binary",
            "schema_capacity": len(bits),
            "weights_dir": str(self.weights_dir),
        }

    def extract_impl(self, input_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._tm is not None

        secret, present, schema = self._tm.decode(Image.open(input_path).convert("RGB"), MODE="binary")
        decoded = str(secret)
        expected_bits = self._bits(context)
        expected = bits_to_string(expected_bits)
        decoded_bits = [int(bit) for bit in decoded if bit in {"0", "1"}]
        matched = decoded == expected
        raw_present = bool(present)
        return {
            "bits": decoded,
            "present": raw_present and matched,
            "raw_present": raw_present,
            "schema": schema,
            "expected_bits": expected,
            "bit_accuracy": bit_accuracy(expected_bits, decoded_bits),
            "matched": matched,
            "model_type": self.model_type,
            "wm_strength": self.wm_strength,
            "payload_bits": len(expected_bits),
            "encoding_mode": "binary",
            "schema_capacity": len(expected_bits),
            "weights_dir": str(self.weights_dir),
        }


@register_watermark
class TrustMarkCWatermark(_TrustMarkBase):
    name = "trustmark-c"
    description = "TrustMark-C compact packaged wrapper."
    default_model_type = "C"


@register_watermark
class TrustMarkQWatermark(_TrustMarkBase):
    name = "trustmark-q"
    description = "TrustMark-Q quality packaged wrapper."
    default_model_type = "Q"
