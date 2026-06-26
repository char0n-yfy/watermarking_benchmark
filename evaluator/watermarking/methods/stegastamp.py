from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from evaluator.watermarking.base import BaseWatermark, WatermarkContext
from evaluator.watermarking.registry import register_watermark
from evaluator.watermarking.utils import (
    bit_accuracy,
    bits_from_message,
    bits_to_numpy,
    bits_to_string,
    normalize_device,
    packaged_algorithm_dir,
    packaged_weights_dir,
    prepend_sys_path,
    require_path,
)


STEGASTAMP_MODULES = [
    "model",
    "stegastamp_pkg",
]


@register_watermark
class StegaStampWatermark(BaseWatermark):
    name = "stegastamp"
    description = "StegaStamp PyTorch package wrapper using packaged weights."

    def __init__(
        self,
        package_parent: str | Path | None = None,
        weights_dir: str | Path | None = None,
        encoder_path: str | Path | None = None,
        decoder_path: str | Path | None = None,
        payload_bits: int = 100,
        **params: Any,
    ) -> None:
        super().__init__(
            package_parent=str(package_parent) if package_parent is not None else None,
            weights_dir=str(weights_dir) if weights_dir is not None else None,
            encoder_path=str(encoder_path) if encoder_path is not None else None,
            decoder_path=str(decoder_path) if decoder_path is not None else None,
            payload_bits=payload_bits,
            **params,
        )
        self.package_parent = require_path(
            package_parent or packaged_algorithm_dir("stegastamp"),
            "StegaStamp package_parent",
        )
        self.weights_dir = require_path(weights_dir or packaged_weights_dir("stegastamp"), "StegaStamp weights_dir")
        self.encoder_path = require_path(
            encoder_path or self.weights_dir / "encoder_best_loss_0.005250_step_66185.pth",
            "StegaStamp encoder_path",
        )
        self.decoder_path = require_path(
            decoder_path or self.weights_dir / "decoder_best_loss_0.005250_step_66185.pth",
            "StegaStamp decoder_path",
        )
        self.payload_bits = int(payload_bits)
        self._loaded = False
        self._loaded_device = None
        self._model = None

    def _load(self, device_name: str) -> None:
        device_name = normalize_device(device_name)
        if self._loaded and self._loaded_device == device_name:
            return

        with prepend_sys_path(self.package_parent, STEGASTAMP_MODULES):
            from stegastamp_pkg.api import load_stegastamp

            model = load_stegastamp(
                str(self.encoder_path),
                str(self.decoder_path),
                device=device_name,
            )

        self._model = model
        self._loaded = True
        self._loaded_device = device_name

    def embed_impl(
        self,
        input_path: Path,
        output_path: Path,
        context: WatermarkContext,
    ) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._model is not None

        bits = bits_from_message(context.message, self.payload_bits, seed=context.seed)
        encoded = self._model.embed(str(input_path), secret=bits_to_numpy(bits))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        encoded.save(output_path, format="PNG")
        return {
            "bits": bits_to_string(bits),
            "payload_bits": self.payload_bits,
            "encoder_path": str(self.encoder_path),
            "decoder_path": str(self.decoder_path),
            "image_size": [400, 400],
        }

    def extract_impl(
        self,
        input_path: Path,
        context: WatermarkContext,
    ) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._model is not None

        decoded = self._model.decode(str(input_path), return_bits=True)
        decoded_message = None
        if isinstance(decoded, tuple):
            decoded_message, decoded_bits = decoded
        else:
            decoded_bits = decoded
        decoded_bits = [int(bit) for bit in decoded_bits.tolist()]

        metadata: dict[str, Any] = {
            "message": decoded_message,
            "bits": bits_to_string(decoded_bits),
            "payload_bits": len(decoded_bits),
            "encoder_path": str(self.encoder_path),
            "decoder_path": str(self.decoder_path),
        }
        if context.message is not None:
            expected = bits_from_message(context.message, len(decoded_bits), seed=context.seed)
            metadata["expected_bits"] = bits_to_string(expected)
            metadata["bit_accuracy"] = bit_accuracy(expected, decoded_bits)
        return metadata
