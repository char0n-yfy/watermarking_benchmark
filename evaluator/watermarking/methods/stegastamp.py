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

    def embed_batch_impl(
        self,
        jobs: list[tuple[Path, Path, WatermarkContext]],
    ) -> list[Mapping[str, Any]]:
        import numpy as np
        import torch
        from PIL import Image
        from torchvision.transforms import functional as TF

        if not jobs:
            return []

        self._load(jobs[0][2].device)
        assert self._model is not None
        device = self._model.device

        image_tensors = []
        bit_lists: list[list[int]] = []
        for input_path, _output_path, context in jobs:
            with Image.open(input_path) as image:
                prepared = image.convert("RGB").resize((400, 400), resample=Image.Resampling.BICUBIC)
            image_tensors.append(TF.to_tensor(prepared))
            bit_lists.append(bits_from_message(context.message, self.payload_bits, seed=context.seed))

        img_batch = torch.stack(image_tensors, dim=0).to(device)
        secret_batch = torch.tensor(np.asarray(bit_lists, dtype=np.float32), dtype=torch.float32, device=device)
        with torch.no_grad():
            residual = self._model.encoder((secret_batch, img_batch))
            encoded_batch = torch.clamp(img_batch + residual, 0, 1).detach().cpu().numpy()

        metadatas: list[Mapping[str, Any]] = []
        for (_input_path, output_path, _context), bits, encoded in zip(jobs, bit_lists, encoded_batch):
            encoded_pixels = (encoded * 255).astype(np.uint8).transpose((1, 2, 0))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(encoded_pixels).save(output_path, format="PNG")
            metadatas.append(
                {
                    "bits": bits_to_string(bits),
                    "payload_bits": self.payload_bits,
                    "encoder_path": str(self.encoder_path),
                    "decoder_path": str(self.decoder_path),
                    "image_size": [400, 400],
                }
            )
        return metadatas

    def extract_batch_impl(
        self,
        jobs: list[tuple[Path, WatermarkContext]],
    ) -> list[Mapping[str, Any]]:
        import sys

        import numpy as np
        import torch
        from PIL import Image
        from torchvision.transforms import functional as TF

        if not jobs:
            return []

        self._load(jobs[0][1].device)
        assert self._model is not None
        device = self._model.device

        image_tensors = []
        for input_path, _context in jobs:
            with Image.open(input_path) as image:
                prepared = image.convert("RGB").resize((400, 400), resample=Image.Resampling.BICUBIC)
            image_tensors.append(TF.to_tensor(prepared))

        img_batch = torch.stack(image_tensors, dim=0).to(device)
        with torch.no_grad():
            decoded_batch = self._model.decoder(img_batch).detach().cpu().numpy()
        decoded_bits_batch = np.round(decoded_batch).astype(np.uint8)

        api_module = sys.modules.get(self._model.__class__.__module__)
        string_from_bits = getattr(api_module, "_string_from_secret_bits", None)

        metadatas: list[Mapping[str, Any]] = []
        for bits_array, (_input_path, context) in zip(decoded_bits_batch, jobs):
            decoded_message = None
            if callable(string_from_bits):
                decoded_message = string_from_bits(bits_array)
            decoded_bits = [int(bit) for bit in bits_array.tolist()]
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
            metadatas.append(metadata)
        return metadatas
