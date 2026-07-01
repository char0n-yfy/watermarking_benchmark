from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from evaluator.watermarking.base import BaseWatermark, WatermarkContext
from evaluator.watermarking.registry import register_watermark
from evaluator.watermarking.utils import (
    bit_accuracy,
    bits_from_message,
    bits_to_string,
    normalize_device,
    packaged_algorithm_dir,
    packaged_weights_dir,
    prepend_sys_path,
    require_path,
)


@register_watermark
class PIMoGWatermark(BaseWatermark):
    name = "pimog"
    description = "PIMoG ScreenShooting 30-bit watermark wrapper using packaged epoch-99 weights."

    def __init__(
        self,
        repo_dir: str | Path | None = None,
        weights_dir: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        payload_bits: int = 30,
        **params: Any,
    ) -> None:
        super().__init__(
            repo_dir=str(repo_dir) if repo_dir is not None else None,
            weights_dir=str(weights_dir) if weights_dir is not None else None,
            checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
            payload_bits=payload_bits,
            **params,
        )
        self.repo_dir = require_path(repo_dir or packaged_algorithm_dir("pimog"), "PIMoG repo_dir")
        self.weights_dir = require_path(weights_dir or packaged_weights_dir("pimog"), "PIMoG weights_dir")
        self.checkpoint_path = require_path(
            checkpoint_path or self.weights_dir / "Encoder_Decoder_Model_mask_99.pth",
            "PIMoG checkpoint",
        )
        self.payload_bits = int(payload_bits)
        if self.payload_bits != 30:
            raise ValueError("The packaged PIMoG checkpoint supports a 30-bit payload")
        self._loaded = False
        self._loaded_device = None
        self._torch = None
        self._tf = None
        self._model = None

    def _load(self, device_name: str) -> None:
        device_name = normalize_device(device_name)
        if self._loaded and self._loaded_device == device_name:
            return

        with prepend_sys_path(self.repo_dir, ["model", "Noise_Layer"]):
            import torch
            import torchvision.transforms as transforms
            from model import Encoder_Decoder

            device = torch.device(device_name)
            model = Encoder_Decoder("ScreenShooting")
            state = torch.load(self.checkpoint_path, map_location="cpu")
            if all(key.startswith("module.") for key in state):
                if device.type == "cuda":
                    wrapped = torch.nn.DataParallel(model).to(device)
                    wrapped.load_state_dict(state, strict=True)
                    model = wrapped.module
                else:
                    stripped = {key.removeprefix("module."): value for key, value in state.items()}
                    model.load_state_dict(stripped, strict=True)
            else:
                model.load_state_dict(state, strict=True)
            model = model.to(device).eval()

        self._torch = torch
        self._tf = transforms
        self._model = model
        self._loaded = True
        self._loaded_device = device_name

    def _transform(self):
        assert self._tf is not None
        return self._tf.Compose(
            [
                self._tf.Resize((128, 128)),
                self._tf.ToTensor(),
                self._tf.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def _message_tensor(self, context: WatermarkContext):
        assert self._torch is not None
        assert self._model is not None
        bits = bits_from_message(context.message, self.payload_bits, seed=context.seed)
        tensor = self._torch.tensor(bits, dtype=self._torch.float32, device=next(self._model.parameters()).device).unsqueeze(0)
        return bits, tensor

    def embed_batch_impl(
        self,
        jobs: list[tuple[Path, Path, WatermarkContext]],
    ) -> list[Mapping[str, Any]]:
        if not jobs:
            return []
        self._load(jobs[0][2].device)
        assert self._torch is not None
        assert self._tf is not None
        assert self._model is not None

        device = next(self._model.parameters()).device
        transform = self._transform()
        images = self._torch.cat(
            [
                transform(Image.open(input_path).convert("RGB")).unsqueeze(0)
                for input_path, _output_path, _context in jobs
            ],
            dim=0,
        ).to(device)
        bits_list = [bits_from_message(context.message, self.payload_bits, seed=context.seed) for _input_path, _output_path, context in jobs]
        messages = self._torch.tensor(bits_list, dtype=self._torch.float32, device=device)

        with self._torch.no_grad():
            encoded = self._model.Encoder(images, messages)

        to_pil = self._tf.ToPILImage()
        for index, (_input_path, output_path, _context) in enumerate(jobs):
            to_pil(((encoded[index].detach().cpu().clamp(-1, 1) + 1) / 2)).save(output_path)

        return [
            {
                "bits": bits_to_string(bits),
                "payload_bits": self.payload_bits,
                "checkpoint_file": str(self.checkpoint_path),
                "weights_dir": str(self.weights_dir),
                "image_size": [128, 128],
            }
            for bits in bits_list
        ]

    def extract_batch_impl(
        self,
        jobs: list[tuple[Path, WatermarkContext]],
    ) -> list[Mapping[str, Any]]:
        if not jobs:
            return []
        self._load(jobs[0][1].device)
        assert self._torch is not None
        assert self._model is not None

        device = next(self._model.parameters()).device
        transform = self._transform()
        images = self._torch.cat(
            [transform(Image.open(input_path).convert("RGB")).unsqueeze(0) for input_path, _context in jobs],
            dim=0,
        ).to(device)
        with self._torch.no_grad():
            decoded = self._model.Decoder(images)
        decoded_batch = decoded.detach().cpu().round().clip(0, 1).int().tolist()

        results: list[Mapping[str, Any]] = []
        for decoded_bits, (_input_path, context) in zip(decoded_batch, jobs):
            metadata: dict[str, Any] = {
                "bits": bits_to_string(decoded_bits),
                "payload_bits": len(decoded_bits),
                "checkpoint_file": str(self.checkpoint_path),
                "weights_dir": str(self.weights_dir),
            }
            if context.message is not None:
                expected = bits_from_message(context.message, len(decoded_bits), seed=context.seed)
                metadata["expected_bits"] = bits_to_string(expected)
                metadata["bit_accuracy"] = bit_accuracy(expected, decoded_bits)
            results.append(metadata)
        return results

    def embed_impl(self, input_path: Path, output_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._torch is not None
        assert self._tf is not None
        assert self._model is not None

        bits, msg = self._message_tensor(context)
        image = self._transform()(Image.open(input_path).convert("RGB")).unsqueeze(0).to(msg.device)
        with self._torch.no_grad():
            encoded = self._model.Encoder(image, msg)
        self._tf.ToPILImage()(((encoded[0].detach().cpu().clamp(-1, 1) + 1) / 2)).save(output_path)
        return {
            "bits": bits_to_string(bits),
            "payload_bits": self.payload_bits,
            "checkpoint_file": str(self.checkpoint_path),
            "weights_dir": str(self.weights_dir),
            "image_size": [128, 128],
        }

    def extract_impl(self, input_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._torch is not None
        assert self._model is not None

        image = self._transform()(Image.open(input_path).convert("RGB")).unsqueeze(0).to(next(self._model.parameters()).device)
        with self._torch.no_grad():
            decoded = self._model.Decoder(image)
        decoded_bits = decoded.detach().cpu().round().clip(0, 1).int()[0].tolist()
        metadata: dict[str, Any] = {
            "bits": bits_to_string(decoded_bits),
            "payload_bits": len(decoded_bits),
            "checkpoint_file": str(self.checkpoint_path),
            "weights_dir": str(self.weights_dir),
        }
        if context.message is not None:
            expected = bits_from_message(context.message, len(decoded_bits), seed=context.seed)
            metadata["expected_bits"] = bits_to_string(expected)
            metadata["bit_accuracy"] = bit_accuracy(expected, decoded_bits)
        return metadata
