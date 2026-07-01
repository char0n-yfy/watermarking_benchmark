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
class MBRSWatermark(BaseWatermark):
    name = "mbrs"
    description = "MBRS 256-bit JPEG-robust watermark wrapper using packaged EC_42 weights."

    def __init__(
        self,
        repo_dir: str | Path | None = None,
        weights_dir: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        payload_bits: int = 256,
        **params: Any,
    ) -> None:
        super().__init__(
            repo_dir=str(repo_dir) if repo_dir is not None else None,
            weights_dir=str(weights_dir) if weights_dir is not None else None,
            checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
            payload_bits=payload_bits,
            **params,
        )
        self.repo_dir = require_path(repo_dir or packaged_algorithm_dir("mbrs"), "MBRS repo_dir")
        self.weights_dir = require_path(weights_dir or packaged_weights_dir("mbrs"), "MBRS weights_dir")
        self.checkpoint_path = require_path(checkpoint_path or self.weights_dir / "EC_42.pth", "MBRS checkpoint")
        self.payload_bits = int(payload_bits)
        if self.payload_bits != 256:
            raise ValueError("The packaged MBRS checkpoint supports a 256-bit payload")
        self._loaded = False
        self._loaded_device = None
        self._torch = None
        self._tf = None
        self._model = None

    def _load(self, device_name: str) -> None:
        device_name = normalize_device(device_name)
        if self._loaded and self._loaded_device == device_name:
            return

        with prepend_sys_path(self.repo_dir, ["network", "utils"]):
            import torch
            import torchvision.transforms as transforms
            from network.Encoder_MP_Decoder import EncoderDecoder

            model = EncoderDecoder(256, 256, self.payload_bits, ["Identity()"])
            model.load_state_dict(torch.load(self.checkpoint_path, map_location="cpu"))
            model = model.to(torch.device(device_name)).eval()

        self._torch = torch
        self._tf = transforms
        self._model = model
        self._loaded = True
        self._loaded_device = device_name

    def _transform(self):
        assert self._tf is not None
        return self._tf.Compose(
            [
                self._tf.Resize((256, 256)),
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
            encoded = self._model.encoder(images, messages)

        to_pil = self._tf.ToPILImage()
        for index, (_input_path, output_path, _context) in enumerate(jobs):
            to_pil(((encoded[index].detach().cpu().clamp(-1, 1) + 1) / 2)).save(output_path)

        return [
            {
                "bits": bits_to_string(bits),
                "payload_bits": self.payload_bits,
                "checkpoint_file": str(self.checkpoint_path),
                "weights_dir": str(self.weights_dir),
                "image_size": [256, 256],
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
            decoded = self._model.decoder(images)
        decoded_batch = decoded.detach().cpu().gt(0.5).int().tolist()

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
            encoded = self._model.encoder(image, msg)
        self._tf.ToPILImage()(((encoded[0].detach().cpu().clamp(-1, 1) + 1) / 2)).save(output_path)
        return {
            "bits": bits_to_string(bits),
            "payload_bits": self.payload_bits,
            "checkpoint_file": str(self.checkpoint_path),
            "weights_dir": str(self.weights_dir),
            "image_size": [256, 256],
        }

    def extract_impl(self, input_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._torch is not None
        assert self._model is not None

        image = self._transform()(Image.open(input_path).convert("RGB")).unsqueeze(0).to(next(self._model.parameters()).device)
        with self._torch.no_grad():
            decoded = self._model.decoder(image)
        decoded_bits = decoded.detach().cpu().gt(0.5).int()[0].tolist()
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
