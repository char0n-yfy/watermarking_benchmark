from __future__ import annotations

import sys
import types
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


def _install_noise_stub(torch_module) -> None:
    from torch import nn

    noise_stub = types.ModuleType("models.Noise")

    class Noise(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = layers

        def forward(self, image, mask):
            if mask is None:
                mask = torch_module.ones(
                    (image.shape[0], 1, image.shape[2], image.shape[3]),
                    device=image.device,
                    dtype=image.dtype,
                )
            return image, mask

    noise_stub.Noise = Noise
    sys.modules["models.Noise"] = noise_stub


@register_watermark
class MaskWMD32Watermark(BaseWatermark):
    name = "maskwm-d32"
    description = "MaskWM-D_32bits packaged wrapper using the single D_32bits checkpoint."

    def __init__(
        self,
        repo_dir: str | Path | None = None,
        weights_dir: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        payload_bits: int = 32,
        **params: Any,
    ) -> None:
        super().__init__(
            repo_dir=str(repo_dir) if repo_dir is not None else None,
            weights_dir=str(weights_dir) if weights_dir is not None else None,
            checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
            payload_bits=payload_bits,
            **params,
        )
        self.repo_dir = require_path(repo_dir or packaged_algorithm_dir("maskwm"), "MaskWM repo_dir")
        self.weights_dir = require_path(weights_dir or packaged_weights_dir("maskwm"), "MaskWM weights_dir")
        self.checkpoint_path = require_path(checkpoint_path or self.weights_dir / "D_32bits.pth", "MaskWM-D_32bits checkpoint")
        self.payload_bits = int(payload_bits)
        if self.payload_bits != 32:
            raise ValueError("Packaged MaskWM-D_32bits expects payload_bits=32")
        self._loaded = False
        self._loaded_device = None
        self._torch = None
        self._F = None
        self._tf = None
        self._TF = None
        self._model = None

    def _load(self, device_name: str) -> None:
        device_name = normalize_device(device_name)
        if self._loaded and self._loaded_device == device_name:
            return

        with prepend_sys_path(self.repo_dir, ["models"]):
            import torch
            import torch.nn.functional as F
            import torchvision.transforms.functional as TF
            from omegaconf import OmegaConf
            from torchvision import transforms

            _install_noise_stub(torch)
            from models.Mask_Model import WatermarkModel

            config = OmegaConf.load(self.repo_dir / "configs" / "model" / "D_32bits.yaml")
            model = WatermarkModel(**config)
            model.load_state_dict(torch.load(self.checkpoint_path, map_location="cpu"), strict=True)
            model = model.to(torch.device(device_name)).eval()

        self._torch = torch
        self._F = F
        self._tf = transforms
        self._TF = TF
        self._model = model
        self._loaded = True
        self._loaded_device = device_name

    def _transform(self):
        assert self._tf is not None
        return self._tf.Compose(
            [
                self._tf.Resize(512),
                self._tf.CenterCrop(512),
                self._tf.ToTensor(),
                self._tf.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def _image_tensor(self, input_path: Path):
        assert self._model is not None
        return self._transform()(Image.open(input_path).convert("RGB")).unsqueeze(0).to(next(self._model.parameters()).device)

    def embed_batch_impl(
        self,
        jobs: list[tuple[Path, Path, WatermarkContext]],
    ) -> list[Mapping[str, Any]]:
        if not jobs:
            return []
        self._load(jobs[0][2].device)
        assert self._torch is not None
        assert self._F is not None
        assert self._TF is not None
        assert self._model is not None

        image = self._torch.cat([self._image_tensor(input_path) for input_path, _output_path, _context in jobs], dim=0)
        image_256 = self._F.interpolate(image, size=[256, 256], mode="bilinear")
        bits_list = [bits_from_message(context.message, self.payload_bits, seed=context.seed) for _input_path, _output_path, context in jobs]
        message = self._torch.tensor(bits_list, dtype=self._torch.float32, device=image.device)
        with self._torch.no_grad():
            wm_image_256 = self._model.encoder(image_256, message, use_jnd=True, jnd_factor=1.3, blue=True)
            wm_image = (self._F.interpolate((wm_image_256 - image_256), size=[512, 512], mode="bilinear") + image).clamp_(-1, 1)

        for index, (_input_path, output_path, _context) in enumerate(jobs):
            self._TF.to_pil_image((wm_image[index].detach().cpu() + 1) / 2).save(output_path)

        return [
            {
                "bits": bits_to_string(bits),
                "payload_bits": self.payload_bits,
                "checkpoint_file": str(self.checkpoint_path),
                "image_size": [512, 512],
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
        assert self._F is not None
        assert self._model is not None

        image = self._torch.cat([self._image_tensor(input_path) for input_path, _context in jobs], dim=0)
        image_256 = self._F.interpolate(image, size=[256, 256], mode="bilinear")
        with self._torch.no_grad():
            decoded, _ = self._model.decoder(image_256)
        decoded_batch = decoded.gt(0.5).int().detach().cpu().tolist()

        results: list[Mapping[str, Any]] = []
        for decoded_bits, (_input_path, context) in zip(decoded_batch, jobs):
            metadata: dict[str, Any] = {
                "bits": bits_to_string(decoded_bits),
                "payload_bits": len(decoded_bits),
                "checkpoint_file": str(self.checkpoint_path),
                "image_size": [512, 512],
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
        assert self._F is not None
        assert self._TF is not None
        assert self._model is not None

        image = self._image_tensor(input_path)
        image_256 = self._F.interpolate(image, size=[256, 256], mode="bilinear")
        bits = bits_from_message(context.message, self.payload_bits, seed=context.seed)
        message = self._torch.tensor(bits, dtype=self._torch.float32, device=image.device).unsqueeze(0)
        with self._torch.no_grad():
            wm_image_256 = self._model.encoder(image_256, message, use_jnd=True, jnd_factor=1.3, blue=True)
            wm_image = (self._F.interpolate((wm_image_256 - image_256), size=[512, 512], mode="bilinear") + image).clamp_(-1, 1)
        self._TF.to_pil_image((wm_image[0].detach().cpu() + 1) / 2).save(output_path)
        return {
            "bits": bits_to_string(bits),
            "payload_bits": self.payload_bits,
            "checkpoint_file": str(self.checkpoint_path),
            "image_size": [512, 512],
        }

    def extract_impl(self, input_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._torch is not None
        assert self._F is not None
        assert self._model is not None

        image = self._image_tensor(input_path)
        image_256 = self._F.interpolate(image, size=[256, 256], mode="bilinear")
        with self._torch.no_grad():
            decoded, _ = self._model.decoder(image_256)
        decoded_bits = decoded.gt(0.5).int().detach().cpu().tolist()[0]
        metadata: dict[str, Any] = {
            "bits": bits_to_string(decoded_bits),
            "payload_bits": len(decoded_bits),
            "checkpoint_file": str(self.checkpoint_path),
            "image_size": [512, 512],
        }
        if context.message is not None:
            expected = bits_from_message(context.message, len(decoded_bits), seed=context.seed)
            metadata["expected_bits"] = bits_to_string(expected)
            metadata["bit_accuracy"] = bit_accuracy(expected, decoded_bits)
        return metadata
