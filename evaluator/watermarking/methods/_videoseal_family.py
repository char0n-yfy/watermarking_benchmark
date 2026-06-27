from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from evaluator.watermarking.base import BaseWatermark, WatermarkContext
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


class VideoSealFamilyWatermark(BaseWatermark):
    algorithm_dir_name = ""
    checkpoint_filename = ""
    default_payload_bits = 256
    display_name = "VideoSeal-family"

    def __init__(
        self,
        repo_dir: str | Path | None = None,
        weights_dir: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        payload_bits: int | None = None,
        **params: Any,
    ) -> None:
        payload_bits = int(payload_bits or self.default_payload_bits)
        super().__init__(
            repo_dir=str(repo_dir) if repo_dir is not None else None,
            weights_dir=str(weights_dir) if weights_dir is not None else None,
            checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
            payload_bits=payload_bits,
            **params,
        )
        if not self.algorithm_dir_name or not self.checkpoint_filename:
            raise ValueError("VideoSeal-family wrappers must define algorithm_dir_name and checkpoint_filename")
        self.repo_dir = require_path(repo_dir or packaged_algorithm_dir(self.algorithm_dir_name), f"{self.display_name} repo_dir")
        self.weights_dir = require_path(
            weights_dir or packaged_weights_dir(self.algorithm_dir_name),
            f"{self.display_name} weights_dir",
        )
        self.checkpoint_path = require_path(
            checkpoint_path or self.weights_dir / self.checkpoint_filename,
            f"{self.display_name} checkpoint",
        )
        self.payload_bits = payload_bits
        self._loaded = False
        self._loaded_device: str | None = None
        self._torch = None
        self._tf = None
        self._model = None

    def _load(self, device_name: str) -> None:
        device_name = normalize_device(device_name)
        if self._loaded and self._loaded_device == device_name:
            return

        with prepend_sys_path(self.repo_dir, ["videoseal"]):
            import torch
            import torchvision.transforms as transforms
            from videoseal.utils.cfg import setup_model_from_checkpoint

            model = setup_model_from_checkpoint(str(self.checkpoint_path)).to(torch.device(device_name)).eval()

        self._torch = torch
        self._tf = transforms
        self._model = model
        self._loaded = True
        self._loaded_device = device_name

    def _message_tensor(self, context: WatermarkContext):
        assert self._torch is not None
        assert self._model is not None
        bits = bits_from_message(context.message, self.payload_bits, seed=context.seed)
        tensor = self._torch.tensor(
            bits,
            dtype=self._torch.float32,
            device=next(self._model.parameters()).device,
        ).unsqueeze(0)
        return bits, tensor

    def embed_impl(self, input_path: Path, output_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._torch is not None
        assert self._tf is not None
        assert self._model is not None

        bits, msg = self._message_tensor(context)
        tensor = self._tf.ToTensor()(Image.open(input_path).convert("RGB")).unsqueeze(0).to(msg.device)
        kwargs: dict[str, Any] = {}
        signature = inspect.signature(self._model.embed)
        if "msgs" in signature.parameters:
            kwargs["msgs"] = msg
        elif "messages" in signature.parameters:
            kwargs["messages"] = msg
        elif "message" in signature.parameters:
            kwargs["message"] = msg
        if "is_video" in signature.parameters:
            kwargs["is_video"] = False

        with self._torch.no_grad():
            outputs = self._model.embed(tensor, **kwargs)
        watermarked = outputs["imgs_w"][0].detach().cpu().clamp(0, 1)
        self._tf.ToPILImage()(watermarked).save(output_path)
        return {
            "bits": bits_to_string(bits),
            "payload_bits": self.payload_bits,
            "checkpoint_file": str(self.checkpoint_path),
            "weights_dir": str(self.weights_dir),
        }

    def extract_impl(self, input_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._torch is not None
        assert self._tf is not None
        assert self._model is not None

        device = next(self._model.parameters()).device
        tensor = self._tf.ToTensor()(Image.open(input_path).convert("RGB")).unsqueeze(0).to(device)
        kwargs: dict[str, Any] = {}
        if "is_video" in inspect.signature(self._model.detect).parameters:
            kwargs["is_video"] = False

        with self._torch.no_grad():
            detected = self._model.detect(tensor, **kwargs)
        decoded_bits = (detected["preds"][0, 1:] > 0).int().detach().cpu().tolist()
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
