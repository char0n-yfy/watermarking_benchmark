from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from evaluator.image_io import save_png_image
from evaluator.watermarking.base import BaseWatermark, WatermarkContext
from evaluator.watermarking.utils import (
    bit_accuracy,
    bits_from_message,
    bits_to_string,
    move_tensor_to_device,
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
    default_lowres_attenuation = False
    display_name = "VideoSeal-family"

    def __init__(
        self,
        repo_dir: str | Path | None = None,
        weights_dir: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        payload_bits: int | None = None,
        lowres_attenuation: bool | None = None,
        **params: Any,
    ) -> None:
        payload_bits = int(payload_bits or self.default_payload_bits)
        self.lowres_attenuation = (
            self.default_lowres_attenuation
            if lowres_attenuation is None
            else bool(lowres_attenuation)
        )
        super().__init__(
            repo_dir=str(repo_dir) if repo_dir is not None else None,
            weights_dir=str(weights_dir) if weights_dir is not None else None,
            checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
            payload_bits=payload_bits,
            lowres_attenuation=self.lowres_attenuation,
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

    def release(self) -> None:
        from evaluator.runtime_cleanup import move_to_cpu, torch_cleanup

        if self._model is not None:
            move_to_cpu(self._model)
        self._model = None
        self._loaded = False
        self._loaded_device = None
        torch_cleanup()

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

    def _embed_kwargs(self, messages):
        assert self._model is not None
        kwargs: dict[str, Any] = {}
        signature = inspect.signature(self._model.embed)
        if "msgs" in signature.parameters:
            kwargs["msgs"] = messages
        elif "messages" in signature.parameters:
            kwargs["messages"] = messages
        elif "message" in signature.parameters:
            kwargs["message"] = messages
        if "is_video" in signature.parameters:
            kwargs["is_video"] = False
        if "lowres_attenuation" in signature.parameters:
            kwargs["lowres_attenuation"] = self.lowres_attenuation
        return kwargs

    def _detect_kwargs(self) -> dict[str, Any]:
        assert self._model is not None
        kwargs: dict[str, Any] = {}
        if "is_video" in inspect.signature(self._model.detect).parameters:
            kwargs["is_video"] = False
        return kwargs

    def _internal_size(self) -> list[int]:
        assert self._model is not None
        size = int(getattr(self._model, "img_size", 256))
        return [size, size]

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

        loaded: list[tuple[int, Path, list[int], Any, Any]] = []
        for index, (input_path, output_path, context) in enumerate(jobs):
            bits = bits_from_message(context.message, self.payload_bits, seed=context.seed)
            message = self._torch.tensor(bits, dtype=self._torch.float32)
            tensor = self._tf.ToTensor()(Image.open(input_path).convert("RGB"))
            loaded.append((index, output_path, bits, tensor, message))

        results: list[Mapping[str, Any] | None] = [None] * len(jobs)
        grouped: dict[tuple[int, int], list[tuple[int, Path, list[int], Any, Any]]] = {}
        for item in loaded:
            grouped.setdefault(tuple(item[3].shape[-2:]), []).append(item)

        to_pil = self._tf.ToPILImage()
        internal_size = self._internal_size()
        device = next(self._model.parameters()).device
        with self._torch.no_grad():
            for items in grouped.values():
                tensors = move_tensor_to_device(self._torch.stack([item[3] for item in items], dim=0), device)
                messages = move_tensor_to_device(self._torch.stack([item[4] for item in items], dim=0), device)
                outputs = self._model.embed(tensors, **self._embed_kwargs(messages))
                watermarked = outputs["imgs_w"].detach().cpu().clamp(0, 1)
                for batch_index, (result_index, output_path, bits, _tensor, _message) in enumerate(items):
                    save_png_image(to_pil(watermarked[batch_index]), output_path)
                    results[result_index] = {
                        "bits": bits_to_string(bits),
                        "payload_bits": self.payload_bits,
                        "checkpoint_file": str(self.checkpoint_path),
                        "weights_dir": str(self.weights_dir),
                        "lowres_attenuation": self.lowres_attenuation,
                        "internalSize": internal_size,
                    }
        return [dict(result or {}) for result in results]

    def extract_batch_impl(
        self,
        jobs: list[tuple[Path, WatermarkContext]],
    ) -> list[Mapping[str, Any]]:
        if not jobs:
            return []
        self._load(jobs[0][1].device)
        assert self._torch is not None
        assert self._tf is not None
        assert self._model is not None

        loaded: list[tuple[int, WatermarkContext, Any]] = []
        for index, (input_path, context) in enumerate(jobs):
            tensor = self._tf.ToTensor()(Image.open(input_path).convert("RGB"))
            loaded.append((index, context, tensor))

        results: list[Mapping[str, Any] | None] = [None] * len(jobs)
        grouped: dict[tuple[int, int], list[tuple[int, WatermarkContext, Any]]] = {}
        for item in loaded:
            grouped.setdefault(tuple(item[2].shape[-2:]), []).append(item)

        with self._torch.no_grad():
            for items in grouped.values():
                device = next(self._model.parameters()).device
                tensors = move_tensor_to_device(self._torch.stack([item[2] for item in items], dim=0), device)
                detected = self._model.detect(tensors, **self._detect_kwargs())
                decoded_batch = (detected["preds"][:, 1:] > 0).int().detach().cpu().tolist()
                internal_size = self._internal_size()
                for (result_index, context, _tensor), decoded_bits in zip(items, decoded_batch):
                    metadata: dict[str, Any] = {
                        "bits": bits_to_string(decoded_bits),
                        "payload_bits": len(decoded_bits),
                        "checkpoint_file": str(self.checkpoint_path),
                        "weights_dir": str(self.weights_dir),
                        "decodeInternalSize": internal_size,
                    }
                    if context.message is not None:
                        expected = bits_from_message(context.message, len(decoded_bits), seed=context.seed)
                        metadata["expected_bits"] = bits_to_string(expected)
                        metadata["bit_accuracy"] = bit_accuracy(expected, decoded_bits)
                    results[result_index] = metadata
        return [dict(result or {}) for result in results]

    def embed_impl(self, input_path: Path, output_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._torch is not None
        assert self._tf is not None
        assert self._model is not None

        bits, msg = self._message_tensor(context)
        tensor = self._tf.ToTensor()(Image.open(input_path).convert("RGB")).unsqueeze(0).to(msg.device)
        with self._torch.no_grad():
            outputs = self._model.embed(tensor, **self._embed_kwargs(msg))
        watermarked = outputs["imgs_w"][0].detach().cpu().clamp(0, 1)
        save_png_image(self._tf.ToPILImage()(watermarked), output_path)
        return {
            "bits": bits_to_string(bits),
            "payload_bits": self.payload_bits,
            "checkpoint_file": str(self.checkpoint_path),
            "weights_dir": str(self.weights_dir),
            "lowres_attenuation": self.lowres_attenuation,
            "internalSize": self._internal_size(),
        }

    def extract_impl(self, input_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._torch is not None
        assert self._tf is not None
        assert self._model is not None

        device = next(self._model.parameters()).device
        tensor = self._tf.ToTensor()(Image.open(input_path).convert("RGB")).unsqueeze(0).to(device)
        with self._torch.no_grad():
            detected = self._model.detect(tensor, **self._detect_kwargs())
        decoded_bits = (detected["preds"][0, 1:] > 0).int().detach().cpu().tolist()
        metadata: dict[str, Any] = {
            "bits": bits_to_string(decoded_bits),
            "payload_bits": len(decoded_bits),
            "checkpoint_file": str(self.checkpoint_path),
            "weights_dir": str(self.weights_dir),
            "decodeInternalSize": self._internal_size(),
        }
        if context.message is not None:
            expected = bits_from_message(context.message, len(decoded_bits), seed=context.seed)
            metadata["expected_bits"] = bits_to_string(expected)
            metadata["bit_accuracy"] = bit_accuracy(expected, decoded_bits)
        return metadata
