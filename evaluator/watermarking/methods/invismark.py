from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from evaluator.watermarking.base import BaseWatermark, WatermarkContext
from evaluator.watermarking.registry import register_watermark
from evaluator.watermarking.utils import (
    bit_accuracy,
    bits_to_string,
    normalize_device,
    packaged_algorithm_dir,
    packaged_weights_dir,
    prepend_sys_path,
    require_path,
)


@register_watermark
class InvisMarkWatermark(BaseWatermark):
    name = "invismark"
    description = "InvisMark 100-bit AI provenance wrapper using packaged paper.ckpt encoder/decoder weights."

    def __init__(
        self,
        repo_dir: str | Path | None = None,
        weights_dir: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        payload_bits: int = 100,
        **params: Any,
    ) -> None:
        super().__init__(
            repo_dir=str(repo_dir) if repo_dir is not None else None,
            weights_dir=str(weights_dir) if weights_dir is not None else None,
            checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
            payload_bits=payload_bits,
            **params,
        )
        self.repo_dir = require_path(repo_dir or packaged_algorithm_dir("invismark"), "InvisMark repo_dir")
        self.weights_dir = require_path(weights_dir or packaged_weights_dir("invismark"), "InvisMark weights_dir")
        self.checkpoint_path = require_path(checkpoint_path or self.weights_dir / "paper.ckpt", "InvisMark checkpoint")
        self.payload_bits = int(payload_bits)
        if self.payload_bits != 100:
            raise ValueError("The packaged InvisMark paper.ckpt expects payload_bits=100")
        self._loaded = False
        self._loaded_device: str | None = None
        self._torch = None
        self._tf = None
        self._config = None
        self._encoder = None
        self._decoder = None

    def _load(self, device_name: str) -> None:
        device_name = normalize_device(device_name)
        if self._loaded and self._loaded_device == device_name:
            return

        with prepend_sys_path(self.repo_dir, ["model", "configs", "utils"]):
            import torch
            from torchvision import transforms

            import model as inv_model

            original_convnext_base = inv_model.torchvision.models.convnext_base

            def convnext_base_without_download(*args, **kwargs):
                kwargs.pop("pretrained", None)
                kwargs["weights"] = None
                return original_convnext_base(*args, **kwargs)

            state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            config = state["config"]
            inv_model.torchvision.models.convnext_base = convnext_base_without_download
            try:
                encoder = inv_model.Encoder(config)
                decoder = inv_model.Extractor(config)
            finally:
                inv_model.torchvision.models.convnext_base = original_convnext_base

            encoder.load_state_dict(state["encoder_state_dict"], strict=True)
            decoder.load_state_dict(state["decoder_state_dict"], strict=True)
            device = torch.device(device_name)
            encoder = encoder.to(device).eval()
            decoder = decoder.to(device).eval()

        self._torch = torch
        self._tf = transforms
        self._config = config
        self._encoder = encoder
        self._decoder = decoder
        self._loaded = True
        self._loaded_device = device_name

    def _payload(self, context: WatermarkContext) -> tuple[list[int], str]:
        if context.message and set(context.message) <= {"0", "1"} and len(context.message) == self.payload_bits:
            return [int(bit) for bit in context.message], "binary"

        material = "|".join(
            [
                context.message or "",
                context.run_id,
                context.sample_id,
                "" if context.seed is None else str(context.seed),
            ]
        )
        uid = uuid.uuid5(uuid.NAMESPACE_URL, material)
        bits: list[int] = []
        for byte in uid.bytes:
            bits.extend((byte >> shift) & 1 for shift in range(7, -1, -1))
        return bits[: self.payload_bits], "uuid5"

    def _image_tensor(self, input_path: Path):
        assert self._torch is not None
        assert self._tf is not None
        assert self._encoder is not None
        tensor = self._tf.ToTensor()(Image.open(input_path).convert("RGB")).unsqueeze(0)
        return tensor.to(next(self._encoder.parameters()).device) * 2.0 - 1.0

    def _resize(self, tensor, size):
        assert self._tf is not None
        return self._tf.Resize(size)(tensor)

    def embed_impl(self, input_path: Path, output_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._torch is not None
        assert self._tf is not None
        assert self._config is not None
        assert self._encoder is not None

        image = self._image_tensor(input_path)
        bits, payload_mode = self._payload(context)
        secret = self._torch.tensor(bits, dtype=self._torch.float32, device=image.device).unsqueeze(0)
        with self._torch.inference_mode():
            small = self._resize(image, tuple(self._config.image_shape))
            encoded_small = self._encoder(small, secret)
            diff = self._resize(encoded_small - small, tuple(image.shape[-2:]))
            watermarked = self._torch.clamp(image + diff, min=-1.0, max=1.0)

        self._tf.ToPILImage()(((watermarked[0].detach().cpu() + 1.0) / 2.0).clamp(0, 1)).save(output_path)
        return {
            "bits": bits_to_string(bits),
            "payload_bits": self.payload_bits,
            "payload_mode": payload_mode,
            "checkpoint_file": str(self.checkpoint_path),
            "weights_dir": str(self.weights_dir),
            "image_size": list(self._config.image_shape),
        }

    def extract_impl(self, input_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._torch is not None
        assert self._config is not None
        assert self._decoder is not None

        image = self._image_tensor(input_path)
        with self._torch.inference_mode():
            pred = self._decoder(self._resize(image, tuple(self._config.image_shape)))
        decoded_bits = (pred[0] >= 0.5).int().detach().cpu().tolist()
        metadata: dict[str, Any] = {
            "bits": bits_to_string(decoded_bits),
            "payload_bits": len(decoded_bits),
            "checkpoint_file": str(self.checkpoint_path),
            "weights_dir": str(self.weights_dir),
            "image_size": list(self._config.image_shape),
        }
        if context.message is not None or context.seed is not None:
            expected, payload_mode = self._payload(context)
            metadata["expected_bits"] = bits_to_string(expected)
            metadata["payload_mode"] = payload_mode
            metadata["bit_accuracy"] = bit_accuracy(expected, decoded_bits)
        return metadata
