from __future__ import annotations

import tempfile
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
class CINWatermark(BaseWatermark):
    name = "cin"
    description = "CIN combined-noise 30-bit watermark wrapper using packaged pretrained weights."

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
        self.repo_dir = require_path(repo_dir or packaged_algorithm_dir("cin"), "CIN repo_dir")
        self.codes_dir = require_path(self.repo_dir / "codes", "CIN codes_dir")
        self.weights_dir = require_path(weights_dir or packaged_weights_dir("cin"), "CIN weights_dir")
        self.checkpoint_path = require_path(
            checkpoint_path or self.weights_dir / "cinNet_nsmNet.pth",
            "CIN checkpoint",
        )
        self.payload_bits = int(payload_bits)
        if self.payload_bits != 30:
            raise ValueError("The packaged CIN checkpoint supports a 30-bit payload")
        self._loaded = False
        self._loaded_device = None
        self._torch = None
        self._tf = None
        self._model = None

    def _load(self, device_name: str) -> None:
        device_name = normalize_device(device_name)
        if self._loaded and self._loaded_device == device_name:
            return

        with prepend_sys_path(self.codes_dir, ["models", "utils"]):
            import torch
            import torchvision.transforms as transforms
            from models.CIN import CIN
            from utils.yml import dict_to_nonedict, parse_yml

            device = torch.device(device_name)
            opt = parse_yml(str(self.codes_dir / "options" / "opt.yml"))
            opt["train"]["batch_size"] = 1
            opt["train"]["resume"]["Empty"] = True
            opt["path"]["folder_temp"] = str(Path(tempfile.gettempdir()) / "watermarking_benchmark_cin")
            if device.type == "cuda":
                opt["train"]["device_ids"] = [0 if device.index is None else device.index]
            else:
                opt["train"]["device_ids"] = []
            opt = dict_to_nonedict(opt)

            model = CIN(opt, device).to(device)
            state = torch.load(self.checkpoint_path, map_location="cpu")["cinNet"]
            if device.type == "cuda":
                model = torch.nn.DataParallel(model, device_ids=opt["train"]["device_ids"])
                model.load_state_dict(state, strict=True)
                model = model.module
            else:
                stripped = {key.removeprefix("module."): value for key, value in state.items()}
                model.load_state_dict(stripped, strict=True)
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
            "image_size": [128, 128],
        }

    def extract_impl(self, input_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._torch is not None
        assert self._model is not None

        image = self._transform()(Image.open(input_path).convert("RGB")).unsqueeze(0).to(next(self._model.parameters()).device)
        with self._torch.no_grad():
            _, msg_fake_1, _, _ = self._model.train_val_decoder(image, "Identity")
        decoded_bits = msg_fake_1.detach().cpu().round().clip(0, 1).int()[0].tolist()
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
