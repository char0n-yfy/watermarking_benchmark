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


HIDDEN_MODULES = [
    "average_meter",
    "model",
    "noise_argparser",
    "noise_layers",
    "options",
    "tensorboard_logger",
    "utils",
    "vgg_loss",
]


@register_watermark
class HiDDeNWatermark(BaseWatermark):
    name = "hidden"
    description = "HiDDeN image watermark wrapper using packaged weights."

    def __init__(
        self,
        repo_dir: str | Path | None = None,
        weights_dir: str | Path | None = None,
        options_file: str | Path | None = None,
        checkpoint_file: str | Path | None = None,
        identity_noise: bool = True,
        **params: Any,
    ) -> None:
        super().__init__(
            repo_dir=str(repo_dir) if repo_dir is not None else None,
            weights_dir=str(weights_dir) if weights_dir is not None else None,
            options_file=str(options_file) if options_file is not None else None,
            checkpoint_file=str(checkpoint_file) if checkpoint_file is not None else None,
            identity_noise=identity_noise,
            **params,
        )
        self.repo_dir = require_path(repo_dir or packaged_algorithm_dir("hidden"), "HiDDeN repo_dir")
        self.weights_dir = require_path(weights_dir or packaged_weights_dir("hidden"), "HiDDeN weights_dir")
        self.options_file = require_path(
            options_file or self.weights_dir / "options-and-config.pickle",
            "HiDDeN options_file",
        )
        self.checkpoint_file = require_path(
            checkpoint_file or self.weights_dir / "combined-noise--epoch-400.pyt",
            "HiDDeN checkpoint_file",
        )
        self.identity_noise = identity_noise
        self._loaded = False
        self._torch = None
        self._np = None
        self._tf = None
        self._hidden_utils = None
        self._model = None
        self._hidden_config = None

    def _load(self, device_name: str) -> None:
        device_name = normalize_device(device_name)
        if self._loaded and str(self._model.device) == device_name:
            return

        with prepend_sys_path(self.repo_dir, HIDDEN_MODULES):
            import numpy as np
            import torch
            import torchvision.transforms.functional as TF
            import utils as hidden_utils
            from model.hidden import Hidden
            from noise_layers.noiser import Noiser

            device = torch.device(device_name)
            _, hidden_config, noise_config = hidden_utils.load_options(str(self.options_file))
            noiser = Noiser([] if self.identity_noise else noise_config, device)
            model = Hidden(hidden_config, device, noiser, None)
            try:
                checkpoint = torch.load(str(self.checkpoint_file), map_location=device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(str(self.checkpoint_file), map_location=device)
            hidden_utils.model_from_checkpoint(model, checkpoint)
            model.encoder_decoder.eval()

        self._torch = torch
        self._np = np
        self._tf = TF
        self._hidden_utils = hidden_utils
        self._model = model
        self._hidden_config = hidden_config
        self._loaded = True

    def _prepare_tensor(self, input_path: Path):
        assert self._hidden_config is not None
        assert self._tf is not None
        assert self._model is not None

        image = Image.open(input_path).convert("RGB")
        original_size = image.size
        width = int(self._hidden_config.W)
        height = int(self._hidden_config.H)
        image = image.resize((width, height), Image.Resampling.BICUBIC)
        tensor = self._tf.to_tensor(image).to(self._model.device)
        tensor = tensor * 2 - 1
        return tensor.unsqueeze(0), original_size

    def embed_batch_impl(
        self,
        jobs: list[tuple[Path, Path, WatermarkContext]],
    ) -> list[Mapping[str, Any]]:
        if not jobs:
            return []
        self._load(jobs[0][2].device)
        assert self._torch is not None
        assert self._np is not None
        assert self._hidden_utils is not None
        assert self._model is not None
        assert self._hidden_config is not None

        prepared = [self._prepare_tensor(input_path) for input_path, _output_path, _context in jobs]
        image_batch = self._torch.cat([item[0] for item in prepared], dim=0)
        original_sizes = [item[1] for item in prepared]
        nbits = int(self._hidden_config.message_length)
        bits_list = [bits_from_message(context.message, nbits, seed=context.seed) for _input_path, _output_path, context in jobs]
        message_batch = self._torch.tensor(bits_list, dtype=self._torch.float32, device=self._model.device)

        with self._torch.no_grad():
            encoded = self._model.encoder_decoder.encoder(image_batch, message_batch)
            decoded = self._model.encoder_decoder.decoder(encoded)

        decoded_batch = decoded.detach().cpu().numpy().round().clip(0, 1).astype(int).tolist()
        encoded_np = self._hidden_utils.tensor_to_image(encoded.detach().cpu())
        results: list[Mapping[str, Any]] = []
        for index, ((_input_path, output_path, _context), bits, decoded_bits, original_size) in enumerate(
            zip(jobs, bits_list, decoded_batch, original_sizes)
        ):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(encoded_np[index]).save(output_path, format="PNG")
            results.append(
                {
                    "bits": bits_to_string(bits),
                    "decoded_bits": bits_to_string(decoded_bits),
                    "bit_accuracy_self_check": bit_accuracy(bits, decoded_bits),
                    "payload_bits": nbits,
                    "image_size": [int(self._hidden_config.W), int(self._hidden_config.H)],
                    "original_size": list(original_size),
                    "checkpoint_file": str(self.checkpoint_file),
                }
            )
        return results

    def extract_batch_impl(
        self,
        jobs: list[tuple[Path, WatermarkContext]],
    ) -> list[Mapping[str, Any]]:
        if not jobs:
            return []
        self._load(jobs[0][1].device)
        assert self._torch is not None
        assert self._model is not None
        assert self._hidden_config is not None

        prepared = [self._prepare_tensor(input_path) for input_path, _context in jobs]
        image_batch = self._torch.cat([item[0] for item in prepared], dim=0)
        original_sizes = [item[1] for item in prepared]
        with self._torch.no_grad():
            decoded = self._model.encoder_decoder.decoder(image_batch)
        decoded_batch = decoded.detach().cpu().numpy().round().clip(0, 1).astype(int).tolist()

        results: list[Mapping[str, Any]] = []
        nbits = int(self._hidden_config.message_length)
        for decoded_bits, original_size, (_input_path, context) in zip(decoded_batch, original_sizes, jobs):
            metadata: dict[str, Any] = {
                "bits": bits_to_string(decoded_bits),
                "payload_bits": nbits,
                "image_size": [int(self._hidden_config.W), int(self._hidden_config.H)],
                "original_size": list(original_size),
                "checkpoint_file": str(self.checkpoint_file),
            }
            if context.message is not None:
                expected = bits_from_message(context.message, nbits, seed=context.seed)
                metadata["expected_bits"] = bits_to_string(expected)
                metadata["bit_accuracy"] = bit_accuracy(expected, decoded_bits)
            results.append(metadata)
        return results

    def embed_impl(
        self,
        input_path: Path,
        output_path: Path,
        context: WatermarkContext,
    ) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._torch is not None
        assert self._np is not None
        assert self._hidden_utils is not None
        assert self._model is not None
        assert self._hidden_config is not None

        if context.seed is not None:
            self._torch.manual_seed(context.seed)
            self._np.random.seed(context.seed)

        image_tensor, original_size = self._prepare_tensor(input_path)
        nbits = int(self._hidden_config.message_length)
        bits = bits_from_message(context.message, nbits, seed=context.seed)
        message_tensor = self._torch.tensor([bits], dtype=self._torch.float32, device=self._model.device)

        with self._torch.no_grad():
            encoded = self._model.encoder_decoder.encoder(image_tensor, message_tensor)
            decoded = self._model.encoder_decoder.decoder(encoded)

        decoded_bits = decoded.detach().cpu().numpy().round().clip(0, 1).astype(int)[0].tolist()
        encoded_np = self._hidden_utils.tensor_to_image(encoded.detach().cpu())[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(encoded_np).save(output_path, format="PNG")

        return {
            "bits": bits_to_string(bits),
            "decoded_bits": bits_to_string(decoded_bits),
            "bit_accuracy_self_check": bit_accuracy(bits, decoded_bits),
            "payload_bits": nbits,
            "image_size": [int(self._hidden_config.W), int(self._hidden_config.H)],
            "original_size": list(original_size),
            "checkpoint_file": str(self.checkpoint_file),
        }

    def extract_impl(
        self,
        input_path: Path,
        context: WatermarkContext,
    ) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._torch is not None
        assert self._model is not None
        assert self._hidden_config is not None

        image_tensor, original_size = self._prepare_tensor(input_path)
        with self._torch.no_grad():
            decoded = self._model.encoder_decoder.decoder(image_tensor)
        decoded_bits = decoded.detach().cpu().numpy().round().clip(0, 1).astype(int)[0].tolist()
        metadata: dict[str, Any] = {
            "bits": bits_to_string(decoded_bits),
            "payload_bits": int(self._hidden_config.message_length),
            "image_size": [int(self._hidden_config.W), int(self._hidden_config.H)],
            "original_size": list(original_size),
            "checkpoint_file": str(self.checkpoint_file),
        }
        if context.message is not None:
            expected = bits_from_message(context.message, int(self._hidden_config.message_length), seed=context.seed)
            metadata["expected_bits"] = bits_to_string(expected)
            metadata["bit_accuracy"] = bit_accuracy(expected, decoded_bits)
        return metadata
