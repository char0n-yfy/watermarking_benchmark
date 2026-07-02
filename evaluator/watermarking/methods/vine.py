from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

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
    require_path,
)


@register_watermark
class VineWatermark(BaseWatermark):
    name = "vine"
    description = "VINE 100-bit diffusion-prior watermark using packaged VINE-R and local SD-Turbo dependencies."

    def __init__(
        self,
        repo_dir: str | Path | None = None,
        weights_dir: str | Path | None = None,
        encoder_dir: str | Path | None = None,
        decoder_dir: str | Path | None = None,
        sd_turbo_dir: str | Path | None = None,
        payload_bits: int = 100,
        sd_variant: str = "fp16",
        **params: Any,
    ) -> None:
        super().__init__(
            repo_dir=str(repo_dir) if repo_dir is not None else None,
            weights_dir=str(weights_dir) if weights_dir is not None else None,
            encoder_dir=str(encoder_dir) if encoder_dir is not None else None,
            decoder_dir=str(decoder_dir) if decoder_dir is not None else None,
            sd_turbo_dir=str(sd_turbo_dir) if sd_turbo_dir is not None else None,
            payload_bits=payload_bits,
            sd_variant=sd_variant,
            **params,
        )
        self.repo_dir = require_path(repo_dir or packaged_algorithm_dir("vine"), "VINE repo_dir")
        self.weights_dir = require_path(weights_dir or packaged_weights_dir("vine"), "VINE weights_dir")
        self.encoder_dir = Path(encoder_dir) if encoder_dir is not None else self.weights_dir / "vine-r-enc"
        self.decoder_dir = Path(decoder_dir) if decoder_dir is not None else self.weights_dir / "vine-r-dec"
        self.sd_turbo_dir = Path(sd_turbo_dir) if sd_turbo_dir is not None else self.weights_dir / "sd-turbo"
        self.payload_bits = int(payload_bits)
        if self.payload_bits != 100:
            raise ValueError("The packaged VINE-R checkpoints expect payload_bits=100")
        self.sd_variant = sd_variant
        self._encoder_loaded = False
        self._decoder_loaded = False
        self._encoder_device: str | None = None
        self._decoder_device: str | None = None
        self._torch = None
        self._tf = None
        self._encoder = None
        self._decoder = None

    @contextmanager
    def _runtime_paths(self, purge_modules: bool = True) -> Iterator[None]:
        paths = [
            self.repo_dir / "third_party",
            self.repo_dir / "diffusers" / "src",
            self.repo_dir,
            self.repo_dir / "vine" / "src",
        ]
        original_path = list(sys.path)
        original_env = {
            "VINE_SD_TURBO_DIR": os.environ.get("VINE_SD_TURBO_DIR"),
            "VINE_SD_TURBO_VARIANT": os.environ.get("VINE_SD_TURBO_VARIANT"),
        }
        if purge_modules:
            for module_name in ("vine", "diffusers", "peft", "vine_turbo", "stega_encoder_decoder"):
                self._purge_module_tree(module_name)
        sys.path[:0] = [str(path.resolve()) for path in paths if path.exists()]
        os.environ["VINE_SD_TURBO_DIR"] = str(self.sd_turbo_dir.resolve())
        os.environ["VINE_SD_TURBO_VARIANT"] = self.sd_variant
        try:
            yield
        finally:
            sys.path[:] = original_path
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    @staticmethod
    def _purge_module_tree(module_name: str) -> None:
        for name in [name for name in sys.modules if name == module_name or name.startswith(module_name + ".")]:
            sys.modules.pop(name, None)

    def _require_model_file(self, directory: Path, label: str) -> Path:
        directory = require_path(directory, f"{label} directory")
        require_path(directory / "config.json", f"{label} config.json")
        return require_path(directory / "model.safetensors", f"{label} model.safetensors")

    def _require_sd_turbo(self) -> None:
        root = require_path(self.sd_turbo_dir, "SD-Turbo directory")
        for rel in (
            "scheduler/scheduler_config.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
            "text_encoder/config.json",
            f"text_encoder/model.{self.sd_variant}.safetensors",
            "unet/config.json",
            f"unet/diffusion_pytorch_model.{self.sd_variant}.safetensors",
            "vae/config.json",
            f"vae/diffusion_pytorch_model.{self.sd_variant}.safetensors",
        ):
            require_path(root / rel, f"SD-Turbo file {rel}")

    def _load_encoder(self, device_name: str) -> None:
        device_name = normalize_device(device_name)
        if self._encoder_loaded and self._encoder_device == device_name:
            return
        self._require_model_file(self.encoder_dir, "VINE encoder")
        self._require_sd_turbo()

        with self._runtime_paths():
            import torch
            from torchvision import transforms
            from vine.src.vine_turbo import VINE_Turbo

            encoder = VINE_Turbo.from_pretrained(
                str(self.encoder_dir.resolve()),
                local_files_only=True,
                device=device_name,
            )
            encoder = encoder.to(torch.device(device_name)).eval()

        self._torch = torch
        self._tf = transforms
        self._encoder = encoder
        self._encoder_loaded = True
        self._encoder_device = device_name

    def _load_decoder(self, device_name: str) -> None:
        device_name = normalize_device(device_name)
        if self._decoder_loaded and self._decoder_device == device_name:
            return
        self._require_model_file(self.decoder_dir, "VINE decoder")

        with self._runtime_paths():
            import torch
            from torchvision import transforms
            from stega_encoder_decoder import CustomConvNeXt

            decoder = CustomConvNeXt.from_pretrained(
                str(self.decoder_dir.resolve()),
                local_files_only=True,
            )
            decoder = decoder.to(torch.device(device_name)).eval()

        self._torch = torch
        self._tf = transforms
        self._decoder = decoder
        self._decoder_loaded = True
        self._decoder_device = device_name

    def release(self) -> None:
        from evaluator.runtime_cleanup import move_to_cpu, torch_cleanup

        if self._encoder is not None:
            move_to_cpu(self._encoder)
        if self._decoder is not None:
            move_to_cpu(self._decoder)
        self._encoder = None
        self._decoder = None
        self._encoder_loaded = False
        self._decoder_loaded = False
        self._encoder_device = None
        self._decoder_device = None
        torch_cleanup()

    @staticmethod
    def _crop_to_square(image: Image.Image) -> Image.Image:
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        return image.crop((left, top, left + side, top + side))

    def _payload(self, context: WatermarkContext) -> list[int]:
        return bits_from_message(context.message, self.payload_bits, seed=context.seed)

    def _embed_metadata(self, bits: list[int], original_size: list[int], output_size: tuple[int, int]) -> dict[str, Any]:
        return {
            "bits": bits_to_string(bits),
            "payload_bits": self.payload_bits,
            "weights_dir": str(self.weights_dir),
            "encoder_dir": str(self.encoder_dir),
            "decoder_dir": str(self.decoder_dir),
            "sd_turbo_dir": str(self.sd_turbo_dir),
            "image_size": [256, 256],
            "original_size": original_size,
            "output_size": list(output_size),
        }

    def extract_batch_impl(
        self,
        jobs: list[tuple[Path, WatermarkContext]],
    ) -> list[Mapping[str, Any]]:
        if not jobs:
            return []
        self._load_decoder(jobs[0][1].device)
        assert self._torch is not None
        assert self._tf is not None
        assert self._decoder is not None

        transform = self._tf.Compose(
            [
                self._tf.Resize(256, interpolation=self._tf.InterpolationMode.BICUBIC),
                self._tf.ToTensor(),
            ]
        )
        tensor = self._torch.cat(
            [transform(Image.open(input_path).convert("RGB")).unsqueeze(0) for input_path, _context in jobs],
            dim=0,
        ).to(next(self._decoder.parameters()).device)

        with self._runtime_paths(purge_modules=False), self._torch.inference_mode():
            pred = self._decoder(tensor)
        decoded_batch = (pred >= 0.5).int().detach().cpu().tolist()

        results: list[Mapping[str, Any]] = []
        for decoded, (_input_path, context) in zip(decoded_batch, jobs):
            metadata: dict[str, Any] = {
                "bits": bits_to_string(decoded),
                "payload_bits": len(decoded),
                "weights_dir": str(self.weights_dir),
                "decoder_dir": str(self.decoder_dir),
                "image_size": [256, 256],
            }
            if context.message is not None or context.seed is not None:
                expected = self._payload(context)
                metadata["expected_bits"] = bits_to_string(expected)
                metadata["bit_accuracy"] = bit_accuracy(expected, decoded)
            results.append(metadata)
        return results

    def embed_batch_impl(
        self,
        jobs: list[tuple[Path, Path, WatermarkContext]],
    ) -> list[Mapping[str, Any]]:
        if not jobs:
            return []
        self._load_encoder(jobs[0][2].device)
        assert self._torch is not None
        assert self._tf is not None
        assert self._encoder is not None

        resize_to_256 = self._tf.Compose(
            [
                self._tf.Resize(256, interpolation=self._tf.InterpolationMode.BICUBIC),
                self._tf.ToTensor(),
            ]
        )
        to_tensor = self._tf.ToTensor()
        to_pil = self._tf.ToPILImage()
        device = next(self._encoder.parameters()).device

        prepared: list[dict[str, Any]] = []
        groups: dict[tuple[int, int], list[int]] = {}
        for index, (input_path, output_path, context) in enumerate(jobs):
            image_pil = Image.open(input_path).convert("RGB")
            original_size = list(image_pil.size)
            if image_pil.size[0] != image_pil.size[1]:
                image_pil = self._crop_to_square(image_pil)
            output_size = image_pil.size
            bits = self._payload(context)
            prepared.append(
                {
                    "image": image_pil,
                    "outputPath": output_path,
                    "originalSize": original_size,
                    "outputSize": output_size,
                    "bits": bits,
                }
            )
            groups.setdefault(output_size, []).append(index)

        results: list[dict[str, Any] | None] = [None] * len(jobs)
        with self._runtime_paths(purge_modules=False), self._torch.inference_mode():
            for output_size, indexes in groups.items():
                small = self._torch.cat(
                    [
                        (2.0 * resize_to_256(prepared[index]["image"]) - 1.0).unsqueeze(0)
                        for index in indexes
                    ],
                    dim=0,
                ).to(device)
                original = self._torch.cat(
                    [
                        (2.0 * to_tensor(prepared[index]["image"]) - 1.0).unsqueeze(0)
                        for index in indexes
                    ],
                    dim=0,
                ).to(device)
                secret = self._torch.tensor(
                    [prepared[index]["bits"] for index in indexes],
                    dtype=self._torch.float32,
                    device=device,
                )

                encoded_small = self._encoder(small, secret)
                residual = self._torch.nn.functional.interpolate(
                    encoded_small - small,
                    size=(output_size[1], output_size[0]),
                    mode="bicubic",
                    align_corners=False,
                )
                encoded = self._torch.clamp(original + residual, min=-1.0, max=1.0)
                encoded = (encoded * 0.5 + 0.5).clamp(0.0, 1.0)

                for batch_index, prepared_index in enumerate(indexes):
                    output_path = prepared[prepared_index]["outputPath"]
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    to_pil(encoded[batch_index].detach().cpu()).save(output_path)
                    metadata = self._embed_metadata(
                        prepared[prepared_index]["bits"],
                        prepared[prepared_index]["originalSize"],
                        prepared[prepared_index]["outputSize"],
                    )
                    metadata["batchOptimized"] = True
                    results[prepared_index] = metadata

        return [result or {} for result in results]

    def embed_impl(self, input_path: Path, output_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        self._load_encoder(context.device)
        assert self._torch is not None
        assert self._tf is not None
        assert self._encoder is not None

        image_pil = Image.open(input_path).convert("RGB")
        original_size = list(image_pil.size)
        if image_pil.size[0] != image_pil.size[1]:
            image_pil = self._crop_to_square(image_pil)
        output_size = image_pil.size

        resize_to_256 = self._tf.Compose(
            [
                self._tf.Resize(256, interpolation=self._tf.InterpolationMode.BICUBIC),
                self._tf.ToTensor(),
            ]
        )
        resize_to_output = self._tf.Resize(output_size, interpolation=self._tf.InterpolationMode.BICUBIC)

        bits = self._payload(context)
        device = next(self._encoder.parameters()).device
        small = (2.0 * resize_to_256(image_pil) - 1.0).unsqueeze(0).to(device)
        original = (2.0 * self._tf.ToTensor()(image_pil) - 1.0).unsqueeze(0).to(device)
        secret = self._torch.tensor(bits, dtype=self._torch.float32, device=device).unsqueeze(0)

        with self._runtime_paths(purge_modules=False), self._torch.inference_mode():
            encoded_small = self._encoder(small, secret)
            residual = resize_to_output(encoded_small - small)
            encoded = self._torch.clamp(original + residual, min=-1.0, max=1.0)
            encoded = (encoded * 0.5 + 0.5).clamp(0.0, 1.0)

        self._tf.ToPILImage()(encoded[0].detach().cpu()).save(output_path)
        return self._embed_metadata(bits, original_size, output_size)

    def extract_impl(self, input_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        self._load_decoder(context.device)
        assert self._torch is not None
        assert self._tf is not None
        assert self._decoder is not None

        image = Image.open(input_path).convert("RGB")
        tensor = self._tf.Compose(
            [
                self._tf.Resize(256, interpolation=self._tf.InterpolationMode.BICUBIC),
                self._tf.ToTensor(),
            ]
        )(image).unsqueeze(0).to(next(self._decoder.parameters()).device)

        with self._runtime_paths(purge_modules=False), self._torch.inference_mode():
            pred = self._decoder(tensor)
        decoded = (pred[0] >= 0.5).int().detach().cpu().tolist()
        metadata: dict[str, Any] = {
            "bits": bits_to_string(decoded),
            "payload_bits": len(decoded),
            "weights_dir": str(self.weights_dir),
            "decoder_dir": str(self.decoder_dir),
            "image_size": [256, 256],
        }
        if context.message is not None or context.seed is not None:
            expected = self._payload(context)
            metadata["expected_bits"] = bits_to_string(expected)
            metadata["bit_accuracy"] = bit_accuracy(expected, decoded)
        return metadata
