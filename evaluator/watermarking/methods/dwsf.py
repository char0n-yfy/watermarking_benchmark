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


DWSF_MODULES = ["networks", "utils"]


@register_watermark
class DWSFWatermark(BaseWatermark):
    name = "dwsf"
    description = "DWSF 30-bit deep dispersed image watermark using packaged encoder, decoder, and segmentation weights."

    def __init__(
        self,
        repo_dir: str | Path | None = None,
        weights_dir: str | Path | None = None,
        encoder_weights: str | Path | None = None,
        decoder_weights: str | Path | None = None,
        seg_weights: str | Path | None = None,
        payload_bits: int = 30,
        psnr: float = 35.0,
        threshold: float = 0.5,
        **params: Any,
    ) -> None:
        super().__init__(
            repo_dir=str(repo_dir) if repo_dir is not None else None,
            weights_dir=str(weights_dir) if weights_dir is not None else None,
            encoder_weights=str(encoder_weights) if encoder_weights is not None else None,
            decoder_weights=str(decoder_weights) if decoder_weights is not None else None,
            seg_weights=str(seg_weights) if seg_weights is not None else None,
            payload_bits=payload_bits,
            psnr=psnr,
            threshold=threshold,
            **params,
        )
        self.repo_dir = require_path(repo_dir or packaged_algorithm_dir("dwsf"), "DWSF repo_dir")
        self.weights_dir = require_path(weights_dir or packaged_weights_dir("dwsf"), "DWSF weights_dir")
        self.encoder_weights = require_path(
            encoder_weights or self.weights_dir / "encoder_best.pth",
            "DWSF encoder_weights",
        )
        self.decoder_weights = require_path(
            decoder_weights or self.weights_dir / "decoder_best.pth",
            "DWSF decoder_weights",
        )
        self.seg_weights = require_path(seg_weights or self.weights_dir / "seg.pth", "DWSF seg_weights")
        self.payload_bits = int(payload_bits)
        if self.payload_bits != 30:
            raise ValueError("The packaged DWSF checkpoints expect payload_bits=30")
        self.psnr = float(psnr)
        self.threshold = float(threshold)
        self._loaded = False
        self._loaded_device: str | None = None
        self._torch = None
        self._np = None
        self._tf = None
        self._encoder_decoder = None
        self._generate_random_coor = None
        self._psnr_clip = None
        self._obtain_wm_blocks = None

    @staticmethod
    def _torch_load(torch_module, path: Path, device):
        try:
            return torch_module.load(str(path), map_location=device, weights_only=True)
        except TypeError:
            return torch_module.load(str(path), map_location=device)

    def _load(self, device_name: str) -> None:
        device_name = normalize_device(device_name)
        if self._loaded and self._loaded_device == device_name:
            return

        with prepend_sys_path(self.repo_dir, DWSF_MODULES):
            import numpy as np
            import torch
            import torch.nn.functional as F
            import torchvision.transforms.functional as TF
            from networks.models.EncoderDecoder import EncoderDecoder
            from utils.img import psnr_clip
            import utils.seg as dwsf_seg
            from utils.util import generate_random_coor

            device = torch.device(device_name)
            dwsf_seg.device = device
            dwsf_seg.init(str(self.seg_weights))

            encoder_decoder = EncoderDecoder(
                H=128,
                W=128,
                message_length=self.payload_bits,
                noise_layers=["Combined([Identity()])"],
            )
            encoder_state = self._torch_load(torch, self.encoder_weights, device)
            decoder_state = self._torch_load(torch, self.decoder_weights, device)
            encoder_decoder.encoder.load_state_dict(encoder_state)
            encoder_decoder.decoder.load_state_dict(decoder_state)
            encoder_decoder.encoder = encoder_decoder.encoder.to(device).eval()
            encoder_decoder.decoder = encoder_decoder.decoder.to(device).eval()

        self._torch = torch
        self._np = np
        self._tf = TF
        self._F = F
        self._encoder_decoder = encoder_decoder
        self._generate_random_coor = generate_random_coor
        self._psnr_clip = psnr_clip
        self._obtain_wm_blocks = dwsf_seg.obtain_wm_blocks
        self._loaded = True
        self._loaded_device = device_name

    def _prepare_tensor(self, input_path: Path):
        assert self._torch is not None
        assert self._tf is not None
        assert self._encoder_decoder is not None

        image = Image.open(input_path).convert("RGB")
        tensor = self._tf.to_tensor(image).unsqueeze(0)
        tensor = tensor.to(next(self._encoder_decoder.encoder.parameters()).device)
        return tensor * 2.0 - 1.0, image.size

    def _prepare_encode_plan(self, image, message):
        assert self._torch is not None
        assert self._F is not None
        assert self._generate_random_coor is not None

        _, _, height, width = image.shape
        h_coor, w_coor, split_size = self._generate_random_coor(height, width, 128)
        blocks = []
        valid_boxes: list[tuple[int, int, int, int]] = []
        for h_idx, w_idx in zip(h_coor, w_coor):
            x1 = h_idx - split_size // 2
            x2 = h_idx + split_size // 2
            y1 = w_idx - split_size // 2
            y2 = w_idx + split_size // 2
            if x1 >= 0 and x2 <= height and y1 >= 0 and y2 <= width:
                blocks.append(image[:, :, x1:x2, y1:y2])
                valid_boxes.append((x1, x2, y1, y2))

        if not blocks:
            raise RuntimeError(f"DWSF did not produce valid embedding blocks for image size {(width, height)}")

        blocks_tensor = self._torch.vstack(blocks)
        original_blocks = blocks_tensor.clone()
        if split_size != 128:
            blocks_tensor = self._F.interpolate(blocks_tensor, (128, 128), mode="bicubic")

        repeated_message = message.repeat((blocks_tensor.shape[0], 1))
        return {
            "image": image,
            "blocks_tensor": blocks_tensor,
            "original_blocks": original_blocks,
            "message": repeated_message,
            "valid_boxes": valid_boxes,
            "split_size": split_size,
        }

    def _compose_watermarked_tensor(self, plan, encoded_blocks):
        assert self._torch is not None
        assert self._F is not None
        assert self._psnr_clip is not None

        image = plan["image"]
        blocks_tensor = plan["blocks_tensor"]
        original_blocks = plan["original_blocks"]
        split_size = int(plan["split_size"])
        valid_boxes = plan["valid_boxes"]

        noise = self._torch.clamp(encoded_blocks - blocks_tensor, -0.2, 0.2)
        if split_size != 128:
            noise = self._F.interpolate(noise, (split_size, split_size), mode="bicubic")

        watermarked = image.clone().detach()
        for idx, (x1, x2, y1, y2) in enumerate(valid_boxes):
            block = original_blocks[idx : idx + 1]
            encoded_block = block + noise[idx : idx + 1]
            encoded_block = self._psnr_clip(encoded_block, block, self.psnr)
            watermarked[:, :, x1:x2, y1:y2] = encoded_block

        return self._torch.clamp(watermarked, -1.0, 1.0), len(valid_boxes), split_size

    def _encode_tensor(self, image, message):
        assert self._encoder_decoder is not None

        plan = self._prepare_encode_plan(image, message)
        encoded_blocks = self._encoder_decoder.encoder(plan["blocks_tensor"], plan["message"])
        return self._compose_watermarked_tensor(plan, encoded_blocks)

    def _save_tensor(self, watermarked, output_path: Path) -> None:
        assert self._tf is not None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = ((watermarked[0].detach().cpu() + 1.0) / 2.0).clamp(0, 1)
        self._tf.to_pil_image(image).save(output_path)

    def _decode_blocks(self, blocks):
        assert self._torch is not None
        assert self._encoder_decoder is not None

        decoded_batches = []
        for start in range(0, int(blocks.shape[0]), 32):
            decoded_batches.append(self._encoder_decoder.decoder(blocks[start : start + 32]))
        decoded = self._torch.vstack(decoded_batches)
        decoded_mean = decoded.mean(dim=0)
        return (decoded_mean > self.threshold).to(self._torch.int64).detach().cpu().tolist()

    def _decode_tensor(self, image):
        assert self._torch is not None
        assert self._obtain_wm_blocks is not None

        blocks = self._obtain_wm_blocks(image)
        decoded_bits = self._decode_blocks(blocks)
        return decoded_bits, int(blocks.shape[0])

    def embed_batch_impl(
        self,
        jobs: list[tuple[Path, Path, WatermarkContext]],
    ) -> list[Mapping[str, Any]]:
        if not jobs:
            return []

        self._load(jobs[0][2].device)
        assert self._torch is not None
        assert self._np is not None
        assert self._encoder_decoder is not None

        records: list[dict[str, Any]] = []
        all_blocks = []
        all_messages = []
        for input_path, output_path, context in jobs:
            if context.seed is not None:
                self._torch.manual_seed(context.seed)
                self._np.random.seed(context.seed)

            image, original_size = self._prepare_tensor(input_path)
            bits = bits_from_message(context.message, self.payload_bits, seed=context.seed)
            message = self._torch.tensor(bits, dtype=self._torch.float32, device=image.device).unsqueeze(0)
            plan = self._prepare_encode_plan(image, message)
            records.append(
                {
                    "output_path": output_path,
                    "original_size": original_size,
                    "bits": bits,
                    "plan": plan,
                }
            )
            all_blocks.append(plan["blocks_tensor"])
            all_messages.append(plan["message"])

        with self._torch.inference_mode():
            encoded_blocks = self._encoder_decoder.encoder(
                self._torch.vstack(all_blocks),
                self._torch.vstack(all_messages),
            )

        metadatas: list[Mapping[str, Any]] = []
        offset = 0
        for record in records:
            plan = record["plan"]
            block_count = int(plan["blocks_tensor"].shape[0])
            watermarked, embedded_blocks, split_size = self._compose_watermarked_tensor(
                plan,
                encoded_blocks[offset : offset + block_count],
            )
            offset += block_count
            self._save_tensor(watermarked, Path(record["output_path"]))
            metadatas.append(
                {
                    "bits": bits_to_string(record["bits"]),
                    "payload_bits": self.payload_bits,
                    "image_size": list(record["original_size"]),
                    "embedded_blocks": embedded_blocks,
                    "split_size": split_size,
                    "psnr_target": self.psnr,
                    "encoder_weights": str(self.encoder_weights),
                    "decoder_weights": str(self.decoder_weights),
                    "seg_weights": str(self.seg_weights),
                    "weights_dir": str(self.weights_dir),
                }
            )
        return metadatas

    def extract_batch_impl(
        self,
        jobs: list[tuple[Path, WatermarkContext]],
    ) -> list[Mapping[str, Any]]:
        if not jobs:
            return []

        self._load(jobs[0][1].device)
        assert self._torch is not None
        assert self._encoder_decoder is not None
        assert self._obtain_wm_blocks is not None

        records: list[dict[str, Any]] = []
        all_blocks = []
        with self._torch.inference_mode():
            for input_path, context in jobs:
                image, original_size = self._prepare_tensor(input_path)
                blocks = self._obtain_wm_blocks(image)
                records.append(
                    {
                        "context": context,
                        "original_size": original_size,
                        "block_count": int(blocks.shape[0]),
                    }
                )
                all_blocks.append(blocks)

            decoded_batches = []
            stacked_blocks = self._torch.vstack(all_blocks)
            for start in range(0, int(stacked_blocks.shape[0]), 32):
                decoded_batches.append(self._encoder_decoder.decoder(stacked_blocks[start : start + 32]))
            decoded = self._torch.vstack(decoded_batches)

        metadatas: list[Mapping[str, Any]] = []
        offset = 0
        for record in records:
            block_count = int(record["block_count"])
            decoded_mean = decoded[offset : offset + block_count].mean(dim=0)
            decoded_bits = (decoded_mean > self.threshold).to(self._torch.int64).detach().cpu().tolist()
            offset += block_count
            context = record["context"]
            metadata: dict[str, Any] = {
                "bits": bits_to_string(decoded_bits),
                "payload_bits": len(decoded_bits),
                "image_size": list(record["original_size"]),
                "decoded_blocks": block_count,
                "threshold": self.threshold,
                "encoder_weights": str(self.encoder_weights),
                "decoder_weights": str(self.decoder_weights),
                "seg_weights": str(self.seg_weights),
                "weights_dir": str(self.weights_dir),
            }
            if context.message is not None or context.seed is not None:
                expected = bits_from_message(context.message, self.payload_bits, seed=context.seed)
                metadata["expected_bits"] = bits_to_string(expected)
                metadata["bit_accuracy"] = bit_accuracy(expected, decoded_bits)
            metadatas.append(metadata)
        return metadatas

    def embed_impl(
        self,
        input_path: Path,
        output_path: Path,
        context: WatermarkContext,
    ) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._torch is not None
        assert self._np is not None
        assert self._tf is not None

        if context.seed is not None:
            self._torch.manual_seed(context.seed)
            self._np.random.seed(context.seed)

        image, original_size = self._prepare_tensor(input_path)
        bits = bits_from_message(context.message, self.payload_bits, seed=context.seed)
        message = self._torch.tensor(bits, dtype=self._torch.float32, device=image.device).unsqueeze(0)

        with self._torch.inference_mode():
            watermarked, embedded_blocks, split_size = self._encode_tensor(image, message)

        self._save_tensor(watermarked, output_path)
        return {
            "bits": bits_to_string(bits),
            "payload_bits": self.payload_bits,
            "image_size": list(original_size),
            "embedded_blocks": embedded_blocks,
            "split_size": split_size,
            "psnr_target": self.psnr,
            "encoder_weights": str(self.encoder_weights),
            "decoder_weights": str(self.decoder_weights),
            "seg_weights": str(self.seg_weights),
            "weights_dir": str(self.weights_dir),
        }

    def extract_impl(
        self,
        input_path: Path,
        context: WatermarkContext,
    ) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._torch is not None

        image, original_size = self._prepare_tensor(input_path)
        with self._torch.inference_mode():
            decoded_bits, decoded_blocks = self._decode_tensor(image)

        metadata: dict[str, Any] = {
            "bits": bits_to_string(decoded_bits),
            "payload_bits": len(decoded_bits),
            "image_size": list(original_size),
            "decoded_blocks": decoded_blocks,
            "threshold": self.threshold,
            "encoder_weights": str(self.encoder_weights),
            "decoder_weights": str(self.decoder_weights),
            "seg_weights": str(self.seg_weights),
            "weights_dir": str(self.weights_dir),
        }
        if context.message is not None or context.seed is not None:
            expected = bits_from_message(context.message, self.payload_bits, seed=context.seed)
            metadata["expected_bits"] = bits_to_string(expected)
            metadata["bit_accuracy"] = bit_accuracy(expected, decoded_bits)
        return metadata
