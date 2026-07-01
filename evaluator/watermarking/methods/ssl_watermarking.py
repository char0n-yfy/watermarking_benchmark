from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
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


SSL_MODULES = [
    "build_normalization_layer",
    "data_augmentation",
    "decode",
    "encode",
    "evaluate",
    "utils",
    "utils_img",
]


class _SingleImageDataset:
    def __init__(self, tensor, label: int = 0) -> None:
        self.tensor = tensor
        self.label = label

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(index)
        return self.tensor, self.label


@register_watermark
class SSLWatermark(BaseWatermark):
    name = "ssl-watermarking"
    description = "SSL latent-space watermark wrapper using packaged weights."

    def __init__(
        self,
        repo_dir: str | Path | None = None,
        weights_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        normlayer_path: str | Path | None = None,
        carrier_path: str | Path | None = None,
        model_name: str = "resnet50",
        payload_bits: int = 30,
        epochs: int = 10,
        batch_size: int = 1,
        target_psnr: float = 40.0,
        lambda_w: float = 50000.0,
        lambda_i: float = 1.0,
        optimizer: str = "Adam,lr=0.01",
        scheduler: str | None = None,
        data_augmentation: str = "all",
        verbose: int = 0,
        **params: Any,
    ) -> None:
        super().__init__(
            repo_dir=str(repo_dir) if repo_dir is not None else None,
            weights_dir=str(weights_dir) if weights_dir is not None else None,
            model_path=str(model_path) if model_path is not None else None,
            normlayer_path=str(normlayer_path) if normlayer_path is not None else None,
            carrier_path=str(carrier_path) if carrier_path is not None else None,
            model_name=model_name,
            payload_bits=payload_bits,
            epochs=epochs,
            batch_size=batch_size,
            target_psnr=target_psnr,
            lambda_w=lambda_w,
            lambda_i=lambda_i,
            optimizer=optimizer,
            scheduler=scheduler,
            data_augmentation=data_augmentation,
            verbose=verbose,
            **params,
        )
        self.repo_dir = require_path(repo_dir or packaged_algorithm_dir("ssl_watermarking"), "SSL repo_dir")
        self.weights_dir = require_path(weights_dir or packaged_weights_dir("ssl_watermarking"), "SSL weights_dir")
        self.model_path = require_path(model_path or self.weights_dir / "dino_r50_plus.pth", "SSL model_path")
        self.normlayer_path = require_path(
            normlayer_path or self.weights_dir / "out2048_coco_orig.pth",
            "SSL normlayer_path",
        )
        self.carrier_path = require_path(
            carrier_path or self.weights_dir / "ssl_carrier_seed2026.pt",
            "SSL carrier_path",
        )
        self.model_name = model_name
        self.payload_bits = int(payload_bits)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.target_psnr = float(target_psnr)
        self.lambda_w = float(lambda_w)
        self.lambda_i = float(lambda_i)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_augmentation = data_augmentation
        self.verbose = int(verbose)
        self._loaded = False
        self._loaded_device = None
        self._torch = None
        self._to_pil_image = None
        self._utils = None
        self._utils_img = None
        self._data_augmentation = None
        self._encode = None
        self._decode = None
        self._model = None
        self._carrier = None

    def _set_module_devices(self, device) -> None:
        assert self._torch is not None
        assert self._utils is not None
        assert self._utils_img is not None
        assert self._encode is not None
        assert self._decode is not None

        self._utils.device = device
        self._utils_img.device = device
        self._encode.device = device
        self._decode.device = device
        self._utils_img.image_mean = self._torch.Tensor(
            self._utils_img.NORMALIZE_IMAGENET.mean
        ).view(-1, 1, 1).to(device)
        self._utils_img.image_std = self._torch.Tensor(
            self._utils_img.NORMALIZE_IMAGENET.std
        ).view(-1, 1, 1).to(device)

    def _load(self, device_name: str) -> None:
        device_name = normalize_device(device_name)
        if self._loaded and self._loaded_device == device_name:
            return

        with prepend_sys_path(self.repo_dir, SSL_MODULES):
            import torch
            from torchvision.transforms import ToPILImage

            import data_augmentation
            import decode
            import encode
            import utils
            import utils_img

            device = torch.device(device_name)
            self._torch = torch
            self._to_pil_image = ToPILImage
            self._utils = utils
            self._utils_img = utils_img
            self._data_augmentation = data_augmentation
            self._encode = encode
            self._decode = decode
            self._set_module_devices(device)

            backbone = utils.build_backbone(path=str(self.model_path), name=self.model_name)
            normlayer = utils.load_normalization_layer(path=str(self.normlayer_path))
            model = utils.NormLayerWrapper(backbone, normlayer)
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.eval()

            carrier = torch.load(str(self.carrier_path), map_location=device)
            carrier = carrier.to(device, non_blocking=True)

        self._model = model
        self._carrier = carrier
        self._loaded = True
        self._loaded_device = device_name

    def _params(self) -> SimpleNamespace:
        return SimpleNamespace(
            batch_size=self.batch_size,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epochs=self.epochs,
            lambda_w=self.lambda_w,
            lambda_i=self.lambda_i,
            target_psnr=self.target_psnr,
            verbose=self.verbose,
        )

    def _data_aug(self):
        assert self._data_augmentation is not None
        mode = str(self.data_augmentation).lower()
        if mode == "none":
            return self._data_augmentation.DifferentiableDataAugmentation()
        return self._data_augmentation.All()

    def extract_batch_impl(
        self,
        jobs: list[tuple[Path, WatermarkContext]],
    ) -> list[Mapping[str, Any]]:
        if not jobs:
            return []
        self._load(jobs[0][1].device)
        assert self._torch is not None
        assert self._decode is not None
        assert self._model is not None
        assert self._carrier is not None

        images = [Image.open(input_path).convert("RGB") for input_path, _context in jobs]
        decoded_data = self._decode.decode_multibit(images, self._carrier, self._model)
        results: list[Mapping[str, Any]] = []
        for decoded_item, (_input_path, context) in zip(decoded_data, jobs):
            decoded_tensor = decoded_item["msg"]
            decoded_bits = [int(bit) for bit in decoded_tensor.type(self._torch.int).tolist()]
            metadata: dict[str, Any] = {
                "bits": bits_to_string(decoded_bits),
                "payload_bits": len(decoded_bits),
                "model_path": str(self.model_path),
                "normlayer_path": str(self.normlayer_path),
                "carrier_path": str(self.carrier_path),
            }
            if context.message is not None:
                expected = bits_from_message(context.message, len(decoded_bits), seed=context.seed)
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
        assert self._to_pil_image is not None
        assert self._utils_img is not None
        assert self._encode is not None
        assert self._model is not None
        assert self._carrier is not None

        if context.seed is not None:
            self._torch.manual_seed(context.seed)

        payload_bits = self.payload_bits
        if payload_bits != int(self._carrier.shape[0]):
            raise ValueError(
                f"Configured payload_bits={payload_bits} but carrier has {self._carrier.shape[0]} rows"
            )
        bits = bits_from_message(context.message, payload_bits, seed=context.seed)
        message_tensor = self._torch.tensor([bits], dtype=self._torch.bool)

        image = Image.open(input_path).convert("RGB")
        image_tensor = self._utils_img.default_transform(image)
        dataset = _SingleImageDataset(image_tensor)
        dataloader = self._torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        pt_imgs_out = self._encode.watermark_multibit(
            dataloader,
            message_tensor,
            self._carrier,
            self._model,
            self._data_aug(),
            self._params(),
        )
        if not pt_imgs_out:
            raise RuntimeError("SSL watermarking did not return an output image")

        output_tensor = self._utils_img.unnormalize_img(pt_imgs_out[0]).detach().cpu().clamp(0, 1)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._to_pil_image()(output_tensor).save(output_path, format="PNG")

        return {
            "bits": bits_to_string(bits),
            "payload_bits": payload_bits,
            "model_path": str(self.model_path),
            "normlayer_path": str(self.normlayer_path),
            "carrier_path": str(self.carrier_path),
            "epochs": self.epochs,
            "target_psnr": self.target_psnr,
        }

    def extract_impl(
        self,
        input_path: Path,
        context: WatermarkContext,
    ) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._torch is not None
        assert self._decode is not None
        assert self._model is not None
        assert self._carrier is not None

        image = Image.open(input_path).convert("RGB")
        decoded_data = self._decode.decode_multibit([image], self._carrier, self._model)
        decoded_tensor = decoded_data[0]["msg"]
        decoded_bits = [int(bit) for bit in decoded_tensor.type(self._torch.int).tolist()]
        metadata: dict[str, Any] = {
            "bits": bits_to_string(decoded_bits),
            "payload_bits": len(decoded_bits),
            "model_path": str(self.model_path),
            "normlayer_path": str(self.normlayer_path),
            "carrier_path": str(self.carrier_path),
        }
        if context.message is not None:
            expected = bits_from_message(context.message, len(decoded_bits), seed=context.seed)
            metadata["expected_bits"] = bits_to_string(expected)
            metadata["bit_accuracy"] = bit_accuracy(expected, decoded_bits)
        return metadata
