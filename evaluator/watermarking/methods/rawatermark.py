from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from evaluator.image_io import save_png_image
from evaluator.watermarking.base import BaseWatermark, WatermarkContext
from evaluator.watermarking.registry import register_watermark
from evaluator.watermarking.utils import (
    move_tensor_to_device,
    normalize_device,
    packaged_algorithm_dir,
    packaged_weights_dir,
    prepend_sys_path,
    require_path,
)


@register_watermark
class RAWatermark(BaseWatermark):
    name = "rawatermark"
    description = "RAWatermark zero-bit detector using packaged wm0 weights."

    def __init__(
        self,
        repo_dir: str | Path | None = None,
        weights_dir: str | Path | None = None,
        wm_index: int = 0,
        **params: Any,
    ) -> None:
        super().__init__(
            repo_dir=str(repo_dir) if repo_dir is not None else None,
            weights_dir=str(weights_dir) if weights_dir is not None else None,
            wm_index=wm_index,
            **params,
        )
        self.repo_dir = require_path(repo_dir or packaged_algorithm_dir("rawatermark"), "RAWatermark repo_dir")
        self.weights_dir = require_path(weights_dir or packaged_weights_dir("rawatermark"), "RAWatermark weights_dir")
        self.wm_index = int(wm_index)
        self._loaded = False
        self._loaded_device = None
        self._torch = None
        self._tf = None
        self._to_pil_image = None
        self._model = None

    def _load(self, device_name: str) -> None:
        device_name = normalize_device(device_name)
        if self._loaded and self._loaded_device == device_name:
            return

        with prepend_sys_path(self.repo_dir, ["scripts"]):
            import torch
            import torchvision.models as tv_models
            import torchvision.transforms as transforms
            from torchvision.transforms.functional import to_pil_image

            original_resnet18 = tv_models.resnet18

            def resnet18_without_download(*model_args, **kwargs):
                kwargs.pop("pretrained", None)
                kwargs.setdefault("weights", None)
                return original_resnet18(*model_args, **kwargs)

            tv_models.resnet18 = resnet18_without_download
            try:
                from scripts import raw as raw_module

                device = torch.device(device_name)
                model = raw_module.RAWatermark(device=device, wm_index=self.wm_index, save_dir=str(self.weights_dir))
                model.classifier.eval()
            finally:
                tv_models.resnet18 = original_resnet18

        self._torch = torch
        self._tf = transforms
        self._to_pil_image = to_pil_image
        self._model = model
        self._loaded = True
        self._loaded_device = device_name

    def _image_tensor_cpu(self, input_path: Path):
        assert self._tf is not None
        transform = self._tf.Compose([self._tf.Resize((512, 512)), self._tf.ToTensor()])
        return transform(Image.open(input_path).convert("RGB"))

    def _image_tensor(self, input_path: Path):
        assert self._model is not None
        return move_tensor_to_device(
            self._image_tensor_cpu(input_path).unsqueeze(0),
            self._model.spa_watermark.device,
        )

    def embed_batch_impl(
        self,
        jobs: list[tuple[Path, Path, WatermarkContext]],
    ) -> list[Mapping[str, Any]]:
        if not jobs:
            return []
        self._load(jobs[0][2].device)
        assert self._torch is not None
        assert self._to_pil_image is not None
        assert self._model is not None

        images = move_tensor_to_device(
            self._torch.stack([self._image_tensor_cpu(input_path) for input_path, _output_path, _context in jobs], dim=0),
            self._model.spa_watermark.device,
        )
        with self._torch.no_grad():
            watermarked = self._model.encode_img(images)
        for index, (_input_path, output_path, _context) in enumerate(jobs):
            save_png_image(self._to_pil_image(watermarked[index].detach().cpu().clamp(0, 1)), output_path)
        return [
            {
                "payload_type": "zero-bit",
                "wm_index": self.wm_index,
                "weights_dir": str(self.weights_dir),
                "image_size": [512, 512],
            }
            for _job in jobs
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

        images = move_tensor_to_device(
            self._torch.stack([self._image_tensor_cpu(input_path) for input_path, _context in jobs], dim=0),
            self._model.spa_watermark.device,
        )
        with self._torch.no_grad():
            pred, _ = self._model.classifier(images)
            probs = self._torch.softmax(pred, dim=1)[:, 1].detach().cpu().tolist()
        return [
            {
                "detection_score": float(prob),
                "payload_type": "zero-bit",
                "wm_index": self.wm_index,
                "weights_dir": str(self.weights_dir),
            }
            for prob in probs
        ]

    def embed_impl(self, input_path: Path, output_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._torch is not None
        assert self._to_pil_image is not None
        assert self._model is not None

        image = self._image_tensor(input_path)
        with self._torch.no_grad():
            watermarked = self._model.encode_img(image)
        save_png_image(self._to_pil_image(watermarked[0].detach().cpu().clamp(0, 1)), output_path)
        return {
            "payload_type": "zero-bit",
            "wm_index": self.wm_index,
            "weights_dir": str(self.weights_dir),
            "image_size": [512, 512],
        }

    def extract_impl(self, input_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._torch is not None
        assert self._model is not None

        image = self._image_tensor(input_path)
        with self._torch.no_grad():
            pred, _ = self._model.classifier(image)
            prob = float(self._torch.softmax(pred, dim=1)[:, 1].item())
        return {
            "detection_score": prob,
            "payload_type": "zero-bit",
            "wm_index": self.wm_index,
            "weights_dir": str(self.weights_dir),
        }
