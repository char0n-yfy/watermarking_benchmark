from __future__ import annotations

import argparse
import json
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
class WAMWatermark(BaseWatermark):
    name = "wam"
    description = "Watermark Anything localized 32-bit watermark wrapper using packaged MIT weights."

    def __init__(
        self,
        repo_dir: str | Path | None = None,
        weights_dir: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        params_path: str | Path | None = None,
        payload_bits: int = 32,
        **params: Any,
    ) -> None:
        super().__init__(
            repo_dir=str(repo_dir) if repo_dir is not None else None,
            weights_dir=str(weights_dir) if weights_dir is not None else None,
            checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
            params_path=str(params_path) if params_path is not None else None,
            payload_bits=payload_bits,
            **params,
        )
        self.repo_dir = require_path(repo_dir or packaged_algorithm_dir("wam"), "WAM repo_dir")
        self.weights_dir = require_path(weights_dir or packaged_weights_dir("wam"), "WAM weights_dir")
        self.checkpoint_path = require_path(
            checkpoint_path or self.weights_dir / "wam_mit.pth",
            "WAM checkpoint",
        )
        self.params_path = require_path(
            params_path or self.weights_dir / "params.json",
            "WAM params.json",
        )
        self.payload_bits = int(payload_bits)
        if self.payload_bits != 32:
            raise ValueError("The packaged WAM model supports a 32-bit payload")
        self._loaded = False
        self._loaded_device = None
        self._torch = None
        self._F = None
        self._save_image = None
        self._default_transform = None
        self._unnormalize_img = None
        self._msg_predict_inference = None
        self._model = None

    def _load(self, device_name: str) -> None:
        device_name = normalize_device(device_name)
        if self._loaded and self._loaded_device == device_name:
            return

        with prepend_sys_path(self.repo_dir, ["watermark_anything"]):
            import omegaconf
            import torch
            import torch.nn.functional as F
            from torchvision.utils import save_image
            from watermark_anything.augmentation.augmenter import Augmenter
            from watermark_anything.data.metrics import msg_predict_inference
            from watermark_anything.data.transforms import default_transform, normalize_img, unnormalize_img
            from watermark_anything.models import Wam, build_embedder, build_extractor
            from watermark_anything.modules.jnd import JND

            params = json.loads(self.params_path.read_text(encoding="utf-8"))
            args = argparse.Namespace(**params)

            def cfg_path(value: str) -> Path:
                path = Path(value)
                return path if path.is_absolute() else self.repo_dir / path

            embedder_cfg = omegaconf.OmegaConf.load(cfg_path(args.embedder_config))
            extractor_cfg = omegaconf.OmegaConf.load(cfg_path(args.extractor_config))
            augmenter_cfg = omegaconf.OmegaConf.load(cfg_path(args.augmentation_config))
            attenuation_cfg = omegaconf.OmegaConf.load(cfg_path(args.attenuation_config))

            embedder = build_embedder(args.embedder_model, embedder_cfg[args.embedder_model], args.nbits)
            extractor = build_extractor(extractor_cfg.model, extractor_cfg[args.extractor_model], args.img_size, args.nbits)
            augmenter = Augmenter(**augmenter_cfg)
            try:
                attenuation = JND(
                    **attenuation_cfg[args.attenuation],
                    preprocess=unnormalize_img,
                    postprocess=normalize_img,
                )
            except Exception:
                attenuation = None

            model = Wam(embedder, extractor, augmenter, attenuation, args.scaling_w, args.scaling_i)
            model.load_state_dict(torch.load(self.checkpoint_path, map_location="cpu"))
            model = model.to(torch.device(device_name)).eval()

        self._torch = torch
        self._F = F
        self._save_image = save_image
        self._default_transform = default_transform
        self._unnormalize_img = unnormalize_img
        self._msg_predict_inference = msg_predict_inference
        self._model = model
        self._loaded = True
        self._loaded_device = device_name

    def _message_tensor(self, context: WatermarkContext):
        assert self._torch is not None
        assert self._model is not None
        bits = bits_from_message(context.message, self.payload_bits, seed=context.seed)
        tensor = self._torch.tensor(bits, dtype=self._torch.float32, device=next(self._model.parameters()).device).unsqueeze(0)
        return bits, tensor

    def embed_impl(self, input_path: Path, output_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._torch is not None
        assert self._default_transform is not None
        assert self._unnormalize_img is not None
        assert self._save_image is not None
        assert self._model is not None

        bits, msg = self._message_tensor(context)
        tensor = self._default_transform(Image.open(input_path).convert("RGB")).unsqueeze(0).to(msg.device)
        with self._torch.no_grad():
            outputs = self._model.embed(tensor, msg)
        self._save_image(self._unnormalize_img(outputs["imgs_w"]), output_path)
        return {
            "bits": bits_to_string(bits),
            "payload_bits": self.payload_bits,
            "checkpoint_file": str(self.checkpoint_path),
            "weights_dir": str(self.weights_dir),
        }

    def extract_impl(self, input_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        self._load(context.device)
        assert self._torch is not None
        assert self._F is not None
        assert self._default_transform is not None
        assert self._msg_predict_inference is not None
        assert self._model is not None

        tensor = self._default_transform(Image.open(input_path).convert("RGB")).unsqueeze(0).to(next(self._model.parameters()).device)
        with self._torch.no_grad():
            preds = self._model.detect(tensor)["preds"]
        mask_preds = self._F.sigmoid(preds[:, 0, :, :])
        bit_preds = preds[:, 1:, :, :]
        decoded = self._msg_predict_inference(bit_preds, mask_preds).cpu().float()[0].int().tolist()

        metadata: dict[str, Any] = {
            "bits": bits_to_string(decoded),
            "payload_bits": len(decoded),
            "checkpoint_file": str(self.checkpoint_path),
            "weights_dir": str(self.weights_dir),
        }
        if context.message is not None:
            expected = bits_from_message(context.message, len(decoded), seed=context.seed)
            metadata["expected_bits"] = bits_to_string(expected)
            metadata["bit_accuracy"] = bit_accuracy(expected, decoded)
        return metadata
