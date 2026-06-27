from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from evaluator.watermarking.base import BaseWatermark, WatermarkContext
from evaluator.watermarking.registry import register_watermark
from evaluator.watermarking.utils import (
    packaged_algorithm_dir,
    packaged_weights_dir,
    prepend_sys_path,
    require_path,
)


class _InvisibleWatermarkBase(BaseWatermark):
    description = "ShieldMnt invisible-watermark wrapper."
    algorithm = ""
    algorithm_dir = ""
    weight_dir_name: str | None = None
    default_payload_bits = 56

    def __init__(
        self,
        repo_dir: str | Path | None = None,
        weights_dir: str | Path | None = None,
        payload_bits: int | None = None,
        **params: Any,
    ) -> None:
        requested_algorithm = params.pop("algorithm", None)
        if requested_algorithm is not None and self._canonical_algorithm(str(requested_algorithm)) != self.algorithm:
            raise ValueError(f"{self.name} is fixed to {self.algorithm}")

        payload_bits = int(payload_bits or self.default_payload_bits)
        if self.algorithm == "rivaGan" and payload_bits != 32:
            raise ValueError("rivaGan only supports a 32-bit payload")

        super().__init__(
            repo_dir=str(repo_dir) if repo_dir is not None else None,
            weights_dir=str(weights_dir) if weights_dir is not None else None,
            algorithm=self.algorithm,
            payload_bits=payload_bits,
            **params,
        )
        self.repo_dir = require_path(
            repo_dir or packaged_algorithm_dir(self.algorithm_dir),
            f"{self.name} repo_dir",
        )
        self.payload_bits = payload_bits
        self.weights_dir: Path | None = None
        self.encoder_path: Path | None = None
        self.decoder_path: Path | None = None
        self._model_loaded = False
        if self.weight_dir_name is not None:
            self.weights_dir = require_path(
                weights_dir or packaged_weights_dir(self.weight_dir_name),
                f"{self.name} weights_dir",
            )
            self.encoder_path = require_path(self.weights_dir / "rivagan_encoder.onnx", "rivaGan encoder_path")
            self.decoder_path = require_path(self.weights_dir / "rivagan_decoder.onnx", "rivaGan decoder_path")

    def _purge_modules(self) -> list[str]:
        if self.algorithm == "rivaGan" and self._model_loaded:
            return []
        return ["imwatermark"]

    @staticmethod
    def _canonical_algorithm(value: str) -> str:
        key = value.replace("-", "").replace("_", "").lower()
        mapping = {
            "dwtdct": "dwtDct",
            "dwtdctsvd": "dwtDctSvd",
            "rivagan": "rivaGan",
        }
        if key not in mapping:
            raise ValueError(f"Unsupported invisible-watermark algorithm: {value}")
        return mapping[key]

    def _message_bytes(self, context: WatermarkContext) -> bytes:
        if self.algorithm == "rivaGan":
            return (context.message or "qing").encode("utf-8")[:4].ljust(4, b"\0")
        max_bytes = max(1, self.payload_bits // 8)
        return (context.message or "test001").encode("utf-8")[:max_bytes]

    @contextmanager
    def _runtime_env(self) -> Iterator[None]:
        if self.weights_dir is None:
            yield
            return

        key = "IMWATERMARK_RIVAGAN_MODEL_DIR"
        old_value = os.environ.get(key)
        os.environ[key] = str(self.weights_dir)
        try:
            yield
        finally:
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value

    def embed_impl(self, input_path: Path, output_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        import cv2
        import numpy as np

        with self._runtime_env(), prepend_sys_path(self.repo_dir, self._purge_modules()):
            from imwatermark import WatermarkEncoder

            if self.algorithm == "rivaGan":
                WatermarkEncoder.loadModel()
                self._model_loaded = True

            data = np.fromfile(str(input_path), dtype=np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Cannot read image: {input_path}")
            encoder = WatermarkEncoder()
            payload = self._message_bytes(context)
            encoder.set_watermark("bytes", payload)
            encoded = encoder.encode(bgr, self.algorithm)
            ok, buffer = cv2.imencode(output_path.suffix or ".png", encoded)
            if not ok:
                raise RuntimeError(f"cv2.imencode failed for {output_path}")
            output_path.write_bytes(buffer.tobytes())

        return {
            "message": payload.decode("utf-8", errors="ignore").rstrip("\0"),
            "payload_bits": len(payload) * 8,
            "algorithm": self.algorithm,
            "weights_dir": None if self.weights_dir is None else str(self.weights_dir),
            "encoder_path": None if self.encoder_path is None else str(self.encoder_path),
            "decoder_path": None if self.decoder_path is None else str(self.decoder_path),
        }

    def extract_impl(self, input_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        import cv2
        import numpy as np

        with self._runtime_env(), prepend_sys_path(self.repo_dir, self._purge_modules()):
            from imwatermark import WatermarkDecoder

            if self.algorithm == "rivaGan":
                WatermarkDecoder.loadModel()
                self._model_loaded = True

            data = np.fromfile(str(input_path), dtype=np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Cannot read image: {input_path}")
            decode_bits = len(self._message_bytes(context)) * 8 if context.message else self.payload_bits
            decoder = WatermarkDecoder("bytes", decode_bits)
            decoded = decoder.decode(bgr, self.algorithm).decode("utf-8", errors="ignore").rstrip("\0")

        expected = self._message_bytes(context).decode("utf-8", errors="ignore").rstrip("\0") if context.message else None
        return {
            "message": decoded,
            "expected_message": expected,
            "match": None if expected is None else decoded == expected,
            "payload_bits": decode_bits,
            "algorithm": self.algorithm,
            "weights_dir": None if self.weights_dir is None else str(self.weights_dir),
            "encoder_path": None if self.encoder_path is None else str(self.encoder_path),
            "decoder_path": None if self.decoder_path is None else str(self.decoder_path),
        }


@register_watermark
class InvisibleWatermarkDwtDct(_InvisibleWatermarkBase):
    name = "invisible-watermark-dwtdct"
    description = "ShieldMnt invisible-watermark using the DWT-DCT algorithm."
    algorithm = "dwtDct"
    algorithm_dir = "dwtDct"


@register_watermark
class InvisibleWatermarkDwtDctSvd(_InvisibleWatermarkBase):
    name = "invisible-watermark-dwtdctsvd"
    description = "ShieldMnt invisible-watermark using the DWT-DCT-SVD algorithm."
    algorithm = "dwtDctSvd"
    algorithm_dir = "dwtDctSvd"


@register_watermark
class InvisibleWatermarkRivaGan(_InvisibleWatermarkBase):
    name = "invisible-watermark-rivagan"
    description = "ShieldMnt invisible-watermark using the RivaGAN algorithm and packaged ONNX weights."
    algorithm = "rivaGan"
    algorithm_dir = "rivaGan"
    weight_dir_name = "rivaGan"
    default_payload_bits = 32
