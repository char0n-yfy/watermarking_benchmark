from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from evaluator.watermarking.base import BaseWatermark, WatermarkContext
from evaluator.watermarking.registry import register_watermark
from evaluator.watermarking.utils import (
    packaged_algorithm_dir,
    prepend_sys_path,
    require_path,
)


class _InvisibleWatermarkBase(BaseWatermark):
    description = "ShieldMnt invisible-watermark wrapper; no neural weights."
    algorithm = ""
    algorithm_dir = ""

    def __init__(
        self,
        repo_dir: str | Path | None = None,
        payload_bits: int = 56,
        **params: Any,
    ) -> None:
        requested_algorithm = params.pop("algorithm", None)
        if requested_algorithm is not None and self._canonical_algorithm(str(requested_algorithm)) != self.algorithm:
            raise ValueError(f"{self.name} is fixed to {self.algorithm}")
        super().__init__(
            repo_dir=str(repo_dir) if repo_dir is not None else None,
            algorithm=self.algorithm,
            payload_bits=payload_bits,
            **params,
        )
        self.repo_dir = require_path(
            repo_dir or packaged_algorithm_dir(self.algorithm_dir),
            f"{self.name} repo_dir",
        )
        self.payload_bits = int(payload_bits)

    @staticmethod
    def _canonical_algorithm(value: str) -> str:
        key = value.replace("-", "").replace("_", "").lower()
        mapping = {
            "dwtdct": "dwtDct",
            "dwtdctsvd": "dwtDctSvd",
        }
        if key not in mapping:
            raise ValueError(f"Unsupported invisible-watermark algorithm: {value}")
        return mapping[key]

    def _message_bytes(self, context: WatermarkContext) -> bytes:
        max_bytes = max(1, self.payload_bits // 8)
        return (context.message or "test001").encode("utf-8")[:max_bytes]

    def embed_impl(self, input_path: Path, output_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        import cv2
        import numpy as np

        with prepend_sys_path(self.repo_dir, ["imwatermark"]):
            from imwatermark import WatermarkEncoder

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
            "message": payload.decode("utf-8", errors="ignore"),
            "payload_bits": len(payload) * 8,
            "algorithm": self.algorithm,
            "checkpoint_file": None,
        }

    def extract_impl(self, input_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        import cv2
        import numpy as np

        with prepend_sys_path(self.repo_dir, ["imwatermark"]):
            from imwatermark import WatermarkDecoder

            data = np.fromfile(str(input_path), dtype=np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Cannot read image: {input_path}")
            decoder = WatermarkDecoder("bytes", self.payload_bits)
            decoded = decoder.decode(bgr, self.algorithm).decode("utf-8", errors="ignore").rstrip("\0")

        expected = self._message_bytes(context).decode("utf-8", errors="ignore").rstrip("\0") if context.message else None
        return {
            "message": decoded,
            "expected_message": expected,
            "match": None if expected is None else decoded == expected,
            "payload_bits": self.payload_bits,
            "algorithm": self.algorithm,
            "checkpoint_file": None,
        }


@register_watermark
class InvisibleWatermarkDwtDct(_InvisibleWatermarkBase):
    name = "invisible-watermark-dwtdct"
    description = "ShieldMnt invisible-watermark using the DWT-DCT algorithm."
    algorithm = "dwtDct"
    algorithm_dir = "invisible_watermark_dwtdct"


@register_watermark
class InvisibleWatermarkDwtDctSvd(_InvisibleWatermarkBase):
    name = "invisible-watermark-dwtdctsvd"
    description = "ShieldMnt invisible-watermark using the DWT-DCT-SVD algorithm."
    algorithm = "dwtDctSvd"
    algorithm_dir = "invisible_watermark_dwtdctsvd"
