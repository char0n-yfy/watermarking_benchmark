from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from evaluator.watermarking.base import BaseWatermark, WatermarkContext
from evaluator.watermarking.registry import register_watermark
from evaluator.watermarking.utils import bit_accuracy, bits_from_message, bits_to_string


def _dct_matrix(n: int = 8):
    import numpy as np

    mat = np.zeros((n, n), dtype=np.float64)
    factor = np.pi / (2.0 * n)
    for k in range(n):
        alpha = np.sqrt(1.0 / n) if k == 0 else np.sqrt(2.0 / n)
        for i in range(n):
            mat[k, i] = alpha * np.cos((2 * i + 1) * k * factor)
    return mat


class _TraditionalWatermark(BaseWatermark):
    payload_bits: int

    def __init__(self, payload_bits: int = 56, alpha: float = 18.0, **params: Any) -> None:
        super().__init__(payload_bits=payload_bits, alpha=alpha, **params)
        self.payload_bits = int(payload_bits)
        self.alpha = float(alpha)

    def _bits(self, context: WatermarkContext) -> list[int]:
        return bits_from_message(context.message, self.payload_bits, seed=context.seed)

    def _metadata(self, bits: list[int], decoded_bits: list[int] | None = None) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "bits": bits_to_string(bits),
            "payload_bits": len(bits),
            "checkpoint_file": None,
        }
        if decoded_bits is not None:
            metadata["decoded_bits"] = bits_to_string(decoded_bits)
            metadata["bit_accuracy"] = bit_accuracy(bits, decoded_bits)
        return metadata


@register_watermark
class TraditionalDctWatermark(_TraditionalWatermark):
    name = "traditional-dct"
    description = "Internal block-DCT QIM image watermark; no neural weights."

    def embed_impl(self, input_path: Path, output_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        import numpy as np

        bits = self._bits(context)
        img = Image.open(input_path).convert("YCbCr")
        y, cb, cr = img.split()
        y_arr = np.asarray(y, dtype=np.float64)
        blocks_y, blocks_x = y_arr.shape[0] // 8, y_arr.shape[1] // 8
        if len(bits) > blocks_y * blocks_x:
            raise ValueError("Message exceeds DCT capacity")

        c = _dct_matrix(8)
        ct = c.T
        coeff_pos = (4, 3)
        for flat_block, bit in enumerate(bits):
            by = (flat_block // blocks_x) * 8
            bx = (flat_block % blocks_x) * 8
            block = y_arr[by : by + 8, bx : bx + 8] - 128.0
            dct = c @ block @ ct
            q = int(np.round(dct[coeff_pos] / self.alpha))
            if q % 2 != int(bit):
                q += 1 if q >= 0 else -1
            dct[coeff_pos] = q * self.alpha
            y_arr[by : by + 8, bx : bx + 8] = ct @ dct @ c + 128.0

        y_out = Image.fromarray(np.clip(y_arr, 0, 255).astype(np.uint8), mode="L")
        Image.merge("YCbCr", (y_out, cb, cr)).convert("RGB").save(output_path)
        return self._metadata(bits)

    def extract_impl(self, input_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        import numpy as np

        expected = self._bits(context)
        img = Image.open(input_path).convert("YCbCr")
        y_arr = np.asarray(img.split()[0], dtype=np.float64)
        blocks_x = y_arr.shape[1] // 8
        c = _dct_matrix(8)
        ct = c.T
        coeff_pos = (4, 3)
        decoded: list[int] = []
        for flat_block in range(self.payload_bits):
            by = (flat_block // blocks_x) * 8
            bx = (flat_block % blocks_x) * 8
            block = y_arr[by : by + 8, bx : bx + 8] - 128.0
            dct = c @ block @ ct
            decoded.append(int(np.round(dct[coeff_pos] / self.alpha)) % 2)
        return self._metadata(expected, decoded)


@register_watermark
class TraditionalSpreadDctWatermark(_TraditionalWatermark):
    name = "traditional-spread-dct"
    description = "Internal two-coefficient spread-spectrum DCT image watermark."

    def embed_impl(self, input_path: Path, output_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        import numpy as np

        bits = self._bits(context)
        img = Image.open(input_path).convert("YCbCr")
        y, cb, cr = img.split()
        y_arr = np.asarray(y, dtype=np.float64)
        blocks_y, blocks_x = y_arr.shape[0] // 8, y_arr.shape[1] // 8
        if len(bits) > blocks_y * blocks_x:
            raise ValueError("Message exceeds DCT capacity")

        c = _dct_matrix(8)
        ct = c.T
        pos_a, pos_b = (3, 4), (4, 3)
        for flat_block, bit in enumerate(bits):
            by = (flat_block // blocks_x) * 8
            bx = (flat_block % blocks_x) * 8
            block = y_arr[by : by + 8, bx : bx + 8] - 128.0
            dct = c @ block @ ct
            mean = (dct[pos_a] + dct[pos_b]) / 2.0
            sign = 1.0 if int(bit) else -1.0
            dct[pos_a] = mean + sign * self.alpha / 2.0
            dct[pos_b] = mean - sign * self.alpha / 2.0
            y_arr[by : by + 8, bx : bx + 8] = ct @ dct @ c + 128.0

        y_out = Image.fromarray(np.clip(y_arr, 0, 255).astype(np.uint8), mode="L")
        Image.merge("YCbCr", (y_out, cb, cr)).convert("RGB").save(output_path)
        return self._metadata(bits)

    def extract_impl(self, input_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        import numpy as np

        expected = self._bits(context)
        img = Image.open(input_path).convert("YCbCr")
        y_arr = np.asarray(img.split()[0], dtype=np.float64)
        blocks_x = y_arr.shape[1] // 8
        c = _dct_matrix(8)
        ct = c.T
        decoded: list[int] = []
        for flat_block in range(self.payload_bits):
            by = (flat_block // blocks_x) * 8
            bx = (flat_block % blocks_x) * 8
            block = y_arr[by : by + 8, bx : bx + 8] - 128.0
            dct = c @ block @ ct
            decoded.append(1 if dct[3, 4] > dct[4, 3] else 0)
        return self._metadata(expected, decoded)


@register_watermark
class TraditionalLsbWatermark(_TraditionalWatermark):
    name = "traditional-lsb"
    description = "Internal blue-channel LSB image watermark; no neural weights."

    def embed_impl(self, input_path: Path, output_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        import numpy as np

        bits = np.asarray(self._bits(context), dtype=np.uint8)
        arr = np.asarray(Image.open(input_path).convert("RGB"), dtype=np.uint8).copy()
        blue = arr[:, :, 2].copy()
        flat = blue.reshape(-1)
        if len(bits) > len(flat):
            raise ValueError("Message exceeds LSB capacity")
        flat[: len(bits)] = (flat[: len(bits)] & 0xFE) | bits
        arr[:, :, 2] = blue
        Image.fromarray(arr, "RGB").save(output_path)
        return self._metadata(bits.astype(int).tolist())

    def extract_impl(self, input_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        import numpy as np

        expected = self._bits(context)
        arr = np.asarray(Image.open(input_path).convert("RGB"), dtype=np.uint8)
        decoded = (arr[:, :, 2].reshape(-1)[: self.payload_bits] & 1).astype(int).tolist()
        return self._metadata(expected, decoded)


@register_watermark
class TraditionalHaarWatermark(_TraditionalWatermark):
    name = "traditional-haar"
    description = "Internal Haar/LL-band QIM image watermark; no neural weights."

    def embed_impl(self, input_path: Path, output_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        import numpy as np

        bits = self._bits(context)
        img = Image.open(input_path).convert("YCbCr")
        y, cb, cr = img.split()
        arr = np.asarray(y, dtype=np.float64)
        h2, w2 = arr.shape[0] - arr.shape[0] % 2, arr.shape[1] - arr.shape[1] % 2
        if len(bits) > (h2 // 2) * (w2 // 2):
            raise ValueError("Message exceeds Haar capacity")

        work = arr[:h2, :w2].copy()
        idx = 0
        for by in range(0, h2, 2):
            for bx in range(0, w2, 2):
                if idx >= len(bits):
                    break
                block = work[by : by + 2, bx : bx + 2]
                avg = float(block.mean())
                q = int(np.round(avg / self.alpha))
                if q % 2 != int(bits[idx]):
                    q += 1 if q >= 0 else -1
                work[by : by + 2, bx : bx + 2] = block + (q * self.alpha - avg)
                idx += 1
            if idx >= len(bits):
                break
        arr[:h2, :w2] = work
        y_out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
        Image.merge("YCbCr", (y_out, cb, cr)).convert("RGB").save(output_path)
        return self._metadata(bits)

    def extract_impl(self, input_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        import numpy as np

        expected = self._bits(context)
        arr = np.asarray(Image.open(input_path).convert("YCbCr").split()[0], dtype=np.float64)
        h2, w2 = arr.shape[0] - arr.shape[0] % 2, arr.shape[1] - arr.shape[1] % 2
        decoded: list[int] = []
        for by in range(0, h2, 2):
            for bx in range(0, w2, 2):
                if len(decoded) >= self.payload_bits:
                    break
                avg = float(arr[by : by + 2, bx : bx + 2].mean())
                decoded.append(int(np.round(avg / self.alpha)) % 2)
            if len(decoded) >= self.payload_bits:
                break
        return self._metadata(expected, decoded)
