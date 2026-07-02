from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Mapping

from evaluator.watermarking.base import BaseWatermark, WatermarkContext
from evaluator.watermarking.registry import register_watermark
from evaluator.watermarking.utils import (
    bit_accuracy,
    bits_to_string,
    packaged_algorithm_dir,
    packaged_weights_dir,
    require_path,
)


JsonDict = dict[str, Any]
_IMWATERMARK_IMPORT_LOCK = threading.RLock()
_IMWATERMARK_MODULES: dict[Path, ModuleType] = {}


def clear_imwatermark_runtime() -> None:
    with _IMWATERMARK_IMPORT_LOCK:
        for module in list(_IMWATERMARK_MODULES.values()):
            try:
                riva_module = importlib.import_module(f"{module.__name__}.rivaGan")
                RivaWatermark = getattr(riva_module, "RivaWatermark")
                RivaWatermark.encoder = None
                RivaWatermark.decoder = None
                RivaWatermark.onnx_providers = None
            except Exception:
                pass


def _load_imwatermark_package(repo_dir: Path) -> ModuleType:
    package_dir = repo_dir.resolve() / "imwatermark"
    init_path = package_dir / "__init__.py"
    if not init_path.exists():
        raise FileNotFoundError(f"imwatermark package does not exist: {package_dir}")

    with _IMWATERMARK_IMPORT_LOCK:
        cached = _IMWATERMARK_MODULES.get(package_dir)
        if cached is not None:
            return cached

        digest = hashlib.sha1(str(package_dir).encode("utf-8")).hexdigest()[:12]
        package_name = f"_wmbench_imwatermark_{digest}"
        spec = importlib.util.spec_from_file_location(
            package_name,
            init_path,
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load imwatermark package from {init_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(package_name, None)
            raise
        _IMWATERMARK_MODULES[package_dir] = module
        return module


def _bytes_to_bits(payload: bytes, nbits: int) -> list[int]:
    bits: list[int] = []
    for byte in payload:
        bits.extend((byte >> shift) & 1 for shift in range(7, -1, -1))
    return bits[:nbits]


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

    def _imwatermark_module(self) -> ModuleType:
        return _load_imwatermark_package(self.repo_dir)

    def _riva_watermark_class(self) -> Any:
        module = self._imwatermark_module()
        riva_module = importlib.import_module(f"{module.__name__}.rivaGan")
        return getattr(riva_module, "RivaWatermark")

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

    def release(self) -> None:
        if self.algorithm == "rivaGan":
            clear_imwatermark_runtime()
        self._model_loaded = False

    def embed_impl(self, input_path: Path, output_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        import cv2
        import numpy as np

        riva_provider_info = None
        with self._runtime_env():
            imwatermark = self._imwatermark_module()
            WatermarkEncoder = getattr(imwatermark, "WatermarkEncoder")

            if self.algorithm == "rivaGan":
                RivaWatermark = self._riva_watermark_class()

                WatermarkEncoder.loadModel()
                self._model_loaded = True
                riva_provider_info = getattr(RivaWatermark, "onnx_providers", None)

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
            "onnx_providers": riva_provider_info,
        }

    def extract_impl(self, input_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        import cv2
        import numpy as np

        expected_payload = self._message_bytes(context) if context.message else None
        decode_bits = len(expected_payload) * 8 if expected_payload is not None else self.payload_bits
        decode_bytes = max(1, (decode_bits + 7) // 8)

        riva_provider_info = None
        with self._runtime_env():
            imwatermark = self._imwatermark_module()
            WatermarkDecoder = getattr(imwatermark, "WatermarkDecoder")

            if self.algorithm == "rivaGan":
                RivaWatermark = self._riva_watermark_class()

                WatermarkDecoder.loadModel()
                self._model_loaded = True
                riva_provider_info = getattr(RivaWatermark, "onnx_providers", None)

            data = np.fromfile(str(input_path), dtype=np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Cannot read image: {input_path}")
            decoder = WatermarkDecoder("bytes", decode_bits)
            decoded_payload = decoder.decode(bgr, self.algorithm)
            if not isinstance(decoded_payload, bytes):
                decoded_payload = bytes(decoded_payload)
            decoded_payload = decoded_payload[:decode_bytes].ljust(decode_bytes, b"\0")

        decoded = decoded_payload.decode("utf-8", errors="ignore").rstrip("\0")
        decoded_bits = _bytes_to_bits(decoded_payload, decode_bits)
        metadata: dict[str, Any] = {
            "message": decoded,
            "bits": bits_to_string(decoded_bits),
            "decoded_bits": bits_to_string(decoded_bits),
            "payload_bits": decode_bits,
            "algorithm": self.algorithm,
            "weights_dir": None if self.weights_dir is None else str(self.weights_dir),
            "encoder_path": None if self.encoder_path is None else str(self.encoder_path),
            "decoder_path": None if self.decoder_path is None else str(self.decoder_path),
            "onnx_providers": riva_provider_info,
        }
        if expected_payload is not None:
            expected_payload = expected_payload[:decode_bytes].ljust(decode_bytes, b"\0")
            expected = expected_payload.decode("utf-8", errors="ignore").rstrip("\0")
            expected_bits = _bytes_to_bits(expected_payload, decode_bits)
            metadata["expected_message"] = expected
            metadata["expected_bits"] = bits_to_string(expected_bits)
            metadata["match"] = decoded == expected
            metadata["bit_accuracy"] = bit_accuracy(expected_bits, decoded_bits)
        else:
            metadata["expected_message"] = None
            metadata["match"] = None
        return metadata


@register_watermark
class InvisibleWatermarkDwtDct(_InvisibleWatermarkBase):
    name = "invisible-watermark-dwtdct"
    description = "ShieldMnt invisible-watermark using the DWT-DCT algorithm."
    algorithm = "dwtDct"
    algorithm_dir = "dwtDct"
    thread_safe_parallel = True


@register_watermark
class InvisibleWatermarkDwtDctSvd(_InvisibleWatermarkBase):
    name = "invisible-watermark-dwtdctsvd"
    description = "ShieldMnt invisible-watermark using the DWT-DCT-SVD algorithm."
    algorithm = "dwtDctSvd"
    algorithm_dir = "dwtDctSvd"
    thread_safe_parallel = True


@register_watermark
class InvisibleWatermarkRivaGan(_InvisibleWatermarkBase):
    name = "invisible-watermark-rivagan"
    description = "ShieldMnt invisible-watermark using the RivaGAN algorithm and packaged ONNX weights."
    algorithm = "rivaGan"
    algorithm_dir = "rivaGan"
    weight_dir_name = "rivaGan"
    default_payload_bits = 32
    _ONNX_ALLOCATION_MARKERS = (
        "AllocateRawInternal",
        "BFCArena",
        "Failed to allocate memory",
        "CUDA out of memory",
        "CUBLAS_STATUS_ALLOC_FAILED",
    )

    @staticmethod
    def _is_onnx_allocation_error(exc: BaseException) -> bool:
        text = f"{type(exc).__name__}: {exc}"
        return any(marker in text for marker in InvisibleWatermarkRivaGan._ONNX_ALLOCATION_MARKERS)

    @contextmanager
    def _onnx_provider_override(self, providers: str) -> Iterator[None]:
        key = "IMWATERMARK_RIVAGAN_ONNX_PROVIDERS"
        old_value = os.environ.get(key)
        os.environ[key] = providers
        try:
            yield
        finally:
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value

    def _cleanup_onnx_runtime(self) -> None:
        clear_imwatermark_runtime()
        self._model_loaded = False
        try:
            from evaluator.runtime_cleanup import torch_cleanup

            torch_cleanup(reset_peak=False)
        except Exception:
            pass

    def _reload_rivagan_model(self) -> Any:
        self._cleanup_onnx_runtime()
        RivaWatermark = self._ensure_model_loaded()
        if RivaWatermark.decoder is None:
            raise RuntimeError("RivaGAN decoder session is not loaded")
        return RivaWatermark

    @staticmethod
    def _short_error(exc: BaseException) -> str:
        return f"{type(exc).__name__}: {exc}"[:500]

    @staticmethod
    def _frame_input(frames: list[Any]) -> Any:
        import numpy as np

        batch_frames = np.stack(frames).astype(np.float32) / 127.5 - 1.0
        return batch_frames.transpose(0, 3, 1, 2)[:, :, None, :, :]

    def _decode_frame_input_with_retry(self, RivaWatermark: Any, frame_input: Any) -> tuple[Any, Mapping[str, Any] | None, list[JsonDict]]:
        retry_events: list[JsonDict] = []
        try:
            return RivaWatermark.decoder.run(None, {"frame": frame_input}), getattr(RivaWatermark, "onnx_providers", None), retry_events
        except Exception as exc:
            if not self._is_onnx_allocation_error(exc):
                raise
            retry_events.append(
                {
                    "providerPolicy": "same_after_cleanup",
                    "reason": self._short_error(exc),
                }
            )

        RivaWatermark = self._reload_rivagan_model()
        try:
            return RivaWatermark.decoder.run(None, {"frame": frame_input}), getattr(RivaWatermark, "onnx_providers", None), retry_events
        except Exception as exc:
            if not self._is_onnx_allocation_error(exc):
                raise
            retry_events.append(
                {
                    "providerPolicy": "cpu_fallback",
                    "reason": self._short_error(exc),
                }
            )

        with self._onnx_provider_override("CPUExecutionProvider"):
            RivaWatermark = self._reload_rivagan_model()
            outputs = RivaWatermark.decoder.run(None, {"frame": frame_input})
            provider_info = getattr(RivaWatermark, "onnx_providers", None)
        self._cleanup_onnx_runtime()
        return outputs, provider_info, retry_events

    def _metadata_from_bits(
        self,
        bits: list[int],
        *,
        decode_bits: int,
        decode_bytes: int,
        expected_payload: bytes | None,
        provider_info: Mapping[str, Any] | None,
        retry_events: list[JsonDict] | None = None,
    ) -> dict[str, Any]:
        import numpy as np

        decoded_payload = bytes(np.packbits(np.asarray(bits, dtype=np.uint8)))[:decode_bytes].ljust(
            decode_bytes,
            b"\0",
        )
        decoded = decoded_payload.decode("utf-8", errors="ignore").rstrip("\0")
        decoded_bits = _bytes_to_bits(decoded_payload, decode_bits)
        metadata: dict[str, Any] = {
            "message": decoded,
            "bits": bits_to_string(decoded_bits),
            "decoded_bits": bits_to_string(decoded_bits),
            "payload_bits": decode_bits,
            "algorithm": self.algorithm,
            "weights_dir": None if self.weights_dir is None else str(self.weights_dir),
            "encoder_path": None if self.encoder_path is None else str(self.encoder_path),
            "decoder_path": None if self.decoder_path is None else str(self.decoder_path),
            "onnx_providers": provider_info,
        }
        if retry_events:
            metadata["onnxRetry"] = retry_events
        if expected_payload is not None:
            expected_payload = expected_payload[:decode_bytes].ljust(decode_bytes, b"\0")
            expected = expected_payload.decode("utf-8", errors="ignore").rstrip("\0")
            expected_bits = _bytes_to_bits(expected_payload, decode_bits)
            metadata["expected_message"] = expected
            metadata["expected_bits"] = bits_to_string(expected_bits)
            metadata["match"] = decoded == expected
            metadata["bit_accuracy"] = bit_accuracy(expected_bits, decoded_bits)
        else:
            metadata["expected_message"] = None
            metadata["match"] = None
        return metadata

    @staticmethod
    def _payload_bits(payload: bytes) -> list[int]:
        return _bytes_to_bits(payload, len(payload) * 8)

    def _ensure_model_loaded(self) -> Any:
        imwatermark = self._imwatermark_module()
        WatermarkEncoder = getattr(imwatermark, "WatermarkEncoder")
        RivaWatermark = self._riva_watermark_class()
        WatermarkEncoder.loadModel()
        self._model_loaded = True
        return RivaWatermark

    def embed_batch_impl(
        self,
        jobs: list[tuple[Path, Path, WatermarkContext]],
    ) -> list[Mapping[str, Any]]:
        import cv2
        import numpy as np

        if not jobs:
            return []

        with self._runtime_env():
            RivaWatermark = self._ensure_model_loaded()
            provider_info = getattr(RivaWatermark, "onnx_providers", None)
            if RivaWatermark.encoder is None:
                raise RuntimeError("RivaGAN encoder session is not loaded")

            frames: list[np.ndarray] = []
            payloads: list[bytes] = []
            groups: dict[tuple[int, int], list[int]] = {}
            for index, (input_path, _output_path, context) in enumerate(jobs):
                data = np.fromfile(str(input_path), dtype=np.uint8)
                bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError(f"Cannot read image: {input_path}")
                frames.append(bgr)
                payload = self._message_bytes(context)
                payloads.append(payload)
                h, w = bgr.shape[:2]
                groups.setdefault((h, w), []).append(index)

            encoded_frames: list[np.ndarray | None] = [None] * len(jobs)
            for indexes in groups.values():
                batch_frames = np.stack([frames[index] for index in indexes]).astype(np.float32) / 127.5 - 1.0
                frame_input = batch_frames.transpose(0, 3, 1, 2)[:, :, None, :, :]
                data_input = np.asarray(
                    [self._payload_bits(payloads[index]) for index in indexes],
                    dtype=np.float32,
                )
                outputs = RivaWatermark.encoder.run(None, {"frame": frame_input, "data": data_input})
                wm_frames = np.clip(outputs[0], -1.0, 1.0)
                wm_frames = ((wm_frames[:, :, 0, :, :].transpose(0, 2, 3, 1) + 1.0) * 127.5).astype(np.uint8)
                for index, encoded in zip(indexes, wm_frames):
                    encoded_frames[index] = encoded

            metadatas: list[Mapping[str, Any]] = []
            for (input_path, output_path, _context), payload, encoded in zip(jobs, payloads, encoded_frames):
                if encoded is None:
                    raise RuntimeError(f"RivaGAN batch encode produced no output for {input_path}")
                ok, buffer = cv2.imencode(output_path.suffix or ".png", encoded)
                if not ok:
                    raise RuntimeError(f"cv2.imencode failed for {output_path}")
                output_path.write_bytes(buffer.tobytes())
                metadatas.append(
                    {
                        "message": payload.decode("utf-8", errors="ignore").rstrip("\0"),
                        "payload_bits": len(payload) * 8,
                        "algorithm": self.algorithm,
                        "weights_dir": None if self.weights_dir is None else str(self.weights_dir),
                        "encoder_path": None if self.encoder_path is None else str(self.encoder_path),
                        "decoder_path": None if self.decoder_path is None else str(self.decoder_path),
                        "onnx_providers": provider_info,
                    }
                )
            return metadatas

    def extract_impl(self, input_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        import cv2
        import numpy as np

        expected_payload = self._message_bytes(context) if context.message else None
        decode_bits = len(expected_payload) * 8 if expected_payload is not None else self.payload_bits
        decode_bytes = max(1, (decode_bits + 7) // 8)

        with self._runtime_env():
            RivaWatermark = self._ensure_model_loaded()
            if RivaWatermark.decoder is None:
                raise RuntimeError("RivaGAN decoder session is not loaded")
            data = np.fromfile(str(input_path), dtype=np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Cannot read image: {input_path}")
            frame_input = self._frame_input([bgr])
            outputs, provider_info, retry_events = self._decode_frame_input_with_retry(RivaWatermark, frame_input)
            decoded = (outputs[0] > 0.52).astype(np.uint8)[0]
            bits = [int(bit) for bit in decoded[:decode_bits]]

        return self._metadata_from_bits(
            bits,
            decode_bits=decode_bits,
            decode_bytes=decode_bytes,
            expected_payload=expected_payload,
            provider_info=provider_info,
            retry_events=retry_events,
        )

    def extract_batch_impl(
        self,
        jobs: list[tuple[Path, WatermarkContext]],
    ) -> list[Mapping[str, Any]]:
        import cv2
        import numpy as np

        if not jobs:
            return []

        with self._runtime_env():
            RivaWatermark = self._ensure_model_loaded()
            provider_info = getattr(RivaWatermark, "onnx_providers", None)
            if RivaWatermark.decoder is None:
                raise RuntimeError("RivaGAN decoder session is not loaded")

            frames: list[np.ndarray] = []
            decode_bits_by_index: list[int] = []
            decode_bytes_by_index: list[int] = []
            expected_payloads: list[bytes | None] = []
            groups: dict[tuple[int, int], list[int]] = {}
            for index, (input_path, context) in enumerate(jobs):
                expected_payload = self._message_bytes(context) if context.message else None
                decode_bits = len(expected_payload) * 8 if expected_payload is not None else self.payload_bits
                decode_bytes = max(1, (decode_bits + 7) // 8)
                data = np.fromfile(str(input_path), dtype=np.uint8)
                bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError(f"Cannot read image: {input_path}")
                frames.append(bgr)
                expected_payloads.append(expected_payload)
                decode_bits_by_index.append(decode_bits)
                decode_bytes_by_index.append(decode_bytes)
                h, w = bgr.shape[:2]
                groups.setdefault((h, w), []).append(index)

            decoded_bits_by_index: list[list[int] | None] = [None] * len(jobs)
            retry_by_index: list[list[JsonDict]] = [[] for _ in jobs]
            provider_info_by_index: list[Mapping[str, Any] | None] = [provider_info for _ in jobs]
            for indexes in groups.values():
                RivaWatermark = self._ensure_model_loaded()
                frame_input = self._frame_input([frames[index] for index in indexes])
                outputs, active_provider_info, retry_events = self._decode_frame_input_with_retry(
                    RivaWatermark,
                    frame_input,
                )
                decoded = (outputs[0] > 0.52).astype(np.uint8)
                for index, bits in zip(indexes, decoded):
                    decoded_bits_by_index[index] = [int(bit) for bit in bits[: decode_bits_by_index[index]]]
                    provider_info_by_index[index] = active_provider_info
                    retry_by_index[index] = list(retry_events)

            metadatas: list[Mapping[str, Any]] = []
            for index, bits in enumerate(decoded_bits_by_index):
                if bits is None:
                    input_path = jobs[index][0]
                    raise RuntimeError(f"RivaGAN batch decode produced no output for {input_path}")
                decode_bits = decode_bits_by_index[index]
                decode_bytes = decode_bytes_by_index[index]
                metadatas.append(
                    self._metadata_from_bits(
                        bits,
                        decode_bits=decode_bits,
                        decode_bytes=decode_bytes,
                        expected_payload=expected_payloads[index],
                        provider_info=provider_info_by_index[index],
                        retry_events=retry_by_index[index],
                    )
                )
            return metadatas
