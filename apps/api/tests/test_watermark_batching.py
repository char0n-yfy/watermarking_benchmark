from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluator.watermarking.base import BaseWatermark, WatermarkContext
from evaluator.watermarking.methods._videoseal_family import VideoSealFamilyWatermark
from evaluator.watermarking.methods.chunkyseal import ChunkySealWatermark
from evaluator.watermarking.methods.cin import CINWatermark
from evaluator.watermarking.methods.dwsf import DWSFWatermark
from evaluator.watermarking.methods.hidden import HiDDeNWatermark
from evaluator.watermarking.methods.invismark import InvisMarkWatermark
from evaluator.watermarking.methods.maskwm import MaskWMD32Watermark
from evaluator.watermarking.methods.mbrs import MBRSWatermark
from evaluator.watermarking.methods.pimog import PIMoGWatermark
from evaluator.watermarking.methods.pixelseal import PixelSealWatermark
from evaluator.watermarking.methods.rawatermark import RAWatermark
from evaluator.watermarking.methods.invisible_watermark import InvisibleWatermarkRivaGan
from evaluator.watermarking.methods.ssl_watermarking import SSLWatermark
from evaluator.watermarking.methods.stegastamp import StegaStampWatermark
from evaluator.watermarking.methods.trustmark import TrustMarkCWatermark, TrustMarkQWatermark
from evaluator.watermarking.methods.videoseal import VideoSealWatermark
from evaluator.watermarking.methods.vine import VineWatermark
from evaluator.watermarking.methods.wam import WAMWatermark


class DummyBatchWatermark(BaseWatermark):
    name = "dummy-batch"

    def __init__(self) -> None:
        super().__init__()
        self.embed_batches: list[int] = []
        self.extract_batches: list[int] = []

    def embed_impl(self, input_path: Path, output_path: Path, context: WatermarkContext):
        output_path.write_text("single", encoding="utf-8")
        return {"mode": "single"}

    def extract_impl(self, input_path: Path, context: WatermarkContext):
        return {"bits": "0", "mode": "single"}

    def embed_batch_impl(self, jobs):
        self.embed_batches.append(len(jobs))
        for _input_path, output_path, _context in jobs:
            output_path.write_text("batch", encoding="utf-8")
        return [{"mode": "batch"} for _job in jobs]

    def extract_batch_impl(self, jobs):
        self.extract_batches.append(len(jobs))
        return [{"bits": "10", "message": "decoded", "mode": "batch"} for _job in jobs]


class FailingBatchWatermark(BaseWatermark):
    name = "failing-batch"

    def embed_impl(self, input_path: Path, output_path: Path, context: WatermarkContext):
        output_path.write_text("single", encoding="utf-8")
        return {"mode": "single"}

    def extract_impl(self, input_path: Path, context: WatermarkContext):
        return {"bits": "1", "mode": "single"}

    def embed_batch_impl(self, jobs):
        raise RuntimeError("batch unavailable")

    def extract_batch_impl(self, jobs):
        raise RuntimeError("batch unavailable")


class SmallOutputWatermark(BaseWatermark):
    name = "small-output"

    def embed_impl(self, input_path: Path, output_path: Path, context: WatermarkContext):
        Image.new("RGB", (64, 64), (120, 160, 200)).save(output_path)
        return {"image_size": [64, 64]}

    def extract_impl(self, input_path: Path, context: WatermarkContext):
        return {"bits": "1", "image_size": [64, 64]}


class WatermarkBatchingTest(unittest.TestCase):
    def test_base_watermark_chunks_batch_impls(self) -> None:
        previous = os.environ.get("WM_BENCH_WATERMARK_BATCH_SIZE")
        os.environ["WM_BENCH_WATERMARK_BATCH_SIZE"] = "2"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                method = DummyBatchWatermark()
                contexts = [
                    WatermarkContext(
                        run_id="run",
                        sample_id=f"sample-{index}",
                        method_name=method.name,
                    )
                    for index in range(5)
                ]
                embed_jobs = []
                extract_jobs = []
                for index, context in enumerate(contexts):
                    input_path = root / f"input-{index}.txt"
                    output_path = root / f"output-{index}.txt"
                    input_path.write_text("input", encoding="utf-8")
                    embed_jobs.append((input_path, output_path, context))
                    extract_jobs.append((output_path, context))

                embed_results = method.embed_many(embed_jobs)
                extract_results = method.extract_many(extract_jobs)

                self.assertEqual(method.embed_batches, [2, 2, 1])
                self.assertEqual(method.extract_batches, [2, 2, 1])
                self.assertTrue(all(result.ok for result in embed_results))
                self.assertTrue(all(result.ok for result in extract_results))
                self.assertTrue(all(result.metadata["mode"] == "batch" for result in embed_results))
                self.assertTrue(all(result.bits == "10" for result in extract_results))
                self.assertTrue(all(result.metadata["executionMode"] == "batch" for result in embed_results))
                self.assertEqual(embed_results[0].metadata["execution"]["configuredBatchSize"], 2)
                self.assertEqual(embed_results[-1].metadata["execution"]["actualBatchSize"], 1)
        finally:
            if previous is None:
                os.environ.pop("WM_BENCH_WATERMARK_BATCH_SIZE", None)
            else:
                os.environ["WM_BENCH_WATERMARK_BATCH_SIZE"] = previous

    def test_base_watermark_uses_stage_specific_batch_sizes(self) -> None:
        previous = {
            key: os.environ.get(key)
            for key in (
                "WM_BENCH_WATERMARK_BATCH_SIZE",
                "WM_BENCH_WATERMARK_EMBED_BATCH_SIZES",
                "WM_BENCH_WATERMARK_EXTRACT_BATCH_SIZES",
            )
        }
        os.environ["WM_BENCH_WATERMARK_BATCH_SIZE"] = "2"
        os.environ["WM_BENCH_WATERMARK_EMBED_BATCH_SIZES"] = "dummy-batch=3"
        os.environ["WM_BENCH_WATERMARK_EXTRACT_BATCH_SIZES"] = "dummy-batch=4"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                method = DummyBatchWatermark()
                contexts = [
                    WatermarkContext(run_id="run", sample_id=f"sample-{index}", method_name=method.name)
                    for index in range(7)
                ]
                embed_jobs = []
                extract_jobs = []
                for index, context in enumerate(contexts):
                    input_path = root / f"input-{index}.txt"
                    output_path = root / f"output-{index}.txt"
                    input_path.write_text("input", encoding="utf-8")
                    embed_jobs.append((input_path, output_path, context))
                    extract_jobs.append((output_path, context))

                embed_results = method.embed_many(embed_jobs)
                extract_results = method.extract_many(extract_jobs)

                self.assertEqual(method.embed_batches, [3, 3, 1])
                self.assertEqual(method.extract_batches, [4, 3])
                self.assertEqual(embed_results[0].metadata["execution"]["configuredBatchSize"], 3)
                self.assertEqual(extract_results[0].metadata["execution"]["configuredBatchSize"], 4)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_base_watermark_records_batch_fallback(self) -> None:
        previous = os.environ.get("WM_BENCH_WATERMARK_BATCH_SIZE")
        os.environ["WM_BENCH_WATERMARK_BATCH_SIZE"] = "2"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                method = FailingBatchWatermark()
                context = WatermarkContext(run_id="run", sample_id="sample", method_name=method.name)
                embed_jobs = []
                extract_jobs = []
                for index in range(3):
                    input_path = root / f"input-{index}.txt"
                    output_path = root / f"output-{index}.txt"
                    input_path.write_text("input", encoding="utf-8")
                    embed_jobs.append((input_path, output_path, context))
                    extract_jobs.append((output_path, context))

                embed_results = method.embed_many(embed_jobs)
                extract_results = method.extract_many(extract_jobs)

                self.assertTrue(all(result.ok for result in embed_results))
                self.assertTrue(all(result.ok for result in extract_results))
                self.assertTrue(all(result.metadata["executionMode"] == "batch_fallback_serial" for result in embed_results))
                self.assertTrue(all(result.metadata["execution"]["fallback"] for result in extract_results))
                self.assertIn("RuntimeError", embed_results[0].metadata["execution"]["fallbackReason"])
        finally:
            if previous is None:
                os.environ.pop("WM_BENCH_WATERMARK_BATCH_SIZE", None)
            else:
                os.environ["WM_BENCH_WATERMARK_BATCH_SIZE"] = previous

    def test_base_watermark_canonicalizes_embed_outputs_and_records_decode_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.png"
            output_path = root / "output.png"
            Image.new("RGB", (300, 200), (40, 50, 60)).save(input_path)
            method = SmallOutputWatermark()
            context = WatermarkContext(run_id="run", sample_id="sample", method_name=method.name)

            embed_result = method.embed(input_path, output_path, context)
            extract_result = method.extract(output_path, context)

            self.assertTrue(embed_result.ok, embed_result.error)
            with Image.open(output_path) as image:
                self.assertEqual(image.size, (512, 512))
            self.assertEqual(embed_result.metadata["inputSize"], [300, 200])
            self.assertEqual(embed_result.metadata["internalSize"], [64, 64])
            self.assertEqual(embed_result.metadata["preCanonicalOutputSize"], [64, 64])
            self.assertEqual(embed_result.metadata["outputSize"], [512, 512])
            self.assertTrue(embed_result.metadata["canonicalizedOutput"])
            self.assertTrue(extract_result.ok, extract_result.error)
            self.assertEqual(extract_result.metadata["decodeInputSize"], [512, 512])
            self.assertEqual(extract_result.metadata["decodeInternalSize"], [64, 64])

    def test_torch_gpu_wrappers_expose_batch_hooks(self) -> None:
        embed_batch_classes = [
            VideoSealWatermark,
            VideoSealFamilyWatermark,
            ChunkySealWatermark,
            PixelSealWatermark,
            CINWatermark,
            HiDDeNWatermark,
            MBRSWatermark,
            PIMoGWatermark,
            MaskWMD32Watermark,
            InvisMarkWatermark,
            WAMWatermark,
            RAWatermark,
            SSLWatermark,
            VineWatermark,
            InvisibleWatermarkRivaGan,
            StegaStampWatermark,
            TrustMarkCWatermark,
            TrustMarkQWatermark,
            DWSFWatermark,
        ]
        extract_batch_classes = [
            *embed_batch_classes,
            SSLWatermark,
            VineWatermark,
        ]

        for cls in embed_batch_classes:
            self.assertIsNot(cls.embed_batch_impl, BaseWatermark.embed_batch_impl, cls.__name__)
        for cls in extract_batch_classes:
            self.assertIsNot(cls.extract_batch_impl, BaseWatermark.extract_batch_impl, cls.__name__)

    def test_invismark_deterministic_payload_uses_uuid4_layout(self) -> None:
        method = InvisMarkWatermark.__new__(InvisMarkWatermark)
        method.payload_bits = 100
        context = WatermarkContext(
            run_id="run",
            sample_id="sample",
            method_name="invismark",
            message="1010101",
            seed=42,
        )

        bits, mode = method._payload(context)
        bits_again, mode_again = method._payload(context)

        self.assertEqual(mode, "deterministic_uuid4")
        self.assertEqual(mode_again, "deterministic_uuid4")
        self.assertEqual(bits, bits_again)
        self.assertEqual(bits[48:52], [0, 1, 0, 0])
        self.assertEqual(bits[64:66], [1, 0])

    def test_videoseal_wrappers_report_model_internal_size(self) -> None:
        fake_model = type("FakeVideoSealModel", (), {"img_size": 256})()
        for cls in (VideoSealWatermark, VideoSealFamilyWatermark):
            method = cls.__new__(cls)
            method._model = fake_model
            self.assertEqual(method._internal_size(), [256, 256])


if __name__ == "__main__":
    unittest.main()
