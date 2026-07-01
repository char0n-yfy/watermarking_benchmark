from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluator.watermarking.base import BaseWatermark, WatermarkContext
from evaluator.watermarking.methods._videoseal_family import VideoSealFamilyWatermark
from evaluator.watermarking.methods.chunkyseal import ChunkySealWatermark
from evaluator.watermarking.methods.cin import CINWatermark
from evaluator.watermarking.methods.hidden import HiDDeNWatermark
from evaluator.watermarking.methods.invismark import InvisMarkWatermark
from evaluator.watermarking.methods.maskwm import MaskWMD32Watermark
from evaluator.watermarking.methods.mbrs import MBRSWatermark
from evaluator.watermarking.methods.pimog import PIMoGWatermark
from evaluator.watermarking.methods.pixelseal import PixelSealWatermark
from evaluator.watermarking.methods.rawatermark import RAWatermark
from evaluator.watermarking.methods.ssl_watermarking import SSLWatermark
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
        finally:
            if previous is None:
                os.environ.pop("WM_BENCH_WATERMARK_BATCH_SIZE", None)
            else:
                os.environ["WM_BENCH_WATERMARK_BATCH_SIZE"] = previous

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


if __name__ == "__main__":
    unittest.main()
