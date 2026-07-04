from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Mapping

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluator.image_batch_io import save_image_batch
from evaluator.watermarking.base import BaseWatermark, WatermarkContext


class _AsyncBatchWatermark(BaseWatermark):
    name = "async-batch-test"

    def embed_batch_impl(
        self,
        jobs: list[tuple[Path, Path, WatermarkContext]],
    ) -> list[Mapping[str, Any]]:
        images = []
        metadatas = []
        for input_path, output_path, _context in jobs:
            image = Image.open(input_path).convert("RGB")
            images.append((image, output_path))
            metadatas.append({"internalSize": list(image.size)})
        save_image_batch(images, save_fn=lambda image, path: image.save(path, format="PNG"))
        return metadatas

    def embed_impl(
        self,
        input_path: Path,
        output_path: Path,
        context: WatermarkContext,
    ) -> Mapping[str, Any]:
        image = Image.open(input_path).convert("RGB")
        image.save(output_path, format="PNG")
        return {"internalSize": list(image.size)}

    def extract_impl(
        self,
        input_path: Path,
        context: WatermarkContext,
    ) -> Mapping[str, Any]:
        return {}


class _FallbackBatchWatermark(_AsyncBatchWatermark):
    name = "fallback-batch-test"

    def embed_batch_impl(
        self,
        jobs: list[tuple[Path, Path, WatermarkContext]],
    ) -> list[Mapping[str, Any]]:
        raise RuntimeError("forced batch failure")

    def embed_impl(
        self,
        input_path: Path,
        output_path: Path,
        context: WatermarkContext,
    ) -> Mapping[str, Any]:
        image = Image.open(input_path).convert("RGB")

        def slow_save(image: Image.Image, path: Path) -> None:
            time.sleep(0.05)
            image.save(path, format="PNG")

        save_image_batch([(image, output_path)], save_fn=slow_save)
        return {"internalSize": list(image.size)}


class WatermarkAsyncSaveTest(unittest.TestCase):
    def test_embed_many_flushes_async_saves_before_protocol_metadata(self) -> None:
        saved_env = {
            key: os.environ.get(key)
            for key in (
                "WM_BENCH_ASYNC_IMAGE_SAVE",
                "WM_BENCH_IMAGE_IO_SAVE_WORKERS",
                "WM_BENCH_WATERMARK_BATCH_SIZE",
            )
        }
        os.environ["WM_BENCH_ASYNC_IMAGE_SAVE"] = "1"
        os.environ["WM_BENCH_IMAGE_IO_SAVE_WORKERS"] = "1"
        os.environ["WM_BENCH_WATERMARK_BATCH_SIZE"] = "2"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                inputs = []
                for index in range(4):
                    path = root / f"in_{index}.png"
                    Image.new("RGB", (16, 12), (index * 30, 20, 40)).save(path, format="PNG")
                    inputs.append(path)

                jobs = [
                    (
                        input_path,
                        root / "out" / f"out_{index}.png",
                        WatermarkContext(
                            run_id="run-test",
                            sample_id=f"sample-{index}",
                            method_name=_AsyncBatchWatermark.name,
                        ),
                    )
                    for index, input_path in enumerate(inputs)
                ]
                results = _AsyncBatchWatermark().embed_many(jobs)

                self.assertEqual(len(results), len(jobs))
                self.assertTrue(all(result.ok for result in results))
                self.assertTrue(all(result.output_path.exists() for result in results))
                execution = results[0].metadata["execution"]
                async_save = execution["config"]["asyncImageSave"]
                self.assertTrue(async_save["active"])
                self.assertEqual(async_save["submitted"], len(jobs))
                self.assertEqual(async_save["flushed"], len(jobs))
        finally:
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_batch_fallback_uses_synchronous_save_semantics(self) -> None:
        saved_env = {
            key: os.environ.get(key)
            for key in (
                "WM_BENCH_ASYNC_IMAGE_SAVE",
                "WM_BENCH_IMAGE_IO_SAVE_WORKERS",
                "WM_BENCH_WATERMARK_BATCH_SIZE",
            )
        }
        os.environ["WM_BENCH_ASYNC_IMAGE_SAVE"] = "1"
        os.environ["WM_BENCH_IMAGE_IO_SAVE_WORKERS"] = "1"
        os.environ["WM_BENCH_WATERMARK_BATCH_SIZE"] = "2"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                input_path = root / "in.png"
                output_path = root / "out.png"
                Image.new("RGB", (16, 12), (100, 110, 120)).save(input_path, format="PNG")
                result = _FallbackBatchWatermark().embed_many(
                    [
                        (
                            input_path,
                            output_path,
                            WatermarkContext(
                                run_id="run-test",
                                sample_id="sample",
                                method_name=_FallbackBatchWatermark.name,
                            ),
                        )
                    ]
                )[0]
                self.assertTrue(result.ok)
                self.assertTrue(output_path.exists())
                self.assertEqual(result.metadata["outputSize"], [512, 512])
                self.assertEqual(result.metadata["executionMode"], "batch_fallback_serial")
        finally:
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
