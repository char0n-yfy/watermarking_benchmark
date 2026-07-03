from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluator.image_batch_io import (
    AsyncImageSavePool,
    ImageBatchPrefetcher,
    load_rgb_batch,
    prefetch_rgb_batch,
    save_image_batch,
    to_tensor_batch,
)
from evaluator.image_protocol import image_size


class ImageBatchIoTest(unittest.TestCase):
    def test_load_tensor_and_save_batch(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = []
            for index, color in enumerate(((10, 20, 30), (40, 50, 60))):
                path = root / f"input_{index}.png"
                Image.new("RGB", (8, 6), color).save(path)
                inputs.append(path)

            images = load_rgb_batch(inputs)
            self.assertEqual([image.size for image in images], [(8, 6), (8, 6)])

            batch = to_tensor_batch(
                images,
                transform=lambda image: torch.as_tensor(list(image.getdata()), dtype=torch.float32).reshape(6, 8, 3).permute(2, 0, 1),
                torch_module=torch,
            )
            self.assertEqual(tuple(batch.shape), (2, 3, 6, 8))

            outputs = [root / "out_0.png", root / "out_1.png"]
            save_image_batch(zip(images, outputs))
            self.assertEqual(image_size(outputs[0]), [8, 6])
            self.assertEqual(image_size(outputs[1]), [8, 6])

    def test_prefetch_rgb_batch_feeds_next_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = []
            for index, color in enumerate(((11, 22, 33), (44, 55, 66))):
                path = root / f"prefetch_{index}.png"
                Image.new("RGB", (8, 6), color).save(path)
                inputs.append(path)

            with ImageBatchPrefetcher(enabled=True) as prefetcher:
                prefetch_rgb_batch(inputs)
                images = load_rgb_batch(inputs)
                profile = prefetcher.profile()

            self.assertEqual([image.size for image in images], [(8, 6), (8, 6)])
            self.assertEqual(profile["submitted"], 1)
            self.assertEqual(profile["hits"], 1)
            self.assertEqual(profile["consumed"], 1)

    def test_async_save_pool_defers_until_flush(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "async.png"
            started = threading.Event()
            release = threading.Event()

            def slow_save(image: Image.Image, path: Path) -> None:
                started.set()
                self.assertTrue(release.wait(timeout=2.0))
                image.save(path, format="PNG")

            image = Image.new("RGB", (8, 6), (70, 80, 90))
            with AsyncImageSavePool(enabled=True) as pool:
                save_image_batch([(image, output)], save_fn=slow_save)
                self.assertTrue(started.wait(timeout=1.0))
                self.assertFalse(output.exists())
                self.assertEqual(pool.submitted, 1)
                release.set()

            self.assertEqual(image_size(output), [8, 6])


if __name__ == "__main__":
    unittest.main()
