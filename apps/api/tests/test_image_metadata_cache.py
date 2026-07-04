from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluator.image_io import save_png_image
from evaluator.image_protocol import clear_image_metadata_cache, image_size


class ImageMetadataCacheTest(unittest.TestCase):
    def test_image_size_cache_hits_and_updates_after_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.png"
            Image.new("RGB", (10, 20), (1, 2, 3)).save(path)

            clear_image_metadata_cache()
            self.assertEqual(image_size(path), [10, 20])
            with patch("evaluator.image_protocol.Image.open", side_effect=AssertionError("cache miss")):
                self.assertEqual(image_size(path), [10, 20])

            save_png_image(Image.new("RGB", (30, 40), (4, 5, 6)), path)
            with patch("evaluator.image_protocol.Image.open", side_effect=AssertionError("cache miss")):
                self.assertEqual(image_size(path), [30, 40])


if __name__ == "__main__":
    unittest.main()
