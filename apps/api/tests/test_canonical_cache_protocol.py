from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.services.experiment_stages import CANONICAL_MANIFEST_NAME, copy_canonical_samples
from evaluator.image_protocol import CANONICAL_OUTPUT_POLICY, CANONICAL_PREPROCESS_POLICY


class CanonicalCacheProtocolTest(unittest.TestCase):
    def test_old_preprocess_manifest_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            output = root / "canonical"
            dataset.mkdir()
            Image.new("RGB", (300, 200), (10, 20, 30)).save(dataset / "sample.png")

            first = copy_canonical_samples(dataset, output, 1)
            self.assertEqual(first[0].metadata["preprocessPolicy"], CANONICAL_PREPROCESS_POLICY)

            manifest_path = output / CANONICAL_MANIFEST_NAME
            records = json.loads(manifest_path.read_text())
            records[0]["preprocessPolicy"] = "letterbox_pad_512"
            records[0]["canonicalOutputPolicy"] = "old"
            records[0]["metadata"]["preprocessPolicy"] = "letterbox_pad_512"
            manifest_path.write_text(json.dumps(records), encoding="utf-8")

            second = copy_canonical_samples(dataset, output, 1)
            refreshed_records = json.loads(manifest_path.read_text())
            self.assertEqual(second[0].metadata["preprocessPolicy"], CANONICAL_PREPROCESS_POLICY)
            self.assertEqual(refreshed_records[0]["preprocessPolicy"], CANONICAL_PREPROCESS_POLICY)
            self.assertEqual(refreshed_records[0]["canonicalOutputPolicy"], CANONICAL_OUTPUT_POLICY)
            self.assertEqual(refreshed_records[0]["canonicalSize"], [512, 512])


if __name__ == "__main__":
    unittest.main()
