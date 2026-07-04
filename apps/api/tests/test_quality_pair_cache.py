from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.services import local_executor


class QualityPairCacheTest(unittest.TestCase):
    def test_record_quality_pairs_reuses_cached_pair_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference_dir = root / "reference"
            target_dir = root / "target"
            reference_dir.mkdir()
            target_dir.mkdir()
            Image.new("RGB", (16, 16), (100, 120, 140)).save(reference_dir / "sample.png")
            Image.new("RGB", (16, 16), (101, 121, 141)).save(target_dir / "sample.png")
            paths = {
                "imageQuality": root / "image_quality.jsonl",
                "runtimeProfile": root / "runtime_profile.jsonl",
            }
            fake_metrics = [{"psnr": 42.0, "ssim": 0.9, "msSsim": 0.91, "nmi": 0.8, "lpips": None, "dists": None}]
            fake_profile = {"stage": "quality", "method": "image_quality", "mode": "hybrid", "jobCount": 1}

            with patch.dict(os.environ, {"WM_BENCH_QUALITY_PAIR_CACHE": "1"}, clear=False):
                cache = local_executor.QualityPairCache()
                with patch(
                    "app.services.local_executor.compute_image_quality_pairs_with_profile",
                    return_value=(fake_metrics, fake_profile),
                ) as compute_mock:
                    first = local_executor._record_quality_pairs(
                        paths,
                        run_id="run",
                        cell_key="cell-a",
                        scope="scope-a",
                        dataset_id="dataset",
                        algorithm_id="algorithm",
                        attack_id=None,
                        attack_method=None,
                        attack_strength=None,
                        seed=1,
                        reference_dir=reference_dir,
                        target_dir=target_dir,
                        quality_pair_cache=cache,
                    )
                    second = local_executor._record_quality_pairs(
                        paths,
                        run_id="run",
                        cell_key="cell-b",
                        scope="scope-b",
                        dataset_id="dataset",
                        algorithm_id="algorithm",
                        attack_id=None,
                        attack_method=None,
                        attack_strength=None,
                        seed=1,
                        reference_dir=reference_dir,
                        target_dir=target_dir,
                        quality_pair_cache=cache,
                    )

            self.assertEqual(compute_mock.call_count, 1)
            self.assertEqual(first[0]["metrics"], second[0]["metrics"])
            profiles = [json.loads(line) for line in paths["runtimeProfile"].read_text().splitlines()]
            self.assertEqual(profiles[1]["metadata"]["execution"]["mode"], "pair_cache")
            self.assertEqual(profiles[1]["metadata"]["execution"]["details"]["cacheHits"], 1)

    def test_quality_pair_cache_is_thread_safe(self) -> None:
        with patch.dict(
            os.environ,
            {"WM_BENCH_QUALITY_PAIR_CACHE": "1", "WM_BENCH_QUALITY_PAIR_CACHE_MAX_ENTRIES": "32"},
            clear=False,
        ):
            cache = local_executor.QualityPairCache()

            def worker(index: int) -> None:
                key = f"pair-{index % 8}"
                cache.set(key, {"psnr": float(index)})
                cached = cache.get(key)
                self.assertIsNotNone(cached)
                _ = cache.stats()

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(worker, range(128)))

            stats = cache.stats()
            self.assertTrue(stats["enabled"])
            self.assertLessEqual(stats["entryCount"], 32)


if __name__ == "__main__":
    unittest.main()
