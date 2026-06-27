from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.services.local_runner import _attack_params
from app.services.resources import (
    get_attack_catalog_item,
    get_watermark_catalog_item,
    list_attack_resources,
    list_watermark_resources,
)
from evaluator.attacks import ATTACK_REGISTRY
from evaluator.watermarking import WATERMARK_REGISTRY


class ResourceCatalogTest(unittest.TestCase):
    def test_all_registered_watermarks_are_exposed_to_frontend(self) -> None:
        resources = list_watermark_resources()
        exposed_methods = {item["method"] for item in resources}

        self.assertEqual(set(WATERMARK_REGISTRY), exposed_methods)
        self.assertGreaterEqual(len(resources), 20)

        for method in WATERMARK_REGISTRY:
            item = get_watermark_catalog_item(method)
            self.assertEqual(item["method"], method)
            self.assertTrue(item["id"].startswith("alg-"))
            self.assertEqual(item["available"], True)

    def test_all_registered_attacks_are_exposed_to_frontend(self) -> None:
        resources = list_attack_resources()
        exposed_methods = {item["method"] for item in resources}

        self.assertEqual(set(ATTACK_REGISTRY), exposed_methods)
        self.assertGreaterEqual(len(resources), 40)

        for method in ATTACK_REGISTRY:
            item = get_attack_catalog_item(method)
            self.assertEqual(item["method"], method)
            self.assertTrue(item["id"].startswith("atk-"))
            self.assertEqual(item["available"], True)

    def test_legacy_attack_presets_remain_resolvable(self) -> None:
        jpeg_smoke = get_attack_catalog_item("atk-jpeg-smoke")
        blur_sweep = get_attack_catalog_item("atk-blur-sweep")
        crop_sweep = get_attack_catalog_item("atk-crop-sweep")

        self.assertEqual(jpeg_smoke["method"], "jpeg")
        self.assertEqual(jpeg_smoke["strengthParam"], "strength")
        self.assertEqual(blur_sweep["method"], "gaussian_blur")
        self.assertEqual(crop_sweep["method"], "resized_crop")

    def test_strength_is_only_injected_for_compatible_attacks(self) -> None:
        jpeg = get_attack_catalog_item("atk-jpeg")
        platform_pipeline = get_attack_catalog_item("atk-cp-platform-pipeline")

        self.assertEqual(_attack_params(jpeg, 0.5), {"strength": 0.5})
        self.assertEqual(_attack_params(platform_pipeline, 0.5), {})


if __name__ == "__main__":
    unittest.main()
