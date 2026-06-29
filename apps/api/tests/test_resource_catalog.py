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
        self.assertGreaterEqual(len(resources), len(ATTACK_REGISTRY))

        for method in ATTACK_REGISTRY:
            item = get_attack_catalog_item(method)
            self.assertEqual(item["method"], method)
            self.assertTrue(item["id"].startswith("atk-"))
            self.assertEqual(item["available"], True)
            expected_category = ATTACK_REGISTRY[method].__module__.split("evaluator.attacks.", 1)[1].split(".", 1)[0]
            self.assertEqual(item["category"], expected_category)
            self.assertIn("categoryLabel", item)
            self.assertEqual(item["categoryPath"], f"evaluator/attacks/{expected_category}")

    def test_consumer_enhancement_attacks_are_grouped_for_frontend(self) -> None:
        resources = list_attack_resources()
        cew_resources = [item for item in resources if item["method"].startswith("cew_")]
        exposed_methods = {item["method"] for item in cew_resources}

        self.assertEqual(len(cew_resources), 16)
        self.assertIn("cew_e1", exposed_methods)
        self.assertIn("cew_d5", exposed_methods)
        self.assertIn("cew_s3", exposed_methods)
        self.assertIn("cew_c4", exposed_methods)
        self.assertNotIn("cew_e1_m", exposed_methods)
        self.assertNotIn("cew_s6", exposed_methods)
        self.assertTrue(all(item["category"] == "consumer_enhancement_workflow_attacks" for item in cew_resources))
        self.assertFalse(get_attack_catalog_item("cew_e1")["requiresGpu"])
        self.assertTrue(get_attack_catalog_item("cew_d5")["requiresGpu"])
        self.assertTrue(get_attack_catalog_item("cew_s1")["requiresGpu"])
        self.assertTrue(get_attack_catalog_item("cew_c1")["requiresGpu"])
        self.assertEqual(get_attack_catalog_item("cew_e1")["strengthParam"], "strength")
        self.assertEqual(get_attack_catalog_item("cew_e1")["strengths"], [0.25, 0.5, 0.75])
        self.assertEqual(get_attack_catalog_item("cew_s1")["strengthParam"], "scale")
        self.assertEqual(get_attack_catalog_item("cew_s1")["strengths"], [2.0, 4.0])

    def test_ctrlregen_and_nfpa_attacks_are_grouped_for_frontend(self) -> None:
        noise_to_image = get_attack_catalog_item("noise_to_image")
        image_to_vedio = get_attack_catalog_item("image_to_vedio")

        self.assertEqual(noise_to_image["category"], "regeneration_attacks")
        self.assertEqual(noise_to_image["strengthParam"], "step")
        self.assertEqual(noise_to_image["strengths"], [0.25, 0.5, 0.75, 1.0])
        self.assertTrue(noise_to_image["requiresGpu"])

        self.assertEqual(image_to_vedio["category"], "regeneration_attacks")
        self.assertEqual(image_to_vedio["strengthParam"], "xy")
        self.assertEqual(image_to_vedio["strengths"], [20.0, 40.0, 60.0])
        self.assertTrue(image_to_vedio["requiresGpu"])

    def test_sharp_viewpoint_rerendering_attack_is_grouped_for_frontend(self) -> None:
        attack = get_attack_catalog_item("3d_viewpoint_rerendering")

        self.assertEqual(attack["category"], "regeneration_attacks")
        self.assertEqual(attack["strengthParam"], "max_disparity")
        self.assertEqual(attack["strengths"], [0.01, 0.02, 0.04])
        self.assertTrue(attack["requiresGpu"])

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
        cew_composite = get_attack_catalog_item("cew_c1")
        cew_edit = get_attack_catalog_item("cew_e1")
        cew_sr = get_attack_catalog_item("cew_s1")

        self.assertEqual(_attack_params(jpeg, 0.5), {"strength": 0.5})
        self.assertEqual(_attack_params(cew_composite, 0.5), {})
        self.assertEqual(_attack_params(cew_edit, 0.75), {"strength": 0.75})
        self.assertEqual(_attack_params(cew_sr, 4.0), {"scale": 4})
        self.assertEqual(_attack_params(get_attack_catalog_item("image_to_vedio"), 40.0), {"xy": 40})
        self.assertEqual(_attack_params(get_attack_catalog_item("noise_to_image"), 0.75), {"step": 0.75})
        self.assertEqual(
            _attack_params(get_attack_catalog_item("3d_viewpoint_rerendering"), 0.02),
            {"max_disparity": 0.02},
        )


if __name__ == "__main__":
    unittest.main()
