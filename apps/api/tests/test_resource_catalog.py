from __future__ import annotations

from collections import Counter
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.services.local_runner import (
    _attack_params,
    _attack_variants_for_attack,
    _strengths_for_attack,
    estimate_selection,
    normalize_selection,
)
from app.services.resources import (
    get_attack_catalog_item,
    get_watermark_catalog_item,
    list_attack_resources,
    list_watermark_resources,
    scan_dataset_resources,
)
from evaluator.attacks import ATTACK_REGISTRY
from evaluator.watermarking import WATERMARK_REGISTRY


class ResourceCatalogTest(unittest.TestCase):
    def test_all_registered_watermarks_are_exposed_to_frontend(self) -> None:
        resources = list_watermark_resources()
        exposed_methods = {item["method"] for item in resources}

        self.assertEqual(set(WATERMARK_REGISTRY), exposed_methods)
        self.assertGreaterEqual(len(resources), 20)
        self.assertEqual(
            {item["category"] for item in resources},
            {"traditional_watermark", "deep_watermark"},
        )

        for method in WATERMARK_REGISTRY:
            item = get_watermark_catalog_item(method)
            self.assertEqual(item["method"], method)
            self.assertTrue(item["id"].startswith("alg-"))
            self.assertEqual(item["available"], True)

    def test_all_registered_attacks_are_exposed_to_frontend(self) -> None:
        resources = list_attack_resources()
        exposed_methods = {item["method"] for item in resources}

        self.assertEqual(set(ATTACK_REGISTRY), exposed_methods)
        self.assertEqual(len(resources), len(ATTACK_REGISTRY))

        for method in ATTACK_REGISTRY:
            item = get_attack_catalog_item(method)
            self.assertEqual(item["method"], method)
            self.assertTrue(item["id"].startswith("atk-"))
            self.assertEqual(item["available"], True)
            expected_module_category = ATTACK_REGISTRY[method].__module__.split("evaluator.attacks.", 1)[1].split(".", 1)[0]
            expected_category = "identity" if method == "identity" else expected_module_category
            self.assertEqual(item["category"], expected_category)
            self.assertIn("categoryLabel", item)
            expected_path = (
                "evaluator/attacks/distortion_attacks"
                if method == "identity"
                else f"evaluator/attacks/{expected_category}"
            )
            self.assertEqual(item["categoryPath"], expected_path)

    def test_dataset_scan_does_not_add_local_root_for_child_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            resources_root = Path(tmpdir)
            child = resources_root / "datasets" / "demo"
            child.mkdir(parents=True)
            Image.new("RGB", (16, 16), (80, 120, 160)).save(child / "sample.png")

            resources = scan_dataset_resources(resources_root)

            self.assertEqual([item.id for item in resources], ["demo"])
            self.assertEqual(resources[0].sample_count, 1)

    def test_dataset_scan_adds_local_root_only_for_direct_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            resources_root = Path(tmpdir)
            datasets_root = resources_root / "datasets"
            child = datasets_root / "demo"
            child.mkdir(parents=True)
            Image.new("RGB", (16, 16), (80, 120, 160)).save(child / "sample.png")
            Image.new("RGB", (16, 16), (160, 120, 80)).save(datasets_root / "root.png")

            resources = scan_dataset_resources(resources_root)

            self.assertEqual([item.id for item in resources], ["local-root", "demo"])
            self.assertEqual(resources[0].sample_count, 1)

    def test_dataset_scan_uses_catalog_id_for_known_display_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            resources_root = Path(tmpdir)
            child = resources_root / "datasets" / "MS COCO"
            child.mkdir(parents=True)
            Image.new("RGB", (16, 16), (80, 120, 160)).save(child / "sample.png")

            resources = scan_dataset_resources(resources_root)

            self.assertEqual([item.id for item in resources], ["ms-coco"])
            self.assertEqual(resources[0].name, "MS COCO")
            self.assertEqual(resources[0].path, child)

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
        self.assertEqual(get_attack_catalog_item("cew_e1")["strengths"], [0.0, 0.5, 1.0])
        self.assertEqual(get_attack_catalog_item("cew_s1")["strengthParam"], "scale")
        self.assertEqual(get_attack_catalog_item("cew_s1")["strengths"], [2.0, 4.0])

    def test_ctrlregen_and_nfpa_attacks_are_grouped_for_frontend(self) -> None:
        noise_to_image = get_attack_catalog_item("noise_to_image")
        image_to_vedio = get_attack_catalog_item("image_to_vedio")

        self.assertEqual(noise_to_image["category"], "regeneration_attacks")
        self.assertEqual(noise_to_image["strengthParam"], "step")
        self.assertEqual(noise_to_image["strengths"], [0.0, 0.5, 1.0])
        self.assertTrue(noise_to_image["requiresGpu"])

        self.assertEqual(image_to_vedio["category"], "regeneration_attacks")
        self.assertEqual(image_to_vedio["strengthParam"], "xy")
        self.assertEqual(image_to_vedio["strengths"], [20.0, 40.0, 60.0])
        self.assertTrue(image_to_vedio["requiresGpu"])

    def test_sharp_viewpoint_rerendering_attack_is_grouped_for_frontend(self) -> None:
        resources = list_attack_resources()
        viewpoint_resources = [
            item for item in resources if item["category"] == "3d_viewpoint_rerendering"
        ]
        attack = get_attack_catalog_item("3d_viewpoint_rerendering_rotate_point")

        self.assertEqual(len(viewpoint_resources), 8)
        self.assertNotIn("3d_viewpoint_rerendering", {item["method"] for item in resources})
        self.assertEqual(attack["category"], "3d_viewpoint_rerendering")
        self.assertTrue(all(item["category"] == "3d_viewpoint_rerendering" for item in viewpoint_resources))
        self.assertEqual(attack["categoryPath"], "evaluator/attacks/3d_viewpoint_rerendering")
        self.assertEqual(attack["strengthParam"], "strength")
        self.assertEqual(attack["strengths"], [0.0, 0.5, 1.0])
        self.assertTrue(attack["requiresGpu"])
        self.assertEqual(attack["viewpointMotion"], "rotate")
        self.assertEqual(attack["viewpointPhasePolicy"], "random_per_sample")
        self.assertEqual(attack["viewpointPhaseChoices"], list(range(8)))
        self.assertEqual(attack["viewpointLookatMode"], "point")
        self.assertEqual(
            Counter(item["viewpointMotion"] for item in viewpoint_resources),
            Counter({"swipe": 2, "shake": 2, "rotate": 2, "rotate_forward": 2}),
        )
        self.assertEqual(
            Counter(item["viewpointLookatMode"] for item in viewpoint_resources),
            Counter({"point": 4, "ahead": 4}),
        )
        self.assertEqual(
            get_attack_catalog_item("3d_viewpoint_rerendering_phase7_ahead")["method"],
            "3d_viewpoint_rerendering_rotate_ahead",
        )
        self.assertEqual(
            get_attack_catalog_item("3d_viewpoint_rerendering_rotate_forward_phase7_ahead")["method"],
            "3d_viewpoint_rerendering_rotate_forward_ahead",
        )
        self.assertEqual(
            get_attack_catalog_item("3d_viewpoint_rerendering_rotate_forward_ahead")["strengthParam"],
            "strength",
        )

    def test_sharp_viewpoint_rerendering_renders_one_random_phase_per_image(self) -> None:
        import numpy as np

        cls = ATTACK_REGISTRY["3d_viewpoint_rerendering_rotate_point"]
        attack = cls(allow_download=False, save_intermediates=False)

        class FakeGaussians:
            def to(self, _device):
                return self

        class FakeIO:
            @staticmethod
            def load_rgb(_path):
                return np.zeros((4, 4, 3), dtype=np.uint8), None, 1.0

        class FakeSceneMetaData:
            def __init__(self, focal_length_px, resolution_px, color_space):
                self.focal_length_px = focal_length_px
                self.resolution_px = resolution_px
                self.color_space = color_space

        class FakeGSplat:
            class GSplatRenderer:
                def __init__(self, color_space):
                    self.color_space = color_space

        selected_sample_ids: list[str] = []
        rendered_phases: list[float] = []

        def select_phase(context):
            selected_sample_ids.append(context.sample_id)
            return 3

        def render_variant(_gaussians, _metadata, *, phase, **_kwargs):
            rendered_phases.append(phase)
            return Image.new("RGB", (4, 4), (10, 20, 30))

        attack._ensure_predictor = lambda _device: None
        attack._predictor = object()
        attack._predictor_device = "cpu"
        attack._checkpoint_path = Path("sharp.pt")
        attack._checkpoint_url = "local"
        attack._source_root = Path("ml_sharp")
        attack._sharp_modules = {
            "io": FakeIO,
            "predict_image": lambda *_args, **_kwargs: FakeGaussians(),
            "SceneMetaData": FakeSceneMetaData,
            "save_ply": lambda *_args, **_kwargs: None,
            "gsplat": FakeGSplat,
        }
        attack._select_phase_index = select_phase
        attack._render_variant = render_variant

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.png"
            output_path = Path(tmpdir) / "output.png"
            Image.new("RGB", (4, 4), (80, 120, 160)).save(input_path)
            metadata = attack.apply(
                input_path,
                output_path,
                context=type(
                    "Context",
                    (),
                    {
                        "run_id": "run",
                        "sample_id": "sample",
                        "attack_name": attack.name,
                        "params": attack.params,
                        "workspace_dir": Path(tmpdir),
                        "device": "cpu",
                        "seed": 42,
                    },
                )(),
            )

            self.assertEqual(selected_sample_ids, ["sample"])
            self.assertEqual(rendered_phases, [3 / 8])
            self.assertEqual(metadata["phase_index"], 3)
            self.assertEqual(metadata["variant_count"], 1)
            self.assertEqual(metadata["variant_output_path"], None)
            self.assertTrue(output_path.exists())

    def test_physical_channel_attacks_are_strength_mapped_without_level_variants(self) -> None:
        resources = list_attack_resources()
        exposed_methods = {item["method"] for item in resources}

        for method in ("screen_shoot", "print_camera", "combined_physical"):
            attack = get_attack_catalog_item(method)
            self.assertEqual(attack["category"], "physical_channel_attacks")
            self.assertEqual(attack["strengthParam"], "strength")
            self.assertEqual(attack["strengths"], [0.0, 0.5, 1.0])
            self.assertFalse(attack["requiresGpu"])

        self.assertNotIn("screen_shoot_mild", exposed_methods)
        self.assertNotIn("screen_shoot_strong_uncorrected", exposed_methods)
        self.assertNotIn("print_camera_strong", exposed_methods)
        self.assertNotIn("combined_physical_strong", exposed_methods)

    def test_identity_attack_is_not_grouped_as_distortion(self) -> None:
        identity = get_attack_catalog_item("identity")

        self.assertEqual(identity["category"], "identity")
        self.assertEqual(identity["categoryLabel"], "Identity")
        self.assertEqual(identity["strengthParam"], None)
        self.assertEqual(identity["strengths"], [0.0])

    def test_legacy_attack_presets_are_hidden_from_frontend_catalog(self) -> None:
        resources = list_attack_resources()
        exposed_ids = {item["id"] for item in resources}

        self.assertNotIn("atk-jpeg-smoke", exposed_ids)
        self.assertNotIn("atk-blur-smoke", exposed_ids)
        self.assertNotIn("atk-jpeg-sweep", exposed_ids)
        self.assertNotIn("atk-blur-sweep", exposed_ids)
        self.assertNotIn("atk-crop-sweep", exposed_ids)
        self.assertEqual(get_attack_catalog_item("atk-jpeg-smoke")["id"], "atk-jpeg")
        self.assertEqual(get_attack_catalog_item("atk-blur-sweep")["id"], "atk-gaussian-blur")

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
        self.assertEqual(_attack_params(get_attack_catalog_item("2x_regen"), 1.0), {"strength": 1.0})
        self.assertEqual(_attack_params(get_attack_catalog_item("4x_regen"), 0.0), {"strength": 0.0})
        self.assertEqual(_attack_params(get_attack_catalog_item("screen_shoot"), 0.5), {"strength": 0.5})
        self.assertEqual(_attack_params(get_attack_catalog_item("combined_physical"), 1.0), {"strength": 1.0})
        self.assertEqual(
            _attack_params(get_attack_catalog_item("3d_viewpoint_rerendering_rotate_point"), 1.0),
            {"strength": 1.0},
        )

    def test_attack_strength_overrides_are_used_by_runner(self) -> None:
        attack_id = "atk-3d-viewpoint-rerendering-rotate-point"
        with tempfile.TemporaryDirectory() as tmpdir:
            normalized = normalize_selection(
                {
                    "datasetIds": ["demo"],
                    "algorithmIds": ["alg-invisible-watermark-dwtdct"],
                    "attackPresetIds": [attack_id],
                    "attackStrengthOverrides": {
                        attack_id: [1.0, "0", 1.0, "bad", float("nan")],
                        "atk-identity": [0.0],
                    },
                    "seeds": [42],
                    "maxSamples": 1,
                },
                Path(tmpdir),
            )

        attack = get_attack_catalog_item(attack_id)
        self.assertEqual(normalized["attackStrengthOverrides"], {attack_id: [0.0, 1.0]})
        self.assertEqual(_strengths_for_attack(normalized, attack_id, attack), [0.0, 1.0])
        self.assertEqual(
            _strengths_for_attack({"attackStrengthOverrides": {}}, attack_id, attack),
            [0.0, 0.5, 1.0],
        )

    def test_attack_strength_overrides_are_used_by_estimator(self) -> None:
        attack_id = "atk-3d-viewpoint-rerendering-rotate-point"
        with tempfile.TemporaryDirectory() as tmpdir:
            resources_root = Path(tmpdir)
            dataset_dir = resources_root / "datasets" / "demo"
            dataset_dir.mkdir(parents=True)
            Image.new("RGB", (16, 16), (80, 120, 160)).save(dataset_dir / "sample.png")

            estimate = estimate_selection(
                {
                    "datasetIds": ["demo"],
                    "algorithmIds": ["alg-invisible-watermark-dwtdct"],
                    "attackPresetIds": [attack_id],
                    "attackStrengthOverrides": {attack_id: [0.0, 1.0]},
                    "seeds": [42, 123],
                    "maxSamples": 1,
                },
                resources_root,
            )

        self.assertEqual(estimate["cellCount"], 4)
        self.assertEqual(estimate["sampleCount"], 1)
        self.assertEqual(estimate["imageOperationCount"], 4)

    def test_attack_param_overrides_are_used_by_regeneration_variants(self) -> None:
        attack_id = "atk-regen-vae"
        with tempfile.TemporaryDirectory() as tmpdir:
            normalized = normalize_selection(
                {
                    "datasetIds": ["demo"],
                    "algorithmIds": ["alg-invisible-watermark-dwtdct"],
                    "attackPresetIds": [attack_id],
                    "attackParamOverrides": {
                        attack_id: [
                            {"vae_model_name": "cheng2020-anchor", "quality": 1},
                            {"vae_model_name": "cheng2020-anchor", "quality": 3},
                            {"quality": float("nan")},
                            {"unknown": None},
                        ],
                    },
                    "seeds": [42],
                    "maxSamples": 1,
                },
                Path(tmpdir),
            )

        attack = get_attack_catalog_item(attack_id)
        variants = _attack_variants_for_attack(normalized, attack_id, attack)
        self.assertEqual(
            normalized["attackParamOverrides"],
            {
                attack_id: [
                    {"vae_model_name": "cheng2020-anchor", "quality": 1},
                    {"vae_model_name": "cheng2020-anchor", "quality": 3},
                ]
            },
        )
        self.assertEqual(len(variants), 2)
        self.assertEqual(variants[0][1], {"vae_model_name": "cheng2020-anchor", "quality": 1})
        self.assertEqual(variants[1][1], {"vae_model_name": "cheng2020-anchor", "quality": 3})

    def test_xy_strength_overrides_are_used_by_regeneration_estimator(self) -> None:
        attack_id = "atk-image-to-vedio"
        with tempfile.TemporaryDirectory() as tmpdir:
            resources_root = Path(tmpdir)
            dataset_dir = resources_root / "datasets" / "demo"
            dataset_dir.mkdir(parents=True)
            Image.new("RGB", (16, 16), (80, 120, 160)).save(dataset_dir / "sample.png")

            estimate = estimate_selection(
                {
                    "datasetIds": ["demo"],
                    "algorithmIds": ["alg-invisible-watermark-dwtdct"],
                    "attackPresetIds": [attack_id],
                    "attackStrengthOverrides": {attack_id: [0, 10, 20, 30, 40, 60]},
                    "seeds": [42],
                    "maxSamples": 1,
                },
                resources_root,
            )

        self.assertEqual(estimate["cellCount"], 6)
        self.assertEqual(_attack_params(get_attack_catalog_item(attack_id), 10.0), {"xy": 10})


if __name__ == "__main__":
    unittest.main()
