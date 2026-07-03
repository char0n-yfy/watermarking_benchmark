from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


class SceneCache3DTest(unittest.TestCase):
    def tearDown(self) -> None:
        module = importlib.import_module("evaluator.attacks.3d_viewpoint_rerendering.attacks")
        module.clear_sharp_scene_cache()
        module.reset_sharp_scene_cache_runtime_min_entries()

    def test_sharp_scene_cache_is_lru_limited(self) -> None:
        module = importlib.import_module("evaluator.attacks.3d_viewpoint_rerendering.attacks")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_a = root / "a.png"
            input_b = root / "b.png"
            checkpoint = root / "checkpoint.pt"
            input_a.write_bytes(b"a")
            input_b.write_bytes(b"b")
            checkpoint.write_bytes(b"weights")

            with patch.dict(
                os.environ,
                {"WM_BENCH_3D_SCENE_CACHE": "1", "WM_BENCH_3D_SCENE_CACHE_MAX_ENTRIES": "1"},
                clear=False,
            ):
                module.clear_sharp_scene_cache()
                key_a = module._sharp_scene_cache_key(
                    input_path=input_a,
                    checkpoint_path=checkpoint,
                    source_root=root,
                    device="cuda:0",
                )
                key_b = module._sharp_scene_cache_key(
                    input_path=input_b,
                    checkpoint_path=checkpoint,
                    source_root=root,
                    device="cuda:0",
                )
                module._set_sharp_scene_cache(key_a, {"scene": "a"})
                self.assertEqual(module._get_sharp_scene_cache(key_a), {"scene": "a"})
                module._set_sharp_scene_cache(key_b, {"scene": "b"})
                self.assertIsNone(module._get_sharp_scene_cache(key_a))
                self.assertEqual(module._get_sharp_scene_cache(key_b), {"scene": "b"})
                self.assertEqual(module.sharp_scene_cache_stats()["entryCount"], 1)

    def test_runtime_min_entries_expands_effective_cache_without_lowering_config(self) -> None:
        module = importlib.import_module("evaluator.attacks.3d_viewpoint_rerendering.attacks")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"weights")
            inputs = []
            for name in ("a.png", "b.png", "c.png"):
                path = root / name
                path.write_bytes(name.encode())
                inputs.append(path)

            with patch.dict(
                os.environ,
                {"WM_BENCH_3D_SCENE_CACHE": "1", "WM_BENCH_3D_SCENE_CACHE_MAX_ENTRIES": "1"},
                clear=False,
            ):
                module.set_sharp_scene_cache_runtime_min_entries(2)
                keys = [
                    module._sharp_scene_cache_key(
                        input_path=path,
                        checkpoint_path=checkpoint,
                        source_root=root,
                        device="cuda:0",
                    )
                    for path in inputs
                ]
                module._set_sharp_scene_cache(keys[0], {"scene": "a"})
                module._set_sharp_scene_cache(keys[1], {"scene": "b"})
                stats = module.sharp_scene_cache_stats()
                self.assertEqual(stats["configuredMaxEntries"], 1)
                self.assertEqual(stats["runtimeMinEntries"], 2)
                self.assertEqual(stats["effectiveMaxEntries"], 2)
                self.assertEqual(stats["entryCount"], 2)
                module._set_sharp_scene_cache(keys[2], {"scene": "c"})
                self.assertIsNone(module._get_sharp_scene_cache(keys[0]))
                self.assertEqual(module.sharp_scene_cache_stats()["entryCount"], 2)

    def test_runtime_min_does_not_enable_disabled_scene_cache(self) -> None:
        module = importlib.import_module("evaluator.attacks.3d_viewpoint_rerendering.attacks")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "a.png"
            checkpoint = root / "checkpoint.pt"
            input_path.write_bytes(b"a")
            checkpoint.write_bytes(b"weights")
            with patch.dict(
                os.environ,
                {"WM_BENCH_3D_SCENE_CACHE": "0", "WM_BENCH_3D_SCENE_CACHE_MAX_ENTRIES": "1"},
                clear=False,
            ):
                module.set_sharp_scene_cache_runtime_min_entries(2)
                key = module._sharp_scene_cache_key(
                    input_path=input_path,
                    checkpoint_path=checkpoint,
                    source_root=root,
                    device="cuda:0",
                )
                module._set_sharp_scene_cache(key, {"scene": "a"})
                self.assertIsNone(module._get_sharp_scene_cache(key))
                self.assertFalse(module.sharp_scene_cache_stats()["enabled"])

    def test_model_cache_params_ignore_runtime_strength_and_configure_runtime_updates_it(self) -> None:
        module = importlib.import_module("evaluator.attacks.3d_viewpoint_rerendering.attacks")
        attack_cls = next(cls for cls in module.VIEWPOINT_ATTACK_CLASSES if cls.name == "3d_viewpoint_rerendering_rotate_point")
        base_params = {
            "checkpoint_path": "/tmp/checkpoint.pt",
            "source_root": "/tmp/source",
            "device": "cuda:0",
            "strength": 0.25,
            "save_intermediates": False,
        }
        changed_params = {**base_params, "strength": 0.75, "max_disparity": 0.1, "save_intermediates": True}
        self.assertEqual(attack_cls.model_cache_params(base_params), attack_cls.model_cache_params(changed_params))

        attack = attack_cls(strength=0.25, save_intermediates=True)
        attack.configure_runtime({"strength": 0.75, "save_intermediates": False, "image_size": 512})
        self.assertEqual(attack.strength, 0.75)
        self.assertFalse(attack.save_intermediates)
        self.assertEqual(attack.image_size, 512)


if __name__ == "__main__":
    unittest.main()
