from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluator.watermarking.utils import packaged_algorithm_dir, prepend_sys_path


class DWSFImportIsolationTest(unittest.TestCase):
    def test_dwsf_utils_package_wins_over_leaked_attack_utils_module(self) -> None:
        dwsf_root = packaged_algorithm_dir("dwsf")
        leaked_attack_root = (
            Path(__file__).resolve().parents[3]
            / "evaluator"
            / "attacks"
            / "regeneration_attacks"
            / "backends"
            / "ctrlregen"
        )
        self.assertTrue((leaked_attack_root / "utils.py").exists())

        saved_path = list(sys.path)
        saved_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "utils"
            or name.startswith("utils.")
            or name == "networks"
            or name.startswith("networks.")
        }
        try:
            for name in list(saved_modules):
                sys.modules.pop(name, None)
            sys.path.insert(0, str(leaked_attack_root))

            with prepend_sys_path(dwsf_root, ["networks", "utils"]):
                utils_spec = importlib.util.find_spec("utils")
                img_spec = importlib.util.find_spec("utils.img")
                networks_spec = importlib.util.find_spec("networks")

            self.assertIsNotNone(utils_spec)
            self.assertIsNotNone(img_spec)
            self.assertIsNotNone(networks_spec)
            self.assertEqual(Path(str(utils_spec.origin)).resolve(), dwsf_root / "utils" / "__init__.py")
            self.assertEqual(Path(str(img_spec.origin)).resolve(), dwsf_root / "utils" / "img.py")
            self.assertEqual(Path(str(networks_spec.origin)).resolve(), dwsf_root / "networks" / "__init__.py")
        finally:
            sys.path[:] = saved_path
            for name in [
                name
                for name in sys.modules
                if name == "utils"
                or name.startswith("utils.")
                or name == "networks"
                or name.startswith("networks.")
            ]:
                sys.modules.pop(name, None)
            sys.modules.update(saved_modules)


if __name__ == "__main__":
    unittest.main()
