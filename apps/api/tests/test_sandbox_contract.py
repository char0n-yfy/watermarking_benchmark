from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worker"))

from sandbox import SandboxSpec, docker_run_command


class SandboxContractTest(unittest.TestCase):
    def test_docker_command_uses_restricted_mounts(self) -> None:
        command = docker_run_command(
            SandboxSpec(
                image_ref="wm/plugin:reviewed",
                input_dir=Path("/data/wm-bench/runs/run-1/input"),
                output_dir=Path("/data/wm-bench/runs/run-1/output"),
                weights_dir=Path("/data/wm-bench/weights/default"),
                gpu_device="0",
                env={"WM_DEVICE": "cuda:0"},
            ),
            ["python", "-m", "plugin.run"],
        )
        joined = " ".join(command)

        self.assertIn("--network none", joined)
        self.assertIn("--read-only", command)
        self.assertIn("target=/workspace/input,readonly", joined)
        self.assertIn("target=/workspace/weights,readonly", joined)
        self.assertIn("target=/workspace/output", joined)
        self.assertIn("--gpus device=0", joined)
        self.assertEqual(command[-3:], ["python", "-m", "plugin.run"])


if __name__ == "__main__":
    unittest.main()
