from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import wmbench


class UnifiedServiceCliTest(unittest.TestCase):
    def parse(self, env_file: Path, *options: str):
        return wmbench.create_parser().parse_args(
            ["--profile", "local", "--env-file", str(env_file), *options, "status"]
        )

    def test_configuration_precedence_is_cli_environment_dotenv_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("API_PORT=8100\nWEB_PORT=3100\nWM_BENCH_DEVICE=dotenv\n", encoding="utf-8")
            args = self.parse(env_file, "--api-port", "8300", "--device", "cli")
            with patch.dict(os.environ, {"API_PORT": "8200", "WEB_PORT": "3200"}, clear=True):
                config = wmbench.build_config(args)
            self.assertEqual(config.api_port, 8300)
            self.assertEqual(config.web_port, 3200)
            self.assertEqual(config.device, "cli")

    def test_environment_overrides_dotenv_without_cli_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("API_PORT=8100\nWM_BENCH_DEVICE=dotenv\n", encoding="utf-8")
            args = self.parse(env_file)
            with patch.dict(os.environ, {"API_PORT": "8200", "WM_BENCH_DEVICE": "environment"}, clear=True):
                config = wmbench.build_config(args)
            self.assertEqual(config.api_port, 8200)
            self.assertEqual(config.device, "environment")
            self.assertEqual(config.env["NEXT_PUBLIC_API_BASE_URL"], "http://localhost:8200")

    def test_custom_web_port_is_added_to_cors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("", encoding="utf-8")
            args = self.parse(env_file, "--web-port", "3456")
            with patch.dict(os.environ, {}, clear=True):
                config = wmbench.build_config(args)
            origins = config.env["WM_BENCH_CORS_ORIGINS"].split(",")
            self.assertIn("http://localhost:3456", origins)
            self.assertIn("http://127.0.0.1:3456", origins)

    def test_autodl_perceptual_helper_lives_in_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env.autodl"
            env_file.write_text("", encoding="utf-8")
            args = wmbench.create_parser().parse_args(
                ["--profile", "autodl", "--env-file", str(env_file), "status"]
            )
            with patch.dict(os.environ, {}, clear=True):
                config = wmbench.build_config(args)

            state: dict[str, object] = {}
            with patch.object(wmbench, "run") as run:
                wmbench.prepare_perceptual_weights(config, state, force=False)

            command = run.call_args.args[0]
            self.assertEqual(command[1], "scripts/prepare_perceptual_weights.py")
            self.assertIn("perceptualWeightsFingerprint", state)

    def test_python_dependency_install_can_be_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            venv_dir = Path(tmp) / "venv"
            python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            python.parent.mkdir(parents=True)
            python.touch()
            env_file.write_text(
                f"WM_BENCH_VENV={venv_dir}\nWM_BENCH_INSTALL_PYTHON_DEPS=0\n",
                encoding="utf-8",
            )
            args = self.parse(env_file)
            with patch.dict(os.environ, {}, clear=True):
                config = wmbench.build_config(args)

            with patch.object(wmbench, "run") as run:
                wmbench.prepare_python(config, {}, force=False)

            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
