from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import env_loader


class EnvLoaderTest(unittest.TestCase):
    def test_load_project_env_reads_repo_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = root / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "# comment",
                        "WM_BENCH_OSS_ENABLED=true",
                        "WM_BENCH_OSS_BUCKET=test-bucket",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.object(env_loader, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True):
                env_loader._LOADED = False
                loaded = env_loader.load_project_env()
                self.assertTrue(loaded)
                self.assertEqual(os.environ["WM_BENCH_OSS_BUCKET"], "test-bucket")

    def test_load_project_env_does_not_override_existing_variables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("WM_BENCH_OSS_BUCKET=from-file\n", encoding="utf-8")
            with patch.object(env_loader, "PROJECT_ROOT", root), patch.dict(
                os.environ, {"WM_BENCH_OSS_BUCKET": "from-shell"}, clear=True
            ):
                env_loader._LOADED = False
                env_loader.load_project_env()
                self.assertEqual(os.environ["WM_BENCH_OSS_BUCKET"], "from-shell")

    def test_load_project_env_uses_configured_dotenv_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            autodl_env = root / ".env.autodl"
            autodl_env.write_text("APP_ENV=autodl\nAPI_PORT=6006\n", encoding="utf-8")
            (root / ".env").write_text("APP_ENV=development\nAPI_PORT=8000\n", encoding="utf-8")
            with patch.object(env_loader, "PROJECT_ROOT", root), patch.dict(
                os.environ, {"WM_BENCH_DOTENV_PATH": str(autodl_env)}, clear=True
            ):
                env_loader._LOADED = False
                loaded = env_loader.load_project_env()
                self.assertTrue(loaded)
                self.assertEqual(os.environ["APP_ENV"], "autodl")
                self.assertEqual(os.environ["API_PORT"], "6006")


if __name__ == "__main__":
    unittest.main()
