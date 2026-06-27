from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.storage import StorageLayout, artifact_uri, safe_segment, sha256_file


class StorageTest(unittest.TestCase):
    def test_safe_segment_removes_path_control_characters(self) -> None:
        self.assertEqual(safe_segment("../run 1"), "run_1")
        self.assertEqual(safe_segment("cell:jpeg/0.5"), "cell_jpeg_0.5")

    def test_run_stage_dir_layout(self) -> None:
        layout = StorageLayout(Path("/data/wm-bench"))

        self.assertEqual(
            layout.run_stage_dir("run 1", "attacked", "cell:1"),
            Path("/data/wm-bench/runs/run_1/attacked/cell_1"),
        )

    def test_checksum_and_artifact_uri(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "file.txt"
            path.write_text("watermark", encoding="utf-8")
            checksum = sha256_file(path)
            uri = artifact_uri(StorageLayout(Path("/data/wm-bench")), "weights", checksum, "model.pt")

        self.assertTrue(checksum.startswith("sha256:"))
        self.assertIn("/weights/", str(uri))
        self.assertTrue(str(uri).endswith("/model.pt"))


if __name__ == "__main__":
    unittest.main()
