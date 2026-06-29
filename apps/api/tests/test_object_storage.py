from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.services.dataset_catalog import build_catalog_item, get_catalog_entry
from app.services.dataset_download import DatasetDownloadService
from app.services.object_storage import ObjectStorageClient, ObjectStorageSettings, parse_manifest_lines


class FakeObjectStorage:
    enabled = True
    files: dict[str, bytes]

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def dataset_compact_key(self, dataset_id: str) -> str:
        return f"wmbench/datasets/{dataset_id}/compact-1000.zip"

    def dataset_manifest_key(self, dataset_id: str) -> str:
        return f"wmbench/datasets/{dataset_id}/manifest.txt"

    def dataset_image_key(self, dataset_id: str, relative_path: str) -> str:
        return f"wmbench/datasets/{dataset_id}/{relative_path.lstrip('/')}"

    def exists(self, key: str) -> bool:
        return key in self.files

    def read_text(self, key: str) -> str:
        return self.files[key].decode("utf-8")

    def download_file(self, key: str, dest: Path, *, on_progress=None) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.files[key])
        if on_progress:
            on_progress(len(self.files[key]), len(self.files[key]), "done")


class ObjectStorageIntegrationTest(unittest.TestCase):
    def test_parse_manifest_lines_supports_http_and_object_keys(self) -> None:
        oss = FakeObjectStorage({})
        text = "\n".join(
            [
                "# comment",
                "https://example.com/a.jpg",
                "images/0002.jpg",
            ]
        )
        targets = parse_manifest_lines(text, dataset_id="ms-coco", oss=oss)  # type: ignore[arg-type]
        self.assertEqual(
            targets,
            [
                "https://example.com/a.jpg",
                "wmbench/datasets/ms-coco/images/0002.jpg",
            ],
        )

    def test_catalog_marks_remote_compact_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oss = FakeObjectStorage({"wmbench/datasets/ms-coco/compact-1000.zip": b"zip"})
            entry = get_catalog_entry("ms-coco")
            item = build_catalog_item(root, entry, oss=oss)  # type: ignore[arg-type]
            self.assertTrue(item["remoteCompactAvailable"])
            self.assertTrue(item["compactAvailable"])
            self.assertFalse(item["installed"])

    def test_compact_download_from_object_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
            archive.close()
            zip_path = Path(archive.name)
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("000001.jpg", b"fake")

            oss = FakeObjectStorage({"wmbench/datasets/ms-coco/compact-1000.zip": zip_path.read_bytes()})
            service = DatasetDownloadService(root, oss=oss)  # type: ignore[arg-type]
            job = service.start_download("ms-coco", mode="compact", sample_count=1000)
            for _ in range(100):
                job = service.get_job(job.id)
                if job.status in {"succeeded", "failed"}:
                    break
                import time

                time.sleep(0.05)

            self.assertEqual(job.status, "succeeded")
            self.assertTrue(job.archive_path)
            self.assertTrue(Path(job.archive_path).exists())

    def test_custom_download_from_object_storage_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = "\n".join(
                [
                    "images/a.jpg",
                    "images/b.jpg",
                    "images/c.jpg",
                ]
            )
            oss = FakeObjectStorage(
                {
                    "wmbench/datasets/ms-coco/manifest.txt": manifest.encode("utf-8"),
                    "wmbench/datasets/ms-coco/images/a.jpg": b"a",
                    "wmbench/datasets/ms-coco/images/b.jpg": b"b",
                    "wmbench/datasets/ms-coco/images/c.jpg": b"c",
                }
            )
            service = DatasetDownloadService(root, oss=oss)  # type: ignore[arg-type]
            job = service.start_download("ms-coco", mode="custom", seed=1, sample_count=2)
            for _ in range(200):
                job = service.get_job(job.id)
                if job.status in {"succeeded", "failed"}:
                    break
                import time

                time.sleep(0.05)

            self.assertEqual(job.status, "succeeded", job.error)
            output_dir = Path(job.output_dir or "")
            self.assertEqual(len(list(output_dir.glob("*"))), 2)

    def test_object_storage_disabled_by_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True), patch(
            "app.core.env_loader.load_project_env", return_value=False
        ):
            from app.services.object_storage import ObjectStorageSettings

            settings = ObjectStorageSettings.from_env()
            self.assertFalse(settings.enabled)


if __name__ == "__main__":
    unittest.main()
