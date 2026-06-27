from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StorageLayout:
    root: Path

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def uploads_dir(self) -> Path:
        return self.root / "uploads"

    @property
    def weights_dir(self) -> Path:
        return self.root / "weights"

    @property
    def plugins_dir(self) -> Path:
        return self.root / "plugins"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    def run_stage_dir(self, run_id: str, stage: str, cell_id: str | None = None) -> Path:
        base = self.runs_dir / safe_segment(run_id) / safe_segment(stage)
        return base if cell_id is None else base / safe_segment(cell_id)


def safe_segment(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)
    cleaned = cleaned.strip("._")
    if not cleaned:
        raise ValueError("path segment is empty after sanitization")
    return cleaned


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def artifact_uri(layout: StorageLayout, artifact_type: str, checksum: str, filename: str) -> Path:
    digest = checksum.split(":", 1)[-1]
    if len(digest) < 12:
        raise ValueError("checksum digest is too short")
    return layout.root / safe_segment(artifact_type) / digest[:2] / digest[2:12] / safe_segment(filename)
