from __future__ import annotations

import random
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx

from app.services.dataset_catalog import (
    COMPACT_SAMPLE_COUNT,
    compact_dir,
    full_dir,
    get_catalog_entry,
)
from app.services.object_storage import ObjectStorageClient, parse_manifest_lines
from app.services.resources import iter_image_paths


DownloadMode = Literal["compact", "custom"]
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


@dataclass
class DatasetDownloadJob:
    id: str
    dataset_id: str
    mode: DownloadMode
    status: JobStatus = "queued"
    progress: int = 0
    total_items: int = 0
    completed_items: int = 0
    seed: int | None = None
    sample_count: int | None = None
    message: str | None = None
    error: str | None = None
    output_dir: str | None = None
    archive_path: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    bytes_downloaded: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "datasetId": self.dataset_id,
            "mode": self.mode,
            "status": self.status,
            "progress": self.progress,
            "totalItems": self.total_items,
            "completedItems": self.completed_items,
            "seed": self.seed,
            "sampleCount": self.sample_count,
            "message": self.message,
            "error": self.error,
            "outputDir": self.output_dir,
            "archivePath": self.archive_path,
            "bytesDownloaded": self.bytes_downloaded,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }


class DatasetDownloadService:
    def __init__(self, resources_root: Path, *, oss: ObjectStorageClient | None = None) -> None:
        self.resources_root = resources_root
        self.oss = oss
        self.cache_root = resources_root / "cache" / "dataset-downloads"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, DatasetDownloadJob] = {}
        self._lock = threading.Lock()

    def get_job(self, job_id: str) -> DatasetDownloadJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"Unknown download job: {job_id}")
        return job

    def list_jobs(self, dataset_id: str | None = None) -> list[DatasetDownloadJob]:
        with self._lock:
            jobs = list(self._jobs.values())
        if dataset_id is not None:
            jobs = [job for job in jobs if job.dataset_id == dataset_id]
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    def start_download(
        self,
        dataset_id: str,
        *,
        mode: DownloadMode,
        seed: int = 42,
        sample_count: int = 100,
    ) -> DatasetDownloadJob:
        if mode == "custom" and sample_count <= 0:
            raise ValueError("sample_count must be positive for custom downloads")
        if mode == "compact" and sample_count != COMPACT_SAMPLE_COUNT:
            sample_count = COMPACT_SAMPLE_COUNT

        job_id = f"dl_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        job = DatasetDownloadJob(
            id=job_id,
            dataset_id=dataset_id,
            mode=mode,
            seed=seed if mode == "custom" else None,
            sample_count=sample_count,
            message="queued",
        )
        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id,),
            name=f"dataset-download-{job_id}",
            daemon=True,
        )
        thread.start()
        return job

    def _set_progress(self, job: DatasetDownloadJob, completed: int, total: int, *, message: str) -> None:
        job.completed_items = completed
        job.total_items = total
        job.message = message
        job.updated_at = time.time()
        if total > 0:
            job.progress = int(round((completed / total) * 100))
        else:
            job.progress = 0

    def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        try:
            job.status = "running"
            job.message = "starting"
            job.updated_at = time.time()
            if job.mode == "compact":
                self._run_compact(job)
            else:
                self._run_custom(job)
            self._set_progress(job, job.total_items, job.total_items, message="completed")
            job.status = "succeeded"
            job.progress = 100
            job.updated_at = time.time()
        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = "failed"
            job.updated_at = time.time()

    def _run_compact(self, job: DatasetDownloadJob) -> None:
        entry = get_catalog_entry(job.dataset_id)
        source = compact_dir(
            self.resources_root,
            job.dataset_id,
            compact_uses_root=entry.compact_uses_root,
        )
        images = iter_image_paths(source)
        if images:
            self._run_compact_from_local(job, images)
            return

        if self.oss and self.oss.enabled:
            object_key = self.oss.dataset_compact_key(job.dataset_id)
            if self.oss.exists(object_key):
                self._run_compact_from_object_storage(job, object_key)
                return

        oss_hint = "wmbench/datasets/<id>/compact-1000.zip"
        if self.oss and self.oss.enabled:
            oss_hint = self.oss.dataset_compact_key(job.dataset_id)

        raise FileNotFoundError(
            f"Compact dataset not found under {source}. "
            f"Place {COMPACT_SAMPLE_COUNT} images in datasets/{job.dataset_id}/compact/, "
            f"upload {oss_hint} to object storage, or enable compactUsesRoot."
        )

    def _run_compact_from_local(self, job: DatasetDownloadJob, images: list[Path]) -> None:
        job_dir = self.cache_root / job.id
        if job_dir.exists():
            shutil.rmtree(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)

        selected = images[: job.sample_count or len(images)]
        total_steps = len(selected) * 2
        step = 0

        for index, image_path in enumerate(selected, start=1):
            target_name = f"{index:06d}{image_path.suffix.lower()}"
            shutil.copy2(image_path, job_dir / target_name)
            step += 1
            self._set_progress(job, step, total_steps, message=f"打包图片 {index}/{len(selected)}")

        archive_path = self.cache_root / f"{job.id}.zip"
        if archive_path.exists():
            archive_path.unlink()
        packed_files = sorted(path for path in job_dir.iterdir() if path.is_file())
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index, image_path in enumerate(packed_files, start=1):
                archive.write(image_path, arcname=image_path.name)
                step += 1
                self._set_progress(job, step, total_steps, message=f"压缩 zip {index}/{len(packed_files)}")

        job.output_dir = str(job_dir)
        job.archive_path = str(archive_path)

    def _run_compact_from_object_storage(self, job: DatasetDownloadJob, object_key: str) -> None:
        archive_path = self.cache_root / f"{job.id}.zip"
        if archive_path.exists():
            archive_path.unlink()

        def on_progress(completed: int, total: int, message: str) -> None:
            total_steps = max(total, 1)
            self._set_progress(job, min(completed, total_steps), total_steps, message=message)
            job.bytes_downloaded = completed

        job.message = "从对象存储下载精简包"
        self.oss.download_file(object_key, archive_path, on_progress=on_progress)
        job.archive_path = str(archive_path)
        self._set_progress(job, 1, 1, message="对象存储精简包已就绪")

    def _run_custom(self, job: DatasetDownloadJob) -> None:
        entry = get_catalog_entry(job.dataset_id)
        output_root = self.resources_root / "datasets" / job.dataset_id / "custom" / f"seed{job.seed}_{job.sample_count}"
        if output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        job.output_dir = str(output_root)

        if entry.manifest_url:
            manifest_text = self._fetch_manifest_text(entry.manifest_url)
            targets = parse_manifest_lines(manifest_text, dataset_id=job.dataset_id, oss=self.oss)
            self._run_custom_from_targets(job, targets, output_root)
            return

        if self.oss and self.oss.enabled:
            manifest_key = self.oss.dataset_manifest_key(job.dataset_id)
            if self.oss.exists(manifest_key):
                manifest_text = self.oss.read_text(manifest_key)
                targets = parse_manifest_lines(manifest_text, dataset_id=job.dataset_id, oss=self.oss)
                self._run_custom_from_targets(job, targets, output_root)
                return

        source = full_dir(self.resources_root, job.dataset_id)
        images = iter_image_paths(source)
        compact = compact_dir(
            self.resources_root,
            job.dataset_id,
            compact_uses_root=entry.compact_uses_root,
        )
        compact_set = {path.resolve() for path in iter_image_paths(compact)}
        pool = [path for path in images if path.resolve() not in compact_set] or images
        if not pool:
            raise FileNotFoundError(
                "No local source images available for custom sampling. "
                f"Add a full dataset under datasets/{job.dataset_id}/full/, "
                f"upload manifest.txt to object storage, or configure manifestUrl in the dataset catalog."
            )

        rng = random.Random(job.seed)
        if len(pool) >= job.sample_count:
            selected = rng.sample(pool, job.sample_count)
        else:
            selected = [rng.choice(pool) for _ in range(job.sample_count)]

        total_steps = len(selected) * 2
        step = 0
        for index, image_path in enumerate(selected, start=1):
            target_name = f"{index:06d}{image_path.suffix.lower()}"
            shutil.copy2(image_path, output_root / target_name)
            step += 1
            self._set_progress(job, step, total_steps, message=f"采样图片 {index}/{len(selected)}")

        archive_path = self.cache_root / f"{job.id}.zip"
        self._create_zip_with_progress(
            job,
            output_root,
            archive_path,
            start_step=len(selected),
            total_steps=total_steps,
        )
        job.archive_path = str(archive_path)

    def _run_custom_from_targets(self, job: DatasetDownloadJob, targets: list[str], output_root: Path) -> None:
        if not targets:
            raise ValueError("Manifest did not contain downloadable entries")

        rng = random.Random(job.seed)
        rng.shuffle(targets)
        selected = targets[: job.sample_count]
        total_steps = len(selected) * 2
        self._set_progress(job, 0, total_steps, message="从 manifest 下载")
        self._download_targets(job, selected, output_root, step_offset=0, total_steps=total_steps)
        archive_path = self.cache_root / f"{job.id}.zip"
        self._create_zip_with_progress(
            job,
            output_root,
            archive_path,
            start_step=len(selected),
            total_steps=total_steps,
        )
        job.archive_path = str(archive_path)

    def _fetch_manifest_text(self, manifest_url: str) -> str:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = client.get(manifest_url)
            response.raise_for_status()
            return response.text

    def _download_targets(
        self,
        job: DatasetDownloadJob,
        targets: list[str],
        output_root: Path,
        *,
        step_offset: int,
        total_steps: int,
    ) -> None:
        for index, target in enumerate(targets, start=1):
            suffix = Path(urlparse(target).path).suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".zip"}:
                suffix = ".jpg"
            dest = output_root / f"{index:06d}{suffix}"
            if target.startswith(("http://", "https://")):
                self._download_http_url(job, target, dest)
            elif self.oss and self.oss.enabled:
                self.oss.download_file(target, dest)
            else:
                raise ValueError(f"Cannot download non-HTTP target without object storage: {target}")
            self._set_progress(
                job,
                step_offset + index,
                total_steps,
                message=f"下载 {index}/{len(targets)}",
            )

    def _download_http_url(self, job: DatasetDownloadJob, url: str, dest: Path) -> None:
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with dest.open("wb") as handle:
                    for chunk in response.iter_bytes(chunk_size=1024 * 256):
                        if chunk:
                            handle.write(chunk)
                            job.bytes_downloaded += len(chunk)

    def _create_zip_with_progress(
        self,
        job: DatasetDownloadJob,
        source_dir: Path,
        archive_path: Path,
        *,
        start_step: int,
        total_steps: int,
    ) -> None:
        if archive_path.exists():
            archive_path.unlink()
        files = sorted(path for path in source_dir.iterdir() if path.is_file())
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index, file_path in enumerate(files, start=1):
                archive.write(file_path, arcname=file_path.name)
                self._set_progress(
                    job,
                    start_step + index,
                    total_steps,
                    message=f"压缩 zip {index}/{len(files)}",
                )
