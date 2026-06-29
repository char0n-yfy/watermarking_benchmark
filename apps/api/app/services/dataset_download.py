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
from app.services.resources import IMAGE_EXTS, iter_image_paths


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

        job_id = self._build_job_id(dataset_id, mode=mode, seed=seed, sample_count=sample_count)
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
            job.status = "succeeded"
            job.progress = 100
            job.updated_at = time.time()
        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = "failed"
            job.updated_at = time.time()

    def _build_job_id(
        self,
        dataset_id: str,
        *,
        mode: DownloadMode,
        seed: int,
        sample_count: int,
    ) -> str:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:6]
        if mode == "compact":
            return f"{dataset_id}__compact__{stamp}_{suffix}"
        return f"{dataset_id}__custom__seed{seed}_{sample_count}__{stamp}_{suffix}"

    def _canonical_cache_archive(
        self,
        dataset_id: str,
        mode: DownloadMode,
        *,
        seed: int | None = None,
        sample_count: int | None = None,
    ) -> Path:
        if mode == "compact":
            return self.cache_root / f"{dataset_id}__compact.zip"
        return self.cache_root / f"{dataset_id}__custom__seed{seed}_{sample_count}.zip"

    def _install_dir_ready(self, install_dir: Path, job: DatasetDownloadJob) -> bool:
        count = len(iter_image_paths(install_dir))
        if count == 0:
            return False
        expected = job.sample_count or (COMPACT_SAMPLE_COUNT if job.mode == "compact" else 1)
        if job.mode == "compact":
            return count >= min(expected, COMPACT_SAMPLE_COUNT)
        return count >= expected

    def _zip_image_count(self, archive_path: Path) -> int:
        with zipfile.ZipFile(archive_path, "r") as archive:
            return sum(
                1
                for name in archive.namelist()
                if not name.endswith("/") and Path(name).suffix.lower() in IMAGE_EXTS
            )

    def _is_valid_cache_archive(self, archive_path: Path, job: DatasetDownloadJob) -> bool:
        if not archive_path.is_file() or archive_path.stat().st_size < 1024:
            return False
        try:
            with zipfile.ZipFile(archive_path, "r") as archive:
                if archive.testzip() is not None:
                    return False
            image_count = self._zip_image_count(archive_path)
        except (OSError, zipfile.BadZipFile):
            return False
        expected = job.sample_count or (COMPACT_SAMPLE_COUNT if job.mode == "compact" else 1)
        if job.mode == "compact":
            return image_count >= min(expected, max(1, COMPACT_SAMPLE_COUNT // 10))
        return image_count >= expected

    def _find_cached_archive(self, job: DatasetDownloadJob) -> Path | None:
        canonical = self._canonical_cache_archive(
            job.dataset_id,
            job.mode,
            seed=job.seed,
            sample_count=job.sample_count,
        )
        if self._is_valid_cache_archive(canonical, job):
            return canonical

        if job.mode == "compact":
            pattern = f"{job.dataset_id}__compact__*.zip"
        else:
            pattern = f"{job.dataset_id}__custom__seed{job.seed}_{job.sample_count}__*.zip"
        for candidate in sorted(self.cache_root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True):
            if self._is_valid_cache_archive(candidate, job):
                return candidate
        return None

    def _mark_already_installed(self, job: DatasetDownloadJob, install_dir: Path) -> None:
        cached = self._find_cached_archive(job)
        count = len(iter_image_paths(install_dir))
        job.output_dir = str(install_dir)
        job.archive_path = str(cached) if cached else None
        self._set_progress(
            job,
            count,
            count,
            message=f"已安装（{count} 张），跳过重复下载",
        )

    def _archive_path(self, job: DatasetDownloadJob) -> Path:
        return self._canonical_cache_archive(
            job.dataset_id,
            job.mode,
            seed=job.seed,
            sample_count=job.sample_count,
        )

    def _staging_dir(self, job: DatasetDownloadJob) -> Path:
        return self.cache_root / job.id

    def _install_dir(self, job: DatasetDownloadJob) -> Path:
        entry = get_catalog_entry(job.dataset_id)
        dataset_root = self.resources_root / "datasets" / job.dataset_id
        if job.mode == "compact":
            if entry.compact_uses_root:
                return dataset_root
            return dataset_root / "compact"
        return dataset_root / "custom" / f"seed{job.seed}_{job.sample_count}"

    def _finalize_from_archive(self, job: DatasetDownloadJob, archive_path: Path) -> None:
        install_dir = self._install_dir(job)
        if self._install_dir_ready(install_dir, job):
            self._mark_already_installed(job, install_dir)
            return

        self._set_progress(job, job.completed_items, max(job.total_items, 1), message="解压到数据集目录")
        self._extract_archive(archive_path, install_dir)
        job.output_dir = str(install_dir)
        job.archive_path = str(archive_path)
        image_count = len(iter_image_paths(install_dir))
        self._set_progress(
            job,
            max(job.total_items, image_count),
            max(job.total_items, image_count),
            message=f"已安装到 {install_dir.relative_to(self.resources_root)}（{image_count} 张）",
        )

    def _extract_archive(self, archive_path: Path, target_dir: Path) -> None:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(target_dir)
        self._maybe_flatten_extracted_dir(target_dir)

    def _maybe_flatten_extracted_dir(self, target_dir: Path) -> None:
        while not iter_image_paths(target_dir):
            subdirs = [path for path in target_dir.iterdir() if path.is_dir()]
            if len(subdirs) != 1:
                return
            nested = subdirs[0]
            for item in nested.iterdir():
                destination = target_dir / item.name
                if destination.exists():
                    if destination.is_dir():
                        shutil.rmtree(destination)
                    else:
                        destination.unlink()
                shutil.move(str(item), str(destination))
            nested.rmdir()

    def _create_zip_from_dir(
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
                    message=f"写入 cache zip {index}/{len(files)}",
                )

    def _run_compact(self, job: DatasetDownloadJob) -> None:
        install_dir = self._install_dir(job)
        if self._install_dir_ready(install_dir, job):
            self._mark_already_installed(job, install_dir)
            return

        cached = self._find_cached_archive(job)
        if cached:
            job.message = "使用 cache 中的精简包"
            self._finalize_from_archive(job, cached)
            return

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
        staging_dir = self._staging_dir(job)
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)

        selected = images[: job.sample_count or len(images)]
        total_steps = len(selected) * 2 + 1
        step = 0

        for index, image_path in enumerate(selected, start=1):
            target_name = f"{index:06d}{image_path.suffix.lower()}"
            shutil.copy2(image_path, staging_dir / target_name)
            step += 1
            self._set_progress(job, step, total_steps, message=f"打包图片 {index}/{len(selected)}")

        archive_path = self._archive_path(job)
        self._create_zip_from_dir(
            job,
            staging_dir,
            archive_path,
            start_step=len(selected),
            total_steps=total_steps,
        )
        self._finalize_from_archive(job, archive_path)

    def _run_compact_from_object_storage(self, job: DatasetDownloadJob, object_key: str) -> None:
        archive_path = self._archive_path(job)
        if archive_path.exists() and not self._is_valid_cache_archive(archive_path, job):
            archive_path.unlink()

        def on_progress(completed: int, total: int, message: str) -> None:
            total_steps = max(total, 1)
            self._set_progress(job, min(completed, total_steps), total_steps, message=message)
            job.bytes_downloaded = completed

        job.message = "下载精简包到 cache"
        self.oss.download_file(object_key, archive_path, on_progress=on_progress)
        self._finalize_from_archive(job, archive_path)

    def _run_custom(self, job: DatasetDownloadJob) -> None:
        install_dir = self._install_dir(job)
        if self._install_dir_ready(install_dir, job):
            self._mark_already_installed(job, install_dir)
            return

        cached = self._find_cached_archive(job)
        if cached:
            job.message = "使用 cache 中的自定义采样包"
            self._finalize_from_archive(job, cached)
            return

        entry = get_catalog_entry(job.dataset_id)
        staging_dir = self._staging_dir(job)
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)

        if entry.manifest_url:
            manifest_text = self._fetch_manifest_text(entry.manifest_url)
            targets = parse_manifest_lines(manifest_text, dataset_id=job.dataset_id, oss=self.oss)
            self._run_custom_from_targets(job, targets, staging_dir)
            return

        if self.oss and self.oss.enabled:
            manifest_key = self.oss.dataset_manifest_key(job.dataset_id)
            if self.oss.exists(manifest_key):
                manifest_text = self.oss.read_text(manifest_key)
                targets = parse_manifest_lines(manifest_text, dataset_id=job.dataset_id, oss=self.oss)
                self._run_custom_from_targets(job, targets, staging_dir)
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

        total_steps = len(selected) * 2 + 1
        step = 0
        for index, image_path in enumerate(selected, start=1):
            target_name = f"{index:06d}{image_path.suffix.lower()}"
            shutil.copy2(image_path, staging_dir / target_name)
            step += 1
            self._set_progress(job, step, total_steps, message=f"采样图片 {index}/{len(selected)}")

        archive_path = self._archive_path(job)
        self._create_zip_from_dir(
            job,
            staging_dir,
            archive_path,
            start_step=len(selected),
            total_steps=total_steps,
        )
        self._finalize_from_archive(job, archive_path)

    def _run_custom_from_targets(self, job: DatasetDownloadJob, targets: list[str], staging_dir: Path) -> None:
        if not targets:
            raise ValueError("Manifest did not contain downloadable entries")

        rng = random.Random(job.seed)
        rng.shuffle(targets)
        selected = targets[: job.sample_count]
        total_steps = len(selected) * 2 + 1
        self._set_progress(job, 0, total_steps, message="从 manifest 下载到 cache")
        self._download_targets(job, selected, staging_dir, step_offset=0, total_steps=total_steps)
        archive_path = self._archive_path(job)
        self._create_zip_from_dir(
            job,
            staging_dir,
            archive_path,
            start_step=len(selected),
            total_steps=total_steps,
        )
        self._finalize_from_archive(job, archive_path)

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

