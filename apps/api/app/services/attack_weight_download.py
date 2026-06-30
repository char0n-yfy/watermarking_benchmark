from __future__ import annotations

import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from app.services.attack_weights import (
    attack_weights_dir_name,
    attack_weights_install_dir,
    attack_weights_installed,
    attack_weights_need_download,
    iter_weight_files,
    methods_for_attack_weights_dir,
)
from app.services.object_storage import ObjectStorageClient


JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


@dataclass
class AttackWeightDownloadJob:
    id: str
    method: str
    weights_dir: str
    status: JobStatus = "queued"
    progress: int = 0
    total_items: int = 0
    completed_items: int = 0
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
            "method": self.method,
            "weightsDir": self.weights_dir,
            "status": self.status,
            "progress": self.progress,
            "totalItems": self.total_items,
            "completedItems": self.completed_items,
            "message": self.message,
            "error": self.error,
            "outputDir": self.output_dir,
            "archivePath": self.archive_path,
            "bytesDownloaded": self.bytes_downloaded,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }


class AttackWeightDownloadService:
    def __init__(self, resources_root: Path, *, oss: ObjectStorageClient | None = None) -> None:
        self.resources_root = resources_root
        self.oss = oss
        self.cache_root = resources_root / "cache" / "attack-weight-downloads"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, AttackWeightDownloadJob] = {}
        self._lock = threading.Lock()

    def get_job(self, job_id: str) -> AttackWeightDownloadJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"Unknown attack weight download job: {job_id}")
        return job

    def start_download(self, method: str) -> AttackWeightDownloadJob:
        if not attack_weights_need_download(method):
            raise ValueError(f"Attack method does not use packaged weights: {method}")
        directory = attack_weights_dir_name(method)
        if directory is None:
            raise ValueError(f"Missing attack weights directory mapping for: {method}")

        job_id = self._build_job_id(directory)
        install_dir = attack_weights_install_dir(self.resources_root, method)
        if attack_weights_installed(install_dir):
            job = AttackWeightDownloadJob(
                id=job_id,
                method=method,
                weights_dir=directory,
                status="succeeded",
            )
            with self._lock:
                self._jobs[job_id] = job
            self._mark_already_installed(job, install_dir)
            job.status = "succeeded"
            job.progress = 100
            return job

        job = AttackWeightDownloadJob(
            id=job_id,
            method=method,
            weights_dir=directory,
        )
        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id,),
            name=f"attack-weight-download-{job_id}",
            daemon=True,
        )
        thread.start()
        return job

    def _build_job_id(self, weights_dir: str) -> str:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:6]
        return f"{weights_dir}__attack-weights__{stamp}_{suffix}"

    def _canonical_cache_archive(self, weights_dir: str) -> Path:
        return self.cache_root / f"{weights_dir}__attack-weights.zip"

    def _set_progress(
        self,
        job: AttackWeightDownloadJob,
        completed: int,
        total: int,
        *,
        message: str | None = None,
    ) -> None:
        job.completed_items = completed
        job.total_items = total
        if message is not None:
            job.message = message
        job.updated_at = time.time()
        job.progress = int(round((completed / total) * 100)) if total > 0 else 0

    def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        try:
            job.status = "running"
            job.message = None
            job.updated_at = time.time()
            self._run_download(job)
            job.status = "succeeded"
            job.progress = 100
            job.updated_at = time.time()
        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.updated_at = time.time()

    def _run_download(self, job: AttackWeightDownloadJob) -> None:
        install_dir = attack_weights_install_dir(self.resources_root, job.method)
        if attack_weights_installed(install_dir):
            self._mark_already_installed(job, install_dir)
            return

        cached = self._find_cached_archive(job.weights_dir)
        if cached:
            self._finalize_from_archive(job, cached, install_dir)
            return

        if not self.oss or not self.oss.enabled:
            raise FileNotFoundError(
                f"攻击算法权重未安装且 OSS 未启用。请手动放入 {install_dir} 或配置 WM_BENCH_OSS_*。"
            )

        object_key = self.oss.attack_weights_key(job.weights_dir)
        if not self.oss.exists(object_key):
            raise FileNotFoundError(
                f"OSS 上未找到攻击算法权重包: {object_key}。"
                f"请上传 wmbench/weights/attacks/{job.weights_dir}/weights.zip"
            )

        archive_path = self._canonical_cache_archive(job.weights_dir)
        if archive_path.exists() and not self._is_valid_cache_archive(archive_path):
            archive_path.unlink()

        def on_progress(completed: int, total: int, message: str) -> None:
            total_steps = max(total, 1)
            self._set_progress(job, min(completed, total_steps), total_steps)
            job.bytes_downloaded = completed

        self.oss.download_file(object_key, archive_path, on_progress=on_progress)
        self._finalize_from_archive(job, archive_path, install_dir)

    def _mark_already_installed(self, job: AttackWeightDownloadJob, install_dir: Path) -> None:
        cached = self._find_cached_archive(job.weights_dir)
        file_count = len(list(install_dir.rglob("*"))) if install_dir.exists() else 0
        job.output_dir = str(install_dir)
        job.archive_path = str(cached) if cached else None
        self._set_progress(
            job,
            max(file_count, 1),
            max(file_count, 1),
        )

    def _find_cached_archive(self, weights_dir: str) -> Path | None:
        for cache_root in self._cache_roots():
            canonical = cache_root / f"{weights_dir}__attack-weights.zip"
            if self._is_valid_cache_archive(canonical):
                return canonical
            legacy = cache_root / f"{weights_dir}__weights.zip"
            if self._is_valid_cache_archive(legacy):
                return legacy
            for candidate in sorted(
                cache_root.glob(f"{weights_dir}__attack-weights__*.zip"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            ):
                if self._is_valid_cache_archive(candidate):
                    return candidate
        return None

    def _cache_roots(self) -> list[Path]:
        legacy = self.resources_root / "cache" / "weight-downloads"
        if legacy == self.cache_root:
            return [self.cache_root]
        return [self.cache_root, legacy]

    def _is_valid_cache_archive(self, archive_path: Path) -> bool:
        if not archive_path.is_file() or archive_path.stat().st_size < 1024:
            return False
        try:
            with zipfile.ZipFile(archive_path, "r") as archive:
                if archive.testzip() is not None:
                    return False
                return any(not name.endswith("/") for name in archive.namelist())
        except (OSError, zipfile.BadZipFile):
            return False

    def _finalize_from_archive(self, job: AttackWeightDownloadJob, archive_path: Path, install_dir: Path) -> None:
        if attack_weights_installed(install_dir):
            self._mark_already_installed(job, install_dir)
            return

        self._set_progress(job, 0, 1)
        self._extract_archive(archive_path, install_dir)
        job.output_dir = str(install_dir)
        job.archive_path = str(archive_path)
        file_count = max(len(iter_weight_files(install_dir)), 1)
        self._set_progress(job, file_count, file_count)

    def _extract_archive(self, archive_path: Path, target_dir: Path) -> None:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(target_dir)
        self._maybe_flatten_wrapper_dir(target_dir)

    def _maybe_flatten_wrapper_dir(self, target_dir: Path) -> None:
        files = [path for path in target_dir.iterdir() if path.is_file()]
        if files:
            return
        subdirs = [path for path in target_dir.iterdir() if path.is_dir()]
        if len(subdirs) != 1:
            return
        nested = subdirs[0]
        if nested.name != target_dir.name:
            return
        for item in nested.iterdir():
            destination = target_dir / item.name
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            shutil.move(str(item), str(destination))
        nested.rmdir()

    def uninstall(self, method: str) -> dict[str, Any]:
        if not attack_weights_need_download(method):
            raise ValueError(f"Attack method does not use packaged weights: {method}")
        directory = attack_weights_dir_name(method)
        if directory is None:
            raise ValueError(f"Missing attack weights directory mapping for: {method}")

        install_dir = attack_weights_install_dir(self.resources_root, method)
        if not install_dir.exists() or not attack_weights_installed(install_dir):
            raise FileNotFoundError(f"Attack weights are not installed: {install_dir}")

        shutil.rmtree(install_dir)
        install_dir.mkdir(parents=True, exist_ok=True)

        shared_methods = methods_for_attack_weights_dir(directory)
        message = "卸载完成"

        return {
            "method": method,
            "weightsDir": directory,
            "installed": False,
            "removedPath": str(install_dir),
            "sharedMethods": shared_methods,
            "message": message,
        }
