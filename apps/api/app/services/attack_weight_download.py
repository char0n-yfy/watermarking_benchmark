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
    AttackWeightSpec,
    attack_method_weights_installed,
    attack_weight_files_ready,
    attack_weight_spec,
    attack_weights_install_dir,
    attack_weights_need_download,
    mark_attack_pack_installed,
    uninstall_attack_weight_pack,
)
from app.services.object_storage import ObjectStorageClient
from app.services.watermark_weights import iter_weight_files


JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


@dataclass
class AttackWeightDownloadJob:
    id: str
    method: str
    weights_dir: str
    weights_pack_id: str
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
    total_bytes: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "method": self.method,
            "weightsDir": self.weights_dir,
            "weightsPackId": self.weights_pack_id,
            "status": self.status,
            "progress": self.progress,
            "totalItems": self.total_items,
            "completedItems": self.completed_items,
            "message": self.message,
            "error": self.error,
            "outputDir": self.output_dir,
            "archivePath": self.archive_path,
            "bytesDownloaded": self.bytes_downloaded,
            "totalBytes": self.total_bytes,
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
        spec = attack_weight_spec(method)
        if spec is None:
            raise ValueError(f"Missing attack weights mapping for: {method}")

        job_id = self._build_job_id(spec.pack_id)
        install_dir = attack_weights_install_dir(self.resources_root, method)
        if attack_method_weights_installed(self.resources_root, method):
            job = AttackWeightDownloadJob(
                id=job_id,
                method=method,
                weights_dir=spec.storage_dir,
                weights_pack_id=spec.pack_id,
                status="succeeded",
            )
            with self._lock:
                self._jobs[job_id] = job
            self._mark_already_installed(job, install_dir, method)
            job.status = "succeeded"
            job.progress = 100
            return job

        job = AttackWeightDownloadJob(
            id=job_id,
            method=method,
            weights_dir=spec.storage_dir,
            weights_pack_id=spec.pack_id,
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

    def _build_job_id(self, pack_id: str) -> str:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:6]
        return f"{pack_id}__attack-weights__{stamp}_{suffix}"

    def _canonical_cache_archive(self, pack_id: str) -> Path:
        return self.cache_root / f"{pack_id}__attack-weights.zip"

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
        job.progress = int(round((completed / total) * 100)) if total > 0 else job.progress

    def _set_byte_progress(
        self,
        job: AttackWeightDownloadJob,
        downloaded: int,
        total: int,
        *,
        message: str | None = None,
    ) -> None:
        job.bytes_downloaded = downloaded
        job.total_bytes = max(total, 0)
        if message is not None:
            job.message = message
        job.updated_at = time.time()
        if total > 0:
            job.completed_items = downloaded
            job.total_items = total
            job.progress = int(min(100, round((downloaded / total) * 100)))
        elif downloaded > 0:
            job.progress = max(job.progress, 1)

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
        spec = attack_weight_spec(job.method)
        if spec is None:
            raise ValueError(f"Missing attack weights mapping for: {job.method}")

        install_dir = attack_weights_install_dir(self.resources_root, job.method)
        if attack_method_weights_installed(self.resources_root, job.method):
            self._mark_already_installed(job, install_dir, job.method)
            return

        if attack_weight_files_ready(install_dir, spec):
            self._complete_install(
                job,
                install_dir,
                spec,
                archive_path=None,
            )
            return

        cached = self._find_cached_archive(spec.pack_id)
        if cached:
            self._finalize_from_archive(job, cached, install_dir, spec)
            return

        if not self.oss or not self.oss.enabled:
            raise FileNotFoundError(
                f"攻击算法权重未安装且 OSS 未启用。请手动放入 {install_dir} 或配置 WM_BENCH_OSS_*。"
            )

        object_key, archive_path = self._resolve_remote_archive(spec)
        if object_key is None or archive_path is None:
            raise FileNotFoundError(
                f"OSS 上未找到攻击算法权重包: {spec.pack_id}。"
                f"请上传 wmbench/weights/attacks/methods/{spec.pack_id}/weights.zip"
            )

        if archive_path.exists() and not self._is_valid_cache_archive(archive_path):
            archive_path.unlink()

        def on_progress(completed: int, total: int, message: str) -> None:
            self._set_byte_progress(job, completed, total, message=message or "正在从 OSS 下载权重包")

        job.status = "running"
        job.message = "正在从 OSS 下载权重包"
        self.oss.download_file(object_key, archive_path, on_progress=on_progress)
        self._finalize_from_archive(job, archive_path, install_dir, spec)

    def _resolve_remote_archive(self, spec: AttackWeightSpec) -> tuple[str | None, Path | None]:
        if not self.oss or not self.oss.enabled:
            return None, None

        method_key = self.oss.attack_weights_key(spec.pack_id)
        if self.oss.exists(method_key):
            return method_key, self._canonical_cache_archive(spec.pack_id)

        legacy_key = self.oss.attack_weights_legacy_key(spec.storage_dir)
        if self.oss.exists(legacy_key):
            return legacy_key, self.cache_root / f"{spec.storage_dir}__legacy-attack-weights.zip"

        return None, None

    def _complete_install(
        self,
        job: AttackWeightDownloadJob,
        install_dir: Path,
        spec: AttackWeightSpec,
        *,
        archive_path: Path | None,
        message: str | None = None,
    ) -> None:
        mark_attack_pack_installed(install_dir, spec.pack_id)
        job.output_dir = str(install_dir)
        job.archive_path = str(archive_path) if archive_path else None
        file_count = max(len(iter_weight_files(install_dir)), 1)
        if message:
            job.message = message
        self._set_progress(job, file_count, file_count)
        if job.total_bytes <= 0 and job.bytes_downloaded <= 0:
            job.progress = 100

    def _mark_already_installed(self, job: AttackWeightDownloadJob, install_dir: Path, method: str) -> None:
        spec = attack_weight_spec(method)
        cached = self._find_cached_archive(spec.pack_id) if spec is not None else None
        file_count = len(list(install_dir.rglob("*"))) if install_dir.exists() else 0
        job.output_dir = str(install_dir)
        job.archive_path = str(cached) if cached else None
        self._set_progress(
            job,
            max(file_count, 1),
            max(file_count, 1),
        )

    def _find_cached_archive(self, pack_id: str) -> Path | None:
        for cache_root in self._cache_roots():
            canonical = cache_root / f"{pack_id}__attack-weights.zip"
            if self._is_valid_cache_archive(canonical):
                return canonical
            legacy = cache_root / f"{pack_id}__weights.zip"
            if self._is_valid_cache_archive(legacy):
                return legacy
            for candidate in sorted(
                cache_root.glob(f"{pack_id}__attack-weights__*.zip"),
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

    def _finalize_from_archive(
        self,
        job: AttackWeightDownloadJob,
        archive_path: Path,
        install_dir: Path,
        spec: AttackWeightSpec,
    ) -> None:
        if attack_method_weights_installed(self.resources_root, job.method):
            self._mark_already_installed(job, install_dir, job.method)
            return

        self._set_progress(job, 0, 1, message="正在解压权重包")
        if not attack_weight_files_ready(install_dir, spec):
            self._extract_archive(archive_path, install_dir)
        self._complete_install(job, install_dir, spec, archive_path=archive_path, message="安装完成")

    def _extract_archive(self, archive_path: Path, target_dir: Path) -> None:
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
        return uninstall_attack_weight_pack(self.resources_root, method)
