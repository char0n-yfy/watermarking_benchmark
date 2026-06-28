from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import httpx


ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class ObjectStorageSettings:
    enabled: bool
    endpoint_url: str | None
    bucket: str
    access_key: str
    secret_key: str
    prefix: str
    region: str
    presign_expires: int

    @classmethod
    def from_env(cls) -> ObjectStorageSettings:
        bucket = os.getenv("WM_BENCH_OSS_BUCKET", "").strip()
        access_key = os.getenv("WM_BENCH_OSS_ACCESS_KEY", "").strip()
        secret_key = os.getenv("WM_BENCH_OSS_SECRET_KEY", "").strip()
        enabled_flag = os.getenv("WM_BENCH_OSS_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
        enabled = enabled_flag and bool(bucket and access_key and secret_key)
        endpoint = os.getenv("WM_BENCH_OSS_ENDPOINT", "").strip() or None
        prefix = os.getenv("WM_BENCH_OSS_PREFIX", "wmbench").strip().strip("/")
        region = os.getenv("WM_BENCH_OSS_REGION", "us-east-1").strip()
        presign_expires = int(os.getenv("WM_BENCH_OSS_PRESIGN_EXPIRES", "3600"))
        return cls(
            enabled=enabled,
            endpoint_url=endpoint,
            bucket=bucket,
            access_key=access_key,
            secret_key=secret_key,
            prefix=prefix,
            region=region,
            presign_expires=presign_expires,
        )


class ObjectStorageClient:
    """S3-compatible object storage (AWS S3, MinIO, Aliyun OSS S3 mode, Tencent COS S3 mode)."""

    def __init__(self, settings: ObjectStorageSettings) -> None:
        self._settings = settings
        self._client = None

    @property
    def settings(self) -> ObjectStorageSettings:
        return self._settings

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    def _get_client(self):
        if not self.enabled:
            raise RuntimeError("Object storage is not configured")
        if self._client is None:
            import boto3
            from botocore.config import Config

            # Aliyun OSS S3-compatible API rejects path-style bucket access.
            config = Config(s3={"addressing_style": "virtual"})
            self._client = boto3.client(
                "s3",
                endpoint_url=self._settings.endpoint_url,
                aws_access_key_id=self._settings.access_key,
                aws_secret_access_key=self._settings.secret_key,
                region_name=self._settings.region,
                config=config,
            )
        return self._client

    def object_key(self, *parts: str) -> str:
        cleaned = [self._settings.prefix] if self._settings.prefix else []
        cleaned.extend(part.strip("/") for part in parts if part)
        return "/".join(cleaned)

    def dataset_compact_key(self, dataset_id: str) -> str:
        return self.object_key("datasets", dataset_id, "compact-1000.zip")

    def dataset_manifest_key(self, dataset_id: str) -> str:
        return self.object_key("datasets", dataset_id, "manifest.txt")

    def dataset_image_key(self, dataset_id: str, relative_path: str) -> str:
        relative = relative_path.lstrip("/")
        if relative.startswith("datasets/"):
            return self.object_key(relative)
        return self.object_key("datasets", dataset_id, relative)

    def exists(self, key: str) -> bool:
        if not self.enabled:
            return False
        try:
            self._get_client().head_object(Bucket=self._settings.bucket, Key=key)
            return True
        except Exception as exc:
            from botocore.exceptions import ClientError

            if isinstance(exc, ClientError):
                code = exc.response.get("Error", {}).get("Code", "")
                if code in {"404", "NoSuchKey", "NotFound"}:
                    return False
            return False

    def presign_get_url(self, key: str, *, expires: int | None = None) -> str:
        client = self._get_client()
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._settings.bucket, "Key": key},
            ExpiresIn=expires or self._settings.presign_expires,
        )

    def read_text(self, key: str) -> str:
        client = self._get_client()
        response = client.get_object(Bucket=self._settings.bucket, Key=key)
        body = response["Body"].read()
        return body.decode("utf-8")

    def download_file(
        self,
        key: str,
        dest: Path,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = self.presign_get_url(key)
        downloaded = 0
        with httpx.Client(timeout=600.0, follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", "0") or 0)
                with dest.open("wb") as handle:
                    for chunk in response.iter_bytes(chunk_size=1024 * 256):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if on_progress:
                            on_progress(downloaded, total, f"downloading {key}")
        if on_progress:
            on_progress(downloaded, downloaded or 1, "download complete")

    def download_via_sdk(
        self,
        key: str,
        dest: Path,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        client = self._get_client()
        response = client.get_object(Bucket=self._settings.bucket, Key=key)
        body = response["Body"]
        total = int(response.get("ContentLength", 0) or 0)
        downloaded = 0
        with dest.open("wb") as handle:
            while True:
                chunk = body.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if on_progress:
                    on_progress(downloaded, total, f"downloading {key}")
        if on_progress:
            on_progress(downloaded, downloaded or 1, "download complete")

    def status(self) -> dict[str, object]:
        if not self.enabled:
            return {
                "enabled": False,
                "bucket": self._settings.bucket or None,
                "prefix": self._settings.prefix,
                "endpoint": self._settings.endpoint_url,
            }
        return {
            "enabled": True,
            "bucket": self._settings.bucket,
            "prefix": self._settings.prefix,
            "endpoint": self._settings.endpoint_url,
            "region": self._settings.region,
        }


def parse_manifest_lines(text: str, *, dataset_id: str, oss: ObjectStorageClient | None = None) -> list[str]:
    """Return download targets: HTTP(S) URLs or resolvable object keys."""
    targets: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split()[0]
        if token.startswith(("http://", "https://")):
            targets.append(token)
            continue
        if token.startswith("s3://"):
            parsed = urlparse(token)
            targets.append(parsed.path.lstrip("/"))
            continue
        if oss is not None:
            targets.append(oss.dataset_image_key(dataset_id, token))
        else:
            targets.append(token)
    return targets


@lru_cache(maxsize=1)
def get_object_storage_client() -> ObjectStorageClient:
    return ObjectStorageClient(ObjectStorageSettings.from_env())
