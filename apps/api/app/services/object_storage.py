from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import httpx


ProgressCallback = Callable[[int, int, str], None]
_EXISTS_CACHE_TTL_SECONDS = 60.0

# Team defaults: clone and start without per-user AccessKey when bucket prefix is public-read.
DEFAULT_OSS_BUCKET = "watermarking-benchmark"
DEFAULT_OSS_ENDPOINT = "https://oss-cn-shanghai.aliyuncs.com"
DEFAULT_OSS_PREFIX = "wmbench"
DEFAULT_OSS_REGION = "oss-cn-shanghai"


@dataclass(frozen=True)
class ObjectStorageSettings:
    enabled: bool
    public_read: bool
    endpoint_url: str | None
    bucket: str
    access_key: str
    secret_key: str
    prefix: str
    region: str
    presign_expires: int

    @classmethod
    def from_env(cls) -> ObjectStorageSettings:
        from ..core.env_loader import load_project_env

        load_project_env(override=True)
        enabled_flag = os.getenv("WM_BENCH_OSS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        public_read = os.getenv("WM_BENCH_OSS_PUBLIC_READ", "true").strip().lower() in {"1", "true", "yes", "on"}
        bucket = os.getenv("WM_BENCH_OSS_BUCKET", DEFAULT_OSS_BUCKET).strip()
        access_key = os.getenv("WM_BENCH_OSS_ACCESS_KEY", "").strip()
        secret_key = os.getenv("WM_BENCH_OSS_SECRET_KEY", "").strip()
        has_auth = bool(access_key and secret_key)
        enabled = enabled_flag and bool(bucket) and (public_read or has_auth)
        endpoint = os.getenv("WM_BENCH_OSS_ENDPOINT", DEFAULT_OSS_ENDPOINT).strip() or None
        prefix = os.getenv("WM_BENCH_OSS_PREFIX", DEFAULT_OSS_PREFIX).strip().strip("/")
        region = os.getenv("WM_BENCH_OSS_REGION", DEFAULT_OSS_REGION).strip()
        presign_expires = int(os.getenv("WM_BENCH_OSS_PRESIGN_EXPIRES", "3600"))
        return cls(
            enabled=enabled,
            public_read=public_read and enabled,
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
        self._exists_cache: dict[str, tuple[float, bool]] = {}

    @property
    def settings(self) -> ObjectStorageSettings:
        return self._settings

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    @property
    def uses_public_read(self) -> bool:
        return self._settings.public_read

    def _get_client(self):
        if not self.enabled or self._settings.public_read:
            raise RuntimeError("Private object storage client is not configured")
        if self._client is None:
            import boto3
            from botocore.config import Config

            # Aliyun OSS S3-compatible API rejects path-style bucket access.
            config = Config(
                s3={"addressing_style": "virtual"},
                connect_timeout=3,
                read_timeout=5,
                retries={"max_attempts": 1},
            )
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

    def public_object_url(self, key: str) -> str:
        """Build a virtual-hosted HTTPS URL for public-read buckets."""
        endpoint = self._settings.endpoint_url or DEFAULT_OSS_ENDPOINT
        parsed = urlparse(endpoint)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid OSS endpoint: {endpoint}")
        host = f"{self._settings.bucket}.{parsed.netloc}"
        return f"{parsed.scheme}://{host}/{key.lstrip('/')}"

    def _download_url(self, key: str) -> str:
        if self._settings.public_read:
            return self.public_object_url(key)
        return self.presign_get_url(key)

    def exists(self, key: str) -> bool:
        if not self.enabled:
            return False
        now = time.monotonic()
        cached = self._exists_cache.get(key)
        if cached is not None and now - cached[0] < _EXISTS_CACHE_TTL_SECONDS:
            return cached[1]

        if self._settings.public_read:
            result = self._exists_public(key)
        else:
            result = self._exists_private(key)

        self._exists_cache[key] = (now, result)
        return result

    def _exists_public(self, key: str) -> bool:
        url = self.public_object_url(key)
        try:
            with httpx.Client(timeout=5.0, follow_redirects=True) as client:
                response = client.head(url)
                if response.status_code == 405:
                    response = client.get(url, headers={"Range": "bytes=0-0"})
                return response.status_code in {200, 206}
        except httpx.HTTPError:
            return False

    def _exists_private(self, key: str) -> bool:
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
        if self._settings.public_read:
            url = self.public_object_url(key)
            with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                return response.text
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
        url = self._download_url(key)
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
        if self._settings.public_read:
            self.download_file(key, dest, on_progress=on_progress)
            return
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
                "mode": "disabled",
                "bucket": self._settings.bucket or None,
                "prefix": self._settings.prefix,
                "endpoint": self._settings.endpoint_url,
            }
        mode = "public-read" if self._settings.public_read else "private"
        payload: dict[str, object] = {
            "enabled": True,
            "mode": mode,
            "bucket": self._settings.bucket,
            "prefix": self._settings.prefix,
            "endpoint": self._settings.endpoint_url,
            "region": self._settings.region,
            "authConfigured": bool(self._settings.access_key and self._settings.secret_key),
        }
        if self._settings.public_read:
            prefix_key = f"{self._settings.prefix}/" if self._settings.prefix else ""
            payload["publicBaseUrl"] = self.public_object_url(prefix_key)
        return payload


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
