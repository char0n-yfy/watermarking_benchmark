from __future__ import annotations

from enum import Enum


class RunStatus(str, Enum):
    DRAFT = "draft"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    PARTIALLY_FAILED = "partially_failed"


class PackageStatus(str, Enum):
    UPLOADED = "uploaded"
    REVIEWED = "reviewed"
    BUILT = "built"
    ENABLED = "enabled"
    REJECTED = "rejected"
