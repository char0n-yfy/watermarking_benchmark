from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class DatasetDownloadCreatePayload(BaseModel):
    mode: Literal["compact", "custom"]
    seed: int = Field(default=42, ge=0)
    sample_count: int = Field(default=100, ge=1, le=10000, alias="sampleCount")

    model_config = {"populate_by_name": True}


class DatasetDownloadJobResponse(BaseModel):
    id: str
    datasetId: str
    mode: str
    status: str
    progress: int
    totalItems: int
    completedItems: int
    seed: Optional[int] = None
    sampleCount: Optional[int] = None
    message: Optional[str] = None
    error: Optional[str] = None
    outputDir: Optional[str] = None
    archivePath: Optional[str] = None
    bytesDownloaded: int = 0
