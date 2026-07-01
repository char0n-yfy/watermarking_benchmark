from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class RecordSchema:
    name: str
    version: int = 1

    def apply(self, payload: Mapping[str, Any]) -> JsonDict:
        return {
            "recordSchema": self.name,
            "schemaVersion": self.version,
            **dict(payload),
        }


SAMPLE_MANIFEST_SCHEMA = RecordSchema("sample_manifest", 1)
RUNTIME_PROFILE_SCHEMA = RecordSchema("runtime_profile", 1)
IMAGE_QUALITY_SCHEMA = RecordSchema("image_quality", 1)
IMAGE_WATERMARK_EMBED_SCHEMA = RecordSchema("image_watermark_embed", 1)
IMAGE_ATTACK_SCHEMA = RecordSchema("image_attack", 1)
IMAGE_DETECTION_SCHEMA = RecordSchema("image_detection", 1)
CELL_MANIFEST_SCHEMA = RecordSchema("cell_manifest", 1)
STAGE_EVENT_SCHEMA = RecordSchema("stage_event", 1)
