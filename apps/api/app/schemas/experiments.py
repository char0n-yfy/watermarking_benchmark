from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ExperimentSelectionPayload(BaseModel):
    datasetIds: list[str] = Field(default_factory=list)
    algorithmIds: list[str] = Field(default_factory=lambda: ["alg-invisible-watermark-dwtdct"])
    attackPresetIds: list[str] = Field(default_factory=lambda: ["atk-identity", "atk-jpeg"])
    attackStrengthOverrides: dict[str, list[float]] = Field(default_factory=dict)
    attackParamOverrides: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    seeds: list[int] = Field(default_factory=lambda: [42])
    maxSamples: int = 1


class ExperimentConfigCreatePayload(BaseModel):
    name: str = "Demo robustness smoke"
    selection: ExperimentSelectionPayload


class ExperimentConfigRenamePayload(BaseModel):
    name: str


class RunCreatePayload(BaseModel):
    config_id: Optional[str] = None
    configId: Optional[str] = None
    name: Optional[str] = None
    taskName: Optional[str] = None
    execute: bool = False

    def resolved_config_id(self) -> str:
        config_id = self.config_id or self.configId
        if not config_id:
            raise ValueError("config_id is required")
        return config_id

    def resolved_name(self) -> str | None:
        name = self.name or self.taskName
        if name is None:
            return None
        stripped = name.strip()
        return stripped or None


class ExperimentSpecDraft(BaseModel):
    name: str
    dataset_version_ids: list[str]
    algorithm_version_ids: list[str]
    attack_preset_ids: list[str]
    seeds: list[int]
    parameter_grid: dict[str, Any] = Field(default_factory=dict)
    max_samples_per_dataset: Optional[int] = None


class ExperimentRunSummary(BaseModel):
    run_id: str
    spec_id: str
    status: str
    cell_count: int
    artifact_root: str
