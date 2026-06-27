export type RunStatus =
  | "draft"
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "partially_failed";

export type ResourceStatus = "enabled" | "reviewed" | "built" | "uploaded";

export interface DatasetVersion {
  id: string;
  name: string;
  sampleCount: number;
  version: string;
}

export interface AlgorithmVersion {
  id: string;
  name: string;
  version: string;
  status: ResourceStatus;
  requiresGpu: boolean;
}

export interface AttackPreset {
  id: string;
  name: string;
  method: string;
  strengths: number[];
}

export interface ModelArtifact {
  id: string;
  name: string;
  checksum: string;
  size: string;
}

export interface ExperimentSelection {
  datasetIds: string[];
  algorithmIds: string[];
  attackPresetIds: string[];
  seeds: number[];
  maxSamples: number;
}

export interface SavedExperimentConfig {
  id: string;
  name: string;
  selection: ExperimentSelection;
  cellCount: number;
  sampleCount: number;
  imageOperationCount: number;
  createdAt: string;
  updatedAt: string;
}

export interface DemoRunRecord {
  id: string;
  configId: string;
  configName: string;
  status: RunStatus;
  cells: number;
  progress: number;
  updatedAt: string;
}
