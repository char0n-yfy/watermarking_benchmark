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
