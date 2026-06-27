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
  path?: string;
}

export interface AlgorithmVersion {
  id: string;
  name: string;
  version: string;
  status: ResourceStatus;
  requiresGpu: boolean;
  method?: string;
  recommended?: boolean;
  available?: boolean;
  params?: Record<string, unknown>;
}

export interface AttackPreset {
  id: string;
  name: string;
  method: string;
  strengths: number[];
  recommended?: boolean;
  available?: boolean;
  params?: Record<string, unknown>;
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
  artifactRoot?: string;
  logPath?: string | null;
  workerId?: string | null;
  cancelRequested?: boolean;
  error?: string | null;
  createdAt?: string;
  startedAt?: string | null;
  finishedAt?: string | null;
}

export interface RunResultCell {
  id: string;
  runId: string;
  cellKey: string;
  status: RunStatus;
  datasetId: string;
  algorithmId: string;
  watermarkMethod: string;
  attackPresetId: string;
  attackMethod: string;
  attackStrength: number;
  seed: number;
  sampleCount: number;
  bitAccuracy: number | null;
  bitErrorRate: number | null;
  elapsedMs: number | null;
  manifestPath: string | null;
  outputDir: string | null;
  error: string | null;
  summary?: Record<string, unknown>;
}

export interface RunAggregate {
  algorithmId: string;
  attackPresetId: string;
  attackStrength: number;
  cellCount: number;
  succeededCells: number;
  failedCells: number;
  meanBitAccuracy: number | null;
  meanBitErrorRate: number | null;
}

export interface RunResults {
  run: DemoRunRecord;
  cells: RunResultCell[];
  summaryPath: string;
  summaryExists: boolean;
  summary?: Record<string, unknown> | null;
  aggregates: RunAggregate[];
}

export interface RunLogs {
  runId: string;
  logPath: string;
  exists: boolean;
  lines: string[];
}

export interface WorkerHeartbeat {
  workerId: string;
  status: string;
  pid: number;
  device: string;
  currentRunId: string | null;
  message: string | null;
  lastSeenAt: string;
}

export interface RuntimeInfo {
  environment: string;
  device: string;
  dataRoot: string;
  resourcesRoot: string;
  runsRoot: string;
  databasePath: string;
  apiHost: string;
  apiPort: number;
  workerPollSeconds: number;
  workers: WorkerHeartbeat[];
}
