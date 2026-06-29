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

export interface DatasetCatalogItem {
  id: string;
  name: string;
  nameZh: string;
  category: string;
  categoryZh: string;
  description: string;
  descriptionZh: string;
  sourceUrl: string;
  manifestUrl?: string | null;
  compactSampleCount: number;
  fullSampleCount: number;
  customPoolCount: number;
  officialTotalImages?: number | null;
  compactAvailable: boolean;
  localAvailable: boolean;
  installed: boolean;
  customDownloadReady: boolean;
  remoteManifestConfigured: boolean;
  remoteCompactAvailable?: boolean;
  remoteCustomAvailable?: boolean;
  objectStorageConfigured?: boolean;
  compactUsesRoot?: boolean;
  rootPath?: string;
  compactPath?: string;
  fullPath?: string;
}

export interface DatasetCatalogResponse {
  categories: Array<{ id: string; nameZh: string }>;
  items: DatasetCatalogItem[];
}

export type DatasetDownloadMode = "compact" | "custom";
export type DatasetDownloadStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface DatasetDownloadJob {
  id: string;
  datasetId: string;
  mode: DatasetDownloadMode;
  status: DatasetDownloadStatus;
  progress: number;
  totalItems: number;
  completedItems: number;
  seed?: number | null;
  sampleCount?: number | null;
  message?: string | null;
  error?: string | null;
  outputDir?: string | null;
  archivePath?: string | null;
  bytesDownloaded?: number;
}

export type WeightDownloadStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface WeightDownloadJob {
  id: string;
  method: string;
  weightsDir: string;
  status: WeightDownloadStatus;
  progress: number;
  totalItems: number;
  completedItems: number;
  message?: string | null;
  error?: string | null;
  outputDir?: string | null;
  archivePath?: string | null;
  bytesDownloaded?: number;
}

export interface AlgorithmVersion {
  id: string;
  name: string;
  version: string;
  status: ResourceStatus;
  requiresGpu: boolean;
  method?: string;
  description?: string;
  category?: string;
  recommended?: boolean;
  available?: boolean;
  params?: Record<string, unknown>;
  weightsDir?: string | null;
  weightsPath?: string | null;
  weightsInstalled?: boolean;
  weightsDownloadReady?: boolean;
  remoteWeightsAvailable?: boolean;
  weightsPackRequired?: boolean;
}

export interface AttackPreset {
  id: string;
  name: string;
  method: string;
  strengths: number[];
  description?: string;
  category?: string;
  categoryLabel?: string;
  categoryPath?: string;
  strengthParam?: string | null;
  requiresGpu?: boolean;
  recommended?: boolean;
  available?: boolean;
  params?: Record<string, unknown>;
  weightsDir?: string | null;
  weightsPath?: string | null;
  weightsInstalled?: boolean;
  weightsDownloadReady?: boolean;
  remoteWeightsAvailable?: boolean;
  weightsPackRequired?: boolean;
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

export interface BenchmarkCategoryScore {
  key: string;
  label: string;
  score: number | null;
  meanNqd: number | null;
  cellCount: number;
  covered: boolean;
}

export interface BenchmarkCoverage {
  requiredCategories: string[];
  coveredCategories: string[];
  missingCategories: string[];
  coveredCategoryCount: number;
  requiredCategoryCount: number;
  coverageRatio: number;
  minSampleCount: number;
  meetsSampleFloor: boolean;
}

export interface BenchmarkLeaderboardRow {
  rank: number;
  algorithmId: string;
  protocolId: string;
  protocolStatus: "official" | "provisional" | string;
  officialEligible: boolean;
  wrs: number | null;
  cleanFidelity: number | null;
  avgNqd: number | null;
  runtimeMs: number | null;
  coverage: BenchmarkCoverage;
  categoryScores: BenchmarkCategoryScore[];
  cellCount: number;
  runId?: string;
  runStatus?: RunStatus;
  configId?: string;
  configName?: string;
  updatedAt?: string;
}

export interface BenchmarkCurvePoint {
  algorithmId: string;
  attackPresetId: string;
  attackMethod: string;
  attackCategory: string;
  attackStrength: number;
  xNqd: number;
  yTprAtFpr: number;
}

export interface BenchmarkScore {
  protocolId: string;
  protocolName: string;
  status: "official" | "provisional" | string;
  officialEligible: boolean;
  wrs: number | null;
  wrsLabel: string;
  fprTarget: number;
  practicalNqdThreshold: number;
  officialMinSamples: number;
  categoryScores: BenchmarkCategoryScore[];
  coverage: BenchmarkCoverage;
  leaderboardRows: BenchmarkLeaderboardRow[];
  curvePoints: BenchmarkCurvePoint[];
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
  score?: BenchmarkScore;
}

export interface RunScoreResponse {
  run: DemoRunRecord;
  score: BenchmarkScore;
  summaryPath: string;
  summaryExists: boolean;
}

export interface BenchmarkProtocol {
  id: string;
  name: string;
  task: string;
  fprTarget: number;
  officialMinSamples: number;
  practicalNqdThreshold: number;
  requiredCategories: Array<{ key: string; label: string; description: string }>;
  qualityMetrics: string[];
  status: string;
}

export interface LeaderboardResponse {
  protocol: BenchmarkProtocol;
  rows: BenchmarkLeaderboardRow[];
  officialRows: BenchmarkLeaderboardRow[];
  provisionalRows: BenchmarkLeaderboardRow[];
  generatedAt: string;
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

export interface SystemMetrics {
  timestamp: string;
  hostName: string;
  platform: string;
  system: string;
  machine: string;
  pythonVersion: string;
  uptimeSeconds: number | null;
  configuredDevice: string;
  cpu: {
    logicalCores: number;
    usagePercent: number | null;
    loadAverage: number[];
  };
  memory: {
    totalBytes: number | null;
    usedBytes: number | null;
    availableBytes: number | null;
    usedPercent: number | null;
  };
  disk: {
    path: string;
    totalBytes: number;
    usedBytes: number;
    freeBytes: number;
    usedPercent: number | null;
    ioReadBytesPerSecond: number | null;
    ioWriteBytesPerSecond: number | null;
    ioTotalBytesPerSecond: number | null;
  };
  gpu: {
    available: boolean;
    devices: Array<{
      index: number;
      name: string;
      utilizationPercent: number | null;
      memoryUsedMiB: number | null;
      memoryTotalMiB: number | null;
      memoryUsedPercent: number | null;
      temperatureC: number | null;
      powerDrawW: number | null;
    }>;
  };
  process: {
    pid: number;
    rssBytes: number | null;
    pythonExecutable: string;
  };
}
