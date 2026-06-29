import type {
  AlgorithmVersion,
  AttackPreset,
  BenchmarkProtocol,
  DatasetCatalogItem,
  DatasetCatalogResponse,
  DatasetDownloadJob,
  DatasetDownloadMode,
  DatasetVersion,
  DemoRunRecord,
  ExperimentSelection,
  LeaderboardResponse,
  ReadinessReport,
  RunEvents,
  RunLogs,
  RunResults,
  RunScoreResponse,
  RuntimeInfo,
  SavedExperimentConfig,
  SystemMetrics,
  WeightDownloadJob,
  ResourceInstallationResult
} from "./types";

const configuredApiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "");

function defaultApiBaseUrl() {
  if (typeof window === "undefined") {
    return process.env.NODE_ENV === "development" ? "http://localhost:8000" : "";
  }

  const { hostname, port, protocol } = window.location;
  const isLocalDevHost = hostname === "localhost" || hostname === "127.0.0.1";
  if (isLocalDevHost && port === "3000") {
    return `${protocol}//${hostname}:8000`;
  }

  return "";
}

export const apiBaseUrl = configuredApiBaseUrl || defaultApiBaseUrl();

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    },
    cache: "no-store"
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `API request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export function fetchDatasets(): Promise<DatasetVersion[]> {
  return requestJson<DatasetVersion[]>("/resources/datasets");
}

export function fetchDatasetCatalog(options?: { remote?: boolean }): Promise<DatasetCatalogResponse> {
  const query = options?.remote ? "?remote=1" : "";
  const request = requestJson<DatasetCatalogResponse>(`/resources/datasets/catalog${query}`);
  if (options?.remote) {
    return request;
  }
  return request.catch(async () => {
    const datasets = await fetchDatasets();
    return {
      categories: [{ id: "local", nameZh: "本地数据集" }],
      items: datasets.map((dataset) => ({
        id: dataset.id,
        name: dataset.name,
        nameZh: dataset.name,
        category: "local",
        categoryZh: "本地数据集",
        description: dataset.path ?? "Locally scanned dataset.",
        descriptionZh: "本地扫描发现的数据集。",
        sourceUrl: "",
        manifestUrl: null,
        compactSampleCount: dataset.sampleCount,
        fullSampleCount: dataset.sampleCount,
        customPoolCount: dataset.sampleCount,
        officialTotalImages: null,
        compactAvailable: dataset.sampleCount > 0,
        localAvailable: dataset.sampleCount > 0,
        installed: true,
        customDownloadReady: dataset.sampleCount > 0,
        remoteManifestConfigured: false,
        compactUsesRoot: true,
        rootPath: dataset.path,
        compactPath: dataset.path,
        fullPath: dataset.path
      }))
    };
  });
}

export function fetchDatasetDetail(datasetId: string): Promise<DatasetCatalogItem> {
  return requestJson<DatasetCatalogItem>(`/resources/datasets/${encodeURIComponent(datasetId)}`);
}

export function startDatasetDownload(
  datasetId: string,
  payload: { mode: DatasetDownloadMode; seed?: number; sampleCount?: number }
): Promise<DatasetDownloadJob> {
  return requestJson<DatasetDownloadJob>(`/resources/datasets/${encodeURIComponent(datasetId)}/downloads`, {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function fetchDatasetDownloadJob(jobId: string): Promise<DatasetDownloadJob> {
  return requestJson<DatasetDownloadJob>(`/resources/datasets/downloads/${encodeURIComponent(jobId)}`);
}

export function datasetDownloadArchiveUrl(jobId: string): string {
  return `${apiBaseUrl}/resources/datasets/downloads/${encodeURIComponent(jobId)}/archive`;
}

export function fetchAlgorithms(options?: { remote?: boolean }): Promise<AlgorithmVersion[]> {
  const query = options?.remote ? "?remote=1" : "";
  return requestJson<AlgorithmVersion[]>(`/resources/watermarks${query}`);
}

export function startWeightDownload(identifier: string): Promise<WeightDownloadJob> {
  return requestJson<WeightDownloadJob>(`/resources/watermarks/${encodeURIComponent(identifier)}/downloads`, {
    method: "POST",
    body: JSON.stringify({})
  });
}

export function fetchWeightDownloadJob(jobId: string): Promise<WeightDownloadJob> {
  return requestJson<WeightDownloadJob>(`/resources/watermarks/downloads/${encodeURIComponent(jobId)}`);
}

export function fetchAttacks(options?: { remote?: boolean }): Promise<AttackPreset[]> {
  const query = options?.remote ? "?remote=1" : "";
  return requestJson<AttackPreset[]>(`/resources/attacks${query}`);
}

export function startAttackWeightDownload(identifier: string): Promise<WeightDownloadJob> {
  return requestJson<WeightDownloadJob>(`/resources/attacks/${encodeURIComponent(identifier)}/downloads`, {
    method: "POST",
    body: JSON.stringify({})
  });
}

export function fetchAttackWeightDownloadJob(jobId: string): Promise<WeightDownloadJob> {
  return requestJson<WeightDownloadJob>(`/resources/attacks/downloads/${encodeURIComponent(jobId)}`);
}

export function uninstallWatermarkInstallation(identifier: string): Promise<ResourceInstallationResult> {
  return requestJson<ResourceInstallationResult>(`/resources/watermarks/${encodeURIComponent(identifier)}/installation`, {
    method: "DELETE"
  });
}

export function uninstallAttackInstallation(identifier: string): Promise<ResourceInstallationResult> {
  return requestJson<ResourceInstallationResult>(`/resources/attacks/${encodeURIComponent(identifier)}/installation`, {
    method: "DELETE"
  });
}

export function uninstallDatasetInstallation(
  datasetId: string,
  options: { mode: DatasetDownloadMode; seed?: number; sampleCount?: number }
): Promise<ResourceInstallationResult> {
  const params = new URLSearchParams({ mode: options.mode });
  if (options.mode === "custom") {
    params.set("seed", String(options.seed ?? 42));
    params.set("sampleCount", String(options.sampleCount ?? 100));
  }
  return requestJson<ResourceInstallationResult>(
    `/resources/datasets/${encodeURIComponent(datasetId)}/installation?${params.toString()}`,
    { method: "DELETE" }
  );
}

export function fetchSavedConfigs(): Promise<SavedExperimentConfig[]> {
  return requestJson<SavedExperimentConfig[]>("/experiment-configs");
}

export function createSavedConfig(name: string, selection: ExperimentSelection): Promise<SavedExperimentConfig> {
  return requestJson<SavedExperimentConfig>("/experiment-configs", {
    method: "POST",
    body: JSON.stringify({ name, selection })
  });
}

export function renameSavedConfig(configId: string, name: string): Promise<SavedExperimentConfig> {
  return requestJson<SavedExperimentConfig>(`/experiment-configs/${configId}`, {
    method: "PATCH",
    body: JSON.stringify({ name })
  });
}

export function deleteSavedConfig(configId: string): Promise<{ id: string; status: string }> {
  return requestJson<{ id: string; status: string }>(`/experiment-configs/${configId}`, {
    method: "DELETE"
  });
}

export function fetchRuns(options?: { scope?: "active" }): Promise<DemoRunRecord[]> {
  const query = options?.scope ? `?scope=${encodeURIComponent(options.scope)}` : "";
  return requestJson<DemoRunRecord[]>(`/runs${query}`);
}

export function fetchRun(runId: string): Promise<DemoRunRecord> {
  return requestJson<DemoRunRecord>(`/runs/${encodeURIComponent(runId)}`);
}

export function createRun(configId: string, name?: string): Promise<DemoRunRecord> {
  return requestJson<DemoRunRecord>("/runs", {
    method: "POST",
    body: JSON.stringify({ configId, name })
  });
}

export function fetchRunResults(runId: string): Promise<RunResults> {
  return requestJson<RunResults>(`/runs/${runId}/results`);
}

export function fetchRunScore(runId: string): Promise<RunScoreResponse> {
  return requestJson<RunScoreResponse>(`/runs/${runId}/score`);
}

export function fetchBenchmarkProtocols(): Promise<BenchmarkProtocol[]> {
  return requestJson<BenchmarkProtocol[]>("/benchmark-protocols");
}

export function fetchLeaderboard(protocolId = "waves-official-detection-v1"): Promise<LeaderboardResponse> {
  return requestJson<LeaderboardResponse>(`/leaderboard?protocol_id=${encodeURIComponent(protocolId)}`);
}

export function cancelRun(runId: string): Promise<DemoRunRecord> {
  return requestJson<DemoRunRecord>(`/runs/${runId}/cancel`, {
    method: "POST"
  });
}

export function resumeRun(runId: string): Promise<DemoRunRecord> {
  return requestJson<DemoRunRecord>(`/runs/${runId}/resume`, {
    method: "POST"
  });
}

export function fetchRunLogs(runId: string): Promise<RunLogs> {
  return requestJson<RunLogs>(`/runs/${runId}/logs`);
}

export function fetchRunEvents(runId: string): Promise<RunEvents> {
  return requestJson<RunEvents>(`/runs/${runId}/events`);
}

export function fetchRuntime(): Promise<RuntimeInfo> {
  return requestJson<RuntimeInfo>("/system/runtime");
}

export function fetchReadiness(): Promise<ReadinessReport> {
  return requestJson<ReadinessReport>("/system/readiness");
}

export function fetchSystemMetrics(): Promise<SystemMetrics> {
  return requestJson<SystemMetrics>("/system/metrics");
}
