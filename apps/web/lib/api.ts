import type {
  AlgorithmVersion,
  AttackPreset,
  DatasetVersion,
  DemoRunRecord,
  ExperimentSelection,
  RunLogs,
  RunResults,
  RuntimeInfo,
  SavedExperimentConfig,
  SystemMetrics
} from "./types";

const configuredApiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "");

export const apiBaseUrl =
  configuredApiBaseUrl || (process.env.NODE_ENV === "development" ? "http://localhost:8000" : "");

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

export function fetchAlgorithms(): Promise<AlgorithmVersion[]> {
  return requestJson<AlgorithmVersion[]>("/resources/watermarks");
}

export function fetchAttacks(): Promise<AttackPreset[]> {
  return requestJson<AttackPreset[]>("/resources/attacks");
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

export function fetchRuns(): Promise<DemoRunRecord[]> {
  return requestJson<DemoRunRecord[]>("/runs");
}

export function createRun(configId: string): Promise<DemoRunRecord> {
  return requestJson<DemoRunRecord>("/runs", {
    method: "POST",
    body: JSON.stringify({ configId })
  });
}

export function fetchRunResults(runId: string): Promise<RunResults> {
  return requestJson<RunResults>(`/runs/${runId}/results`);
}

export function cancelRun(runId: string): Promise<DemoRunRecord> {
  return requestJson<DemoRunRecord>(`/runs/${runId}/cancel`, {
    method: "POST"
  });
}

export function fetchRunLogs(runId: string): Promise<RunLogs> {
  return requestJson<RunLogs>(`/runs/${runId}/logs`);
}

export function fetchRuntime(): Promise<RuntimeInfo> {
  return requestJson<RuntimeInfo>("/system/runtime");
}

export function fetchSystemMetrics(): Promise<SystemMetrics> {
  return requestJson<SystemMetrics>("/system/metrics");
}
