import { estimateMatrix } from "@/lib/matrix";
import { attacks, datasets } from "@/lib/mock-data";
import type { DemoRunRecord, ExperimentSelection, SavedExperimentConfig } from "@/lib/types";

const configsKey = "wm-bench-demo-configs-v2";
const runsKey = "wm-bench-demo-runs-v2";

export const defaultSelection: ExperimentSelection = {
  datasetIds: ["local-root"],
  algorithmIds: ["alg-traditional-lsb"],
  attackPresetIds: ["atk-identity", "atk-jpeg-smoke"],
  seeds: [42],
  maxSamples: 1
};

function nowLabel() {
  return new Date().toLocaleString([], {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  });
}

function estimateSelection(selection: ExperimentSelection) {
  return estimateMatrix(selection, datasets, attacks);
}

export function buildSavedConfig(name: string, selection: ExperimentSelection): SavedExperimentConfig {
  const estimate = estimateSelection(selection);
  const timestamp = new Date().toISOString();
  return {
    id: `cfg-${Date.now()}`,
    name: name.trim() || "Untitled experiment config",
    selection,
    cellCount: estimate.cellCount,
    sampleCount: estimate.sampleCount,
    imageOperationCount: estimate.imageOperationCount,
    createdAt: timestamp,
    updatedAt: timestamp
  };
}

export const defaultSavedConfigs: SavedExperimentConfig[] = [
  buildSavedConfig("Demo robustness smoke", defaultSelection)
].map((config) => ({
  ...config,
  id: "cfg-demo-smoke",
  createdAt: "2026-06-27T00:00:00.000Z",
  updatedAt: "2026-06-27T00:00:00.000Z"
}));

export const defaultRunRecords: DemoRunRecord[] = [
  {
    id: "run_20260627_001",
    configId: "cfg-demo-smoke",
    configName: "Demo robustness smoke",
    status: "running",
    cells: defaultSavedConfigs[0].cellCount,
    progress: 62,
    updatedAt: "19:45"
  },
  {
    id: "run_20260626_004",
    configId: "cfg-demo-smoke",
    configName: "Demo robustness smoke",
    status: "succeeded",
    cells: 3,
    progress: 100,
    updatedAt: "Yesterday"
  }
];

function readJson<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") {
    return fallback;
  }
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function writeJson<T>(key: string, value: T) {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(key, JSON.stringify(value));
}

export function loadSavedConfigs(): SavedExperimentConfig[] {
  return readJson(configsKey, defaultSavedConfigs);
}

export function saveSavedConfigs(configs: SavedExperimentConfig[]) {
  writeJson(configsKey, configs);
}

export function addSavedConfig(config: SavedExperimentConfig): SavedExperimentConfig[] {
  const configs = [config, ...loadSavedConfigs()];
  saveSavedConfigs(configs);
  return configs;
}

export function loadRunRecords(): DemoRunRecord[] {
  return readJson(runsKey, defaultRunRecords);
}

export function saveRunRecords(runs: DemoRunRecord[]) {
  writeJson(runsKey, runs);
}

export function createRunRecord(config: SavedExperimentConfig): DemoRunRecord {
  return {
    id: `run_${new Date().toISOString().slice(0, 10).replaceAll("-", "")}_${String(Date.now()).slice(-4)}`,
    configId: config.id,
    configName: config.name,
    status: "queued",
    cells: config.cellCount,
    progress: 0,
    updatedAt: nowLabel()
  };
}
