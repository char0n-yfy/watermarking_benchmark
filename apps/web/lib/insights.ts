import type {
  AlgorithmVersion,
  AttackPreset,
  DatasetVersion,
  DemoRunRecord,
  RunAggregate,
  RunResults,
  RunStatus,
  SavedExperimentConfig
} from "./types";

export const terminalRunStatuses = new Set<RunStatus>([
  "succeeded",
  "failed",
  "paused",
  "cancelled",
  "partially_failed"
]);

export interface RunStats {
  queued: number;
  running: number;
  completed: number;
  failed: number;
}

export interface ActiveRunRow {
  id: string;
  configName: string;
  datasetLabel: string;
  algorithmLabel: string;
  attackLabel: string;
  progress: number;
  status: RunStatus;
  startedAt: string | null | undefined;
  updatedAt: string;
  cells: number;
}

export interface RunLeaderboardRow {
  rank: number;
  algorithmId: string;
  overallScore: number | null;
  meanBitAccuracy: number | null;
  meanBitErrorRate: number | null;
  cellCount: number;
}

export interface CurvePoint {
  strength: number;
  accuracy: number;
}

export interface CurveSeries {
  algorithmId: string;
  points: CurvePoint[];
}

export function summarizeRuns(runs: DemoRunRecord[]): RunStats {
  return {
    queued: runs.filter((run) => run.status === "queued").length,
    running: runs.filter((run) => run.status === "running").length,
    completed: runs.filter((run) => run.status === "succeeded").length,
    failed: runs.filter(
      (run) => run.status === "failed" || run.status === "partially_failed" || run.status === "cancelled"
    ).length
  };
}

export function buildActiveRunRows(
  runs: DemoRunRecord[],
  configs: SavedExperimentConfig[],
  datasets: DatasetVersion[],
  algorithms: AlgorithmVersion[],
  attacks: AttackPreset[]
): ActiveRunRow[] {
  const configMap = new Map(configs.map((config) => [config.id, config]));
  const datasetMap = new Map(datasets.map((dataset) => [dataset.id, dataset.name]));
  const algorithmMap = new Map(algorithms.map((algorithm) => [algorithm.id, algorithm.name]));
  const attackMap = new Map(attacks.map((attack) => [attack.id, attack.name]));

  return runs
    .filter((run) => run.status === "queued" || run.status === "running")
    .map((run) => {
      const config = configMap.get(run.configId);
      return {
        id: run.id,
        configName: run.configName,
        datasetLabel: labelFromIds(config?.selection.datasetIds ?? [], datasetMap),
        algorithmLabel: labelFromIds(config?.selection.algorithmIds ?? [], algorithmMap),
        attackLabel: labelFromIds(config?.selection.attackPresetIds ?? [], attackMap),
        progress: run.progress,
        status: run.status,
        startedAt: run.startedAt,
        updatedAt: run.updatedAt,
        cells: run.cells
      };
    });
}

export function rankAggregates(aggregates: RunAggregate[]): RunLeaderboardRow[] {
  const grouped = new Map<string, { accuracies: number[]; errorRates: number[]; cellCount: number }>();
  for (const item of aggregates) {
    const current = grouped.get(item.algorithmId) ?? {
      accuracies: [],
      errorRates: [],
      cellCount: 0
    };
    if (item.meanBitAccuracy != null) {
      current.accuracies.push(item.meanBitAccuracy);
    }
    if (item.meanBitErrorRate != null) {
      current.errorRates.push(item.meanBitErrorRate);
    }
    current.cellCount += item.cellCount;
    grouped.set(item.algorithmId, current);
  }

  return Array.from(grouped.entries())
    .map(([algorithmId, value]) => {
      const meanBitAccuracy = mean(value.accuracies);
      const meanBitErrorRate = mean(value.errorRates);
      return {
        rank: 0,
        algorithmId,
        overallScore: meanBitAccuracy,
        meanBitAccuracy,
        meanBitErrorRate,
        cellCount: value.cellCount
      };
    })
    .sort((a, b) => (b.overallScore ?? -1) - (a.overallScore ?? -1))
    .map((row, index) => ({ ...row, rank: index + 1 }));
}

export function buildCurveSeries(results: RunResults | null): CurveSeries[] {
  if (!results) {
    return [];
  }
  const grouped = new Map<string, CurvePoint[]>();
  for (const item of results.aggregates) {
    if (item.meanBitAccuracy == null) {
      continue;
    }
    const points = grouped.get(item.algorithmId) ?? [];
    points.push({
      strength: item.attackStrength,
      accuracy: item.meanBitAccuracy
    });
    grouped.set(item.algorithmId, points);
  }

  return Array.from(grouped.entries())
    .map(([algorithmId, points]) => ({
      algorithmId,
      points: points.sort((a, b) => a.strength - b.strength)
    }))
    .filter((series) => series.points.length >= 2);
}

export function formatMetric(value: number | null | undefined, digits = 3): string {
  return value == null ? "n/a" : value.toFixed(digits);
}

export function statusBadgeClass(status: RunStatus): string {
  if (status === "succeeded") {
    return "badge ok";
  }
  if (status === "failed" || status === "partially_failed" || status === "cancelled") {
    return "badge error";
  }
  return "badge warn";
}

function labelFromIds(ids: string[], lookup: Map<string, string>): string {
  if (ids.length === 0) {
    return "n/a";
  }
  const first = lookup.get(ids[0]) ?? ids[0];
  return ids.length === 1 ? first : `${first} +${ids.length - 1}`;
}

function mean(values: number[]): number | null {
  if (values.length === 0) {
    return null;
  }
  return values.reduce((total, value) => total + value, 0) / values.length;
}
