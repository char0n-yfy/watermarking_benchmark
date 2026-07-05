import type { DemoRunRecord, RunAggregate, RunPhaseKey, RunPhaseState, RunResults, RunStatus } from "./types";

export const terminalRunStatuses = new Set<RunStatus>([
  "succeeded",
  "failed",
  "paused",
  "cancelled",
  "partially_failed"
]);

export interface RunStats {
  running: number;
  completed: number;
}

export interface ActiveRunRow {
  id: string;
  experimentName: string;
  stageKey: RunPhaseKey | string;
  stageLabel: string;
  cellProgress: { current: number; total: number; percent: number };
  phaseProgress: { current: number; total: number; percent: number };
  status: RunStatus;
  startedAt: string | null | undefined;
  updatedAt: string;
  cells: number;
}

const phaseOrder: RunPhaseKey[] = [
  "canonical",
  "watermark_embed",
  "attack",
  "watermark_extract",
  "quality",
  "summary"
];

const fallbackPhaseLabels: Record<RunPhaseKey, string> = {
  canonical: "采样 canonical 数据集",
  watermark_embed: "嵌入水印",
  attack: "攻击",
  watermark_extract: "提取",
  quality: "评估质量",
  summary: "汇总"
};

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
    running: runs.filter((run) => run.status === "running").length,
    completed: runs.filter((run) => run.status === "succeeded").length
  };
}

export function buildActiveRunRows(runs: DemoRunRecord[]): ActiveRunRow[] {
  return runs
    .filter((run) => run.status === "running" || run.status === "paused")
    .map((run) => {
      const stage = currentRunPhase(run);
      const stageIndex = Math.max(0, phaseOrder.indexOf(stage.key as RunPhaseKey));
      const phaseCurrent = stageIndex >= 0 ? stageIndex + 1 : 1;
      const cellProgress = phaseCellProgress(run, stage);
      return {
        id: run.id,
        experimentName: run.taskName || run.configName || run.id,
        stageKey: stage.key,
        stageLabel: stage.label || fallbackPhaseLabels[stage.key as RunPhaseKey] || stage.key,
        cellProgress,
        phaseProgress: progressDoc(phaseCurrent, phaseOrder.length),
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

function mean(values: number[]): number | null {
  if (values.length === 0) {
    return null;
  }
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function currentRunPhase(run: DemoRunRecord): RunPhaseState {
  const phases = run.phases ?? [];
  const currentKey = run.currentPhase;
  const explicit = phases.find((phase) => phase.key === currentKey);
  if (explicit) {
    return explicit;
  }
  for (const key of [...phaseOrder].reverse()) {
    const phase = phases.find((item) => item.key === key);
    if (phase && phase.status !== "pending") {
      return phase;
    }
  }
  const fallbackKey = (currentKey || "canonical") as RunPhaseKey;
  return {
    key: fallbackKey,
    label: fallbackPhaseLabels[fallbackKey] ?? fallbackKey,
    status: run.status,
    current: 0,
    total: 0,
    percent: 0
  };
}

function phaseCellProgress(run: DemoRunRecord, phase: RunPhaseState) {
  const phaseCell = phase.cellProgress;
  if (phaseCell && Number(phaseCell.total) > 0) {
    return progressDoc(Number(phaseCell.current ?? 0), Number(phaseCell.total ?? 0));
  }
  const counters = phase.counters ?? {};
  const counterCurrent = Number(counters.phaseCellsDone ?? counters.resultUnitsDone ?? 0);
  const counterTotal = Number(counters.phaseCellsTotal ?? run.cells ?? 0);
  if (counterTotal > 0) {
    return progressDoc(counterCurrent, counterTotal);
  }
  return progressDoc(Math.round((Number(run.progress ?? 0) / 100) * Number(run.cells ?? 0)), Number(run.cells ?? 0));
}

function progressDoc(current: number, total: number) {
  const safeCurrent = Math.max(0, Math.round(Number.isFinite(current) ? current : 0));
  const safeTotal = Math.max(safeCurrent, Math.round(Number.isFinite(total) ? total : 0));
  const percent = safeTotal > 0 ? Math.round((safeCurrent / safeTotal) * 100) : 0;
  return {
    current: safeCurrent,
    total: safeTotal,
    percent: Math.max(0, Math.min(100, percent))
  };
}
