"use client";

import { useEffect, useMemo, useState } from "react";
import {
  BarChart3,
  CheckCircle2,
  Clock3,
  FolderOpen,
  PauseCircle,
  PlayCircle,
  RefreshCw,
  RotateCcw,
  Save,
  SlidersHorizontal,
  Square,
  TerminalSquare,
  XCircle,
  Zap
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { useLanguage } from "@/components/LanguageProvider";
import {
  cancelParallelTuning,
  cancelRun,
  createRun,
  fetchAlgorithms,
  fetchAttacks,
  fetchDatasetCatalog,
  fetchLatestParallelTuning,
  fetchParallelTuning,
  fetchRun,
  fetchRunEvents,
  fetchRunLogs,
  fetchRuns,
  fetchSavedConfigs,
  pauseRun,
  resumeRun,
  saveParallelTuning,
  startParallelTuning
} from "@/lib/api";
import { localizedDate } from "@/lib/i18n";
import type {
  DemoRunRecord,
  ParallelTuningEvent,
  ParallelTuningJob,
  RunEvents,
  RunLogs,
  RunStageEvent,
  SavedExperimentConfig
} from "@/lib/types";

type StartMode = "new" | "resume";
type TuningMode = "quick" | "full";

type ProgressStep = {
  key: string;
  label: string;
  current: number;
  total: number;
  percent: number;
  meta: string;
};

type ExecutionSummary = {
  taskName: string;
  runId: string;
  status: DemoRunRecord["status"];
  progress: number;
  cells: number;
  completedCells: number;
  succeededCells: number;
  failedCells: number;
  remainingCells: number;
  configName: string;
  workerId?: string | null;
  artifactRoot?: string;
  createdAt?: string;
  startedAt?: string | null;
  finishedAt?: string | null;
  updatedAt?: string;
  durationMs: number | null;
  selection?: {
    datasets: number;
    watermarks: number;
    attacks: number;
    seeds: number;
    maxSamples: number;
    sampleCount: number;
    imageOperationCount: number;
  };
  latestEvent?: {
    title: string;
    meta: string;
    timestamp?: string;
  } | null;
  log?: {
    path: string;
    lineCount: number;
    lastLine: string;
    tailLines: string[];
  } | null;
  note: string;
};

type AttackOutcome = {
  attackId: string;
  succeeded: number;
  failed: number;
  latestParam: string;
};

type CurrentExecution = {
  datasetId: string;
  algorithmId: string;
  attackId: string;
  attackParam: string;
  cellKey: string;
};

type MonitorStats = {
  completedCells: number;
  succeededCells: number;
  failedCells: number;
  attackOutcomes: AttackOutcome[];
  current: CurrentExecution;
};

type TuningForm = {
  mode: TuningMode;
  sampleCount: number;
  warmupCount: number;
  minBatchSize: number;
  maxBatchSize: number;
  minWorkerCount: number;
  maxWorkerCount: number;
  repeatCount: number;
  minImprovementRatio: number;
  tuneWatermarks: boolean;
  tuneAttacks: boolean;
  includeViewpoint3dAttacks: boolean;
  tuneQuality: boolean;
};

type TuningPoint = {
  key: string;
  stage: string;
  label: string;
  candidate: number;
  throughput: number;
  kind: "batch" | "workers";
  ok: boolean;
};

const terminalStatuses = new Set<DemoRunRecord["status"]>(["succeeded", "failed", "paused", "cancelled", "partially_failed"]);
const resumableStatuses = new Set<DemoRunRecord["status"]>(["paused", "failed", "partially_failed"]);
const finalCellStatuses = new Set(["succeeded", "failed", "skipped", "paused", "cancelled"]);
const finalWatermarkStatuses = new Set(["succeeded", "failed", "skipped", "paused", "cancelled"]);
const rawArtifactFiles = [
  "run_plan.json",
  "cell_manifest.jsonl",
  "image_quality.jsonl",
  "image_detection.jsonl",
  "runtime_profile.jsonl",
  "stage_events.jsonl",
  "run_status.json"
];

const quickTuningDefaults: TuningForm = {
  mode: "quick",
  sampleCount: 16,
  warmupCount: 2,
  minBatchSize: 1,
  maxBatchSize: 16,
  minWorkerCount: 1,
  maxWorkerCount: 32,
  repeatCount: 1,
  minImprovementRatio: 0.03,
  tuneWatermarks: true,
  tuneAttacks: true,
  includeViewpoint3dAttacks: false,
  tuneQuality: true
};

const fullTuningDefaults: TuningForm = {
  mode: "full",
  sampleCount: 64,
  warmupCount: 4,
  minBatchSize: 1,
  maxBatchSize: 64,
  minWorkerCount: 1,
  maxWorkerCount: 64,
  repeatCount: 3,
  minImprovementRatio: 0.03,
  tuneWatermarks: true,
  tuneAttacks: true,
  includeViewpoint3dAttacks: false,
  tuneQuality: true
};

function badgeClass(status: DemoRunRecord["status"]) {
  if (status === "running" || status === "succeeded") {
    return "badge ok";
  }
  if (status === "failed" || status === "partially_failed" || status === "cancelled") {
    return "badge error";
  }
  return "badge warn";
}

function isActiveRun(status: DemoRunRecord["status"]) {
  return status === "queued" || status === "running";
}

function isResumableRun(status: DemoRunRecord["status"]) {
  return resumableStatuses.has(status);
}

function isRestartableTerminalRun(status: DemoRunRecord["status"]) {
  return isResumableRun(status);
}

function isTerminalRun(status: DemoRunRecord["status"]) {
  return terminalStatuses.has(status);
}

function isPausableRun(run: DemoRunRecord) {
  return isActiveRun(run.status) && !run.cancelRequested;
}

function isCancellableRun(run: DemoRunRecord) {
  return isActiveRun(run.status) && run.stopIntent !== "cancel";
}

function stopIntentNotice(
  run: DemoRunRecord,
  labels: {
    pauseRequestedNotice: string;
    cancelRequestedNotice: string;
  }
) {
  if (!run.cancelRequested) {
    return null;
  }
  return run.stopIntent === "cancel" ? labels.cancelRequestedNotice : labels.pauseRequestedNotice;
}

function tuningStatusClass(status: string | undefined) {
  if (status === "succeeded") {
    return "badge ok";
  }
  if (status === "failed" || status === "cancelled") {
    return "badge error";
  }
  return "badge warn";
}

function taskName(run: DemoRunRecord) {
  return run.taskName || run.configName || run.id;
}

function progressWidth(progress: number) {
  return `${Math.max(0, Math.min(100, progress))}%`;
}

function positiveInteger(value: number, fallback: number) {
  return Number.isFinite(value) && value > 0 ? Math.round(value) : fallback;
}

function candidateRange(minValue: number, maxValue: number, extras: number[] = []) {
  const minCandidate = positiveInteger(Math.min(minValue, maxValue), 1);
  const maxCandidate = positiveInteger(Math.max(minValue, maxValue), minCandidate);
  const values = new Set<number>(extras.filter((value) => value >= minCandidate && value <= maxCandidate));
  let value = 1;
  while (value <= maxCandidate) {
    if (value >= minCandidate) {
      values.add(value);
    }
    value *= 2;
  }
  values.add(minCandidate);
  values.add(maxCandidate);
  return [...values].sort((left, right) => left - right);
}

function buildTuningPayload(form: TuningForm) {
  const maxBatchSize = positiveInteger(form.maxBatchSize, form.mode === "full" ? 64 : 16);
  const maxWorkerCount = positiveInteger(form.maxWorkerCount, form.mode === "full" ? 64 : 32);
  const batchCandidates = candidateRange(form.minBatchSize, maxBatchSize);
  const workerCandidates = candidateRange(form.minWorkerCount, maxWorkerCount, [24, 32, 48, 64, 96, 128]);
  const sampleCount = Math.max(positiveInteger(form.sampleCount, maxBatchSize), maxBatchSize, Math.max(...batchCandidates));
  return {
    mode: form.mode,
    sampleCount,
    warmupCount: positiveInteger(form.warmupCount, 2),
    batchCandidates,
    workerCandidates,
    repeatCount: positiveInteger(form.repeatCount, form.mode === "full" ? 3 : 1),
    maxBatchSize,
    maxWorkerCount,
    autoExpandCandidates: form.mode === "full",
    minImprovementRatio: Math.max(0, form.minImprovementRatio),
    boundaryPatience: form.mode === "full" ? 2 : 1,
    tuneWatermarks: form.tuneWatermarks,
    tuneAttacks: form.tuneAttacks,
    includeViewpoint3dAttacks: form.tuneAttacks && form.includeViewpoint3dAttacks,
    tuneQuality: form.tuneQuality
  };
}

function percent(current: number, total: number) {
  if (total <= 0) {
    return 0;
  }
  return Math.round((Math.max(0, current) / total) * 100);
}

function latestEvent(events: RunEvents | null) {
  if (!events?.events.length) {
    return null;
  }
  return events.events[events.events.length - 1];
}

function eventTitle(event: RunStageEvent | null) {
  if (!event) {
    return "n/a";
  }
  return [event.stage, event.status].filter(Boolean).join(" · ") || "event";
}

function eventMeta(event: RunStageEvent | null) {
  if (!event) {
    return "n/a";
  }
  const items = [
    event.datasetId ? `dataset=${event.datasetId}` : null,
    event.algorithmId ? `wm=${event.algorithmId}` : null,
    event.attackPresetId ? `attack=${event.attackPresetId}` : null,
    typeof event.attackStrength === "number" ? `strength=${event.attackStrength}` : null,
    event.cellKey ? `cell=${event.cellKey}` : null
  ];
  return items.filter(Boolean).join("  ") || event.error || "n/a";
}

function latestMapped<T>(events: RunStageEvent[], mapValue: (event: RunStageEvent) => T | null | undefined) {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const value = mapValue(events[index]);
    if (value !== null && value !== undefined && value !== "") {
      return value;
    }
  }
  return null;
}

function formatNumber(value: number) {
  return Number.isInteger(value) ? value.toString() : Number(value.toFixed(4)).toString();
}

function durationMsBetween(start?: string | null, end?: string | null) {
  if (!start || !end) {
    return null;
  }
  const startMs = Date.parse(start);
  const endMs = Date.parse(end);
  if (Number.isNaN(startMs) || Number.isNaN(endMs)) {
    return null;
  }
  return Math.max(0, endMs - startMs);
}

function formatDurationMs(durationMs: number | null, language: "zh" | "en") {
  if (durationMs === null) {
    return "n/a";
  }
  const totalSeconds = Math.max(0, Math.round(durationMs / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (language === "zh") {
    if (hours > 0) {
      return `${hours}小时 ${minutes}分钟 ${seconds}秒`;
    }
    if (minutes > 0) {
      return `${minutes}分钟 ${seconds}秒`;
    }
    return `${seconds}秒`;
  }
  if (hours > 0) {
    return `${hours}h ${minutes}m ${seconds}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
}

function formatParamValue(value: unknown): string {
  if (typeof value === "number") {
    return formatNumber(value);
  }
  if (typeof value === "string" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => formatParamValue(item)).join(", ")}]`;
  }
  if (value && typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function paramsLabel(params: unknown) {
  if (!params || typeof params !== "object" || Array.isArray(params)) {
    return null;
  }
  const entries = Object.entries(params as Record<string, unknown>).filter(([, value]) => value !== null && value !== undefined);
  if (!entries.length) {
    return null;
  }
  return entries.map(([key, value]) => `${key}=${formatParamValue(value)}`).join(", ");
}

function strengthFromEvent(event: RunStageEvent) {
  if (typeof event.attackStrength === "number") {
    return formatNumber(event.attackStrength);
  }
  const parts = typeof event.cellKey === "string" ? event.cellKey.split("__") : [];
  return parts.length >= 4 && parts[3] ? parts[3] : null;
}

function attackParamLabel(event: RunStageEvent) {
  const params = paramsLabel(event.attackParams);
  if (params) {
    return params;
  }
  const strength = strengthFromEvent(event);
  return strength ? `strength=${strength}` : null;
}

function humanizeId(id: string) {
  return id.replace(/^(atk|alg|ds)-/, "").replace(/[_-]+/g, " ");
}

function displayName(id: string, names: Record<string, string>) {
  if (!id || id === "n/a") {
    return "n/a";
  }
  return names[id] ?? humanizeId(id);
}

function eventNumber(event: ParallelTuningEvent, key: string) {
  const value = event[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function tuningPoints(job: ParallelTuningJob | null): TuningPoint[] {
  const events = job?.events ?? [];
  return events
    .map((event, index) => {
      const throughput = eventNumber(event, "imagesPerSecond");
      const batchSize = eventNumber(event, "batchSize");
      const workers = eventNumber(event, "workers");
      if (throughput == null || (batchSize == null && workers == null)) {
        return null;
      }
      const method = typeof event.method === "string" ? event.method : String(event.message ?? event.stage ?? "candidate");
      const kind = batchSize != null ? "batch" : "workers";
      const candidate = batchSize ?? workers ?? 1;
      return {
        key: `${event.timestamp ?? "point"}-${index}`,
        stage: String(event.stage ?? "step"),
        label: method,
        candidate,
        throughput,
        kind,
        ok: event.ok !== false
      } satisfies TuningPoint;
    })
    .filter((point): point is TuningPoint => Boolean(point));
}

function uniqueEventCount(events: RunStageEvent[], predicate: (event: RunStageEvent) => boolean, keyOf: (event: RunStageEvent) => string | null) {
  const keys = new Set<string>();
  events.forEach((event) => {
    if (!predicate(event)) {
      return;
    }
    const key = keyOf(event);
    if (key) {
      keys.add(key);
    }
  });
  return keys.size;
}

function variantCountForAttack(config: SavedExperimentConfig | undefined, attackId: string) {
  const params = config?.selection.attackParamOverrides?.[attackId];
  if (params?.length) {
    return params.length;
  }
  const strengths = config?.selection.attackStrengthOverrides?.[attackId];
  if (strengths?.length) {
    return strengths.length;
  }
  return 1;
}

function selectionSummary(config: SavedExperimentConfig | undefined): ExecutionSummary["selection"] {
  if (!config) {
    return undefined;
  }
  return {
    datasets: config.selection.datasetIds.length,
    watermarks: config.selection.algorithmIds.length,
    attacks: config.selection.attackPresetIds.length,
    seeds: config.selection.seeds.length,
    maxSamples: config.selection.maxSamples,
    sampleCount: config.sampleCount,
    imageOperationCount: config.imageOperationCount
  };
}

function lastLogLine(logs: RunLogs | null) {
  if (!logs?.exists) {
    return "";
  }
  for (let index = logs.lines.length - 1; index >= 0; index -= 1) {
    const line = logs.lines[index]?.trim();
    if (line) {
      return line;
    }
  }
  return "";
}

function logTailLines(logs: RunLogs | null, maxLines = 6) {
  if (!logs?.exists) {
    return [];
  }
  return logs.lines
    .map((line) => line.trimEnd())
    .filter((line) => line.trim())
    .slice(-maxLines);
}

function completedCellsFromRun(run: DemoRunRecord) {
  if (run.cells <= 0) {
    return 0;
  }
  return Math.max(0, Math.min(run.cells, Math.round((run.progress / 100) * run.cells)));
}

function attackIdFromEvent(event: RunStageEvent, attackIds: string[]) {
  if (typeof event.attackPresetId === "string" && event.attackPresetId) {
    return event.attackPresetId;
  }
  const cellKey = typeof event.cellKey === "string" ? event.cellKey : "";
  return attackIds.find((attackId) => cellKey.includes(`__${attackId}__`)) ?? null;
}

function buildMonitorStats(
  run: DemoRunRecord | null,
  config: SavedExperimentConfig | undefined,
  events: RunEvents | null
): MonitorStats {
  const eventList = events?.events ?? [];
  const attackIds = config?.selection.attackPresetIds ?? [];
  const finalEvents = new Map<string, RunStageEvent>();

  eventList.forEach((event) => {
    if (event.stage !== "cell" || !finalCellStatuses.has(String(event.status)) || typeof event.cellKey !== "string") {
      return;
    }
    finalEvents.set(event.cellKey, event);
  });

  let succeededCells = 0;
  let failedCells = 0;
  const outcomes = new Map<string, AttackOutcome>();

  finalEvents.forEach((event) => {
    const attackId = attackIdFromEvent(event, attackIds) ?? "n/a";
    const current =
      outcomes.get(attackId) ??
      ({
        attackId,
        succeeded: 0,
        failed: 0,
        latestParam: attackParamLabel(event) ?? "n/a"
      } satisfies AttackOutcome);

    if (event.status === "succeeded" || event.status === "skipped") {
      succeededCells += 1;
      current.succeeded += 1;
    } else {
      failedCells += 1;
      current.failed += 1;
    }
    current.latestParam = attackParamLabel(event) ?? current.latestParam;
    outcomes.set(attackId, current);
  });

  const currentAttackId = latestMapped(eventList, (event) => attackIdFromEvent(event, attackIds)) ?? attackIds[0] ?? "n/a";

  return {
    completedCells: finalEvents.size,
    succeededCells,
    failedCells,
    attackOutcomes: [...outcomes.values()].sort((left, right) => left.attackId.localeCompare(right.attackId)),
    current: {
      datasetId:
        latestMapped(eventList, (event) => (typeof event.datasetId === "string" ? event.datasetId : null)) ??
        config?.selection.datasetIds[0] ??
        "n/a",
      algorithmId:
        latestMapped(eventList, (event) => (typeof event.algorithmId === "string" ? event.algorithmId : null)) ??
        config?.selection.algorithmIds[0] ??
        "n/a",
      attackId: currentAttackId,
      attackParam:
        latestMapped(eventList, (event) => (attackIdFromEvent(event, attackIds) ? attackParamLabel(event) : null)) ?? "n/a",
      cellKey:
        latestMapped(eventList, (event) => (typeof event.cellKey === "string" ? event.cellKey : null)) ??
        (run?.id ? `${run.id}:pending` : "n/a")
    }
  };
}

function buildProgressSteps(
  run: DemoRunRecord | null,
  config: SavedExperimentConfig | undefined,
  events: RunEvents | null,
  labels: {
    taskProgress: string;
    datasetProgress: string;
    watermarkProgress: string;
    attackProgress: string;
    hyperProgress: string;
    currentAttack: string;
    waitingForStage: string;
  }
): ProgressStep[] {
  const selection = config?.selection;
  const datasetCount = Math.max(0, selection?.datasetIds.length ?? 0);
  const algorithmCount = Math.max(0, selection?.algorithmIds.length ?? 0);
  const attackIds = selection?.attackPresetIds ?? [];
  const seedCount = Math.max(0, selection?.seeds.length ?? 0);
  const eventList = events?.events ?? [];
  const taskProgress = run?.progress ?? 0;

  const datasetDone = uniqueEventCount(
    eventList,
    (event) => event.stage === "dataset" && event.status === "finished",
    (event) => (typeof event.datasetId === "string" ? event.datasetId : null)
  );
  const watermarkTotal = datasetCount * algorithmCount * seedCount;
  const watermarkDone = uniqueEventCount(
    eventList,
    (event) => event.stage === "watermark_embed" && finalWatermarkStatuses.has(String(event.status)),
    (event) => (typeof event.cellKey === "string" ? event.cellKey : null)
  );
  const attackDone = uniqueEventCount(
    eventList,
    (event) => event.stage === "cell" && finalCellStatuses.has(String(event.status)),
    (event) => (typeof event.cellKey === "string" ? event.cellKey : null)
  );

  const currentAttack =
    [...eventList]
      .reverse()
      .map((event) => attackIdFromEvent(event, attackIds))
      .find(Boolean) ??
    attackIds[0] ??
    "";
  const currentAttackTotal = currentAttack
    ? datasetCount * algorithmCount * seedCount * variantCountForAttack(config, currentAttack)
    : 0;
  const currentAttackDone = currentAttack
    ? uniqueEventCount(
        eventList,
        (event) =>
          event.stage === "cell" &&
          finalCellStatuses.has(String(event.status)) &&
          attackIdFromEvent(event, attackIds) === currentAttack,
        (event) => (typeof event.cellKey === "string" ? event.cellKey : null)
      )
    : 0;

  return [
    {
      key: "task",
      label: labels.taskProgress,
      current: taskProgress,
      total: 100,
      percent: taskProgress,
      meta: `${taskProgress}%`
    },
    {
      key: "dataset",
      label: labels.datasetProgress,
      current: datasetDone,
      total: datasetCount,
      percent: percent(datasetDone, datasetCount),
      meta: datasetCount ? `${datasetDone}/${datasetCount}` : labels.waitingForStage
    },
    {
      key: "watermark",
      label: labels.watermarkProgress,
      current: watermarkDone,
      total: watermarkTotal,
      percent: percent(watermarkDone, watermarkTotal),
      meta: watermarkTotal ? `${watermarkDone}/${watermarkTotal}` : labels.waitingForStage
    },
    {
      key: "attack",
      label: labels.attackProgress,
      current: attackDone,
      total: run?.cells ?? 0,
      percent: percent(attackDone, run?.cells ?? 0),
      meta: run?.cells ? `${attackDone}/${run.cells}` : labels.waitingForStage
    },
    {
      key: "hyper",
      label: labels.hyperProgress,
      current: currentAttackDone,
      total: currentAttackTotal,
      percent: percent(currentAttackDone, currentAttackTotal),
      meta: currentAttack
        ? `${labels.currentAttack}: ${currentAttack} · ${currentAttackDone}/${currentAttackTotal}`
        : labels.waitingForStage
    }
  ];
}

function makeSummary(
  run: DemoRunRecord,
  note: string,
  options: {
    statusOverride?: DemoRunRecord["status"];
    config?: SavedExperimentConfig;
    events?: RunEvents | null;
    logs?: RunLogs | null;
  } = {}
): ExecutionSummary {
  const stats = buildMonitorStats(run, options.config, options.events ?? null);
  const completedCells = Math.max(stats.completedCells, completedCellsFromRun(run));
  const latest = latestEvent(options.events ?? null);
  const logLine = lastLogLine(options.logs ?? null);
  const tailLines = logTailLines(options.logs ?? null);
  return {
    taskName: taskName(run),
    runId: run.id,
    status: options.statusOverride ?? run.status,
    progress: run.progress,
    cells: run.cells,
    completedCells,
    succeededCells: stats.succeededCells,
    failedCells: stats.failedCells,
    remainingCells: Math.max(0, run.cells - completedCells),
    configName: run.configName,
    workerId: run.workerId,
    artifactRoot: run.artifactRoot,
    createdAt: run.createdAt,
    startedAt: run.startedAt,
    finishedAt: run.finishedAt,
    updatedAt: run.updatedAt,
    durationMs: durationMsBetween(run.startedAt, run.finishedAt ?? run.updatedAt),
    selection: selectionSummary(options.config),
    latestEvent: latest
      ? {
          title: eventTitle(latest),
          meta: eventMeta(latest),
          timestamp: latest.timestamp
        }
      : null,
    log: options.logs
      ? {
          path: options.logs.logPath,
          lineCount: options.logs.exists ? options.logs.lines.length : 0,
          lastLine: logLine,
          tailLines
        }
      : null,
    note
  };
}

function terminalRunNote(
  status: DemoRunRecord["status"],
  labels: {
    runFinishedNotice: string;
    stopSavedNotice: string;
    runCancelledNotice: string;
    runFailedNotice: string;
  }
) {
  if (status === "succeeded") {
    return labels.runFinishedNotice;
  }
  if (status === "paused") {
    return labels.stopSavedNotice;
  }
  if (status === "cancelled") {
    return labels.runCancelledNotice;
  }
  return labels.runFailedNotice;
}

function runStatusLabel(
  status: DemoRunRecord["status"],
  statusLabels: Record<string, string>
) {
  return statusLabels[status] ?? status;
}

export default function RunsPage() {
  const { language, t } = useLanguage();
  const [configs, setConfigs] = useState<SavedExperimentConfig[]>([]);
  const [activeRuns, setActiveRuns] = useState<DemoRunRecord[]>([]);
  const [selectedConfigId, setSelectedConfigId] = useState("");
  const [selectedResumeRunId, setSelectedResumeRunId] = useState("");
  const [taskNameInput, setTaskNameInput] = useState("");
  const [startMode, setStartMode] = useState<StartMode>("new");
  const [startDialogOpen, setStartDialogOpen] = useState(false);
  const [monitorRunId, setMonitorRunId] = useState("");
  const [monitorRun, setMonitorRun] = useState<DemoRunRecord | null>(null);
  const [logs, setLogs] = useState<RunLogs | null>(null);
  const [events, setEvents] = useState<RunEvents | null>(null);
  const [lastSummary, setLastSummary] = useState<ExecutionSummary | null>(null);
  const [cancelConfirmOpen, setCancelConfirmOpen] = useState(false);
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [resourceNames, setResourceNames] = useState<Record<string, string>>({});
  const [tuningJob, setTuningJob] = useState<ParallelTuningJob | null>(null);
  const [tuningDialogOpen, setTuningDialogOpen] = useState(false);
  const [tuningForm, setTuningForm] = useState<TuningForm>(quickTuningDefaults);
  const [tuningBusy, setTuningBusy] = useState(false);
  const [tuningNotice, setTuningNotice] = useState("");

  const selectedConfig = useMemo(
    () => configs.find((config) => config.id === selectedConfigId),
    [configs, selectedConfigId]
  );
  const selectedResumeRun = useMemo(
    () => activeRuns.find((run) => run.id === selectedResumeRunId),
    [activeRuns, selectedResumeRunId]
  );
  const monitoredConfig = useMemo(
    () => configs.find((config) => config.id === monitorRun?.configId),
    [configs, monitorRun?.configId]
  );
  const currentEvent = latestEvent(events);
  const progressSteps = useMemo(
    () =>
      buildProgressSteps(monitorRun, monitoredConfig, events, {
        taskProgress: t.runs.taskProgress,
        datasetProgress: t.runs.datasetProgress,
        watermarkProgress: t.runs.watermarkProgress,
        attackProgress: t.runs.attackProgress,
        hyperProgress: t.runs.hyperProgress,
        currentAttack: t.runs.currentAttack,
        waitingForStage: t.runs.waitingForStage
      }),
    [
      events,
      monitorRun,
      monitoredConfig,
      t.runs.attackProgress,
      t.runs.currentAttack,
      t.runs.datasetProgress,
      t.runs.hyperProgress,
      t.runs.taskProgress,
      t.runs.waitingForStage,
      t.runs.watermarkProgress
    ]
  );
  const monitorStats = useMemo(() => buildMonitorStats(monitorRun, monitoredConfig, events), [events, monitorRun, monitoredConfig]);
  const successfulAttackOutcomes = useMemo(
    () => monitorStats.attackOutcomes.filter((outcome) => outcome.succeeded > 0),
    [monitorStats.attackOutcomes]
  );
  const failedAttackOutcomes = useMemo(
    () => monitorStats.attackOutcomes.filter((outcome) => outcome.failed > 0),
    [monitorStats.attackOutcomes]
  );
  const tuningRunning = tuningJob?.status === "running";
  const tuningEnvEntries = Object.entries(tuningJob?.summary?.envUpdates ?? {});
  const tuningChartPoints = useMemo(() => tuningPoints(tuningJob).slice(-28), [tuningJob]);
  const tuningEvents = (tuningJob?.events ?? []).slice(-8).reverse();
  const effectiveTuningPayload = useMemo(() => buildTuningPayload(tuningForm), [tuningForm]);

  const refreshBase = async () => {
    const [loadedConfigs, loadedRuns, latestTuning] = await Promise.all([
      fetchSavedConfigs(),
      fetchRuns({ scope: "unfinished" }),
      fetchLatestParallelTuning().catch(() => null)
    ]);
    const manageableRuns = loadedRuns.filter((run) => isActiveRun(run.status) || isResumableRun(run.status));
    setConfigs(loadedConfigs);
    setActiveRuns(manageableRuns);
    if (latestTuning) {
      setTuningJob(latestTuning);
    }
    setSelectedConfigId((current) => {
      if (current && loadedConfigs.some((config) => config.id === current)) {
        return current;
      }
      return loadedConfigs[0]?.id ?? "";
    });
    setSelectedResumeRunId((current) => {
      if (current && manageableRuns.some((run) => run.id === current)) {
        return current;
      }
      return manageableRuns[0]?.id ?? "";
    });
    return { loadedConfigs, manageableRuns };
  };

  useEffect(() => {
    let cancelled = false;
    const loadResourceNames = async () => {
      const [datasetCatalog, algorithms, attacks] = await Promise.all([
        fetchDatasetCatalog().catch(() => null),
        fetchAlgorithms().catch(() => []),
        fetchAttacks().catch(() => [])
      ]);
      if (cancelled) {
        return;
      }
      const nextResourceNames: Record<string, string> = {};
      datasetCatalog?.items.forEach((dataset) => {
        nextResourceNames[dataset.id] = language === "zh" ? dataset.nameZh || dataset.name : dataset.name || dataset.nameZh;
      });
      algorithms.forEach((algorithm) => {
        nextResourceNames[algorithm.id] = algorithm.name;
      });
      attacks.forEach((attack) => {
        nextResourceNames[attack.id] = attack.name;
      });
      setResourceNames(nextResourceNames);
    };
    loadResourceNames().catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [language]);

  useEffect(() => {
    if (monitorRunId || startDialogOpen || lastSummary) {
      return;
    }
    const runningRun = activeRuns.find((run) => run.status === "running" && !run.cancelRequested);
    if (runningRun) {
      setMonitorRunId(runningRun.id);
      setMonitorRun(runningRun);
      setNotice("");
    }
  }, [activeRuns, lastSummary, monitorRunId, startDialogOpen]);

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      refreshBase().catch(() => {
        if (!cancelled) {
          setNotice(t.runs.apiUnavailable);
        }
      });
    };
    load();
    const timer = window.setInterval(load, 2500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [t.runs.apiUnavailable]);

  useEffect(() => {
    if (!monitorRunId) {
      setMonitorRun(null);
      setLogs(null);
      setEvents(null);
      return;
    }

    let cancelled = false;
    const loadMonitor = async () => {
      try {
        const [runValue, eventValue, logValue] = await Promise.all([
          fetchRun(monitorRunId),
          fetchRunEvents(monitorRunId).catch(() => null),
          fetchRunLogs(monitorRunId).catch(() => null)
        ]);
        if (cancelled) {
          return;
        }
        setMonitorRun(runValue);
        setEvents(eventValue);
        setLogs(logValue);
        if (isTerminalRun(runValue.status)) {
          const summaryConfig = configs.find((config) => config.id === runValue.configId);
          const note = terminalRunNote(runValue.status, t.runs);
          setLastSummary(makeSummary(runValue, note, { config: summaryConfig, events: eventValue, logs: logValue }));
          setNotice("");
          setMonitorRunId("");
          setCancelConfirmOpen(false);
          refreshBase().catch(() => undefined);
        }
      } catch {
        if (!cancelled) {
          setNotice(t.runs.apiUnavailable);
        }
      }
    };
    loadMonitor();
    const timer = window.setInterval(loadMonitor, 2500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [
    configs,
    monitorRunId,
    t.runs.apiUnavailable,
    t.runs.runCancelledNotice,
    t.runs.runFailedNotice,
    t.runs.runFinishedNotice,
    t.runs.stopSavedNotice
  ]);

  useEffect(() => {
    if (!tuningRunning || !tuningJob?.id) {
      return;
    }
    const timer = window.setInterval(() => {
      fetchParallelTuning(tuningJob.id)
        .then(setTuningJob)
        .catch(() => undefined);
    }, 1800);
    return () => window.clearInterval(timer);
  }, [tuningJob?.id, tuningRunning]);

  const openStartDialog = () => {
    setStartDialogOpen(true);
    setNotice("");
    setTaskNameInput((current) => current || `${t.runs.defaultTaskName} ${new Date().toLocaleString()}`);
    refreshBase()
      .then(({ manageableRuns }) => {
        if (!manageableRuns.length) {
          setStartMode("new");
        }
      })
      .catch(() => undefined);
  };

  const submitStartDialog = async () => {
    setBusy(true);
    setNotice("");
    try {
      if (startMode === "new") {
        if (!selectedConfig) {
          setNotice(t.runs.configRequired);
          return;
        }
        const trimmedName = taskNameInput.trim();
        if (!trimmedName) {
          setNotice(t.runs.taskNameRequired);
          return;
        }
        const nextRun = await createRun(selectedConfig.id, trimmedName);
        setMonitorRunId(nextRun.id);
        setMonitorRun(nextRun);
        setLastSummary(null);
        setStartDialogOpen(false);
        setNotice(t.runs.createdTaskNotice);
      } else {
        if (!selectedResumeRun) {
          setNotice(t.runs.resumeTaskRequired);
          return;
        }
        const updated = isActiveRun(selectedResumeRun.status)
          ? selectedResumeRun
          : await resumeRun(selectedResumeRun.id);
        setMonitorRunId(updated.id);
        setMonitorRun(updated);
        setLastSummary(null);
        setStartDialogOpen(false);
        setNotice(t.runs.resumedTaskNotice);
      }
      refreshBase().catch(() => undefined);
    } catch {
      setNotice(startMode === "new" ? t.runs.createTaskFailed : t.runs.resumeTaskFailed);
    } finally {
      setBusy(false);
    }
  };

  const pauseCurrentRun = async () => {
    if (!monitorRun) {
      return;
    }
    setBusy(true);
    try {
      const updated = await pauseRun(monitorRun.id);
      setMonitorRun(updated);
      if (isTerminalRun(updated.status)) {
        const summary = makeSummary(updated, terminalRunNote(updated.status, t.runs), {
          config: monitoredConfig,
          events,
          logs
        });
        setLastSummary(summary);
        setNotice("");
        setMonitorRunId("");
        setMonitorRun(null);
        setLogs(null);
        setEvents(null);
      } else {
        setNotice(t.runs.pauseRequestedNotice);
      }
      refreshBase().catch(() => undefined);
    } catch {
      setNotice(t.runs.stopFailed);
    } finally {
      setBusy(false);
    }
  };

  const requestCancelCurrentRun = () => {
    if (!monitorRun) {
      return;
    }
    setCancelConfirmOpen(true);
  };

  const confirmCancelCurrentRun = async () => {
    if (!monitorRun) {
      return;
    }
    setBusy(true);
    try {
      const updated = await cancelRun(monitorRun.id);
      setCancelConfirmOpen(false);
      setMonitorRun(updated);
      if (isTerminalRun(updated.status)) {
        const summary = makeSummary(updated, terminalRunNote(updated.status, t.runs), {
          config: monitoredConfig,
          events,
          logs
        });
        setLastSummary(summary);
        setNotice("");
        setMonitorRunId("");
        setMonitorRun(null);
        setLogs(null);
        setEvents(null);
      } else {
        setNotice(t.runs.cancelRequestedNotice);
      }
      refreshBase().catch(() => undefined);
    } catch {
      setNotice(t.runs.cancelFailed);
    } finally {
      setBusy(false);
    }
  };

  const resumeSummaryRun = async () => {
    if (!lastSummary) {
      return;
    }
    setBusy(true);
    setNotice("");
    try {
      const updated = await resumeRun(lastSummary.runId);
      setMonitorRunId(updated.id);
      setMonitorRun(updated);
      setLastSummary(null);
      setStartDialogOpen(false);
      setNotice(t.runs.resumedTaskNotice);
      refreshBase().catch(() => undefined);
    } catch {
      setNotice(t.runs.resumeTaskFailed);
    } finally {
      setBusy(false);
    }
  };

  const openTuningDialog = () => {
    setTuningDialogOpen(true);
    setTuningNotice("");
  };

  const setTuningMode = (mode: TuningMode) => {
    setTuningForm(mode === "full" ? fullTuningDefaults : quickTuningDefaults);
  };

  const updateTuningForm = (updates: Partial<TuningForm>) => {
    setTuningForm((current) => ({ ...current, ...updates }));
  };

  const submitTuningDialog = async () => {
    setTuningBusy(true);
    setTuningNotice("");
    try {
      const payload = buildTuningPayload(tuningForm);
      const job = await startParallelTuning(payload);
      setTuningJob(job);
      setTuningDialogOpen(false);
      setTuningNotice(`调参任务已启动，sampleCount=${payload.sampleCount}，最大 batch=${payload.maxBatchSize}。`);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (error) {
      setTuningNotice(error instanceof Error ? error.message : "调参任务启动失败。");
    } finally {
      setTuningBusy(false);
    }
  };

  const persistTuning = async () => {
    if (!tuningJob?.id) {
      return;
    }
    setTuningBusy(true);
    setTuningNotice("");
    try {
      const saved = await saveParallelTuning(tuningJob.id);
      setTuningNotice(
        `已保存 ${saved.savedKeys.length} 个参数到 ${saved.envPath}，并已写入运行时配置，后续实验会直接使用。`
      );
      setTuningJob(await fetchParallelTuning(tuningJob.id));
    } catch (error) {
      setTuningNotice(error instanceof Error ? error.message : "保存参数失败。");
    } finally {
      setTuningBusy(false);
    }
  };

  const stopTuning = async () => {
    if (!tuningJob?.id) {
      return;
    }
    setTuningBusy(true);
    setTuningNotice("");
    try {
      const updated = await cancelParallelTuning(tuningJob.id);
      setTuningJob(updated);
      setTuningNotice("调参任务已停止，已完成的候选记录保留在该任务目录中。");
    } catch (error) {
      setTuningNotice(error instanceof Error ? error.message : "停止调参失败。");
    } finally {
      setTuningBusy(false);
    }
  };

  const closeSummaryDialog = () => {
    const summaryRunId = lastSummary?.runId;
    setLastSummary(null);
    setNotice("");
    setMonitorRunId("");
    setMonitorRun(null);
    setLogs(null);
    setEvents(null);
    setStartDialogOpen(false);
    if (summaryRunId) {
      setActiveRuns((current) => current.filter((run) => run.id !== summaryRunId || (run.status === "running" && !run.cancelRequested)));
    }
    refreshBase().catch(() => undefined);
  };

  const formatOptionalDate = (value?: string | null) => (value ? localizedDate(language, value) : "n/a");
  const statusLabels = t.common.status as Record<string, string>;
  const currentStopNotice = monitorRun ? stopIntentNotice(monitorRun, t.runs) : null;
  const startActionDisabled =
    busy ||
    (startMode === "new"
      ? !selectedConfig || !taskNameInput.trim()
      : !selectedResumeRun || (!isActiveRun(selectedResumeRun.status) && !isResumableRun(selectedResumeRun.status)));

  return (
    <AppShell active="runs">
      <div className="topbar">
        <div className="title-block">
          <h1>{t.runs.title}</h1>
          <p>{t.runs.subtitle}</p>
        </div>
        <div className="toolbar">
          <button className="button" onClick={() => refreshBase()} title={t.common.updated} type="button">
            <RefreshCw size={16} />
          </button>
        </div>
      </div>

      <section className="panel run-tuning-panel">
        <div className="panel-header">
          <div>
            <h2>并行参数自动调优</h2>
            <p>
              {tuningJob
                ? `${tuningJob.id} · ${tuningJob.message ?? tuningJob.status}`
                : "在开始实验之外单独搜索 batch size 与 CPU worker 参数。"}
            </p>
          </div>
          <span className={tuningStatusClass(tuningJob?.status)}>{tuningJob?.status ?? "idle"}</span>
        </div>
        <div className="panel-body run-tuning-body">
          <div className="run-tuning-toolbar">
            <button className="button primary" disabled={tuningBusy || tuningRunning} onClick={openTuningDialog} type="button">
              <SlidersHorizontal size={16} />
              开始调参
            </button>
            <button
              className="button"
              disabled={tuningBusy || tuningJob?.status !== "succeeded" || tuningEnvEntries.length === 0}
              onClick={persistTuning}
              type="button"
            >
              <Save size={16} />
              保存参数
            </button>
            <button className="button danger" disabled={tuningBusy || !tuningRunning} onClick={stopTuning} type="button">
              <Square size={16} />
              停止调参
            </button>
            <button
              className="button"
              disabled={!tuningJob?.id}
              onClick={() => tuningJob?.id && fetchParallelTuning(tuningJob.id).then(setTuningJob)}
              type="button"
            >
              <RefreshCw size={16} />
              刷新调参
            </button>
          </div>

          <div className="run-tuning-progress-card">
            <div className="run-overview-title">
              <span>搜索进度</span>
              <strong>{tuningJob?.progress ?? 0}%</strong>
            </div>
            <div className="progress-track run-progress-large">
              <div className="progress-bar" style={{ width: progressWidth(tuningJob?.progress ?? 0) }} />
            </div>
            <p>{tuningJob?.message ?? "等待启动调参任务。"}</p>
          </div>

          {tuningNotice ? <div className="risk ok">{tuningNotice}</div> : null}
          {tuningJob?.error ? <div className="risk error">{tuningJob.error}</div> : null}

          <div className="run-tuning-grid">
            <section className="run-tuning-card chart">
              <div className="run-tuning-card-head">
                <div>
                  <strong>吞吐量趋势</strong>
                  <span>每个候选点使用 images/sec，完整模式取重复测量中位数。</span>
                </div>
                <BarChart3 size={17} />
              </div>
              <ThroughputChart points={tuningChartPoints} />
            </section>

            <section className="run-tuning-card">
              <div className="run-tuning-card-head">
                <div>
                  <strong>运行过程</strong>
                  <span>最近的候选测量事件</span>
                </div>
                <Zap size={17} />
              </div>
              {tuningEvents.length ? (
                <ul className="tuning-event-list run-tuning-events">
                  {tuningEvents.map((event, index) => (
                    <li key={`${event.timestamp ?? "event"}-${index}`}>
                      <span>{event.stage ?? "step"}</span>
                      <strong>{event.message ?? "running"}</strong>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="empty compact-empty">暂无调参事件。</div>
              )}
            </section>
          </div>

          <section className="run-tuning-card env">
            <div className="run-tuning-card-head">
              <div>
                <strong>推荐参数</strong>
                <span>{tuningJob?.summary?.reportPath ? `报告：${tuningJob.summary.reportPath}` : "调参完成后生成可保存的 .env 参数。"}</span>
              </div>
            </div>
            {tuningEnvEntries.length ? (
              <div className="env-suggestion-list run-env-suggestion-list">
                {tuningEnvEntries.map(([key, value]) => (
                  <div key={key} className="env-suggestion-row">
                    <code>{key}</code>
                    <span>{value}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty compact-empty">完成搜索后显示 summary。</div>
            )}
          </section>
        </div>
      </section>

      {monitorRun ? (
        <section className="panel run-execution-panel">
          <div className="panel-header">
            <div>
              <h2>{t.runs.monitorTitle}</h2>
              <p>{t.runs.monitorSubtitle}</p>
            </div>
            <span className={badgeClass(monitorRun.status)}>{runStatusLabel(monitorRun.status, statusLabels)}</span>
          </div>
          <div className="panel-body run-execution-body">
            <div className="run-monitor-toolbar">
              <div>
                <span>{t.runs.selectedTask}</span>
                <strong>{taskName(monitorRun)}</strong>
                <code>{monitorRun.id}</code>
                {currentStopNotice ? <small className="run-pause-state">{currentStopNotice}</small> : null}
              </div>
              <div className="run-monitor-actions">
                {isPausableRun(monitorRun) ? (
                  <button className="button" disabled={busy} onClick={pauseCurrentRun} type="button">
                    <PauseCircle size={15} />
                    {t.runs.pauseAndSave}
                  </button>
                ) : null}
                {isCancellableRun(monitorRun) ? (
                  <button className="button danger" disabled={busy || monitorRun.stopIntent === "cancel"} onClick={requestCancelCurrentRun} type="button">
                    <XCircle size={15} />
                    {t.runs.cancelExperiment}
                  </button>
                ) : null}
              </div>
            </div>

            <div className="run-monitor-overview">
              <section className="run-overview-card primary">
                <div className="run-overview-title">
                  <span>{t.runs.experimentProgress}</span>
                  <strong>{monitorRun.progress}%</strong>
                </div>
                <div className="progress-track run-progress-large">
                  <div className="progress-bar" style={{ width: progressWidth(monitorRun.progress) }} />
                </div>
                <div className="run-count-grid">
                  <Metric label={t.runs.completedCells} value={`${monitorStats.completedCells}/${monitorRun.cells}`} />
                  <Metric label={t.runs.successfulCells} value={monitorStats.succeededCells.toString()} />
                  <Metric label={t.runs.failedCells} value={monitorStats.failedCells.toString()} />
                </div>
                <p>{t.runs.cellExplanation}</p>
              </section>

              <section className="run-overview-card">
                <div className="run-overview-title">
                  <span>{t.runs.currentExecution}</span>
                  <strong>{eventTitle(currentEvent)}</strong>
                </div>
                <div className="run-current-grid">
                  <CurrentField
                    label={t.runs.currentDataset}
                    raw={monitorStats.current.datasetId}
                    value={displayName(monitorStats.current.datasetId, resourceNames)}
                  />
                  <CurrentField
                    label={t.runs.currentWatermark}
                    raw={monitorStats.current.algorithmId}
                    value={displayName(monitorStats.current.algorithmId, resourceNames)}
                  />
                  <CurrentField
                    label={t.runs.currentAttack}
                    raw={monitorStats.current.attackId}
                    value={displayName(monitorStats.current.attackId, resourceNames)}
                  />
                  <CurrentField label={t.runs.currentAttackParam} raw={monitorStats.current.cellKey} value={monitorStats.current.attackParam} />
                </div>
                <div className="run-stage-line">
                  <Clock3 size={15} />
                  <span>{currentEvent?.timestamp ? localizedDate(language, currentEvent.timestamp) : t.runs.waitingForStage}</span>
                  <code>{eventMeta(currentEvent)}</code>
                </div>
              </section>
            </div>

            <div className="run-attack-status-grid">
              <AttackOutcomeList
                emptyText={t.runs.noSuccessfulAttacks}
                kind="success"
                names={resourceNames}
                outcomes={successfulAttackOutcomes}
                title={t.runs.successfulAttacks}
              />
              <AttackOutcomeList
                emptyText={t.runs.noFailedAttacks}
                kind="failed"
                names={resourceNames}
                outcomes={failedAttackOutcomes}
                title={t.runs.failedAttacks}
              />
            </div>

            <div className="run-progress-stack">
              <div className="run-progress-section-head">
                <strong>{t.runs.stageProgress}</strong>
                <span>{t.runs.attackStatusHint}</span>
              </div>
              {progressSteps.map((step) => (
                <ProgressMeter key={step.key} step={step} />
              ))}
            </div>

            <div className="run-meta-grid run-monitor-meta-grid">
              <Metric label={t.common.config} value={monitorRun.configName} />
              <Metric label={t.runs.matrixCells} value={monitorRun.cells.toString()} />
              <Metric label={t.runs.worker} value={monitorRun.workerId ?? "n/a"} />
              <Metric label={t.runs.updated} value={formatOptionalDate(monitorRun.updatedAt)} />
              <Metric label={t.runs.started} value={formatOptionalDate(monitorRun.startedAt)} />
              <Metric label={t.runs.finished} value={formatOptionalDate(monitorRun.finishedAt)} />
            </div>

            {notice ? <div className="risk ok">{notice}</div> : null}
            {monitorRun.error ? <div className="risk error">{monitorRun.error}</div> : null}

            <div className="run-log-shell">
              <div className="run-log-head">
                <div>
                  <TerminalSquare size={16} />
                  <strong>{t.runs.liveLog}</strong>
                </div>
                <span>{logs?.logPath ?? monitorRun.logPath ?? "worker.log"}</span>
              </div>
              <pre className="log-preview run-live-log">
                {logs?.exists ? logs.lines.join("\n") || "empty log" : t.runs.noLogYet}
              </pre>
            </div>

            <div className="run-events-list run-execution-events">
              <div className="run-artifacts-head">
                <CheckCircle2 size={15} />
                <span>{t.runs.latestEvents}</span>
              </div>
              {events?.events.length ? (
                events.events.slice(-10).reverse().map((event, index) => (
                  <div className="run-event-row" key={`${event.timestamp ?? index}-${index}`}>
                    <strong>{eventTitle(event)}</strong>
                    <span>{event.timestamp ? localizedDate(language, event.timestamp) : "n/a"}</span>
                    <code>{eventMeta(event)}</code>
                  </div>
                ))
              ) : (
                <div className="empty compact-empty">{t.runs.noEvents}</div>
              )}
            </div>
          </div>
        </section>
      ) : (
        <section className="panel run-start-panel">
          <div className="panel-body run-start-body">
            <div className="run-start-copy">
              <h2>{t.runs.startExperiment}</h2>
              <p>{t.runs.startExperimentHint}</p>
            </div>
            <div className="run-start-actions">
              <button className="button primary run-start-button" onClick={openStartDialog} type="button">
                <PlayCircle size={18} />
                {t.runs.startExperiment}
              </button>
            </div>

            {notice ? <div className="risk ok">{notice}</div> : null}

          </div>
        </section>
      )}

      {tuningDialogOpen ? (
        <div className="modal-backdrop" role="presentation">
          <div aria-modal="true" className="config-modal tuning-config-modal" role="dialog">
            <div className="modal-header">
              <div>
                <h2>并行参数搜索</h2>
                <p>配置搜索范围和搜索模式。sampleCount 会自动不小于最大 batch。</p>
              </div>
              <button className="icon-button" onClick={() => setTuningDialogOpen(false)} title="关闭" type="button">
                ×
              </button>
            </div>
            <div className="modal-body tuning-config-body">
              <div className="run-mode-grid tuning-mode-grid">
                <button
                  className={tuningForm.mode === "quick" ? "run-mode-card selected" : "run-mode-card"}
                  onClick={() => setTuningMode("quick")}
                  type="button"
                >
                  <SlidersHorizontal size={18} />
                  <strong>快速模式</strong>
                  <span>固定候选集合，单次测量，适合快速得到一版可用参数。</span>
                </button>
                <button
                  className={tuningForm.mode === "full" ? "run-mode-card selected" : "run-mode-card"}
                  onClick={() => setTuningMode("full")}
                  type="button"
                >
                  <BarChart3 size={18} />
                  <strong>完整模式</strong>
                  <span>自动扩展候选、检测吞吐边界、重复测量并用中位数选最优。</span>
                </button>
              </div>

              <div className="tuning-config-grid">
                <section className="run-dialog-section">
                  <h3>样本与测量</h3>
                  <div className="field-grid">
                    <div className="field">
                      <label htmlFor="tuning-samples">sampleCount</label>
                      <input
                        id="tuning-samples"
                        min={2}
                        onChange={(event) => updateTuningForm({ sampleCount: Number(event.target.value) })}
                        type="number"
                        value={tuningForm.sampleCount}
                      />
                    </div>
                    <div className="field">
                      <label htmlFor="tuning-warmup">warmupCount</label>
                      <input
                        id="tuning-warmup"
                        min={1}
                        onChange={(event) => updateTuningForm({ warmupCount: Number(event.target.value) })}
                        type="number"
                        value={tuningForm.warmupCount}
                      />
                    </div>
                    <div className="field">
                      <label htmlFor="tuning-repeat">repeatCount</label>
                      <input
                        id="tuning-repeat"
                        min={1}
                        onChange={(event) => updateTuningForm({ repeatCount: Number(event.target.value) })}
                        type="number"
                        value={tuningForm.repeatCount}
                      />
                    </div>
                    <div className="field">
                      <label htmlFor="tuning-improve">最小提升比例</label>
                      <input
                        id="tuning-improve"
                        min={0}
                        onChange={(event) => updateTuningForm({ minImprovementRatio: Number(event.target.value) })}
                        step={0.01}
                        type="number"
                        value={tuningForm.minImprovementRatio}
                      />
                    </div>
                  </div>
                </section>

                <section className="run-dialog-section">
                  <h3>搜索范围</h3>
                  <div className="field-grid">
                    <div className="field">
                      <label htmlFor="tuning-batch-min">最小 batch</label>
                      <input
                        id="tuning-batch-min"
                        min={1}
                        onChange={(event) => updateTuningForm({ minBatchSize: Number(event.target.value) })}
                        type="number"
                        value={tuningForm.minBatchSize}
                      />
                    </div>
                    <div className="field">
                      <label htmlFor="tuning-batch-max">最大 batch</label>
                      <input
                        id="tuning-batch-max"
                        min={1}
                        onChange={(event) => updateTuningForm({ maxBatchSize: Number(event.target.value) })}
                        type="number"
                        value={tuningForm.maxBatchSize}
                      />
                    </div>
                    <div className="field">
                      <label htmlFor="tuning-worker-min">最小 workers</label>
                      <input
                        id="tuning-worker-min"
                        min={1}
                        onChange={(event) => updateTuningForm({ minWorkerCount: Number(event.target.value) })}
                        type="number"
                        value={tuningForm.minWorkerCount}
                      />
                    </div>
                    <div className="field">
                      <label htmlFor="tuning-worker-max">最大 workers</label>
                      <input
                        id="tuning-worker-max"
                        min={1}
                        onChange={(event) => updateTuningForm({ maxWorkerCount: Number(event.target.value) })}
                        type="number"
                        value={tuningForm.maxWorkerCount}
                      />
                    </div>
                  </div>
                </section>

                <section className="run-dialog-section tuning-stage-section">
                  <h3>调参阶段</h3>
                  <label>
                    <input
                      checked={tuningForm.tuneWatermarks}
                      onChange={(event) => updateTuningForm({ tuneWatermarks: event.target.checked })}
                      type="checkbox"
                    />
                    水印嵌入/解码
                  </label>
                  <label>
                    <input
                      checked={tuningForm.tuneAttacks}
                      onChange={(event) => updateTuningForm({ tuneAttacks: event.target.checked })}
                      type="checkbox"
                    />
                    攻击方法
                  </label>
                  <label>
                    <input
                      checked={tuningForm.tuneAttacks && tuningForm.includeViewpoint3dAttacks}
                      disabled={!tuningForm.tuneAttacks}
                      onChange={(event) => updateTuningForm({ includeViewpoint3dAttacks: event.target.checked })}
                      type="checkbox"
                    />
                    <span className="tuning-stage-option-copy">
                      <strong>3D 视角重渲染</strong>
                      <small>默认排除；勾选后只调 rotate + 看向中心代表项，并应用到全部 3D 变体。</small>
                    </span>
                  </label>
                  <label>
                    <input
                      checked={tuningForm.tuneQuality}
                      onChange={(event) => updateTuningForm({ tuneQuality: event.target.checked })}
                      type="checkbox"
                    />
                    quality 指标
                  </label>
                </section>

                <section className="run-dialog-section tuning-effective-section">
                  <h3>实际提交参数</h3>
                  <div className="run-meta-grid">
                    <Metric label="sampleCount" value={String(effectiveTuningPayload.sampleCount)} />
                    <Metric label="最大 batch" value={String(effectiveTuningPayload.maxBatchSize)} />
                    <Metric label="最大 workers" value={String(effectiveTuningPayload.maxWorkerCount)} />
                    <Metric label="重复次数" value={String(effectiveTuningPayload.repeatCount)} />
                    <Metric label="3D 攻击" value={effectiveTuningPayload.includeViewpoint3dAttacks ? "包含" : "排除"} />
                  </div>
                  <code>batchCandidates={effectiveTuningPayload.batchCandidates.join(",")}</code>
                  <code>workerCandidates={effectiveTuningPayload.workerCandidates.join(",")}</code>
                </section>
              </div>
            </div>
            <div className="modal-footer">
              <button className="button" onClick={() => setTuningDialogOpen(false)} type="button">
                取消
              </button>
              <button className="button primary" disabled={tuningBusy} onClick={submitTuningDialog} type="button">
                <Zap size={16} />
                开始搜索
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {startDialogOpen ? (
        <div className="modal-backdrop" role="presentation">
          <div aria-modal="true" className="config-modal run-start-modal" role="dialog">
            <div className="modal-header">
              <div>
                <h2>{t.runs.startDialogTitle}</h2>
                <p>{t.runs.startDialogHint}</p>
              </div>
              <button className="icon-button" onClick={() => setStartDialogOpen(false)} title={t.runs.cancel} type="button">
                ×
              </button>
            </div>
            <div className="modal-body run-start-modal-body">
              <div className="run-mode-grid">
                <button
                  className={startMode === "new" ? "run-mode-card selected" : "run-mode-card"}
                  onClick={() => setStartMode("new")}
                  type="button"
                >
                  <PlayCircle size={18} />
                  <strong>{t.runs.newTask}</strong>
                  <span>{t.runs.newTaskHint}</span>
                </button>
                <button
                  className={startMode === "resume" ? "run-mode-card selected" : "run-mode-card"}
                  disabled={!activeRuns.length}
                  onClick={() => setStartMode("resume")}
                  type="button"
                >
                  <RotateCcw size={18} />
                  <strong>{t.runs.continueTask}</strong>
                  <span>{t.runs.continueTaskHint}</span>
                </button>
              </div>

              {startMode === "new" ? (
                <div className="run-dialog-section">
                  <div className="field">
                    <label htmlFor="run-task-name">{t.runs.taskName}</label>
                    <input
                      id="run-task-name"
                      onChange={(event) => setTaskNameInput(event.target.value)}
                      placeholder={t.runs.taskNamePlaceholder}
                      value={taskNameInput}
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="run-config">{t.runs.selectConfig}</label>
                    <select
                      id="run-config"
                      onChange={(event) => setSelectedConfigId(event.target.value)}
                      value={selectedConfigId}
                    >
                      {configs.map((config) => (
                        <option key={config.id} value={config.id}>
                          {config.name}
                        </option>
                      ))}
                    </select>
                  </div>
                  {selectedConfig ? <ConfigPreview config={selectedConfig} t={t} /> : <div className="empty">{t.runs.noConfigs}</div>}
                </div>
              ) : (
                <div className="run-dialog-section">
                  <div className="run-resume-list">
                    {activeRuns.length ? (
                      activeRuns.map((run) => (
                        <button
                          className={run.id === selectedResumeRunId ? "run-resume-card selected" : "run-resume-card"}
                          key={run.id}
                          onClick={() => setSelectedResumeRunId(run.id)}
                          type="button"
                        >
                          <div>
                            <strong>{taskName(run)}</strong>
                            <span>{run.configName}</span>
                            <code>{run.id}</code>
                          </div>
                          <div>
                            <span className={badgeClass(run.status)}>{runStatusLabel(run.status, statusLabels)}</span>
                            <small>{run.progress}%</small>
                          </div>
                        </button>
                      ))
                    ) : (
                      <div className="empty">{t.runs.noUnfinishedTasks}</div>
                    )}
                  </div>
                </div>
              )}
            </div>
            <div className="modal-footer">
              <button className="button" onClick={() => setStartDialogOpen(false)} type="button">
                {t.runs.cancel}
              </button>
              <button className="button primary" disabled={startActionDisabled} onClick={submitStartDialog} type="button">
                {startMode === "resume" ? <RotateCcw size={16} /> : <PlayCircle size={16} />}
                {startMode === "resume" ? t.runs.resume : t.runs.beginExecution}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {cancelConfirmOpen && monitorRun ? (
        <div className="modal-backdrop" role="presentation">
          <div aria-modal="true" className="config-modal run-confirm-modal" role="dialog">
            <div className="modal-header">
              <div>
                <h2>{t.runs.cancelConfirmTitle}</h2>
                <p>{taskName(monitorRun)}</p>
              </div>
              <button className="icon-button" onClick={() => setCancelConfirmOpen(false)} title={t.runs.cancel} type="button">
                ×
              </button>
            </div>
            <div className="modal-body run-confirm-body">
              <div className="risk warn">{t.runs.cancelConfirmBody}</div>
              <code>{monitorRun.id}</code>
            </div>
            <div className="modal-footer">
              <button className="button" disabled={busy} onClick={() => setCancelConfirmOpen(false)} type="button">
                {t.runs.keepRunning}
              </button>
              <button className="button danger" disabled={busy} onClick={confirmCancelCurrentRun} type="button">
                <XCircle size={16} />
                {t.runs.confirmCancelExperiment}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {lastSummary ? (
        <div className="modal-backdrop" role="presentation">
          <div aria-modal="true" className="config-modal run-summary-modal" role="dialog">
            <div className="modal-header">
              <div>
                <h2>{t.runs.executionSummary}</h2>
                <p>{lastSummary.note}</p>
              </div>
              <button className="icon-button" onClick={closeSummaryDialog} title={t.runs.closeSummary} type="button">
                ×
              </button>
            </div>
            <div className="modal-body run-summary-modal-body">
              <div className="run-summary-head">
                <div>
                  <strong>{lastSummary.taskName}</strong>
                  <code>{lastSummary.runId}</code>
                </div>
                <span className={badgeClass(lastSummary.status)}>{runStatusLabel(lastSummary.status, statusLabels)}</span>
              </div>
              <div className="run-meta-grid">
                <Metric label={t.common.progress} value={`${lastSummary.progress}%`} />
                <Metric label={t.runs.duration} value={formatDurationMs(lastSummary.durationMs, language)} />
                <Metric label={t.runs.completedCells} value={`${lastSummary.completedCells}/${lastSummary.cells}`} />
                <Metric label={t.runs.successfulCells} value={lastSummary.succeededCells.toString()} />
                <Metric label={t.runs.failedCells} value={lastSummary.failedCells.toString()} />
                <Metric label={t.runs.remainingCells} value={lastSummary.remainingCells.toString()} />
                <Metric label={t.common.config} value={lastSummary.configName} />
                <Metric label={t.runs.worker} value={lastSummary.workerId ?? "n/a"} />
                <Metric label={t.runs.created} value={formatOptionalDate(lastSummary.createdAt)} />
                <Metric label={t.runs.started} value={formatOptionalDate(lastSummary.startedAt)} />
                <Metric label={t.runs.finished} value={formatOptionalDate(lastSummary.finishedAt)} />
                <Metric label={t.runs.updated} value={formatOptionalDate(lastSummary.updatedAt)} />
                <Metric label={t.runs.matrixCells} value={lastSummary.cells.toString()} />
              </div>
              {lastSummary.selection ? (
                <div className="run-summary-section">
                  <div className="run-artifacts-head">
                    <CheckCircle2 size={15} />
                    <span>{t.runs.selectionScope}</span>
                  </div>
                  <div className="run-meta-grid run-summary-compact-grid">
                    <Metric label={t.runs.datasets} value={lastSummary.selection.datasets.toString()} />
                    <Metric label={t.runs.watermarks} value={lastSummary.selection.watermarks.toString()} />
                    <Metric label={t.runs.attacks} value={lastSummary.selection.attacks.toString()} />
                    <Metric label={t.runs.seeds} value={lastSummary.selection.seeds.toString()} />
                    <Metric label={t.common.samples} value={lastSummary.selection.sampleCount.toString()} />
                    <Metric label={t.console.ops} value={lastSummary.selection.imageOperationCount.toString()} />
                  </div>
                </div>
              ) : null}
              <div className="run-summary-section">
                <div className="run-artifacts-head">
                  <TerminalSquare size={15} />
                  <span>{t.runs.latestEvent}</span>
                </div>
                {lastSummary.latestEvent ? (
                  <div className="run-event-row summary-event-row">
                    <strong>{lastSummary.latestEvent.title}</strong>
                    <span>{lastSummary.latestEvent.timestamp ? localizedDate(language, lastSummary.latestEvent.timestamp) : "n/a"}</span>
                    <code>{lastSummary.latestEvent.meta}</code>
                  </div>
                ) : (
                  <div className="empty compact-empty">{t.runs.noEvents}</div>
                )}
              </div>
              <div className="run-summary-section">
                <div className="run-artifacts-head">
                  <TerminalSquare size={15} />
                  <span>{t.runs.logSummary}</span>
                </div>
                <div className="run-meta-grid run-summary-compact-grid">
                  <Metric label={t.runs.logPath} value={lastSummary.log?.path ?? lastSummary.artifactRoot ?? "n/a"} />
                  <Metric label={t.runs.logLines} value={(lastSummary.log?.lineCount ?? 0).toString()} />
                </div>
                <div className="run-summary-log-tail">
                  <span>{t.runs.logTail}</span>
                  <pre>{lastSummary.log?.tailLines.length ? lastSummary.log.tailLines.join("\n") : t.runs.noLastLogLine}</pre>
                </div>
              </div>
              <div className="run-artifacts">
                <div className="run-artifacts-head">
                  <FolderOpen size={15} />
                  <span>{t.runs.rawArtifacts}</span>
                </div>
                <code>{lastSummary.artifactRoot ?? "n/a"}</code>
                <div className="artifact-chip-grid">
                  {rawArtifactFiles.map((file) => (
                    <span key={file}>{file}</span>
                  ))}
                </div>
              </div>
            </div>
            <div className="modal-footer">
              {isRestartableTerminalRun(lastSummary.status) ? (
                <button className="button" disabled={busy} onClick={resumeSummaryRun} type="button">
                  <RotateCcw size={16} />
                  {t.runs.resumeFromCheckpoint}
                </button>
              ) : null}
              <button className="button primary" onClick={closeSummaryDialog} type="button">
                {t.runs.closeSummary}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </AppShell>
  );
}

function ProgressMeter({ step }: { step: ProgressStep }) {
  return (
    <div className="run-progress-meter">
      <div className="run-progress-head">
        <span>{step.label}</span>
        <strong>{step.meta}</strong>
      </div>
      <div className="progress-track">
        <div className="progress-bar" style={{ width: progressWidth(step.percent) }} />
      </div>
    </div>
  );
}

function ThroughputChart({ points }: { points: TuningPoint[] }) {
  const width = 720;
  const height = 240;
  const padding = { top: 16, right: 18, bottom: 42, left: 46 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const maxThroughput = Math.max(1, ...points.map((point) => point.throughput));
  const visiblePoints = points.slice(-18);
  const barGap = 7;
  const barWidth = visiblePoints.length
    ? Math.max(8, (plotWidth - barGap * Math.max(0, visiblePoints.length - 1)) / visiblePoints.length)
    : 18;

  if (!visiblePoints.length) {
    return <div className="empty compact-empty">搜索开始后显示吞吐量曲线。</div>;
  }

  return (
    <div className="throughput-chart">
      <svg aria-label="throughput chart" preserveAspectRatio="none" viewBox={`0 0 ${width} ${height}`}>
        <line className="chart-axis" x1={padding.left} x2={width - padding.right} y1={height - padding.bottom} y2={height - padding.bottom} />
        <line className="chart-axis" x1={padding.left} x2={padding.left} y1={padding.top} y2={height - padding.bottom} />
        {[0.25, 0.5, 0.75, 1].map((ratio) => {
          const y = padding.top + plotHeight * (1 - ratio);
          return (
            <g key={ratio}>
              <line className="chart-grid-line" x1={padding.left} x2={width - padding.right} y1={y} y2={y} />
              <text className="chart-label" x={8} y={y + 4}>
                {(maxThroughput * ratio).toFixed(1)}
              </text>
            </g>
          );
        })}
        {visiblePoints.map((point, index) => {
          const x = padding.left + index * (barWidth + barGap);
          const barHeight = Math.max(3, (point.throughput / maxThroughput) * plotHeight);
          const y = height - padding.bottom - barHeight;
          return (
            <g key={point.key}>
              <rect
                className={point.kind === "batch" ? "chart-bar batch" : "chart-bar workers"}
                height={barHeight}
                rx={3}
                width={barWidth}
                x={x}
                y={y}
              />
              <text className="chart-x-label" textAnchor="middle" x={x + barWidth / 2} y={height - 20}>
                {point.kind === "batch" ? `b${point.candidate}` : `w${point.candidate}`}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="throughput-chart-meta">
        <span>
          <i className="batch" /> batch size
        </span>
        <span>
          <i className="workers" /> workers
        </span>
        <strong>最高 {maxThroughput.toFixed(2)} img/s</strong>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function CurrentField({ label, raw, value }: { label: string; raw: string; value: string }) {
  return (
    <div className="run-current-field">
      <span>{label}</span>
      <strong>{value}</strong>
      {raw && raw !== "n/a" && raw !== value ? <code>{raw}</code> : null}
    </div>
  );
}

function AttackOutcomeList({
  emptyText,
  kind,
  names,
  outcomes,
  title
}: {
  emptyText: string;
  kind: "success" | "failed";
  names: Record<string, string>;
  outcomes: AttackOutcome[];
  title: string;
}) {
  return (
    <section className={`run-attack-status-card ${kind}`}>
      <div className="run-attack-status-head">
        <strong>{title}</strong>
        <span>{outcomes.length}</span>
      </div>
      {outcomes.length ? (
        <div className="run-attack-list">
          {outcomes.map((outcome) => (
            <div className="run-attack-row" key={`${kind}-${outcome.attackId}`}>
              <div>
                <strong>{displayName(outcome.attackId, names)}</strong>
                <code>{outcome.attackId}</code>
              </div>
              <div>
                <span>{kind === "success" ? outcome.succeeded : outcome.failed}</span>
                <small>{outcome.latestParam}</small>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="run-attack-empty">{emptyText}</div>
      )}
    </section>
  );
}

function ConfigPreview({ config, t }: { config: SavedExperimentConfig; t: ReturnType<typeof useLanguage>["t"] }) {
  return (
    <div className="run-config-summary">
      <strong>{t.runs.configSummary}</strong>
      <span>{config.name}</span>
      <small>{config.id}</small>
      <div className="stats">
        <div className="stat">
          <span>{t.console.cells}</span>
          <strong>{config.cellCount}</strong>
        </div>
        <div className="stat">
          <span>{t.common.samples}</span>
          <strong>{config.sampleCount}</strong>
        </div>
        <div className="stat">
          <span>{t.console.ops}</span>
          <strong>{config.imageOperationCount}</strong>
        </div>
      </div>
    </div>
  );
}
