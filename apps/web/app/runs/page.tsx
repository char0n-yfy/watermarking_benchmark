"use client";

import { useEffect, useMemo, useState } from "react";
import {
  CheckCircle2,
  Clock3,
  FolderOpen,
  PlayCircle,
  RefreshCw,
  RotateCcw,
  Square,
  TerminalSquare
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { useLanguage } from "@/components/LanguageProvider";
import {
  cancelRun,
  createRun,
  fetchAlgorithms,
  fetchAttacks,
  fetchDatasetCatalog,
  fetchRun,
  fetchRunEvents,
  fetchRunLogs,
  fetchRuns,
  fetchSavedConfigs,
  resumeRun
} from "@/lib/api";
import { localizedDate } from "@/lib/i18n";
import type { DemoRunRecord, RunEvents, RunLogs, RunStageEvent, SavedExperimentConfig } from "@/lib/types";

type StartMode = "new" | "resume";

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
  artifactRoot?: string;
  updatedAt?: string;
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

const activeStatuses = new Set(["queued", "running"]);
const terminalStatuses = new Set(["succeeded", "failed", "cancelled", "partially_failed"]);
const stoppableStatuses = new Set(["queued", "running"]);
const finalCellStatuses = new Set(["succeeded", "failed", "skipped", "cancelled"]);
const finalWatermarkStatuses = new Set(["succeeded", "failed", "skipped", "cancelled"]);
const rawArtifactFiles = [
  "run_plan.json",
  "cell_manifest.jsonl",
  "image_quality.jsonl",
  "image_detection.jsonl",
  "runtime_profile.jsonl",
  "stage_events.jsonl",
  "run_status.json"
];

function badgeClass(status: DemoRunRecord["status"]) {
  if (status === "running" || status === "succeeded") {
    return "badge ok";
  }
  if (status === "failed" || status === "partially_failed" || status === "cancelled") {
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

function makeSummary(run: DemoRunRecord, note: string, statusOverride?: DemoRunRecord["status"]): ExecutionSummary {
  return {
    taskName: taskName(run),
    runId: run.id,
    status: statusOverride ?? run.status,
    progress: run.progress,
    cells: run.cells,
    artifactRoot: run.artifactRoot,
    updatedAt: run.updatedAt,
    note
  };
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
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [resourceNames, setResourceNames] = useState<Record<string, string>>({});

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

  const refreshBase = async () => {
    const [loadedConfigs, loadedRuns] = await Promise.all([fetchSavedConfigs(), fetchRuns({ scope: "active" })]);
    setConfigs(loadedConfigs);
    setActiveRuns(loadedRuns);
    setSelectedConfigId((current) => {
      if (current && loadedConfigs.some((config) => config.id === current)) {
        return current;
      }
      return loadedConfigs[0]?.id ?? "";
    });
    setSelectedResumeRunId((current) => {
      if (current && loadedRuns.some((run) => run.id === current)) {
        return current;
      }
      return loadedRuns[0]?.id ?? "";
    });
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
        if (terminalStatuses.has(runValue.status)) {
          const note =
            runValue.status === "succeeded"
              ? t.runs.runFinishedNotice
              : runValue.status === "cancelled"
                ? t.runs.stopSavedNotice
                : t.runs.runFailedNotice;
          setLastSummary(makeSummary(runValue, note));
          setNotice("");
          setMonitorRunId("");
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
  }, [monitorRunId, t.runs.apiUnavailable, t.runs.runFailedNotice, t.runs.runFinishedNotice, t.runs.stopSavedNotice]);

  const openStartDialog = () => {
    setStartDialogOpen(true);
    setNotice("");
    setTaskNameInput((current) => current || `${t.runs.defaultTaskName} ${new Date().toLocaleString()}`);
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
        const updated = activeStatuses.has(selectedResumeRun.status)
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

  const stopCurrentRun = async () => {
    if (!monitorRun) {
      return;
    }
    setBusy(true);
    try {
      const updated = await cancelRun(monitorRun.id);
      const summary = makeSummary(updated, t.runs.stopSavedNotice, "cancelled");
      setLastSummary(summary);
      setNotice("");
      setMonitorRunId("");
      setMonitorRun(null);
      setLogs(null);
      setEvents(null);
      refreshBase().catch(() => undefined);
    } catch {
      setNotice(t.runs.stopFailed);
    } finally {
      setBusy(false);
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

      {monitorRun ? (
        <section className="panel run-execution-panel">
          <div className="panel-header">
            <div>
              <h2>{t.runs.monitorTitle}</h2>
              <p>{t.runs.monitorSubtitle}</p>
            </div>
            <span className={badgeClass(monitorRun.status)}>{statusLabels[monitorRun.status] ?? monitorRun.status}</span>
          </div>
          <div className="panel-body run-execution-body">
            <div className="run-monitor-toolbar">
              <div>
                <span>{t.runs.selectedTask}</span>
                <strong>{taskName(monitorRun)}</strong>
                <code>{monitorRun.id}</code>
              </div>
              {stoppableStatuses.has(monitorRun.status) ? (
                <button className="button danger" disabled={busy} onClick={stopCurrentRun} type="button">
                  <Square size={15} />
                  {t.runs.stopAndSave}
                </button>
              ) : null}
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
                            <span className={badgeClass(run.status)}>{statusLabels[run.status] ?? run.status}</span>
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
              <button className="button primary" disabled={busy} onClick={submitStartDialog} type="button">
                <PlayCircle size={16} />
                {t.runs.beginExecution}
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
                <span className={badgeClass(lastSummary.status)}>{statusLabels[lastSummary.status] ?? lastSummary.status}</span>
              </div>
              <div className="run-meta-grid">
                <Metric label={t.common.progress} value={`${lastSummary.progress}%`} />
                <Metric label={t.runs.matrixCells} value={lastSummary.cells.toString()} />
                <Metric label={t.runs.updated} value={formatOptionalDate(lastSummary.updatedAt)} />
                <Metric label={t.runs.artifactRoot} value={lastSummary.artifactRoot ?? "n/a"} />
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
