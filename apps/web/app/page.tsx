"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock3,
  Filter,
  PlayCircle,
  RefreshCw,
  Trophy
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { useLanguage } from "@/components/LanguageProvider";
import { RobustnessCurve } from "@/components/RobustnessCurve";
import {
  fetchAlgorithms,
  fetchAttacks,
  fetchDatasets,
  fetchRunResults,
  fetchRuns,
  fetchRuntime,
  fetchSavedConfigs
} from "@/lib/api";
import {
  buildActiveRunRows,
  formatMetric,
  statusBadgeClass,
  summarizeRuns
} from "@/lib/insights";
import type {
  AlgorithmVersion,
  AttackPreset,
  DatasetVersion,
  DemoRunRecord,
  RunResults,
  RuntimeInfo,
  SavedExperimentConfig
} from "@/lib/types";

function latestResultCandidate(runs: DemoRunRecord[]) {
  return (
    runs.find((run) => run.status === "succeeded" || run.status === "partially_failed") ??
    runs.find((run) => run.status !== "queued" && run.status !== "running") ??
    null
  );
}

export default function ExperimentConsole() {
  const { t } = useLanguage();
  const [configs, setConfigs] = useState<SavedExperimentConfig[]>([]);
  const [runs, setRuns] = useState<DemoRunRecord[]>([]);
  const [runtime, setRuntime] = useState<RuntimeInfo | null>(null);
  const [datasets, setDatasets] = useState<DatasetVersion[]>([]);
  const [algorithms, setAlgorithms] = useState<AlgorithmVersion[]>([]);
  const [attacks, setAttacks] = useState<AttackPreset[]>([]);
  const [latestResults, setLatestResults] = useState<RunResults | null>(null);
  const [autoRefreshSeconds, setAutoRefreshSeconds] = useState(10);
  const [notice, setNotice] = useState("");

  const loadDashboard = async () => {
    const [loadedConfigs, loadedRuns, loadedRuntime, loadedDatasets, loadedAlgorithms, loadedAttacks] =
      await Promise.all([
        fetchSavedConfigs(),
        fetchRuns(),
        fetchRuntime(),
        fetchDatasets(),
        fetchAlgorithms(),
        fetchAttacks()
      ]);
    setConfigs(loadedConfigs);
    setRuns(loadedRuns);
    setRuntime(loadedRuntime);
    setDatasets(loadedDatasets);
    setAlgorithms(loadedAlgorithms);
    setAttacks(loadedAttacks);
    setNotice("");

    const latestRun = latestResultCandidate(loadedRuns);
    if (latestRun) {
      try {
        setLatestResults(await fetchRunResults(latestRun.id));
      } catch {
        setLatestResults(null);
      }
    } else {
      setLatestResults(null);
    }
  };

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      loadDashboard().catch(() => {
        if (!cancelled) {
          setNotice("API 未启动或不可访问，请先启动 FastAPI 服务。");
        }
      });
    };
    load();
    if (autoRefreshSeconds <= 0) {
      return () => {
        cancelled = true;
      };
    }
    const timer = window.setInterval(load, autoRefreshSeconds * 1000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [autoRefreshSeconds]);

  const stats = useMemo(() => summarizeRuns(runs), [runs]);
  const activeRows = useMemo(
    () => buildActiveRunRows(runs, configs, datasets, algorithms, attacks),
    [runs, configs, datasets, algorithms, attacks]
  );
  const systemHealthy = Boolean(runtime) && !(runtime?.workers ?? []).some((worker) => worker.status === "error");
  const workerLabel = runtime?.workers.length
    ? runtime.workers.map((worker) => `${worker.status}:${worker.device}`).join(", ")
    : "no worker";

  return (
    <AppShell active="console">
      <div className="topbar console-topbar">
        <div className="title-block console-title">
          <h1>{t.console.title}</h1>
          <span className={systemHealthy ? "status-dot ok" : "status-dot warn"}>
            {systemHealthy ? t.console.systemHealthy : t.console.systemDegraded}
          </span>
        </div>
        <div className="toolbar">
          <button className="button" onClick={loadDashboard} type="button">
            <RefreshCw size={16} />
            {t.common.refresh}
          </button>
          <label className="select-button">
            <span>{t.console.autoRefresh}</span>
            <select
              onChange={(event) => setAutoRefreshSeconds(Number(event.target.value))}
              value={autoRefreshSeconds}
            >
              <option value={10}>10s</option>
              <option value={30}>30s</option>
              <option value={0}>Off</option>
            </select>
          </label>
          <button className="button" disabled type="button">
            <Filter size={16} />
            {t.console.filters}
          </button>
        </div>
      </div>

      <section className="metric-card-grid">
        <div className="metric-card">
          <div>
            <span>{t.console.queuedRuns}</span>
            <strong>{stats.queued}</strong>
          </div>
          <Clock3 size={20} />
        </div>
        <div className="metric-card">
          <div>
            <span>{t.console.runningRuns}</span>
            <strong>{stats.running}</strong>
          </div>
          <PlayCircle size={20} />
        </div>
        <div className="metric-card">
          <div>
            <span>{t.console.completedRuns}</span>
            <strong>{stats.completed}</strong>
          </div>
          <CheckCircle2 size={20} />
        </div>
        <div className="metric-card danger">
          <div>
            <span>{t.console.failedRuns}</span>
            <strong>{stats.failed}</strong>
          </div>
          <AlertTriangle size={20} />
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>{t.console.activeRuns}</h2>
          <Activity size={16} />
        </div>
        <div className="panel-body table-scroll">
          <table className="table compact-table">
            <thead>
              <tr>
                <th>{t.runs.run}</th>
                <th>{t.common.config}</th>
                <th>{t.common.dataset}</th>
                <th>{t.common.algorithm}</th>
                <th>{t.common.attackPreset}</th>
                <th>{t.common.progress}</th>
                <th>{t.runs.status}</th>
                <th>{t.console.startedAt}</th>
              </tr>
            </thead>
            <tbody>
              {activeRows.map((run) => (
                <tr key={run.id}>
                  <td>{run.id}</td>
                  <td>{run.configName}</td>
                  <td>{run.datasetLabel}</td>
                  <td>{run.algorithmLabel}</td>
                  <td>{run.attackLabel}</td>
                  <td>
                    <div className="progress-cell">
                      <div className="progress-track">
                        <div className="progress-bar" style={{ width: `${run.progress}%` }} />
                      </div>
                      <span>{run.progress}%</span>
                    </div>
                  </td>
                  <td>
                    <span className={statusBadgeClass(run.status)}>{t.common.status[run.status]}</span>
                  </td>
                  <td>{run.startedAt ?? run.updatedAt}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {activeRows.length === 0 ? <div className="empty compact-empty">{t.console.noActiveRuns}</div> : null}
          {notice ? <div className="risk warn">{notice}</div> : null}
        </div>
      </section>

      <section className="console-bottom-grid">
        <div className="panel">
          <div className="panel-header">
            <h2>{t.console.robustnessCurves}</h2>
          </div>
          <div className="panel-body">
            <RobustnessCurve emptyText={t.console.needMultipleStrengths} results={latestResults} />
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h2>{t.console.leaderboardPreview}</h2>
            <Trophy size={16} />
          </div>
          <div className="panel-body">
            <div className="leaderboard-mini-empty">
              <Trophy size={22} />
              <p>{t.console.officialLeaderboardPending}</p>
            </div>
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h2>{t.console.datasetsSummary}</h2>
          </div>
          <div className="panel-body table-scroll">
            <table className="table compact-table">
              <thead>
                <tr>
                  <th>{t.common.dataset}</th>
                  <th>{t.common.samples}</th>
                  <th>{t.resources.status}</th>
                </tr>
              </thead>
              <tbody>
                {datasets.slice(0, 5).map((dataset) => (
                  <tr key={dataset.id}>
                    <td>{dataset.name}</td>
                    <td>{dataset.sampleCount.toLocaleString()}</td>
                    <td>
                      <span className="badge ok">{t.common.enabled}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {datasets.length === 0 ? <div className="empty compact-empty">{t.common.noData}</div> : null}
          </div>
        </div>
      </section>

      <p className="dashboard-note">
        {t.console.monitorNote} {workerLabel ? `(${workerLabel})` : ""}
        {latestResults?.aggregates[0]?.meanBitAccuracy != null
          ? ` · Latest Bit Acc. ${formatMetric(latestResults.aggregates[0].meanBitAccuracy)}`
          : ""}
      </p>
    </AppShell>
  );
}
