"use client";

import { useEffect, useMemo, useState } from "react";
import { FileText, PlayCircle, Square, RefreshCw } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { useLanguage } from "@/components/LanguageProvider";
import { cancelRun, createRun, fetchRunLogs, fetchRuns, fetchSavedConfigs } from "@/lib/api";
import { localizedDate } from "@/lib/i18n";
import type { DemoRunRecord, RunLogs, SavedExperimentConfig } from "@/lib/types";

const terminalStatuses = new Set(["succeeded", "failed", "cancelled", "partially_failed"]);

function badgeClass(status: DemoRunRecord["status"]) {
  if (status === "succeeded") {
    return "badge ok";
  }
  if (status === "failed" || status === "partially_failed" || status === "cancelled") {
    return "badge error";
  }
  return "badge warn";
}

export default function RunsPage() {
  const { language, t } = useLanguage();
  const [configs, setConfigs] = useState<SavedExperimentConfig[]>([]);
  const [runs, setRuns] = useState<DemoRunRecord[]>([]);
  const [selectedConfigId, setSelectedConfigId] = useState("");
  const [selectedRunId, setSelectedRunId] = useState("");
  const [logs, setLogs] = useState<RunLogs | null>(null);
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const selectedConfig = useMemo(
    () => configs.find((config) => config.id === selectedConfigId),
    [configs, selectedConfigId]
  );

  const refresh = async () => {
    const [loadedConfigs, loadedRuns] = await Promise.all([fetchSavedConfigs(), fetchRuns()]);
    setConfigs(loadedConfigs);
    setRuns(loadedRuns);
    setSelectedConfigId((current) => current || loadedConfigs[0]?.id || "");
  };

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      refresh().catch(() => {
        if (!cancelled) {
          setNotice("API 未启动或不可访问，运行队列无法读取。");
        }
      });
    };
    load();
    const timer = window.setInterval(load, 2500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  const runSelectedConfig = async () => {
    if (!selectedConfig) {
      return;
    }
    setBusy(true);
    setNotice(t.runs.queuedNotice);
    try {
      const nextRun = await createRun(selectedConfig.id);
      setRuns((current) => [nextRun, ...current.filter((run) => run.id !== nextRun.id)]);
      setSelectedRunId(nextRun.id);
    } catch {
      setNotice("提交失败，请确认 FastAPI 服务已启动。");
    } finally {
      setBusy(false);
    }
  };

  const cancelSelectedRun = async (runId: string) => {
    setBusy(true);
    try {
      const updated = await cancelRun(runId);
      setRuns((current) => current.map((run) => (run.id === runId ? updated : run)));
      setNotice("已发送取消请求。");
    } catch {
      setNotice("取消失败，请检查 API 状态。");
    } finally {
      setBusy(false);
    }
  };

  const loadLogs = async (runId: string) => {
    setSelectedRunId(runId);
    try {
      setLogs(await fetchRunLogs(runId));
      setNotice("");
    } catch {
      setNotice("读取日志失败，请检查 run 是否存在。");
    }
  };

  return (
    <AppShell active="runs">
      <div className="topbar">
        <div className="title-block">
          <h1>{t.runs.title}</h1>
          <p>{t.runs.subtitle}</p>
        </div>
        <div className="toolbar">
          <button className="button" onClick={() => refresh()} title={t.common.updated} type="button">
            <RefreshCw size={16} />
          </button>
        </div>
      </div>

      <div className="runs-grid">
        <div className="panel">
          <div className="panel-header">
            <h2>{t.runs.launcher}</h2>
            <PlayCircle size={16} />
          </div>
          <div className="panel-body launcher-panel">
            {configs.length === 0 ? (
              <div className="empty">{t.runs.noConfigs}</div>
            ) : (
              <>
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
                {selectedConfig ? (
                  <div className="stats">
                    <div className="stat">
                      <span>{t.console.cells}</span>
                      <strong>{selectedConfig.cellCount}</strong>
                    </div>
                    <div className="stat">
                      <span>{t.common.samples}</span>
                      <strong>{selectedConfig.sampleCount}</strong>
                    </div>
                    <div className="stat">
                      <span>{t.console.ops}</span>
                      <strong>{selectedConfig.imageOperationCount}</strong>
                    </div>
                  </div>
                ) : null}
                <button
                  className="button primary"
                  disabled={busy}
                  onClick={runSelectedConfig}
                  type="button"
                >
                  <PlayCircle size={16} />
                  {t.runs.execute}
                </button>
                {notice ? <div className="risk ok">{notice}</div> : null}
              </>
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h2>{t.runs.recent}</h2>
          </div>
          <div className="panel-body">
            <table className="table">
              <thead>
                <tr>
                  <th>{t.runs.run}</th>
                  <th>{t.common.config}</th>
                  <th>{t.runs.status}</th>
                  <th>{t.common.progress}</th>
                  <th>{t.runs.worker}</th>
                  <th>{t.runs.updated}</th>
                  <th>{t.runs.logs}</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr key={run.id}>
                    <td>{run.id}</td>
                    <td>{run.configName}</td>
                    <td>
                      <span className={badgeClass(run.status)}>{t.common.status[run.status]}</span>
                    </td>
                    <td>{run.progress}%</td>
                    <td>{run.workerId ?? "n/a"}</td>
                    <td>{localizedDate(language, run.updatedAt)}</td>
                    <td>
                      <div className="table-actions">
                        <button className="icon-button" onClick={() => loadLogs(run.id)} type="button">
                          <FileText size={15} />
                        </button>
                        {!terminalStatuses.has(run.status) ? (
                          <button
                            className="icon-button danger"
                            disabled={busy}
                            onClick={() => cancelSelectedRun(run.id)}
                            type="button"
                          >
                            <Square size={14} />
                          </button>
                        ) : null}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {runs.length === 0 ? <div className="empty">{t.common.noData}</div> : null}
          </div>
        </div>
      </div>

      {selectedRunId ? (
        <div className="panel log-panel">
          <div className="panel-header">
            <h2>
              {t.runs.logs} · {selectedRunId}
            </h2>
          </div>
          <div className="panel-body">
            {logs ? (
              <>
                <div className="risk ok">
                  {t.runs.logPath}: {logs.logPath}
                </div>
                <pre className="log-preview">
                  {logs.exists ? logs.lines.join("\n") || "empty log" : "log file not created yet"}
                </pre>
              </>
            ) : (
              <div className="empty">{t.common.noData}</div>
            )}
          </div>
        </div>
      ) : null}
    </AppShell>
  );
}
