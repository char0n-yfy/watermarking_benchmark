"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { Activity, Archive, ArrowRight, CheckCircle2, Clock3, PlayCircle, XCircle } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { useLanguage } from "@/components/LanguageProvider";
import { loadRunRecords, loadSavedConfigs } from "@/lib/demo-store";
import type { DemoRunRecord, SavedExperimentConfig } from "@/lib/types";

export default function ExperimentConsole() {
  const { t } = useLanguage();
  const [configs, setConfigs] = useState<SavedExperimentConfig[]>([]);
  const [runs, setRuns] = useState<DemoRunRecord[]>([]);

  useEffect(() => {
    setConfigs(loadSavedConfigs());
    setRuns(loadRunRecords());
  }, []);

  const stats = useMemo(
    () => ({
      running: runs.filter((run) => run.status === "running").length,
      queued: runs.filter((run) => run.status === "queued").length,
      succeeded: runs.filter((run) => run.status === "succeeded").length,
      failed: runs.filter((run) => run.status === "failed" || run.status === "partially_failed").length
    }),
    [runs]
  );

  const activeRuns = runs.filter((run) => run.status === "running" || run.status === "queued");

  return (
    <AppShell active="console">
      <div className="topbar">
        <div className="title-block">
          <h1>{t.console.title}</h1>
          <p>{t.console.subtitle}</p>
        </div>
        <div className="toolbar">
          <Link className="button" href="/configs">
            <Archive size={16} />
            {t.console.openConfigs}
          </Link>
          <Link className="button primary" href="/runs">
            <PlayCircle size={16} />
            {t.console.openRuns}
          </Link>
        </div>
      </div>

      <section className="dashboard-grid">
        <div className="panel dashboard-main">
          <div className="panel-header">
            <h2>{t.console.activeQueue}</h2>
            <Activity size={16} />
          </div>
          <div className="panel-body">
            <div className="stats dashboard-stats">
              <div className="stat">
                <span>{t.console.running}</span>
                <strong>{stats.running}</strong>
              </div>
              <div className="stat">
                <span>{t.console.queued}</span>
                <strong>{stats.queued}</strong>
              </div>
              <div className="stat">
                <span>{t.console.completed}</span>
                <strong>{stats.succeeded}</strong>
              </div>
              <div className="stat">
                <span>{t.console.failed}</span>
                <strong>{stats.failed}</strong>
              </div>
            </div>
            <div className="run-list">
              {activeRuns.length === 0 ? <div className="empty">{t.common.noData}</div> : null}
              {activeRuns.map((run) => (
                <div className="run-card" key={run.id}>
                  <div className="run-card-main">
                    <div className="run-status-icon">
                      {run.status === "running" ? <Activity size={17} /> : <Clock3 size={17} />}
                    </div>
                    <div>
                      <strong>{run.configName}</strong>
                      <span>
                        {run.id} · {run.cells} {t.console.cells}
                      </span>
                    </div>
                  </div>
                  <div className="progress-block">
                    <span>{run.progress}%</span>
                    <div className="progress-track">
                      <div className="progress-bar" style={{ width: `${run.progress}%` }} />
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <p className="dashboard-note">{t.console.monitorNote}</p>
          </div>
        </div>

        <aside className="panel">
          <div className="panel-header">
            <h2>{t.console.savedConfigs}</h2>
            <Archive size={16} />
          </div>
          <div className="panel-body resource-list">
            {configs.map((config) => (
              <Link className="resource-item link-item" href="/runs" key={config.id}>
                <div>
                  <strong>{config.name}</strong>
                  <span>
                    {config.cellCount} {t.console.cells} · {config.imageOperationCount.toLocaleString()}{" "}
                    {t.console.ops}
                  </span>
                </div>
                <ArrowRight size={15} />
              </Link>
            ))}
          </div>
        </aside>

        <aside className="panel">
          <div className="panel-header">
            <h2>{t.console.recentActivity}</h2>
            <CheckCircle2 size={16} />
          </div>
          <div className="panel-body resource-list">
            {runs.slice(0, 4).map((run) => (
              <div className="resource-item" key={run.id}>
                <div>
                  <strong>{run.configName}</strong>
                  <span>
                    {run.updatedAt} · {run.progress}% {t.common.progress}
                  </span>
                </div>
                <span className={run.status === "succeeded" ? "badge ok" : "badge warn"}>
                  {run.status === "failed" ? <XCircle size={12} /> : null}
                  {t.common.status[run.status]}
                </span>
              </div>
            ))}
          </div>
        </aside>
      </section>
    </AppShell>
  );
}
