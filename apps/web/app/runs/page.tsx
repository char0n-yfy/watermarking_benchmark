"use client";

import { useEffect, useMemo, useState } from "react";
import { PlayCircle } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { useLanguage } from "@/components/LanguageProvider";
import {
  createRunRecord,
  loadRunRecords,
  loadSavedConfigs,
  saveRunRecords
} from "@/lib/demo-store";
import { localizedDate } from "@/lib/i18n";
import type { DemoRunRecord, SavedExperimentConfig } from "@/lib/types";

export default function RunsPage() {
  const { language, t } = useLanguage();
  const [configs, setConfigs] = useState<SavedExperimentConfig[]>([]);
  const [runs, setRuns] = useState<DemoRunRecord[]>([]);
  const [selectedConfigId, setSelectedConfigId] = useState("");
  const [notice, setNotice] = useState("");
  const selectedConfig = useMemo(
    () => configs.find((config) => config.id === selectedConfigId),
    [configs, selectedConfigId]
  );

  useEffect(() => {
    const loadedConfigs = loadSavedConfigs();
    setConfigs(loadedConfigs);
    setRuns(loadRunRecords());
    setSelectedConfigId(loadedConfigs[0]?.id ?? "");
  }, []);

  const runSelectedConfig = () => {
    if (!selectedConfig) {
      return;
    }
    const nextRun = createRunRecord(selectedConfig);
    const nextRuns = [nextRun, ...runs];
    setRuns(nextRuns);
    saveRunRecords(nextRuns);
    setNotice(t.runs.queuedNotice);
  };

  return (
    <AppShell active="runs">
      <div className="topbar">
        <div className="title-block">
          <h1>{t.runs.title}</h1>
          <p>{t.runs.subtitle}</p>
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
                <button className="button primary" onClick={runSelectedConfig} type="button">
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
                  <th>{t.runs.cells}</th>
                  <th>{t.runs.updated}</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr key={run.id}>
                    <td>{run.id}</td>
                    <td>{run.configName}</td>
                    <td>
                      <span className={run.status === "succeeded" ? "badge ok" : "badge warn"}>
                        {t.common.status[run.status]}
                      </span>
                    </td>
                    <td>{run.progress}%</td>
                    <td>{run.cells}</td>
                    <td>{localizedDate(language, run.updatedAt)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
