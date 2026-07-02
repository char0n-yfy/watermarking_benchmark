"use client";

import { AppShell } from "@/components/AppShell";
import { useLanguage } from "@/components/LanguageProvider";

const entities = [
  "experiment_configs",
  "experiment_runs",
  "experiment_cells",
  "worker_heartbeats"
];

export default function SchemaPage() {
  const { t } = useLanguage();

  return (
    <AppShell active="schema">
      <div className="topbar">
        <div className="title-block">
          <h1>{t.schema.title}</h1>
          <p>{t.schema.subtitle}</p>
        </div>
      </div>
      <div className="panel">
        <div className="panel-header">
          <h2>{t.schema.coreTables}</h2>
        </div>
        <div className="panel-body resource-list">
          {entities.map((entity) => (
            <div className="resource-item" key={entity}>
              <div>
                <strong>{entity}</strong>
                <span>{t.schema.tableDescription}</span>
              </div>
              <span className="badge">{t.common.table}</span>
            </div>
          ))}
        </div>
      </div>
    </AppShell>
  );
}
