"use client";

import { AppShell } from "@/components/AppShell";
import { useLanguage } from "@/components/LanguageProvider";

const entities = [
  "users",
  "datasets",
  "dataset_versions",
  "samples",
  "algorithm_packages",
  "algorithm_versions",
  "model_artifacts",
  "attack_methods",
  "attack_presets",
  "experiment_specs",
  "experiment_runs",
  "experiment_cells",
  "artifacts",
  "metric_summaries",
  "sandbox_builds"
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
