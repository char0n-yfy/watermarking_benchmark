"use client";

import { useEffect, useMemo, useState } from "react";
import { Archive, Braces, Database, Gauge, GitBranch, RotateCcw, Shield } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { useLanguage } from "@/components/LanguageProvider";
import { buildSavedConfig, defaultSelection, addSavedConfig, loadSavedConfigs } from "@/lib/demo-store";
import { localizedName } from "@/lib/i18n";
import { algorithms, attacks, datasets } from "@/lib/mock-data";
import { estimateMatrix } from "@/lib/matrix";
import type { ExperimentSelection, SavedExperimentConfig } from "@/lib/types";

function toggle(values: string[], value: string) {
  return values.includes(value) ? values.filter((item) => item !== value) : [...values, value];
}

export default function ConfigsPage() {
  const { language, t } = useLanguage();
  const [selection, setSelection] = useState<ExperimentSelection>(defaultSelection);
  const [configName, setConfigName] = useState("Demo robustness smoke");
  const [savedConfigs, setSavedConfigs] = useState<SavedExperimentConfig[]>([]);
  const [message, setMessage] = useState("");
  const estimate = useMemo(() => estimateMatrix(selection, datasets, attacks), [selection]);
  const specPreview = {
    name: configName,
    dataset_versions: selection.datasetIds,
    algorithm_versions: selection.algorithmIds,
    attack_presets: selection.attackPresetIds,
    seeds: selection.seeds,
    max_samples_per_dataset: selection.maxSamples,
    materialized_cells: estimate.cellCount
  };

  useEffect(() => {
    setSavedConfigs(loadSavedConfigs());
  }, []);

  const saveConfig = () => {
    const config = buildSavedConfig(configName, selection);
    setSavedConfigs(addSavedConfig(config));
    setMessage(t.configs.savedToast);
  };

  return (
    <AppShell active="configs">
      <div className="topbar">
        <div className="title-block">
          <h1>{t.configs.title}</h1>
          <p>{t.configs.subtitle}</p>
        </div>
        <div className="toolbar">
          <button
            className="button"
            title={t.console.reset}
            onClick={() => {
              setSelection(defaultSelection);
              setConfigName("Demo robustness smoke");
            }}
          >
            <RotateCcw size={16} />
          </button>
          <button className="button primary" onClick={saveConfig} title={t.configs.saveConfig}>
            <Archive size={16} />
            {t.configs.saveConfig}
          </button>
        </div>
      </div>

      <section className="console-grid config-grid">
        <div className="panel">
          <div className="panel-header">
            <h2>{t.configs.savedConfigs}</h2>
            <Archive size={16} />
          </div>
          <div className="panel-body resource-list">
            {savedConfigs.length === 0 ? <div className="empty">{t.configs.empty}</div> : null}
            {savedConfigs.map((config) => (
              <div className="resource-item" key={config.id}>
                <div>
                  <strong>{config.name}</strong>
                  <span>
                    {config.cellCount} {t.console.cells} · {config.sampleCount.toLocaleString()}{" "}
                    {t.common.samples}
                  </span>
                </div>
                <span className="badge ok">{t.common.config}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h2>{t.console.matrix}</h2>
            <GitBranch size={16} />
          </div>
          <div className="panel-body matrix">
            <div className="field config-name-field">
              <label htmlFor="config-name">{t.configs.nameLabel}</label>
              <input
                id="config-name"
                onChange={(event) => setConfigName(event.target.value)}
                placeholder={t.configs.namePlaceholder}
                value={configName}
              />
            </div>

            <div className="matrix-row">
              <div className="matrix-label">
                <Database size={15} />
                {t.console.datasets}
              </div>
              <div className="option-grid">
                {datasets.map((dataset) => (
                  <label className="check-tile" key={dataset.id}>
                    <input
                      checked={selection.datasetIds.includes(dataset.id)}
                      onChange={() =>
                        setSelection((current) => ({
                          ...current,
                          datasetIds: toggle(current.datasetIds, dataset.id)
                        }))
                      }
                      type="checkbox"
                    />
                    <span>{localizedName(language, dataset.id, dataset.name)}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="matrix-row">
              <div className="matrix-label">
                <Shield size={15} />
                {t.console.algorithms}
              </div>
              <div className="option-grid">
                {algorithms.map((algorithm) => (
                  <label className="check-tile" key={algorithm.id}>
                    <input
                      checked={selection.algorithmIds.includes(algorithm.id)}
                      disabled={algorithm.status !== "enabled"}
                      onChange={() =>
                        setSelection((current) => ({
                          ...current,
                          algorithmIds: toggle(current.algorithmIds, algorithm.id)
                        }))
                      }
                      type="checkbox"
                    />
                    <span>{algorithm.name}</span>
                    {algorithm.requiresGpu ? <span className="badge warn">{t.common.gpu}</span> : null}
                  </label>
                ))}
              </div>
            </div>

            <div className="matrix-row">
              <div className="matrix-label">
                <Gauge size={15} />
                {t.console.attacks}
              </div>
              <div className="option-grid">
                {attacks.map((attack) => (
                  <label className="check-tile" key={attack.id}>
                    <input
                      checked={selection.attackPresetIds.includes(attack.id)}
                      onChange={() =>
                        setSelection((current) => ({
                          ...current,
                          attackPresetIds: toggle(current.attackPresetIds, attack.id)
                        }))
                      }
                      type="checkbox"
                    />
                    <span>{localizedName(language, attack.id, attack.name)}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="matrix-row">
              <div className="matrix-label">
                <Braces size={15} />
                {t.console.parameters}
              </div>
              <div className="field-grid">
                <div className="field">
                  <label htmlFor="seeds">{t.console.seeds}</label>
                  <input
                    id="seeds"
                    value={selection.seeds.join(",")}
                    onChange={(event) =>
                      setSelection((current) => ({
                        ...current,
                        seeds: event.target.value
                          .split(",")
                          .map((seed) => Number(seed.trim()))
                          .filter((seed) => Number.isFinite(seed))
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="max-samples">{t.console.maxSamples}</label>
                  <input
                    id="max-samples"
                    min={1}
                    onChange={(event) =>
                      setSelection((current) => ({
                        ...current,
                        maxSamples: Number(event.target.value)
                      }))
                    }
                    type="number"
                    value={selection.maxSamples}
                  />
                </div>
              </div>
            </div>
            {message ? <div className="risk ok">{message}</div> : null}
          </div>
        </div>

        <aside className="panel">
          <div className="panel-header">
            <h2>{t.configs.specPreview}</h2>
            <Braces size={16} />
          </div>
          <div className="panel-body">
            <div className="stats">
              <div className="stat">
                <span>{t.console.cells}</span>
                <strong>{estimate.cellCount}</strong>
              </div>
              <div className="stat">
                <span>{t.common.samples}</span>
                <strong>{estimate.sampleCount}</strong>
              </div>
              <div className="stat">
                <span>{t.console.ops}</span>
                <strong>{estimate.imageOperationCount}</strong>
              </div>
            </div>
            <div className={`risk ${estimate.level}`}>
              {estimate.level === "ok" ? t.console.okRisk : t.console.warnRisk}
            </div>
            <pre className="json-preview">{JSON.stringify(specPreview, null, 2)}</pre>
          </div>
        </aside>
      </section>
    </AppShell>
  );
}
