"use client";

import { useEffect, useMemo, useState } from "react";
import { Archive, Braces, Database, Gauge, GitBranch, RotateCcw, Shield } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { useLanguage } from "@/components/LanguageProvider";
import { defaultSelection } from "@/lib/demo-store";
import {
  createSavedConfig,
  fetchAlgorithms,
  fetchAttacks,
  fetchDatasets,
  fetchSavedConfigs
} from "@/lib/api";
import { localizedName } from "@/lib/i18n";
import {
  algorithms as fallbackAlgorithms,
  attacks as fallbackAttacks,
  datasets as fallbackDatasets
} from "@/lib/mock-data";
import { estimateMatrix } from "@/lib/matrix";
import type {
  AlgorithmVersion,
  AttackPreset,
  DatasetVersion,
  ExperimentSelection,
  SavedExperimentConfig
} from "@/lib/types";

function toggle(values: string[], value: string) {
  return values.includes(value) ? values.filter((item) => item !== value) : [...values, value];
}

function matchesResource(
  query: string,
  resource: { id: string; name: string; method?: string; category?: string; description?: string }
) {
  const normalizedQuery = query.trim().toLowerCase();
  if (!normalizedQuery) {
    return true;
  }
  return [resource.id, resource.name, resource.method, resource.category, resource.description]
    .filter(Boolean)
    .join(" ")
    .toLowerCase()
    .includes(normalizedQuery);
}

export default function ConfigsPage() {
  const { language, t } = useLanguage();
  const [selection, setSelection] = useState<ExperimentSelection>(defaultSelection);
  const [configName, setConfigName] = useState("Demo robustness smoke");
  const [savedConfigs, setSavedConfigs] = useState<SavedExperimentConfig[]>([]);
  const [datasets, setDatasets] = useState<DatasetVersion[]>(fallbackDatasets);
  const [algorithms, setAlgorithms] = useState<AlgorithmVersion[]>(fallbackAlgorithms);
  const [attacks, setAttacks] = useState<AttackPreset[]>(fallbackAttacks);
  const [algorithmFilter, setAlgorithmFilter] = useState("");
  const [attackFilter, setAttackFilter] = useState("");
  const [message, setMessage] = useState("");
  const estimate = useMemo(() => estimateMatrix(selection, datasets, attacks), [selection, datasets, attacks]);
  const filteredAlgorithms = useMemo(
    () => algorithms.filter((algorithm) => matchesResource(algorithmFilter, algorithm)),
    [algorithmFilter, algorithms]
  );
  const filteredAttacks = useMemo(
    () => attacks.filter((attack) => matchesResource(attackFilter, attack)),
    [attackFilter, attacks]
  );
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
    let cancelled = false;
    Promise.all([fetchDatasets(), fetchAlgorithms(), fetchAttacks(), fetchSavedConfigs()])
      .then(([apiDatasets, apiAlgorithms, apiAttacks, apiConfigs]) => {
        if (cancelled) {
          return;
        }
        if (apiDatasets.length > 0) {
          setDatasets(apiDatasets);
          setSelection((current) => {
            const validDatasetIds = new Set(apiDatasets.map((dataset) => dataset.id));
            const hasValidDataset = current.datasetIds.some((id) => validDatasetIds.has(id));
            return hasValidDataset ? current : { ...current, datasetIds: [apiDatasets[0].id] };
          });
        } else {
          setDatasets([]);
          setSelection((current) => ({ ...current, datasetIds: [] }));
          setMessage("resources/datasets 下还没有可用图片，请先解压数据集。");
        }
        setAlgorithms(apiAlgorithms.length > 0 ? apiAlgorithms : fallbackAlgorithms);
        setAttacks(apiAttacks.length > 0 ? apiAttacks : fallbackAttacks);
        setSavedConfigs(apiConfigs);
      })
      .catch(() => {
        if (!cancelled) {
          setSavedConfigs([]);
          setMessage("API 未启动，当前只显示占位资源，无法保存配置。");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const saveConfig = async () => {
    try {
      const config = await createSavedConfig(configName, selection);
      setSavedConfigs([config, ...savedConfigs]);
      setMessage(t.configs.savedToast);
    } catch {
      setMessage("API 保存失败，请先启动 FastAPI 服务后再保存。");
    }
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
          <button
            className="button primary"
            disabled={
              selection.datasetIds.length === 0 ||
              selection.algorithmIds.length === 0 ||
              selection.attackPresetIds.length === 0
            }
            onClick={saveConfig}
            title={t.configs.saveConfig}
          >
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
              <div className="selector-stack">
                <div className="selector-tools">
                  <input
                    aria-label="Filter watermark algorithms"
                    onChange={(event) => setAlgorithmFilter(event.target.value)}
                    placeholder={language === "zh" ? "筛选水印算法" : "Filter watermark algorithms"}
                    value={algorithmFilter}
                  />
                  <span className="count-pill">
                    {selection.algorithmIds.length}/{algorithms.length}
                  </span>
                </div>
                <div className="option-grid dense-options">
                  {filteredAlgorithms.map((algorithm) => (
                    <label className="check-tile resource-check-tile" key={algorithm.id}>
                      <input
                        checked={selection.algorithmIds.includes(algorithm.id)}
                        disabled={algorithm.status !== "enabled" || algorithm.available === false}
                        onChange={() =>
                          setSelection((current) => ({
                            ...current,
                            algorithmIds: toggle(current.algorithmIds, algorithm.id)
                          }))
                        }
                        type="checkbox"
                      />
                      <span className="tile-copy">
                        <strong>{algorithm.name}</strong>
                        <small>
                          {algorithm.method ?? algorithm.id}
                          {algorithm.category ? ` · ${algorithm.category}` : ""}
                        </small>
                      </span>
                      {algorithm.requiresGpu ? <span className="badge warn">{t.common.gpu}</span> : null}
                      {algorithm.available === false ? <span className="badge error">Missing</span> : null}
                    </label>
                  ))}
                </div>
                {filteredAlgorithms.length === 0 ? <div className="empty compact-empty">{t.common.noData}</div> : null}
              </div>
            </div>

            <div className="matrix-row">
              <div className="matrix-label">
                <Gauge size={15} />
                {t.console.attacks}
              </div>
              <div className="selector-stack">
                <div className="selector-tools">
                  <input
                    aria-label="Filter attacks"
                    onChange={(event) => setAttackFilter(event.target.value)}
                    placeholder={language === "zh" ? "筛选攻击方法" : "Filter attacks"}
                    value={attackFilter}
                  />
                  <span className="count-pill">
                    {selection.attackPresetIds.length}/{attacks.length}
                  </span>
                </div>
                <div className="option-grid dense-options">
                  {filteredAttacks.map((attack) => (
                    <label className="check-tile resource-check-tile" key={attack.id}>
                      <input
                        checked={selection.attackPresetIds.includes(attack.id)}
                        disabled={attack.available === false}
                        onChange={() =>
                          setSelection((current) => ({
                            ...current,
                            attackPresetIds: toggle(current.attackPresetIds, attack.id)
                          }))
                        }
                        type="checkbox"
                      />
                      <span className="tile-copy">
                        <strong>{localizedName(language, attack.id, attack.name)}</strong>
                        <small>
                          {attack.method}
                          {attack.category ? ` · ${attack.category}` : ""}
                          {attack.strengths.length > 1 ? ` · ${attack.strengths.length} strengths` : ""}
                        </small>
                      </span>
                      {attack.requiresGpu ? <span className="badge warn">{t.common.gpu}</span> : null}
                      {attack.available === false ? <span className="badge error">Missing</span> : null}
                    </label>
                  ))}
                </div>
                {filteredAttacks.length === 0 ? <div className="empty compact-empty">{t.common.noData}</div> : null}
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
