"use client";

import { useEffect, useMemo, useState } from "react";
import { Archive, Braces, Database, Edit3, Gauge, Plus, Save, Search, Shield, Trash2, X } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { useLanguage } from "@/components/LanguageProvider";
import {
  createSavedConfig,
  deleteSavedConfig,
  fetchAlgorithms,
  fetchAttacks,
  fetchDatasets,
  fetchSavedConfigs,
  renameSavedConfig
} from "@/lib/api";
import { localizedName } from "@/lib/i18n";
import { estimateMatrix } from "@/lib/matrix";
import type {
  AlgorithmVersion,
  AttackPreset,
  DatasetVersion,
  ExperimentSelection,
  SavedExperimentConfig
} from "@/lib/types";

const emptySelection: ExperimentSelection = {
  datasetIds: [],
  algorithmIds: [],
  attackPresetIds: [],
  seeds: [42],
  maxSamples: 100
};

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

function categoryLabel(language: string, category: string) {
  const zh: Record<string, string> = {
    "consumer-enhancement": "消费级增强",
    distortion: "经典失真",
    geometric: "几何变换",
    identity: "无攻击",
    regeneration: "再生成",
    removal: "移除攻击"
  };
  const en: Record<string, string> = {
    "consumer-enhancement": "Consumer enhancement",
    distortion: "Distortion",
    geometric: "Geometric",
    identity: "Identity",
    regeneration: "Regeneration",
    removal: "Removal"
  };
  const labels = language === "zh" ? zh : en;
  return labels[category] ?? category;
}

function attackCategory(attack: AttackPreset) {
  return attack.category || attack.method.split("_")[0] || "other";
}

function parseSeeds(value: string) {
  return value
    .split(",")
    .map((seed) => Number(seed.trim()))
    .filter((seed) => Number.isFinite(seed));
}

function idsFromQuery(paramName: string, validIds: Set<string>): string[] | null {
  if (typeof window === "undefined") {
    return null;
  }
  const raw = new URLSearchParams(window.location.search).get(paramName);
  if (raw == null) {
    return null;
  }
  return raw
    .split(",")
    .map((value) => value.trim())
    .filter((value) => value && validIds.has(value));
}

export default function ConfigsPage() {
  const { language, t } = useLanguage();
  const [selection, setSelection] = useState<ExperimentSelection>(emptySelection);
  const [configName, setConfigName] = useState("");
  const [seedText, setSeedText] = useState(emptySelection.seeds.join(","));
  const [savedConfigs, setSavedConfigs] = useState<SavedExperimentConfig[]>([]);
  const [datasets, setDatasets] = useState<DatasetVersion[]>([]);
  const [algorithms, setAlgorithms] = useState<AlgorithmVersion[]>([]);
  const [attacks, setAttacks] = useState<AttackPreset[]>([]);
  const [algorithmFilter, setAlgorithmFilter] = useState("");
  const [attackFilter, setAttackFilter] = useState("");
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [renameTarget, setRenameTarget] = useState<SavedExperimentConfig | null>(null);
  const [renameName, setRenameName] = useState("");
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
  const attackGroups = useMemo(() => {
    const groups = new Map<string, AttackPreset[]>();
    for (const attack of filteredAttacks) {
      const category = attackCategory(attack);
      groups.set(category, [...(groups.get(category) ?? []), attack]);
    }
    return [...groups.entries()].sort(([left], [right]) => left.localeCompare(right));
  }, [filteredAttacks]);

  const canSave =
    configName.trim().length > 0 &&
    selection.datasetIds.length > 0 &&
    selection.algorithmIds.length > 0 &&
    selection.attackPresetIds.length > 0 &&
    selection.seeds.length > 0 &&
    selection.maxSamples > 0;

  const copy =
    language === "zh"
      ? {
          addConfig: "新增配置",
          cancel: "取消",
          createTitle: "新增实验配置",
          datasetSettings: "数据集配置",
          draftEmpty: "当前没有正在编辑的实验配置。点击新增配置后，在弹窗中选择数据集、水印算法和攻击算法。",
          filterAlgorithms: "筛选水印算法",
          filterAttacks: "筛选攻击方法",
          deleteConfig: "删除",
          deleteConfirm: "确定删除这个实验配置？已有运行记录不会被删除。",
          renameConfig: "重命名",
          renameTitle: "重命名配置",
          save: "保存配置",
          modalHint: "保存后，该配置会进入运行页，可提交给 worker 执行。"
        }
      : {
          addConfig: "New config",
          cancel: "Cancel",
          createTitle: "New experiment config",
          datasetSettings: "Dataset settings",
          draftEmpty: "No experiment config is being edited. Create one in the dialog.",
          filterAlgorithms: "Filter watermark algorithms",
          filterAttacks: "Filter attacks",
          deleteConfig: "Delete",
          deleteConfirm: "Delete this experiment config? Existing runs will be kept.",
          renameConfig: "Rename",
          renameTitle: "Rename config",
          save: "Save config",
          modalHint: "After saving, launch this config from the Runs page."
        };

  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchDatasets(), fetchAlgorithms(), fetchAttacks(), fetchSavedConfigs()])
      .then(([apiDatasets, apiAlgorithms, apiAttacks, apiConfigs]) => {
        if (cancelled) {
          return;
        }
        const validDatasetIds = new Set(apiDatasets.map((dataset) => dataset.id));
        const validAlgorithmIds = new Set(apiAlgorithms.map((algorithm) => algorithm.id));
        const validAttackIds = new Set(apiAttacks.map((attack) => attack.id));
        const queryDatasetIds = idsFromQuery("datasetIds", validDatasetIds);
        const queryAlgorithmIds = idsFromQuery("algorithmIds", validAlgorithmIds);
        const queryAttackIds = idsFromQuery("attackPresetIds", validAttackIds);
        const hasQuerySelection =
          Boolean(queryDatasetIds && queryDatasetIds.length > 0) ||
          Boolean(queryAlgorithmIds && queryAlgorithmIds.length > 0) ||
          Boolean(queryAttackIds && queryAttackIds.length > 0);

        setDatasets(apiDatasets);
        setAlgorithms(apiAlgorithms);
        setAttacks(apiAttacks);
        setSavedConfigs(apiConfigs);

        if (hasQuerySelection) {
          setSelection((current) => ({
            ...current,
            datasetIds: queryDatasetIds ?? current.datasetIds.filter((id) => validDatasetIds.has(id)),
            algorithmIds: queryAlgorithmIds ?? current.algorithmIds.filter((id) => validAlgorithmIds.has(id)),
            attackPresetIds: queryAttackIds ?? current.attackPresetIds.filter((id) => validAttackIds.has(id))
          }));
          setIsCreateOpen(true);
          setMessage(t.configs.prefilledFromResources);
          return;
        }

        if (apiDatasets.length === 0) {
          setMessage("resources/datasets 下还没有可用图片，请先解压数据集。");
        }
      })
      .catch(() => {
        if (!cancelled) {
          setMessage("API 未启动或资源接口不可访问，暂时无法创建配置。");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const openCreateModal = () => {
    setSelection(emptySelection);
    setSeedText(emptySelection.seeds.join(","));
    setConfigName("");
    setAlgorithmFilter("");
    setAttackFilter("");
    setMessage("");
    setIsCreateOpen(true);
  };

  const saveConfig = async () => {
    if (!canSave) {
      setMessage("请至少选择一个数据集、一个水印算法、一个攻击算法，并填写配置名称。");
      return;
    }
    try {
      const config = await createSavedConfig(configName.trim(), selection);
      setSavedConfigs([config, ...savedConfigs]);
      setIsCreateOpen(false);
      setMessage(t.configs.savedToast);
    } catch {
      setMessage("API 保存失败，请先启动 FastAPI 服务后再保存。");
    }
  };

  const openRenameModal = (config: SavedExperimentConfig) => {
    setRenameTarget(config);
    setRenameName(config.name);
    setMessage("");
  };

  const renameConfig = async () => {
    if (!renameTarget) {
      return;
    }
    const nextName = renameName.trim();
    if (!nextName) {
      setMessage("配置名称不能为空。");
      return;
    }
    try {
      const updated = await renameSavedConfig(renameTarget.id, nextName);
      setSavedConfigs((configs) => configs.map((config) => (config.id === updated.id ? updated : config)));
      setRenameTarget(null);
      setRenameName("");
      setMessage(language === "zh" ? "配置已重命名。" : "Config renamed.");
    } catch {
      setMessage("重命名失败，请确认 API 服务可用。");
    }
  };

  const deleteConfig = async (config: SavedExperimentConfig) => {
    if (!window.confirm(copy.deleteConfirm)) {
      return;
    }
    try {
      await deleteSavedConfig(config.id);
      setSavedConfigs((configs) => configs.filter((item) => item.id !== config.id));
      setMessage(language === "zh" ? "配置已删除。" : "Config deleted.");
    } catch {
      setMessage("删除失败，请确认 API 服务可用。");
    }
  };

  return (
    <AppShell active="configs">
      <div className="topbar">
        <div className="title-block">
          <h1>{t.configs.title}</h1>
          <p>{t.configs.subtitle}</p>
        </div>
        <button className="button primary" onClick={openCreateModal} type="button">
          <Plus size={16} />
          {copy.addConfig}
        </button>
      </div>

      <section className="configs-home-grid">
        <div className="panel">
          <div className="panel-header">
            <h2>{t.configs.savedConfigs}</h2>
            <Archive size={16} />
          </div>
          <div className="panel-body resource-list">
            {savedConfigs.length === 0 ? (
              <div className="config-empty-state">
                <Braces size={34} />
                <h2>{t.configs.empty}</h2>
                <p>{copy.draftEmpty}</p>
                <button className="button primary" onClick={openCreateModal} type="button">
                  <Plus size={16} />
                  {copy.addConfig}
                </button>
              </div>
            ) : null}
            {savedConfigs.map((config) => (
              <div className="resource-item config-list-item" key={config.id}>
                <div>
                  <strong>{config.name}</strong>
                  <span>
                    {config.cellCount} {t.console.cells} · {config.sampleCount.toLocaleString()}{" "}
                    {t.common.samples}
                  </span>
                </div>
                <div className="config-actions">
                  <button
                    aria-label={`${copy.renameConfig}: ${config.name}`}
                    className="icon-button"
                    onClick={() => openRenameModal(config)}
                    title={copy.renameConfig}
                    type="button"
                  >
                    <Edit3 size={15} />
                  </button>
                  <button
                    aria-label={`${copy.deleteConfig}: ${config.name}`}
                    className="icon-button danger"
                    onClick={() => deleteConfig(config)}
                    title={copy.deleteConfig}
                    type="button"
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {message ? <div className="risk ok config-page-message">{message}</div> : null}

      {isCreateOpen ? (
        <div aria-modal="true" className="modal-backdrop" role="dialog">
          <div className="config-modal">
            <div className="modal-header">
              <div>
                <h2>{copy.createTitle}</h2>
                <p>{copy.modalHint}</p>
              </div>
              <button aria-label="Close" className="icon-button" onClick={() => setIsCreateOpen(false)} type="button">
                <X size={16} />
              </button>
            </div>

            <div className="modal-body config-modal-body">
              <div className="field config-name-field">
                <label htmlFor="config-name">{t.configs.nameLabel}</label>
                <input
                  id="config-name"
                  onChange={(event) => setConfigName(event.target.value)}
                  placeholder={t.configs.namePlaceholder}
                  value={configName}
                />
              </div>

              <section className="modal-section">
                <div className="modal-section-heading">
                  <Database size={16} />
                  <h3>{copy.datasetSettings}</h3>
                  <span className="count-pill">
                    {selection.datasetIds.length}/{datasets.length}
                  </span>
                </div>
                <div className="field-grid">
                  <div className="field">
                    <label htmlFor="seeds">{t.console.seeds}</label>
                    <input
                      id="seeds"
                      onChange={(event) => {
                        setSeedText(event.target.value);
                        setSelection((current) => ({ ...current, seeds: parseSeeds(event.target.value) }));
                      }}
                      placeholder="42, 123, 2026"
                      value={seedText}
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
                <div className="option-grid dense-options">
                  {datasets.map((dataset) => (
                    <label className="check-tile resource-check-tile" key={dataset.id}>
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
                      <span className="tile-copy">
                        <strong>{localizedName(language, dataset.id, dataset.name)}</strong>
                        <small>
                          {dataset.sampleCount.toLocaleString()} {t.common.samples}
                        </small>
                      </span>
                    </label>
                  ))}
                </div>
                {datasets.length === 0 ? <div className="empty compact-empty">{t.common.noData}</div> : null}
              </section>

              <section className="modal-section">
                <div className="modal-section-heading">
                  <Shield size={16} />
                  <h3>{t.console.algorithms}</h3>
                  <span className="count-pill">
                    {selection.algorithmIds.length}/{algorithms.length}
                  </span>
                </div>
                <div className="selector-tools">
                  <div className="field-icon-input">
                    <Search size={15} />
                    <input
                      aria-label="Filter watermark algorithms"
                      onChange={(event) => setAlgorithmFilter(event.target.value)}
                      placeholder={copy.filterAlgorithms}
                      value={algorithmFilter}
                    />
                  </div>
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
              </section>

              <section className="modal-section">
                <div className="modal-section-heading">
                  <Gauge size={16} />
                  <h3>{t.console.attacks}</h3>
                  <span className="count-pill">
                    {selection.attackPresetIds.length}/{attacks.length}
                  </span>
                </div>
                <div className="selector-tools">
                  <div className="field-icon-input">
                    <Search size={15} />
                    <input
                      aria-label="Filter attacks"
                      onChange={(event) => setAttackFilter(event.target.value)}
                      placeholder={copy.filterAttacks}
                      value={attackFilter}
                    />
                  </div>
                </div>
                <div className="attack-group-list">
                  {attackGroups.map(([category, categoryAttacks]) => (
                    <div className="attack-group" key={category}>
                      <div className="attack-group-title">
                        <strong>{categoryLabel(language, category)}</strong>
                        <span>{categoryAttacks.length}</span>
                      </div>
                      <div className="option-grid dense-options">
                        {categoryAttacks.map((attack) => (
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
                                {attack.strengths.length > 1 ? ` · ${attack.strengths.length} strengths` : ""}
                              </small>
                            </span>
                            {attack.requiresGpu ? <span className="badge warn">{t.common.gpu}</span> : null}
                            {attack.available === false ? <span className="badge error">Missing</span> : null}
                          </label>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
                {filteredAttacks.length === 0 ? <div className="empty compact-empty">{t.common.noData}</div> : null}
              </section>
            </div>

            <div className="modal-footer">
              <div className="config-estimate">
                <span>
                  {estimate.cellCount} {t.console.cells}
                </span>
                <span>
                  {estimate.sampleCount.toLocaleString()} {t.common.samples}
                </span>
                <span>
                  {estimate.imageOperationCount.toLocaleString()} {t.console.ops}
                </span>
              </div>
              <div className="toolbar">
                <button className="button" onClick={() => setIsCreateOpen(false)} type="button">
                  {copy.cancel}
                </button>
                <button className="button primary" disabled={!canSave} onClick={saveConfig} type="button">
                  <Save size={16} />
                  {copy.save}
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {renameTarget ? (
        <div aria-modal="true" className="modal-backdrop compact-modal-backdrop" role="dialog">
          <div className="rename-modal">
            <div className="modal-header">
              <div>
                <h2>{copy.renameTitle}</h2>
                <p>{renameTarget.id}</p>
              </div>
              <button aria-label="Close" className="icon-button" onClick={() => setRenameTarget(null)} type="button">
                <X size={16} />
              </button>
            </div>
            <div className="modal-body">
              <div className="field">
                <label htmlFor="rename-config-name">{t.configs.nameLabel}</label>
                <input
                  autoFocus
                  id="rename-config-name"
                  onChange={(event) => setRenameName(event.target.value)}
                  value={renameName}
                />
              </div>
            </div>
            <div className="modal-footer">
              <span />
              <div className="toolbar">
                <button className="button" onClick={() => setRenameTarget(null)} type="button">
                  {copy.cancel}
                </button>
                <button className="button primary" disabled={!renameName.trim()} onClick={renameConfig} type="button">
                  <Save size={16} />
                  {copy.save}
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </AppShell>
  );
}
