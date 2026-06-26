"use client";

import { useMemo, useState } from "react";
import {
  Archive,
  Boxes,
  Braces,
  Database,
  Gauge,
  GitBranch,
  Play,
  RotateCcw,
  Shield
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { useLanguage } from "@/components/LanguageProvider";
import { localizedName } from "@/lib/i18n";
import { algorithms, artifacts, attacks, datasets } from "@/lib/mock-data";
import { estimateMatrix } from "@/lib/matrix";
import type { ExperimentSelection } from "@/lib/types";

const initialSelection: ExperimentSelection = {
  datasetIds: ["ds-demo-v1"],
  algorithmIds: ["alg-dct-qim-001"],
  attackPresetIds: ["atk-identity", "atk-jpeg-sweep"],
  seeds: [42],
  maxSamples: 64
};

function toggle(values: string[], value: string) {
  return values.includes(value) ? values.filter((item) => item !== value) : [...values, value];
}

export default function ExperimentConsole() {
  const { language, t } = useLanguage();
  const [selection, setSelection] = useState<ExperimentSelection>(initialSelection);
  const estimate = useMemo(() => estimateMatrix(selection, datasets, attacks), [selection]);
  const specPreview = {
    name: "robustness-console-draft",
    dataset_versions: selection.datasetIds,
    algorithm_versions: selection.algorithmIds,
    attack_presets: selection.attackPresetIds,
    seeds: selection.seeds,
    max_samples_per_dataset: selection.maxSamples,
    materialized_cells: estimate.cellCount
  };

  return (
    <AppShell active="console">
      <div className="topbar">
        <div className="title-block">
          <h1>{t.console.title}</h1>
          <p>{t.console.subtitle}</p>
        </div>
        <div className="toolbar">
          <button className="button" title={t.console.reset} onClick={() => setSelection(initialSelection)}>
            <RotateCcw size={16} />
          </button>
          <button className="button" title={t.console.save}>
            <Archive size={16} />
          </button>
          <button className="button primary" title={t.console.materialize}>
            <Play size={16} />
          </button>
        </div>
      </div>

      <section className="console-grid">
        <div className="panel">
          <div className="panel-header">
            <h2>{t.console.resources}</h2>
            <Boxes size={16} />
          </div>
          <div className="panel-body resource-list">
            {datasets.map((dataset) => (
              <div className="resource-item" key={dataset.id}>
                <div>
                  <strong>{localizedName(language, dataset.id, dataset.name)}</strong>
                  <span>
                    {dataset.sampleCount.toLocaleString()} {t.common.samples} · {dataset.version}
                  </span>
                </div>
                <span className="badge ok">{t.common.dataset}</span>
              </div>
            ))}
            {artifacts.map((artifact) => (
              <div className="resource-item" key={artifact.id}>
                <div>
                  <strong>{artifact.name}</strong>
                  <span>{artifact.size} · {artifact.checksum}</span>
                </div>
                <span className="badge">{t.common.weight}</span>
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
          </div>
        </div>

        <aside className="panel">
          <div className="panel-header">
            <h2>{t.console.inspector}</h2>
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
