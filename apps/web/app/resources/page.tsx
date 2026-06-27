"use client";

import { useEffect, useState } from "react";
import { Database, PackageCheck } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { useLanguage } from "@/components/LanguageProvider";
import { fetchAlgorithms, fetchAttacks, fetchDatasets } from "@/lib/api";
import { localizedName } from "@/lib/i18n";
import {
  algorithms as fallbackAlgorithms,
  artifacts,
  attacks as fallbackAttacks,
  datasets as fallbackDatasets
} from "@/lib/mock-data";
import type { AlgorithmVersion, AttackPreset, DatasetVersion } from "@/lib/types";

export default function ResourcesPage() {
  const { language, t } = useLanguage();
  const [datasets, setDatasets] = useState<DatasetVersion[]>(fallbackDatasets);
  const [algorithms, setAlgorithms] = useState<AlgorithmVersion[]>(fallbackAlgorithms);
  const [attacks, setAttacks] = useState<AttackPreset[]>(fallbackAttacks);

  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchDatasets(), fetchAlgorithms(), fetchAttacks()])
      .then(([apiDatasets, apiAlgorithms, apiAttacks]) => {
        if (cancelled) {
          return;
        }
        setDatasets(apiDatasets);
        setAlgorithms(apiAlgorithms.length > 0 ? apiAlgorithms : fallbackAlgorithms);
        setAttacks(apiAttacks.length > 0 ? apiAttacks : fallbackAttacks);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <AppShell active="resources">
      <div className="topbar">
        <div className="title-block">
          <h1>{t.resources.title}</h1>
          <p>{t.resources.subtitle}</p>
        </div>
      </div>

      <section className="panel">
        <div className="panel-header">
          <h2>{t.console.datasetsSummary}</h2>
          <Database size={16} />
        </div>
        <div className="panel-body table-scroll">
          <table className="table compact-table">
            <thead>
              <tr>
                <th>{t.resources.name}</th>
                <th>{t.common.samples}</th>
                <th>Version</th>
                <th>Path</th>
                <th>{t.resources.status}</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((dataset) => (
                <tr key={dataset.id}>
                  <td>{localizedName(language, dataset.id, dataset.name)}</td>
                  <td>{dataset.sampleCount.toLocaleString()}</td>
                  <td>{dataset.version}</td>
                  <td className="path-cell">{dataset.path ?? "n/a"}</td>
                  <td>
                    <span className="badge ok">{t.common.enabled}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {datasets.length === 0 ? <div className="empty compact-empty">{t.common.noData}</div> : null}
        </div>
      </section>

      <section className="panel resources-detail-panel">
        <div className="panel-header">
          <h2>{t.resources.catalog}</h2>
          <PackageCheck size={16} />
        </div>
        <div className="panel-body table-scroll">
          <table className="table">
            <thead>
              <tr>
                <th>{t.resources.name}</th>
                <th>{t.resources.type}</th>
                <th>{t.resources.status}</th>
                <th>{t.resources.details}</th>
              </tr>
            </thead>
            <tbody>
              {algorithms.map((algorithm) => (
                <tr key={algorithm.id}>
                  <td>{algorithm.name}</td>
                  <td>{t.common.algorithm}</td>
                  <td>
                    <span className={algorithm.status === "enabled" ? "badge ok" : "badge warn"}>
                      {t.common.status[algorithm.status]}
                    </span>
                  </td>
                  <td>{algorithm.requiresGpu ? t.common.gpu : t.common.cpu}</td>
                </tr>
              ))}
              {attacks.map((attack) => (
                <tr key={attack.id}>
                  <td>{localizedName(language, attack.id, attack.name)}</td>
                  <td>{t.common.attackPreset}</td>
                  <td>
                    <span className="badge ok">{t.common.enabled}</span>
                  </td>
                  <td>{attack.method}</td>
                </tr>
              ))}
              {artifacts.map((artifact) => (
                <tr key={artifact.id}>
                  <td>{artifact.name}</td>
                  <td>{t.common.weight}</td>
                  <td>
                    <span className="badge">{t.common.indexed}</span>
                  </td>
                  <td>{artifact.size}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </AppShell>
  );
}
