"use client";

import { AppShell } from "@/components/AppShell";
import { useLanguage } from "@/components/LanguageProvider";
import { localizedName } from "@/lib/i18n";
import { algorithms, artifacts, attacks, datasets } from "@/lib/mock-data";

export default function ResourcesPage() {
  const { language, t } = useLanguage();

  return (
    <AppShell active="resources">
      <div className="topbar">
        <div className="title-block">
          <h1>{t.resources.title}</h1>
          <p>{t.resources.subtitle}</p>
        </div>
      </div>
      <div className="panel">
        <div className="panel-header">
          <h2>{t.resources.catalog}</h2>
        </div>
        <div className="panel-body">
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
              {datasets.map((dataset) => (
                <tr key={dataset.id}>
                  <td>{localizedName(language, dataset.id, dataset.name)}</td>
                  <td>{t.common.dataset}</td>
                  <td>
                    <span className="badge ok">{t.common.versioned}</span>
                  </td>
                  <td>
                    {dataset.sampleCount.toLocaleString()} {t.common.samples}
                  </td>
                </tr>
              ))}
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
      </div>
    </AppShell>
  );
}
