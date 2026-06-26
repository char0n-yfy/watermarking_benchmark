"use client";

import { AppShell } from "@/components/AppShell";
import { useLanguage } from "@/components/LanguageProvider";
import { localizedDate, localizedName } from "@/lib/i18n";
import { recentRuns } from "@/lib/mock-data";

export default function RunsPage() {
  const { language, t } = useLanguage();

  return (
    <AppShell active="runs">
      <div className="topbar">
        <div className="title-block">
          <h1>{t.runs.title}</h1>
          <p>{t.runs.subtitle}</p>
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
                <th>{t.runs.status}</th>
                <th>{t.runs.cells}</th>
                <th>{t.runs.updated}</th>
              </tr>
            </thead>
            <tbody>
              {recentRuns.map((run) => (
                <tr key={run.id}>
                  <td>{localizedName(language, run.id, run.name)}</td>
                  <td>
                    <span className={run.status === "succeeded" ? "badge ok" : "badge warn"}>
                      {t.common.status[run.status]}
                    </span>
                  </td>
                  <td>{run.cells}</td>
                  <td>{localizedDate(language, run.updatedAt)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </AppShell>
  );
}
