"use client";

import { AppShell } from "@/components/AppShell";
import { useLanguage } from "@/components/LanguageProvider";

const rows = [
  { cell: "demo · dct-qim · identity · 42", tpr: "0.99", psnr: "inf", manifest: "attack_manifest.json" },
  { cell: "demo · dct-qim · jpeg:0.50 · 42", tpr: "0.87", psnr: "32.4", manifest: "attack_manifest.json" },
  { cell: "demo · dct-qim · blur:0.40 · 42", tpr: "0.74", psnr: "28.9", manifest: "attack_manifest.json" }
];

export default function ResultsPage() {
  const { t } = useLanguage();

  return (
    <AppShell active="results">
      <div className="topbar">
        <div className="title-block">
          <h1>{t.results.title}</h1>
          <p>{t.results.subtitle}</p>
        </div>
      </div>
      <div className="panel">
        <div className="panel-header">
          <h2>{t.results.matrixCells}</h2>
        </div>
        <div className="panel-body">
          <table className="table">
            <thead>
              <tr>
                <th>{t.results.cell}</th>
                <th>TPR</th>
                <th>PSNR</th>
                <th>{t.results.manifest}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.cell}>
                  <td>{row.cell}</td>
                  <td>{row.tpr}</td>
                  <td>{row.psnr}</td>
                  <td>{row.manifest}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </AppShell>
  );
}
