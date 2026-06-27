"use client";

import { useEffect, useMemo, useState } from "react";
import { BarChart3, Trophy } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { useLanguage } from "@/components/LanguageProvider";
import { RobustnessCurve } from "@/components/RobustnessCurve";
import { fetchRunResults, fetchRuns } from "@/lib/api";
import { formatMetric, rankAggregates } from "@/lib/insights";
import type { RunResults } from "@/lib/types";

export default function ResultsPage() {
  const { t } = useLanguage();
  const [results, setResults] = useState<RunResults | null>(null);
  const [notice, setNotice] = useState("");
  const ranking = useMemo(() => rankAggregates(results?.aggregates ?? []), [results]);

  useEffect(() => {
    let cancelled = false;
    fetchRuns()
      .then((runs) => {
        const latest =
          runs.find((run) => run.status === "succeeded" || run.status === "partially_failed") ??
          runs.find((run) => run.status !== "queued") ??
          runs[0];
        return latest ? fetchRunResults(latest.id) : null;
      })
      .then((nextResults) => {
        if (!cancelled && nextResults) {
          setResults(nextResults);
        }
        if (!cancelled && !nextResults) {
          setNotice("暂无真实运行结果，请先在 Runs 页面提交并启动 worker。");
        }
      })
      .catch(() => {
        if (!cancelled) {
          setNotice("API 未启动或暂无真实运行结果。");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <AppShell active="results">
      <div className="topbar">
        <div className="title-block">
          <h1>{t.results.title}</h1>
          <p>{t.results.subtitle}</p>
        </div>
      </div>

      <section className="results-grid">
        <div className="panel">
          <div className="panel-header">
            <h2>{t.results.runLeaderboard}</h2>
            <Trophy size={16} />
          </div>
          <div className="panel-body table-scroll">
            {notice ? <div className="risk warn">{notice}</div> : null}
            {results ? (
              <table className="table compact-table">
                <thead>
                  <tr>
                    <th>{t.common.rank}</th>
                    <th>{t.common.algorithm}</th>
                    <th>{t.common.overallScore}</th>
                    <th>Bit Acc.</th>
                    <th>BER</th>
                    <th>{t.runs.cells}</th>
                  </tr>
                </thead>
                <tbody>
                  {ranking.map((row) => (
                    <tr key={row.algorithmId}>
                      <td>{row.rank}</td>
                      <td>{row.algorithmId}</td>
                      <td>{formatMetric(row.overallScore)}</td>
                      <td>{formatMetric(row.meanBitAccuracy)}</td>
                      <td>{formatMetric(row.meanBitErrorRate)}</td>
                      <td>{row.cellCount}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="empty compact-empty">{t.common.noData}</div>
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h2>{t.console.robustnessCurves}</h2>
            <BarChart3 size={16} />
          </div>
          <div className="panel-body">
            <RobustnessCurve emptyText={t.console.needMultipleStrengths} results={results} />
          </div>
        </div>
      </section>

      <div className="panel results-detail-panel">
        <div className="panel-header">
          <h2>{t.results.aggregates}</h2>
        </div>
        <div className="panel-body">
          {results ? (
            <>
              <div className="stats">
                <div className="stat">
                  <span>{t.runs.run}</span>
                  <strong>{results.run.id}</strong>
                </div>
                <div className="stat">
                  <span>{t.runs.status}</span>
                  <strong>{t.common.status[results.run.status]}</strong>
                </div>
                <div className="stat">
                  <span>{t.console.cells}</span>
                  <strong>{results.cells.length}</strong>
                </div>
              </div>
              <table className="table">
                <thead>
                  <tr>
                    <th>{t.common.algorithm}</th>
                    <th>{t.common.attackPreset}</th>
                    <th>Strength</th>
                    <th>Mean Bit Acc.</th>
                    <th>Mean BER</th>
                    <th>{t.runs.cells}</th>
                  </tr>
                </thead>
                <tbody>
                  {results.aggregates.map((item) => (
                    <tr key={`${item.algorithmId}-${item.attackPresetId}-${item.attackStrength}`}>
                      <td>{item.algorithmId}</td>
                      <td>{item.attackPresetId}</td>
                      <td>{item.attackStrength}</td>
                      <td>{formatMetric(item.meanBitAccuracy)}</td>
                      <td>{formatMetric(item.meanBitErrorRate)}</td>
                      <td>
                        {item.succeededCells}/{item.cellCount}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {results.aggregates.length === 0 ? <div className="empty">{t.common.noData}</div> : null}
            </>
          ) : (
            <div className="empty">{t.common.noData}</div>
          )}
        </div>
      </div>

      <div className="panel results-detail-panel">
        <div className="panel-header">
          <h2>{t.results.matrixCells}</h2>
        </div>
        <div className="panel-body table-scroll">
          {results ? (
            <table className="table">
              <thead>
                <tr>
                  <th>{t.results.cell}</th>
                  <th>Bit Acc.</th>
                  <th>BER</th>
                  <th>{t.runs.status}</th>
                  <th>{t.results.manifest}</th>
                </tr>
              </thead>
              <tbody>
                {results.cells.map((cell) => (
                  <tr key={cell.id}>
                    <td>
                      {cell.datasetId} · {cell.watermarkMethod} · {cell.attackMethod}:
                      {cell.attackStrength} · {cell.seed}
                    </td>
                    <td>{formatMetric(cell.bitAccuracy)}</td>
                    <td>{formatMetric(cell.bitErrorRate)}</td>
                    <td>{t.common.status[cell.status]}</td>
                    <td>{cell.manifestPath ?? "n/a"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="empty">{t.common.noData}</div>
          )}
        </div>
      </div>
    </AppShell>
  );
}
