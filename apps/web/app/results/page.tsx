"use client";

import { useEffect, useMemo, useState } from "react";
import { BarChart3, Gauge, Trophy } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { BenchmarkRadar } from "@/components/BenchmarkRadar";
import { useLanguage } from "@/components/LanguageProvider";
import { RobustnessCurve } from "@/components/RobustnessCurve";
import { fetchRunResults, fetchRunScore, fetchRuns } from "@/lib/api";
import { formatMetric, rankAggregates } from "@/lib/insights";
import type { BenchmarkScore, RunResults } from "@/lib/types";

export default function ResultsPage() {
  const { t } = useLanguage();
  const [results, setResults] = useState<RunResults | null>(null);
  const [score, setScore] = useState<BenchmarkScore | null>(null);
  const [notice, setNotice] = useState("");
  const legacyRanking = useMemo(() => rankAggregates(results?.aggregates ?? []), [results]);
  const scoreRows = score?.leaderboardRows ?? [];

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
          setScore(nextResults.score ?? null);
          if (!nextResults.score) {
            fetchRunScore(nextResults.run.id)
              .then((scoreResponse) => {
                if (!cancelled) {
                  setScore(scoreResponse.score);
                }
              })
              .catch(() => undefined);
          }
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
            <h2>{t.results.benchmarkScore}</h2>
            <Trophy size={16} />
          </div>
          <div className="panel-body table-scroll">
            {notice ? <div className="risk warn">{notice}</div> : null}
            {scoreRows.length > 0 ? (
              <table className="table compact-table">
                <thead>
                  <tr>
                    <th>{t.common.rank}</th>
                    <th>{t.common.algorithm}</th>
                    <th>{t.common.wrs}</th>
                    <th>{t.results.cleanFidelity}</th>
                    <th>{t.results.nqd}</th>
                    <th>{t.common.coverage}</th>
                    <th>{t.runs.cells}</th>
                  </tr>
                </thead>
                <tbody>
                  {scoreRows.map((row) => (
                    <tr key={row.algorithmId}>
                      <td>{row.rank}</td>
                      <td>{row.algorithmId}</td>
                      <td>{row.wrs == null ? "n/a" : row.wrs.toFixed(1)}</td>
                      <td>{formatMetric(row.cleanFidelity)}</td>
                      <td>{formatMetric(row.avgNqd)}</td>
                      <td>
                        {row.coverage.coveredCategoryCount}/{row.coverage.requiredCategoryCount}
                      </td>
                      <td>{row.cellCount}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : results ? (
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
                  {legacyRanking.map((row) => (
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
            <h2>{t.results.radar}</h2>
            <BarChart3 size={16} />
          </div>
          <div className="panel-body">
            <BenchmarkRadar categories={score?.categoryScores ?? []} emptyText={t.console.needMultipleStrengths} />
          </div>
        </div>
      </section>

      <section className="results-grid">
        <div className="panel">
          <div className="panel-header">
            <h2>{t.console.robustnessCurves}</h2>
            <BarChart3 size={16} />
          </div>
          <div className="panel-body">
            <RobustnessCurve emptyText={t.console.needMultipleStrengths} results={results} score={score} />
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h2>{t.results.coverage}</h2>
            <Gauge size={16} />
          </div>
          <div className="panel-body coverage-grid">
            {score ? (
              score.categoryScores.map((category) => (
                <div className={category.covered ? "coverage-item covered" : "coverage-item"} key={category.key}>
                  <span>{category.label}</span>
                  <strong>{category.score == null ? "n/a" : category.score.toFixed(2)}</strong>
                </div>
              ))
            ) : (
              <div className="empty compact-empty">{t.common.noData}</div>
            )}
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
                  <span>{t.common.wrs}</span>
                  <strong>{score?.wrs == null ? "n/a" : score.wrs.toFixed(1)}</strong>
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
                    <th>{t.results.tprAtFpr}</th>
                    <th>{t.results.nqd}</th>
                    <th>{t.runs.cells}</th>
                  </tr>
                </thead>
                <tbody>
                  {results.aggregates.map((item) => {
                    const scorePoint = score?.curvePoints.find(
                      (point) =>
                        point.algorithmId === item.algorithmId &&
                        point.attackPresetId === item.attackPresetId &&
                        point.attackStrength === item.attackStrength
                    );
                    return (
                      <tr key={`${item.algorithmId}-${item.attackPresetId}-${item.attackStrength}`}>
                        <td>{item.algorithmId}</td>
                        <td>{item.attackPresetId}</td>
                        <td>{item.attackStrength}</td>
                        <td>{formatMetric(item.meanBitAccuracy)}</td>
                        <td>{formatMetric(item.meanBitErrorRate)}</td>
                        <td>{formatMetric(scorePoint?.yTprAtFpr)}</td>
                        <td>{formatMetric(scorePoint?.xNqd)}</td>
                        <td>
                          {item.succeededCells}/{item.cellCount}
                        </td>
                      </tr>
                    );
                  })}
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
                  <th>{t.results.tprAtFpr}</th>
                  <th>{t.runs.status}</th>
                  <th>{t.results.manifest}</th>
                </tr>
              </thead>
              <tbody>
                {results.cells.map((cell) => {
                  const scoring = cell.summary?.scoring as { tprAtFpr?: number } | undefined;
                  return (
                    <tr key={cell.id}>
                      <td>
                        {cell.datasetId} · {cell.watermarkMethod} · {cell.attackMethod}: {cell.attackStrength} ·{" "}
                        {cell.seed}
                      </td>
                      <td>{formatMetric(cell.bitAccuracy)}</td>
                      <td>{formatMetric(cell.bitErrorRate)}</td>
                      <td>{formatMetric(scoring?.tprAtFpr)}</td>
                      <td>{t.common.status[cell.status]}</td>
                      <td>{cell.manifestPath ?? "n/a"}</td>
                    </tr>
                  );
                })}
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
