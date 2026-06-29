"use client";

import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  BarChart3,
  Bug,
  CheckCircle2,
  Download,
  Filter,
  Gauge,
  Info,
  Layers3,
  Search,
  Trophy
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { BenchmarkRadar } from "@/components/BenchmarkRadar";
import { useLanguage } from "@/components/LanguageProvider";
import { RobustnessCurve } from "@/components/RobustnessCurve";
import { fetchRunResults, fetchRunScore, fetchRuns } from "@/lib/api";
import { formatMetric, rankAggregates, statusBadgeClass } from "@/lib/insights";
import type {
  BenchmarkCategoryScore,
  BenchmarkLeaderboardRow,
  BenchmarkScore,
  RunAggregate,
  RunResultCell,
  RunResults,
  RunStatus
} from "@/lib/types";

type ResultsTab = "overview" | "attack" | "quality" | "debug";
type StatusFilter = RunStatus | "all";

interface ScoringSummary {
  attackCategory?: string;
  attackMethod?: string;
  attackPresetId?: string;
  attackStrength?: number;
  cleanFidelity?: number;
  detectionThreshold?: number | null;
  elapsedMs?: number;
  empiricalFpr?: number | null;
  failureStage?: string;
  normalizedQualityDegradation?: number | null;
  practicalForWrs?: boolean;
  tprAtFpr?: number | null;
}

const RESULT_TABS: ResultsTab[] = ["overview", "attack", "quality", "debug"];

export default function ResultsPage() {
  const { t } = useLanguage();
  const [results, setResults] = useState<RunResults | null>(null);
  const [score, setScore] = useState<BenchmarkScore | null>(null);
  const [notice, setNotice] = useState("");
  const [activeTab, setActiveTab] = useState<ResultsTab>("overview");
  const [selectedAlgorithmIds, setSelectedAlgorithmIds] = useState<string[]>([]);
  const [debugStatus, setDebugStatus] = useState<StatusFilter>("all");
  const [debugAttack, setDebugAttack] = useState("all");
  const [debugSearch, setDebugSearch] = useState("");
  const [failedOnly, setFailedOnly] = useState(false);

  const legacyRanking = useMemo(() => rankAggregates(results?.aggregates ?? []), [results]);
  const scoreRows = score?.leaderboardRows ?? [];
  const algorithmIds = useMemo(() => collectAlgorithmIds(results, scoreRows, legacyRanking), [legacyRanking, results, scoreRows]);

  useEffect(() => {
    setSelectedAlgorithmIds((current) => {
      const next = current.filter((id) => algorithmIds.includes(id));
      return next.length > 0 ? next : algorithmIds.slice(0, 3);
    });
  }, [algorithmIds]);

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
          setNotice(t.results.noRealResults);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setNotice(t.results.apiUnavailable);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [t.results.apiUnavailable, t.results.noRealResults]);

  const selectedSet = useMemo(() => new Set(selectedAlgorithmIds), [selectedAlgorithmIds]);
  const selectedScoreRows = useMemo(
    () => scoreRows.filter((row) => selectedSet.size === 0 || selectedSet.has(row.algorithmId)),
    [scoreRows, selectedSet]
  );
  const selectedLegacyRows = useMemo(
    () => legacyRanking.filter((row) => selectedSet.size === 0 || selectedSet.has(row.algorithmId)),
    [legacyRanking, selectedSet]
  );
  const summary = useMemo(() => buildRunSummary(results, score), [results, score]);
  const aggregateRows = useMemo(
    () =>
      (results?.aggregates ?? [])
        .filter((item) => selectedSet.size === 0 || selectedSet.has(item.algorithmId))
        .map((item) => ({ aggregate: item, point: findScorePoint(score, item) })),
    [results, score, selectedSet]
  );
  const debugAttackOptions = useMemo(
    () => Array.from(new Set((results?.cells ?? []).map((cell) => cell.attackPresetId))).sort(),
    [results]
  );
  const debugCells = useMemo(
    () =>
      (results?.cells ?? []).filter((cell) => {
        const scoring = cellScoring(cell);
        const haystack = [
          cell.id,
          cell.cellKey,
          cell.datasetId,
          cell.algorithmId,
          cell.watermarkMethod,
          cell.attackPresetId,
          cell.attackMethod,
          cell.error,
          scoring?.failureStage
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();
        const queryMatch = !debugSearch.trim() || haystack.includes(debugSearch.trim().toLowerCase());
        const statusMatch = debugStatus === "all" || cell.status === debugStatus;
        const attackMatch = debugAttack === "all" || cell.attackPresetId === debugAttack;
        const algorithmMatch = selectedSet.size === 0 || selectedSet.has(cell.algorithmId);
        const failedMatch = !failedOnly || cell.status === "failed" || cell.status === "partially_failed" || Boolean(cell.error);
        return queryMatch && statusMatch && attackMatch && algorithmMatch && failedMatch;
      }),
    [debugAttack, debugSearch, debugStatus, failedOnly, results, selectedSet]
  );

  const radarSeries = selectedScoreRows.map((row) => ({
    id: row.algorithmId,
    label: row.algorithmId,
    categories: row.categoryScores
  }));
  const radarCategories = score?.categoryScores ?? selectedScoreRows[0]?.categoryScores ?? [];

  return (
    <AppShell active="results">
      <div className="topbar">
        <div className="title-block">
          <h1>{t.results.title}</h1>
          <p>{t.results.subtitle}</p>
        </div>
        <div className="toolbar">
          <button className="button" disabled={!results} onClick={() => exportResultsCsv(results, score)} type="button">
            <Download size={16} />
            {t.results.exportCsv}
          </button>
        </div>
      </div>

      {notice ? <div className="risk warn">{notice}</div> : null}

      <section className="results-summary-grid">
        <SummaryCard label={t.runs.run} value={summary.runId} meta={summary.configName} />
        <SummaryCard label={t.runs.status} value={summary.statusLabel} meta={`${summary.progress}% ${t.common.progress}`} />
        <SummaryCard label={t.common.wrs} value={summary.wrs} meta={summary.protocolStatus} />
        <SummaryCard label={t.common.coverage} value={summary.coverage} meta={summary.coverageMeta} />
        <SummaryCard label={t.results.completedCells} value={summary.completedCells} meta={`${summary.totalCells} ${t.runs.cells}`} />
        <SummaryCard label={t.common.samples} value={summary.sampleCount} meta={`${summary.algorithmCount} ${t.console.algorithms} · ${summary.attackCount} ${t.console.attacks}`} />
      </section>

      <section className="result-confidence-card">
        <div className={score?.officialEligible ? "confidence-icon ok" : "confidence-icon warn"}>
          {score?.officialEligible ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}
        </div>
        <div>
          <strong>{score?.officialEligible ? t.results.officialReady : t.results.provisionalWarning}</strong>
          <p>
            {score
              ? `${t.results.sampleFloor}: ${score.coverage.minSampleCount}/${score.officialMinSamples}. ${t.results.missingCategories}: ${
                  score.coverage.missingCategories.length ? score.coverage.missingCategories.join(", ") : t.results.none
                }`
              : t.results.noScoreYet}
          </p>
        </div>
      </section>

      <section className="result-tabs" aria-label={t.results.resultViews}>
        {RESULT_TABS.map((tab) => (
          <button
            className={activeTab === tab ? "result-tab active" : "result-tab"}
            key={tab}
            onClick={() => setActiveTab(tab)}
            type="button"
          >
            {tabIcon(tab)}
            {tabLabel(tab, t)}
          </button>
        ))}
      </section>

      {activeTab !== "debug" ? (
        <AlgorithmSelector
          algorithmIds={algorithmIds}
          selectedAlgorithmIds={selectedAlgorithmIds}
          setSelectedAlgorithmIds={setSelectedAlgorithmIds}
          title={t.results.selectedAlgorithms}
        />
      ) : null}

      {activeTab === "overview" ? (
        <OverviewTab
          legacyRows={selectedLegacyRows}
          radarCategories={radarCategories}
          radarSeries={radarSeries}
          results={results}
          score={score}
          scoreRows={selectedScoreRows}
        />
      ) : null}

      {activeTab === "attack" ? <AttackAnalysisTab aggregateRows={aggregateRows} score={score} /> : null}

      {activeTab === "quality" ? (
        <QualityTab results={results} score={score} selectedAlgorithmIds={selectedAlgorithmIds} scoreRows={selectedScoreRows} />
      ) : null}

      {activeTab === "debug" ? (
        <DebugTab
          attackOptions={debugAttackOptions}
          debugAttack={debugAttack}
          debugCells={debugCells}
          debugSearch={debugSearch}
          debugStatus={debugStatus}
          failedOnly={failedOnly}
          results={results}
          selectedAlgorithmIds={selectedAlgorithmIds}
          setDebugAttack={setDebugAttack}
          setDebugSearch={setDebugSearch}
          setDebugStatus={setDebugStatus}
          setFailedOnly={setFailedOnly}
          setSelectedAlgorithmIds={setSelectedAlgorithmIds}
        />
      ) : null}
    </AppShell>
  );

  function OverviewTab({
    legacyRows,
    radarCategories,
    radarSeries,
    results,
    score,
    scoreRows
  }: {
    legacyRows: ReturnType<typeof rankAggregates>;
    radarCategories: BenchmarkCategoryScore[];
    radarSeries: Array<{ id: string; label: string; categories: BenchmarkCategoryScore[] }>;
    results: RunResults | null;
    score: BenchmarkScore | null;
    scoreRows: BenchmarkLeaderboardRow[];
  }) {
    return (
      <>
        <section className="results-grid">
          <div className="panel">
            <div className="panel-header">
              <h2>{t.results.benchmarkScore}</h2>
              <Trophy size={16} />
            </div>
            <div className="panel-body table-scroll">
              {scoreRows.length > 0 ? (
                <ScoreRowsTable rows={scoreRows} />
              ) : results ? (
                <LegacyRowsTable rows={legacyRows} />
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
              <BenchmarkRadar categories={radarCategories} emptyText={t.console.needMultipleStrengths} series={radarSeries} />
            </div>
          </div>
        </section>

        <section className="results-grid">
          <div className="panel">
            <div className="panel-header">
              <h2>{t.results.scoreBreakdown}</h2>
              <Info size={16} />
            </div>
            <div className="panel-body score-breakdown-grid">
              {(score?.categoryScores ?? []).map((category) => (
                <div className={category.covered ? "score-breakdown-card covered" : "score-breakdown-card"} key={category.key}>
                  <span>{category.label}</span>
                  <strong>{category.score == null ? "n/a" : category.score.toFixed(2)}</strong>
                  <small>
                    {category.cellCount} {t.runs.cells} · NQD {formatMetric(category.meanNqd)}
                  </small>
                </div>
              ))}
              {!score ? <div className="empty compact-empty">{t.common.noData}</div> : null}
            </div>
          </div>

          <CoverageMatrix rows={scoreRows} score={score} />
        </section>
      </>
    );
  }

  function AttackAnalysisTab({
    aggregateRows,
    score
  }: {
    aggregateRows: Array<{ aggregate: RunAggregate; point: ReturnType<typeof findScorePoint> }>;
    score: BenchmarkScore | null;
  }) {
    return (
      <section className="panel results-detail-panel">
        <div className="panel-header">
          <h2>{t.results.attackAnalysis}</h2>
          <Gauge size={16} />
        </div>
        <div className="panel-body table-scroll">
          <table className="table">
            <thead>
              <tr>
                <th>{t.common.algorithm}</th>
                <th>{t.common.attackPreset}</th>
                <th>{t.results.category}</th>
                <th>{t.results.strength}</th>
                <th>Bit Acc.</th>
                <th>BER</th>
                <th>{t.results.tprAtFpr}</th>
                <th>{t.results.nqd}</th>
                <th>{t.runs.cells}</th>
              </tr>
            </thead>
            <tbody>
              {aggregateRows.map(({ aggregate, point }) => (
                <tr key={`${aggregate.algorithmId}-${aggregate.attackPresetId}-${aggregate.attackStrength}`}>
                  <td>{aggregate.algorithmId}</td>
                  <td>{aggregate.attackPresetId}</td>
                  <td>{point?.attackCategory ?? categoryForScore(score, aggregate.attackPresetId) ?? "n/a"}</td>
                  <td>{aggregate.attackStrength}</td>
                  <td>{formatMetric(aggregate.meanBitAccuracy)}</td>
                  <td>{formatMetric(aggregate.meanBitErrorRate)}</td>
                  <td>{formatMetric(point?.yTprAtFpr)}</td>
                  <td>{formatMetric(point?.xNqd)}</td>
                  <td>
                    {aggregate.succeededCells}/{aggregate.cellCount}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {aggregateRows.length === 0 ? <div className="empty compact-empty">{t.common.noData}</div> : null}
        </div>
      </section>
    );
  }

  function QualityTab({
    results,
    score,
    selectedAlgorithmIds,
    scoreRows
  }: {
    results: RunResults | null;
    score: BenchmarkScore | null;
    selectedAlgorithmIds: string[];
    scoreRows: BenchmarkLeaderboardRow[];
  }) {
    return (
      <>
        <section className="results-grid">
          <div className="panel">
            <div className="panel-header">
              <h2>{t.results.qualityRobustness}</h2>
              <BarChart3 size={16} />
            </div>
            <div className="panel-body">
              <RobustnessCurve
                emptyText={t.console.needMultipleStrengths}
                results={results}
                score={score}
                selectedAlgorithmIds={selectedAlgorithmIds}
              />
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <h2>{t.results.auxiliaryMetrics}</h2>
              <Gauge size={16} />
            </div>
            <div className="panel-body score-breakdown-grid">
              {scoreRows.map((row) => (
                <div className="score-breakdown-card" key={row.algorithmId}>
                  <span>{row.algorithmId}</span>
                  <strong>{row.wrs == null ? "n/a" : row.wrs.toFixed(1)}</strong>
                  <small>
                    {t.results.cleanFidelity} {formatMetric(row.cleanFidelity)} · NQD {formatMetric(row.avgNqd)} ·{" "}
                    {row.runtimeMs == null ? "n/a" : `${(row.runtimeMs / 1000).toFixed(2)}s`}
                  </small>
                </div>
              ))}
              {scoreRows.length === 0 ? <div className="empty compact-empty">{t.common.noData}</div> : null}
            </div>
          </div>
        </section>

        <section className="panel results-detail-panel">
          <div className="panel-header">
            <h2>{t.results.qualityPoints}</h2>
          </div>
          <div className="panel-body table-scroll">
            <table className="table compact-table">
              <thead>
                <tr>
                  <th>{t.common.algorithm}</th>
                  <th>{t.common.attackPreset}</th>
                  <th>{t.results.category}</th>
                  <th>{t.results.strength}</th>
                  <th>{t.results.tprAtFpr}</th>
                  <th>{t.results.nqd}</th>
                </tr>
              </thead>
              <tbody>
                {(score?.curvePoints ?? [])
                  .filter((point) => selectedAlgorithmIds.length === 0 || selectedAlgorithmIds.includes(point.algorithmId))
                  .map((point) => (
                    <tr key={`${point.algorithmId}-${point.attackPresetId}-${point.attackStrength}-${point.xNqd}`}>
                      <td>{point.algorithmId}</td>
                      <td>{point.attackPresetId}</td>
                      <td>{point.attackCategory}</td>
                      <td>{point.attackStrength}</td>
                      <td>{formatMetric(point.yTprAtFpr)}</td>
                      <td>{formatMetric(point.xNqd)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
            {!score?.curvePoints?.length ? <div className="empty compact-empty">{t.common.noData}</div> : null}
          </div>
        </section>
      </>
    );
  }

  function DebugTab({
    attackOptions,
    debugAttack,
    debugCells,
    debugSearch,
    debugStatus,
    failedOnly,
    results,
    selectedAlgorithmIds,
    setDebugAttack,
    setDebugSearch,
    setDebugStatus,
    setFailedOnly,
    setSelectedAlgorithmIds
  }: {
    attackOptions: string[];
    debugAttack: string;
    debugCells: RunResultCell[];
    debugSearch: string;
    debugStatus: StatusFilter;
    failedOnly: boolean;
    results: RunResults | null;
    selectedAlgorithmIds: string[];
    setDebugAttack: (value: string) => void;
    setDebugSearch: (value: string) => void;
    setDebugStatus: (value: StatusFilter) => void;
    setFailedOnly: (value: boolean) => void;
    setSelectedAlgorithmIds: (value: string[] | ((current: string[]) => string[])) => void;
  }) {
    return (
      <>
        <AlgorithmSelector
          algorithmIds={algorithmIds}
          selectedAlgorithmIds={selectedAlgorithmIds}
          setSelectedAlgorithmIds={setSelectedAlgorithmIds}
          title={t.results.selectedAlgorithms}
        />
        <section className="panel results-detail-panel">
          <div className="panel-header">
            <h2>{t.results.debugCells}</h2>
            <Bug size={16} />
          </div>
          <div className="panel-body">
            <div className="debug-filter-bar">
              <div className="field-icon-input">
                <Search size={15} />
                <input
                  aria-label={t.results.searchCells}
                  onChange={(event) => setDebugSearch(event.target.value)}
                  placeholder={t.results.searchCells}
                  value={debugSearch}
                />
              </div>
              <select onChange={(event) => setDebugStatus(event.target.value as StatusFilter)} value={debugStatus}>
                <option value="all">{t.results.allStatuses}</option>
                <option value="succeeded">{t.common.status.succeeded}</option>
                <option value="failed">{t.common.status.failed}</option>
                <option value="partially_failed">{t.common.status.partially_failed}</option>
                <option value="cancelled">{t.common.status.cancelled}</option>
              </select>
              <select onChange={(event) => setDebugAttack(event.target.value)} value={debugAttack}>
                <option value="all">{t.results.allAttacks}</option>
                {attackOptions.map((attack) => (
                  <option key={attack} value={attack}>
                    {attack}
                  </option>
                ))}
              </select>
              <label className="toggle-row inline-toggle">
                <input checked={failedOnly} onChange={(event) => setFailedOnly(event.target.checked)} type="checkbox" />
                <span>{t.results.failedOnly}</span>
              </label>
            </div>

            <div className="table-scroll">
              <table className="table">
                <thead>
                  <tr>
                    <th>{t.results.cell}</th>
                    <th>{t.common.algorithm}</th>
                    <th>{t.common.attackPreset}</th>
                    <th>Bit Acc.</th>
                    <th>BER</th>
                    <th>{t.results.tprAtFpr}</th>
                    <th>{t.runs.status}</th>
                    <th>{t.results.failureStage}</th>
                    <th>{t.results.manifest}</th>
                  </tr>
                </thead>
                <tbody>
                  {debugCells.map((cell) => {
                    const scoring = cellScoring(cell);
                    return (
                      <tr key={cell.id}>
                        <td>
                          <strong>{cell.datasetId}</strong>
                          <span className="subtle-cell">
                            {cell.watermarkMethod} · {cell.attackMethod}: {cell.attackStrength} · seed {cell.seed}
                          </span>
                        </td>
                        <td>{cell.algorithmId}</td>
                        <td>{cell.attackPresetId}</td>
                        <td>{formatMetric(cell.bitAccuracy)}</td>
                        <td>{formatMetric(cell.bitErrorRate)}</td>
                        <td>{formatMetric(scoring?.tprAtFpr)}</td>
                        <td>
                          <span className={statusBadgeClass(cell.status)}>{t.common.status[cell.status]}</span>
                        </td>
                        <td>{scoring?.failureStage ?? (cell.error ? "unknown" : "n/a")}</td>
                        <td className="path-cell" title={cell.manifestPath ?? undefined}>
                          {cell.manifestPath ?? "n/a"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {debugCells.length === 0 ? <div className="empty compact-empty">{t.common.noData}</div> : null}
            </div>

            {results?.run.logPath ? (
              <div className="debug-detail">
                <strong>{t.runs.logPath}</strong>
                <code>{results.run.logPath}</code>
              </div>
            ) : null}
          </div>
        </section>
      </>
    );
  }

  function ScoreRowsTable({ rows }: { rows: BenchmarkLeaderboardRow[] }) {
    if (rows.length === 0) {
      return <div className="empty compact-empty">{t.common.noData}</div>;
    }
    return (
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
          {rows.map((row) => (
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
    );
  }

  function LegacyRowsTable({ rows }: { rows: ReturnType<typeof rankAggregates> }) {
    if (rows.length === 0) {
      return <div className="empty compact-empty">{t.common.noData}</div>;
    }
    return (
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
          {rows.map((row) => (
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
    );
  }

  function CoverageMatrix({ rows, score }: { rows: BenchmarkLeaderboardRow[]; score: BenchmarkScore | null }) {
    const categories = score?.categoryScores ?? [];
    return (
      <div className="panel">
        <div className="panel-header">
          <h2>{t.results.coverageMatrix}</h2>
          <Layers3 size={16} />
        </div>
        <div className="panel-body coverage-matrix-wrap">
          {categories.length > 0 && rows.length > 0 ? (
            <div className="coverage-matrix">
              <div className="coverage-matrix-row header">
                <span>{t.results.category}</span>
                {rows.map((row) => (
                  <strong key={row.algorithmId}>{row.algorithmId}</strong>
                ))}
              </div>
              {categories.map((category) => (
                <div className="coverage-matrix-row" key={category.key}>
                  <span>{category.label}</span>
                  {rows.map((row) => {
                    const algorithmCategory = row.categoryScores.find((item) => item.key === category.key);
                    return (
                      <i
                        className={algorithmCategory?.covered ? "coverage-cell covered" : "coverage-cell"}
                        key={`${row.algorithmId}-${category.key}`}
                        title={algorithmCategory?.score == null ? "n/a" : algorithmCategory.score.toFixed(2)}
                      >
                        {algorithmCategory?.covered ? "✓" : "—"}
                      </i>
                    );
                  })}
                </div>
              ))}
            </div>
          ) : (
            <div className="empty compact-empty">{t.common.noData}</div>
          )}
        </div>
      </div>
    );
  }

  function AlgorithmSelector({
    algorithmIds,
    selectedAlgorithmIds,
    setSelectedAlgorithmIds,
    title
  }: {
    algorithmIds: string[];
    selectedAlgorithmIds: string[];
    setSelectedAlgorithmIds: (value: string[] | ((current: string[]) => string[])) => void;
    title: string;
  }) {
    if (algorithmIds.length === 0) {
      return null;
    }
    return (
      <section className="algorithm-selector">
        <div>
          <Filter size={15} />
          <strong>{title}</strong>
        </div>
        <div className="selector-chip-row">
          {algorithmIds.map((algorithmId) => (
            <button
              className={selectedAlgorithmIds.includes(algorithmId) ? "selector-chip active" : "selector-chip"}
              key={algorithmId}
              onClick={() =>
                setSelectedAlgorithmIds((current) =>
                  current.includes(algorithmId)
                    ? current.filter((id) => id !== algorithmId)
                    : [...current, algorithmId]
                )
              }
              type="button"
            >
              {algorithmId}
            </button>
          ))}
        </div>
      </section>
    );
  }

  function SummaryCard({ label, meta, value }: { label: string; meta: string; value: string }) {
    return (
      <div className="result-summary-card">
        <span>{label}</span>
        <strong title={value}>{value}</strong>
        <small title={meta}>{meta}</small>
      </div>
    );
  }
}

function buildRunSummary(results: RunResults | null, score: BenchmarkScore | null) {
  const cells = results?.cells ?? [];
  const algorithmCount = new Set(cells.map((cell) => cell.algorithmId)).size;
  const attackCount = new Set(cells.map((cell) => cell.attackPresetId)).size;
  const sampleCount = cells.reduce((total, cell) => total + cell.sampleCount, 0);
  const succeededCells = cells.filter((cell) => cell.status === "succeeded").length;
  const failedCells = cells.filter((cell) => cell.status === "failed" || cell.status === "partially_failed").length;
  const totalCells = cells.length || results?.run.cells || 0;
  return {
    algorithmCount,
    attackCount,
    completedCells: `${succeededCells}/${totalCells}`,
    configName: results?.run.configName ?? "n/a",
    coverage: score ? `${score.coverage.coveredCategoryCount}/${score.coverage.requiredCategoryCount}` : "n/a",
    coverageMeta: score ? `${Math.round(score.coverage.coverageRatio * 100)}%` : "n/a",
    failedCells,
    progress: Math.round(results?.run.progress ?? 0).toString(),
    protocolStatus: score?.status ?? "n/a",
    runId: results?.run.id ?? "n/a",
    sampleCount: sampleCount.toLocaleString(),
    statusLabel: results?.run.status ?? "n/a",
    totalCells: totalCells.toString(),
    wrs: score?.wrs == null ? "n/a" : score.wrs.toFixed(1)
  };
}

function collectAlgorithmIds(
  results: RunResults | null,
  scoreRows: BenchmarkLeaderboardRow[],
  legacyRows: ReturnType<typeof rankAggregates>
) {
  const ids = new Set<string>();
  scoreRows.forEach((row) => ids.add(row.algorithmId));
  legacyRows.forEach((row) => ids.add(row.algorithmId));
  results?.aggregates.forEach((item) => ids.add(item.algorithmId));
  results?.cells.forEach((cell) => ids.add(cell.algorithmId));
  return Array.from(ids).sort();
}

function cellScoring(cell: RunResultCell): ScoringSummary | undefined {
  return cell.summary?.scoring as ScoringSummary | undefined;
}

function findScorePoint(score: BenchmarkScore | null, item: RunAggregate) {
  return score?.curvePoints.find(
    (point) =>
      point.algorithmId === item.algorithmId &&
      point.attackPresetId === item.attackPresetId &&
      point.attackStrength === item.attackStrength
  );
}

function categoryForScore(score: BenchmarkScore | null, attackPresetId: string) {
  return score?.curvePoints.find((point) => point.attackPresetId === attackPresetId)?.attackCategory;
}

function tabIcon(tab: ResultsTab) {
  if (tab === "overview") {
    return <Trophy size={15} />;
  }
  if (tab === "attack") {
    return <Gauge size={15} />;
  }
  if (tab === "quality") {
    return <BarChart3 size={15} />;
  }
  return <Bug size={15} />;
}

function tabLabel(tab: ResultsTab, t: ReturnType<typeof useLanguage>["t"]) {
  if (tab === "overview") {
    return t.results.overview;
  }
  if (tab === "attack") {
    return t.results.attackAnalysis;
  }
  if (tab === "quality") {
    return t.results.qualityRobustness;
  }
  return t.results.debugCells;
}

function exportResultsCsv(results: RunResults | null, score: BenchmarkScore | null) {
  if (!results) {
    return;
  }
  const rows = [
    ["run_id", "cell_id", "algorithm_id", "attack_preset_id", "attack_strength", "status", "bit_accuracy", "ber", "tpr_at_fpr", "nqd", "manifest_path"],
    ...results.cells.map((cell) => {
      const scoring = cellScoring(cell);
      return [
        results.run.id,
        cell.id,
        cell.algorithmId,
        cell.attackPresetId,
        String(cell.attackStrength),
        cell.status,
        String(cell.bitAccuracy ?? ""),
        String(cell.bitErrorRate ?? ""),
        String(scoring?.tprAtFpr ?? ""),
        String(scoring?.normalizedQualityDegradation ?? ""),
        cell.manifestPath ?? ""
      ];
    })
  ];
  if (score) {
    rows.push([]);
    rows.push(["protocol_id", score.protocolId, "status", score.status, "wrs", String(score.wrs ?? "")]);
  }
  const csv = rows.map((row) => row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${results.run.id}-results.csv`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
