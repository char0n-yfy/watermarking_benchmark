"use client";

import { useEffect, useMemo, useState } from "react";
import { BarChart3, Download, Info, LoaderCircle, TriangleAlert } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { PageState } from "@/components/PageState";
import { curveDomainColor } from "@/components/RobustnessCurve";
import { useLanguage } from "@/components/LanguageProvider";
import { chartBarFill } from "@/lib/chart-colors";
import { fetchAlgorithms, fetchLeaderboard, fetchRuns } from "@/lib/api";
import { resolveWatermarkDisplayName } from "@/lib/watermark-display";
import type { AlgorithmVersion, BenchmarkLeaderboardRow, DemoRunRecord, LeaderboardResponse } from "@/lib/types";

type RankingMetric = "robustness" | "complexity" | "fidelity" | "composite";
type CompositeWeightKey = Exclude<RankingMetric, "composite">;
type CompositeWeights = Record<CompositeWeightKey, number>;

const RANKING_METRICS: RankingMetric[] = ["robustness", "complexity", "fidelity", "composite"];
const COMPOSITE_WEIGHT_KEYS: CompositeWeightKey[] = ["robustness", "fidelity", "complexity"];
const DEFAULT_COMPOSITE_WEIGHTS: CompositeWeights = {
  robustness: 50,
  fidelity: 30,
  complexity: 20
} as const;
const COMPOSITE_WEIGHTS_STORAGE_KEY = "wm-bench-leaderboard-composite-weights";
const DEMO_DEFAULT_RUN_ID = "run_20260704_173157_b4b766";

export default function LeaderboardPage() {
  const { language, t } = useLanguage();
  const [leaderboard, setLeaderboard] = useState<LeaderboardResponse | null>(null);
  const [algorithms, setAlgorithms] = useState<AlgorithmVersion[]>([]);
  const [runs, setRuns] = useState<DemoRunRecord[]>([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [rankingMetric, setRankingMetric] = useState<RankingMetric>("composite");
  const [compositeWeights, setCompositeWeights] = useState<CompositeWeights>({ ...DEFAULT_COMPOSITE_WEIGHTS });
  const [compositeWeightsLoaded, setCompositeWeightsLoaded] = useState(false);
  const [selectedRankingRow, setSelectedRankingRow] = useState<BenchmarkLeaderboardRow | null>(null);
  const [runsLoading, setRunsLoading] = useState(true);
  const [leaderboardState, setLeaderboardState] = useState<"idle" | "loading" | "ready" | "error">("idle");

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(COMPOSITE_WEIGHTS_STORAGE_KEY);
      if (stored) {
        setCompositeWeights(normalizeCompositeWeights(JSON.parse(stored)));
      }
    } catch {
      // Ignore malformed or unavailable browser storage and keep the defaults.
    } finally {
      setCompositeWeightsLoaded(true);
    }
  }, []);

  useEffect(() => {
    if (!compositeWeightsLoaded) {
      return;
    }
    try {
      window.localStorage.setItem(COMPOSITE_WEIGHTS_STORAGE_KEY, JSON.stringify(compositeWeights));
    } catch {
      // Weight adjustment still works when browser storage is unavailable.
    }
  }, [compositeWeights, compositeWeightsLoaded]);

  useEffect(() => {
    let cancelled = false;
    setRunsLoading(true);
    Promise.all([fetchRuns(), fetchAlgorithms().catch(() => [] as AlgorithmVersion[])])
      .then(([nextRuns, nextAlgorithms]) => {
        if (cancelled) {
          return;
        }
        const leaderboardRuns = nextRuns.filter(
          (run) => run.status === "succeeded" || run.status === "partially_failed"
        );
        setRuns(leaderboardRuns);
        setSelectedRunId((current) => {
          if (current && leaderboardRuns.some((run) => run.id === current)) {
            return current;
          }
          return (
            leaderboardRuns.find((run) => run.id === DEMO_DEFAULT_RUN_ID)?.id ??
            leaderboardRuns[0]?.id ??
            ""
          );
        });
        setAlgorithms(nextAlgorithms);
        setRunsLoading(false);
      })
      .catch(() => {
        if (!cancelled) {
          setRunsLoading(false);
          setLeaderboardState("error");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedRunId) {
      setLeaderboard(null);
      setSelectedRankingRow(null);
      if (!runsLoading) {
        setLeaderboardState((current) => (current === "error" ? current : "idle"));
      }
      return;
    }
    let cancelled = false;
    setLeaderboard(null);
    setSelectedRankingRow(null);
    setLeaderboardState("loading");
    fetchLeaderboard("wrs-v2-detection-v1", selectedRunId)
      .then((nextLeaderboard) => {
        if (!cancelled) {
          setLeaderboard(nextLeaderboard);
          setLeaderboardState("ready");
        }
      })
      .catch(() => {
        if (!cancelled) {
          setLeaderboard(null);
          setLeaderboardState("error");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [runsLoading, selectedRunId]);

  const rows = leaderboard?.rows ?? [];
  const algorithmNames = useMemo(() => buildAlgorithmNames(algorithms), [algorithms]);
  const algorithmColorDomain = useMemo(() => rows.map((row) => row.algorithmId), [rows]);

  return (
    <AppShell active="leaderboard">
      <div className="topbar run-picker-topbar">
        <div className="title-block">
          <h1>{t.leaderboard.title}</h1>
          <p>{t.leaderboard.subtitle}</p>
        </div>
        <div className="toolbar run-picker-toolbar">
          <select
            aria-label={language === "zh" ? "选择实验" : "Select experiment"}
            className="run-result-select"
            disabled={runs.length === 0}
            onChange={(event) => setSelectedRunId(event.target.value)}
            value={selectedRunId}
          >
            {runs.map((run) => (
              <option key={run.id} value={run.id}>
                {leaderboardRunLabel(run, language)} / {t.common.status[run.status]}
              </option>
            ))}
          </select>
          <button className="button" disabled={rows.length === 0} onClick={() => exportLeaderboardCsv(rows)} type="button">
            <Download size={16} />
            {t.results.exportCsv}
          </button>
        </div>
      </div>

      <section className="leaderboard-grid">
        {rows.length === 0 ? (
          <PageState
            description={
              runsLoading || leaderboardState === "loading"
                ? language === "zh" ? "正在读取固定评测协议下的算法排名。" : "Reading algorithm rankings for the selected protocol."
                : leaderboardState === "error"
                  ? language === "zh" ? "排行榜接口暂时不可用，请确认评分产物已生成后重试。" : "The leaderboard endpoint is unavailable. Confirm that scoring artifacts exist, then retry."
                  : language === "zh" ? "当前实验尚未生成可排名的评分行。请选择其他实验，或先完成正式评分。" : "This experiment has no rankable score rows yet. Select another experiment or finish official scoring."
            }
            icon={runsLoading || leaderboardState === "loading" ? LoaderCircle : leaderboardState === "error" ? TriangleAlert : Info}
            title={
              runsLoading || leaderboardState === "loading"
                ? language === "zh" ? "正在加载天梯图" : "Loading leaderboard"
                : leaderboardState === "error"
                  ? language === "zh" ? "无法读取天梯图" : "Unable to load leaderboard"
                  : language === "zh" ? "暂无可排名数据" : "No rankable data"
            }
            tone={runsLoading || leaderboardState === "loading" ? "loading" : leaderboardState === "error" ? "error" : "empty"}
          />
        ) : (
          <>
            <AlgorithmEvaluationRanking
              algorithmColorDomain={algorithmColorDomain}
              algorithmNames={algorithmNames}
              compositeWeights={compositeWeights}
              language={language}
              metric={rankingMetric}
              onChangeCompositeWeights={setCompositeWeights}
              onPick={setSelectedRankingRow}
              rows={rows}
              selectedAlgorithmId={selectedRankingRow?.algorithmId}
              setMetric={setRankingMetric}
            />

            {selectedRankingRow ? (
              <div className="panel leaderboard-ranking-detail">
                <div className="panel-header">
                  <h2>{displayAlgorithm(selectedRankingRow.algorithmId, algorithmNames)}</h2>
                  <span className={selectedRankingRow.officialEligible ? "badge ok" : "badge warn"}>
                    {selectedRankingRow.officialEligible ? t.common.official : t.common.provisional}
                  </span>
                </div>
                <div className="panel-body metric-list">
                  <div className="metric-row">
                    <span>{language === "zh" ? "鲁棒性" : "Robustness"}</span>
                    <strong>{formatPercent(robustnessScore(selectedRankingRow))}</strong>
                  </div>
                  <div className="metric-row">
                    <span>{language === "zh" ? "保真度" : "Fidelity"}</span>
                    <strong>{formatPercent(normalizedFidelityScore(selectedRankingRow, rows))}</strong>
                  </div>
                  <div className="metric-row">
                    <span>{language === "zh" ? "算法复杂度" : "Algorithm complexity"}</span>
                    <strong>{formatPercent(algorithmSimplicityScore(selectedRankingRow, rows))}</strong>
                  </div>
                </div>
              </div>
            ) : null}
          </>
        )}
      </section>
    </AppShell>
  );
}

function leaderboardRunLabel(run: DemoRunRecord, language: string): string {
  const rawName = run.taskName?.trim() || run.configName?.trim() || run.id;
  if (!/^Imported run\b/i.test(rawName)) {
    return rawName;
  }
  const datasetIds = run.selection?.datasetIds ?? [];
  const dataset = datasetIds.length === 1 && datasetIds[0].toLowerCase() === "imagenet" ? "ImageNet" : datasetIds[0] || "dataset";
  const sampleCount = Number(run.selection?.maxSamples ?? 0);
  if (sampleCount <= 0) {
    return rawName;
  }
  return language === "zh" ? `${dataset} ${sampleCount.toLocaleString()} 张图片测评` : `${dataset} ${sampleCount.toLocaleString()}-image experiment`;
}

function AlgorithmEvaluationRanking({
  algorithmColorDomain,
  algorithmNames,
  compositeWeights,
  language,
  metric,
  onChangeCompositeWeights,
  onPick,
  rows,
  selectedAlgorithmId,
  setMetric
}: {
  algorithmColorDomain: string[];
  algorithmNames: Record<string, string>;
  compositeWeights: CompositeWeights;
  language: string;
  metric: RankingMetric;
  onChangeCompositeWeights: (weights: CompositeWeights) => void;
  onPick: (row: BenchmarkLeaderboardRow) => void;
  rows: BenchmarkLeaderboardRow[];
  selectedAlgorithmId?: string;
  setMetric: (metric: RankingMetric) => void;
}) {
  const [draftWeights, setDraftWeights] = useState<CompositeWeights>(compositeWeights);

  useEffect(() => {
    setDraftWeights(compositeWeights);
  }, [compositeWeights]);

  if (rows.length === 0) {
    return null;
  }
  const rankedRows = rankRowsByMetric(rows, metric, compositeWeights);
  const maxScore = Math.max(...rankedRows.map((row) => rankingMetricScore(row, metric, rows, compositeWeights)), 1);
  const labels = rankingLabels(language);
  const firstHandle = draftWeights.robustness;
  const secondHandle = draftWeights.robustness + draftWeights.fidelity;
  const commitDraftWeights = () => onChangeCompositeWeights(draftWeights);
  const updateFirstHandle = (value: number) => {
    const next = Math.min(clampWeight(value), secondHandle);
    setDraftWeights(weightsFromHandles(next, secondHandle));
  };
  const updateSecondHandle = (value: number) => {
    const next = Math.max(firstHandle, clampWeight(value));
    setDraftWeights(weightsFromHandles(firstHandle, next));
  };
  return (
    <section className="panel interactive-bars-panel overview-ladder-panel leaderboard-ranking-panel">
      <div className="panel-header">
        <h2>{language === "zh" ? "算法评估排名" : "Algorithm evaluation ranking"}</h2>
        <BarChart3 size={16} />
      </div>
      <div className="ranking-mode-tabs overview-evaluation-tabs">
        {RANKING_METRICS.map((key) => (
          <button className={metric === key ? "active" : ""} key={key} onClick={() => setMetric(key)} type="button">
            {labels[key]}
          </button>
        ))}
      </div>
      {metric === "composite" ? (
        <div className="composite-weight-editor">
          <div className="composite-weight-header">
            <div>
              <strong>{language === "zh" ? "综合评分权重" : "Composite score weights"}</strong>
              <span>
                {language === "zh" ? "拖动两个滑块调整三项占比，松开后更新排名" : "Drag both handles to set the split; ranking updates on release"}
              </span>
            </div>
            <button
              className="composite-weight-reset"
              onClick={() => {
                const defaults = { ...DEFAULT_COMPOSITE_WEIGHTS };
                setDraftWeights(defaults);
                onChangeCompositeWeights(defaults);
              }}
              type="button"
            >
              {language === "zh" ? "恢复默认 50/30/20" : "Reset to 50/30/20"}
            </button>
          </div>
          <div className="composite-weight-values" aria-live="polite">
            {COMPOSITE_WEIGHT_KEYS.map((key) => (
              <div className={`composite-weight-value ${key}`} key={key}>
                <i aria-hidden="true" />
                <span>{labels[key]}</span>
                <strong>{draftWeights[key]}%</strong>
              </div>
            ))}
          </div>
          <div className="composite-weight-slider">
            <div aria-hidden="true" className="composite-weight-track">
              <i className="robustness" style={{ width: `${draftWeights.robustness}%` }} />
              <i className="fidelity" style={{ width: `${draftWeights.fidelity}%` }} />
              <i className="complexity" style={{ width: `${draftWeights.complexity}%` }} />
            </div>
            <input
              aria-label={language === "zh" ? "鲁棒性与保真度分界" : "Robustness and fidelity boundary"}
              className="composite-weight-range first"
              max="100"
              min="0"
              onBlur={commitDraftWeights}
              onChange={(event) => updateFirstHandle(Number(event.target.value))}
              onKeyUp={commitDraftWeights}
              onPointerUp={commitDraftWeights}
              step="1"
              type="range"
              value={firstHandle}
            />
            <input
              aria-label={language === "zh" ? "保真度与算法复杂度分界" : "Fidelity and complexity boundary"}
              className="composite-weight-range second"
              max="100"
              min="0"
              onBlur={commitDraftWeights}
              onChange={(event) => updateSecondHandle(Number(event.target.value))}
              onKeyUp={commitDraftWeights}
              onPointerUp={commitDraftWeights}
              step="1"
              type="range"
              value={secondHandle}
            />
          </div>
          <div className="composite-weight-scale" aria-hidden="true">
            <span>0%</span>
            <span>50%</span>
            <span>100%</span>
          </div>
        </div>
      ) : null}
      <div className="panel-body interactive-bars">
        {rankedRows.map((row, index) => {
          const value = rankingMetricScore(row, metric, rows, compositeWeights);
          const width = `${Math.max(3, (value / maxScore) * 100)}%`;
          const color = curveDomainColor(algorithmColorDomain, row.algorithmId);
          return (
            <button
              className={selectedAlgorithmId === row.algorithmId ? "interactive-bar-row active" : "interactive-bar-row"}
              key={row.algorithmId}
              onClick={() => onPick(row)}
              type="button"
            >
              <span>{index + 1}. {displayAlgorithm(row.algorithmId, algorithmNames)}</span>
              <i style={{ width, background: chartBarFill(color) }} />
              <strong>{formatRankingValue(row, metric, rows, compositeWeights)}</strong>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function buildAlgorithmNames(algorithms: AlgorithmVersion[]): Record<string, string> {
  const names: Record<string, string> = {};
  for (const algorithm of algorithms) {
    const label = displayAlgorithm(algorithm.id, {}, algorithm.name, algorithm.method);
    names[algorithm.id] = label;
    if (algorithm.method) {
      names[algorithm.method] = label;
    }
  }
  return names;
}

function displayAlgorithm(
  id: string,
  resourceNames: Record<string, string> = {},
  fallbackName?: string,
  methodOverride?: string
): string {
  const method = methodOverride ?? id.replace(/^alg[-_]/, "");
  return resourceNames[id] ?? resourceNames[method] ?? resolveWatermarkDisplayName(method, fallbackName || displayTokenLabel(method));
}

function displayTokenLabel(value: string): string {
  return value
    .replace(/^alg[-_]/, "")
    .split(/[-_]/)
    .filter(Boolean)
    .map((part) => {
      const lower = part.toLowerCase();
      if (["dct", "dwt", "svd", "ssl", "wam", "cin", "dwsf"].includes(lower)) {
        return lower.toUpperCase();
      }
      return lower.charAt(0).toUpperCase() + lower.slice(1);
    })
    .join("-");
}

function rankingLabels(language: string): Record<RankingMetric, string> {
  return {
    robustness: language === "zh" ? "鲁棒性" : "Robustness",
    complexity: language === "zh" ? "算法复杂度" : "Algorithm complexity",
    fidelity: language === "zh" ? "保真度" : "Fidelity",
    composite: language === "zh" ? "综合评分" : "Composite score"
  };
}

function rankRowsByMetric(
  rows: BenchmarkLeaderboardRow[],
  metric: RankingMetric,
  compositeWeights: CompositeWeights
): BenchmarkLeaderboardRow[] {
  return [...rows].sort((left, right) => {
    const leftScore = rankingMetricScore(left, metric, rows, compositeWeights);
    const rightScore = rankingMetricScore(right, metric, rows, compositeWeights);
    return rightScore - leftScore || left.algorithmId.localeCompare(right.algorithmId);
  });
}

function rankingMetricScore(
  row: BenchmarkLeaderboardRow,
  metric: RankingMetric,
  rows: BenchmarkLeaderboardRow[],
  compositeWeights: CompositeWeights
): number {
  if (metric === "robustness") {
    return (robustnessScore(row) ?? 0) * 100;
  }
  if (metric === "fidelity") {
    return (normalizedFidelityScore(row, rows) ?? 0) * 100;
  }
  if (metric === "complexity") {
    return (algorithmSimplicityScore(row, rows) ?? 0) * 100;
  }
  return (compositeScore(row, rows, compositeWeights) ?? 0) * 100;
}

function formatRankingValue(
  row: BenchmarkLeaderboardRow,
  metric: RankingMetric,
  rows: BenchmarkLeaderboardRow[],
  compositeWeights: CompositeWeights
): string {
  if (metric === "robustness") {
    return formatPercent(robustnessScore(row));
  }
  if (metric === "fidelity") {
    return formatPercent(normalizedFidelityScore(row, rows));
  }
  if (metric === "complexity") {
    return formatPercent(algorithmSimplicityScore(row, rows));
  }
  return formatPercent(compositeScore(row, rows, compositeWeights));
}

function compositeScore(
  row: BenchmarkLeaderboardRow,
  rows: BenchmarkLeaderboardRow[],
  compositeWeights: CompositeWeights
): number | null {
  const components: Array<{ value: number; weight: number }> = [];
  const robustness = robustnessScore(row);
  if (robustness != null) {
    components.push({ value: robustness, weight: compositeWeights.robustness });
  }
  const fidelity = normalizedFidelityScore(row, rows);
  if (fidelity != null) {
    components.push({ value: fidelity, weight: compositeWeights.fidelity });
  }
  const complexity = algorithmSimplicityScore(row, rows);
  if (complexity != null) {
    components.push({ value: complexity, weight: compositeWeights.complexity });
  }
  const weightSum = components.reduce((total, item) => total + item.weight, 0);
  if (components.length === 0 || weightSum <= 0) {
    return null;
  }
  return components.reduce((total, item) => total + item.value * item.weight, 0) / weightSum;
}

function clampWeight(value: number): number {
  return Number.isFinite(value) ? Math.max(0, Math.min(100, Math.round(value))) : 0;
}

function weightsFromHandles(firstPosition: number, secondPosition: number): CompositeWeights {
  const first = clampWeight(firstPosition);
  const second = Math.max(first, clampWeight(secondPosition));
  return {
    robustness: first,
    fidelity: second - first,
    complexity: 100 - second
  };
}

function normalizeCompositeWeights(value: unknown): CompositeWeights {
  if (!value || typeof value !== "object") {
    return { ...DEFAULT_COMPOSITE_WEIGHTS };
  }
  const candidate = value as Partial<Record<CompositeWeightKey, unknown>>;
  const robustness = Number(candidate.robustness);
  const fidelity = Number(candidate.fidelity);
  const complexity = Number(candidate.complexity);
  if (![robustness, fidelity, complexity].every((item) => Number.isFinite(item) && item >= 0)) {
    return { ...DEFAULT_COMPOSITE_WEIGHTS };
  }
  const total = robustness + fidelity + complexity;
  if (total <= 0) {
    return { ...DEFAULT_COMPOSITE_WEIGHTS };
  }
  const first = (robustness / total) * 100;
  const second = ((robustness + fidelity) / total) * 100;
  return weightsFromHandles(first, second);
}

function robustnessScore(row: BenchmarkLeaderboardRow): number | null {
  return meanNumber(
    row.categoryScores
      .filter((category) => category.covered)
      .map((category) => category.score)
  );
}

function normalizedFidelityScore(row: BenchmarkLeaderboardRow, rows: BenchmarkLeaderboardRow[]): number | null {
  const value = finiteOrNull(row.cleanFidelity);
  const values = rows.map((item) => finiteOrNull(item.cleanFidelity)).filter((item): item is number => item != null);
  if (value == null || values.length === 0) {
    return null;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (max <= min) {
    return 1;
  }
  return Math.max(0, Math.min(1, (value - min) / (max - min)));
}

function algorithmSimplicityScore(row: BenchmarkLeaderboardRow, rows: BenchmarkLeaderboardRow[]): number | null {
  const runtime = finiteOrNull(row.runtimeMs);
  const runtimes = rows.map((item) => finiteOrNull(item.runtimeMs)).filter((value): value is number => value != null && value > 0);
  if (runtime == null || runtime <= 0 || runtimes.length === 0) {
    return null;
  }
  const min = Math.min(...runtimes);
  const max = Math.max(...runtimes);
  if (max <= min) {
    return 1;
  }
  const logMin = Math.log1p(min);
  const logMax = Math.log1p(max);
  const ratio = (Math.log1p(runtime) - logMin) / Math.max(0.0001, logMax - logMin);
  return Math.max(0, Math.min(1, 1 - ratio));
}

function formatPercent(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? "n/a" : (value * 100).toFixed(1);
}

function finiteOrNull(value: number | null | undefined): number | null {
  return value == null || !Number.isFinite(value) ? null : value;
}

function meanNumber(values: Array<number | null | undefined>): number | null {
  const finite = values.filter((value): value is number => value != null && Number.isFinite(value));
  if (!finite.length) {
    return null;
  }
  return finite.reduce((total, value) => total + value, 0) / finite.length;
}

function exportLeaderboardCsv(rows: LeaderboardResponse["rows"]) {
  const csvRows = [
    ["rank", "algorithm_id", "protocol_status", "wrs", "clean_fidelity", "avg_nqd", "coverage", "runtime_ms", "run_id"],
    ...rows.map((row) => [
      String(row.rank),
      row.algorithmId,
      row.protocolStatus,
      String(row.wrs ?? ""),
      String(row.cleanFidelity ?? ""),
      String(row.avgNqd ?? ""),
      `${row.coverage.coveredCategoryCount}/${row.coverage.requiredCategoryCount}`,
      String(row.runtimeMs ?? ""),
      row.runId ?? ""
    ])
  ];
  const csv = csvRows.map((row) => row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "wm-bench-leaderboard.csv";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
