"use client";

import { useEffect, useMemo, useState } from "react";
import { BarChart3, Clock3, Download, History } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { useLanguage } from "@/components/LanguageProvider";
import { fetchAlgorithms, fetchLeaderboard, fetchRuns } from "@/lib/api";
import { resolveWatermarkDisplayName } from "@/lib/watermark-display";
import type { AlgorithmVersion, BenchmarkLeaderboardRow, DemoRunRecord, LeaderboardResponse } from "@/lib/types";

type RankingMetric = "robustness" | "complexity" | "fidelity" | "composite";

const RANKING_METRICS: RankingMetric[] = ["robustness", "composite", "fidelity", "complexity"];
const COMPOSITE_WEIGHTS = {
  robustness: 0.5,
  fidelity: 0.3,
  complexity: 0.2
} as const;

type AlgorithmHistory = {
  algorithmId: string;
  primary: BenchmarkLeaderboardRow;
  runs: BenchmarkLeaderboardRow[];
};

export default function LeaderboardPage() {
  const { language, t } = useLanguage();
  const [leaderboard, setLeaderboard] = useState<LeaderboardResponse | null>(null);
  const [algorithms, setAlgorithms] = useState<AlgorithmVersion[]>([]);
  const [runs, setRuns] = useState<DemoRunRecord[]>([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [rankingMetric, setRankingMetric] = useState<RankingMetric>("composite");
  const [selectedAlgorithmId, setSelectedAlgorithmId] = useState("");

  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchRuns(), fetchAlgorithms().catch(() => [] as AlgorithmVersion[])])
      .then(([nextRuns, nextAlgorithms]) => {
        if (cancelled) {
          return;
        }
        const leaderboardRuns = nextRuns.filter(
          (run) => run.status === "succeeded" || run.status === "partially_failed"
        );
        setRuns(leaderboardRuns);
        setSelectedRunId((current) =>
          current && leaderboardRuns.some((run) => run.id === current) ? current : (leaderboardRuns[0]?.id ?? "")
        );
        setAlgorithms(nextAlgorithms);
      })
      .catch(() => {
        // Keep the page shell available; empty charts already show no-data states.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedRunId) {
      setLeaderboard(null);
      setSelectedAlgorithmId("");
      return;
    }
    let cancelled = false;
    setLeaderboard(null);
    setSelectedAlgorithmId("");
    fetchLeaderboard("wrs-v2-detection-v1", selectedRunId)
      .then((nextLeaderboard) => {
        if (!cancelled) {
          setLeaderboard(nextLeaderboard);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setLeaderboard(null);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedRunId]);

  const allRows = leaderboard?.rows ?? [];
  const algorithmHistories = useMemo(() => buildAlgorithmHistories(allRows), [allRows]);
  const rows = useMemo(() => algorithmHistories.map((item) => item.primary), [algorithmHistories]);
  const algorithmNames = useMemo(() => buildAlgorithmNames(algorithms), [algorithms]);
  const selectedHistory = algorithmHistories.find((item) => item.algorithmId === selectedAlgorithmId) ?? null;

  useEffect(() => {
    if (!rows.length) {
      setSelectedAlgorithmId("");
      return;
    }
    setSelectedAlgorithmId((current) => {
      if (current && rows.some((row) => row.algorithmId === current)) {
        return current;
      }
      return rankRowsByMetric(rows, rankingMetric)[0]?.algorithmId ?? "";
    });
  }, [rankingMetric, rows]);

  return (
    <AppShell active="leaderboard">
      <div className="topbar">
        <div className="title-block">
          <h1>{t.leaderboard.title}</h1>
          <p>{t.leaderboard.subtitle}</p>
        </div>
        <div className="toolbar">
          <select
            aria-label={language === "zh" ? "选择实验" : "Select experiment"}
            className="run-result-select"
            disabled={runs.length === 0}
            onChange={(event) => setSelectedRunId(event.target.value)}
            value={selectedRunId}
          >
            {runs.map((run) => (
              <option key={run.id} value={run.id}>
                {leaderboardRunLabel(run, language)} / {run.status}
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
        <AlgorithmEvaluationRanking
          algorithmNames={algorithmNames}
          language={language}
          metric={rankingMetric}
          onPick={(row) => setSelectedAlgorithmId(row.algorithmId)}
          rows={rows}
          selectedAlgorithmId={selectedAlgorithmId}
          setMetric={setRankingMetric}
        />

        {selectedHistory ? (
          <div className="panel leaderboard-ranking-detail">
            <div className="panel-header">
              <h2>{displayAlgorithm(selectedHistory.algorithmId, algorithmNames)}</h2>
              <span className={selectedHistory.primary.officialEligible ? "badge ok" : "badge warn"}>
                {selectedHistory.primary.officialEligible ? t.common.official : t.common.provisional}
              </span>
            </div>
            <div className="leaderboard-score-scope">
              <span>{language === "zh" ? "官方协议指标" : "Official protocol"}</span>
              <strong>WRS-v2</strong>
              <p>{language === "zh" ? "用于正式鲁棒性结论与对外报告。" : "Use for formal robustness claims and reporting."}</p>
            </div>
            <div className="panel-body metric-list leaderboard-detail-metrics">
              <div className="metric-row">
                <span>{language === "zh" ? "官方 WRS" : "Official WRS"}</span>
                <strong>{selectedHistory.primary.wrs == null ? "n/a" : selectedHistory.primary.wrs.toFixed(1)}</strong>
              </div>
              <div className="metric-row">
                <span>{language === "zh" ? "产品综合分" : "Product composite"}</span>
                <strong>{formatPercent(compositeScore(selectedHistory.primary, rows))}</strong>
              </div>
              <div className="metric-row">
                <span>{t.results.cleanFidelity}</span>
                <strong>{formatPercent(normalizedFidelityScore(selectedHistory.primary, rows))}</strong>
              </div>
              <div className="metric-row">
                <span>{language === "zh" ? "算法复杂度" : "Algorithm complexity"}</span>
                <strong>{formatPercent(algorithmSimplicityScore(selectedHistory.primary, rows))}</strong>
              </div>
            </div>
            <div className="leaderboard-history-heading">
              <span><History size={15} /> {language === "zh" ? "Run 历史" : "Run history"}</span>
              <strong>{selectedHistory.runs.length}</strong>
            </div>
            <div className="leaderboard-run-history">
              {selectedHistory.runs.map((row, index) => (
                <div className={index === 0 ? "leaderboard-run-item current" : "leaderboard-run-item"} key={`${row.runId ?? "run"}-${row.updatedAt ?? index}`}>
                  <div>
                    <strong>{row.configName || row.runId || (language === "zh" ? "未命名运行" : "Unnamed run")}</strong>
                    <span><Clock3 size={12} /> {formatRunTimestamp(row.updatedAt, language)}</span>
                  </div>
                  <div>
                    <b>WRS {row.wrs == null ? "n/a" : row.wrs.toFixed(1)}</b>
                    <span className={row.officialEligible ? "run-status official" : "run-status provisional"}>
                      {(t.common.status as Record<string, string>)[row.runStatus ?? ""] ?? row.runStatus ?? row.protocolStatus}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : null}
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
  algorithmNames,
  language,
  metric,
  onPick,
  rows,
  selectedAlgorithmId,
  setMetric
}: {
  algorithmNames: Record<string, string>;
  language: string;
  metric: RankingMetric;
  onPick: (row: BenchmarkLeaderboardRow) => void;
  rows: BenchmarkLeaderboardRow[];
  selectedAlgorithmId?: string;
  setMetric: (metric: RankingMetric) => void;
}) {
  if (rows.length === 0) {
    return null;
  }
  const rankedRows = rankRowsByMetric(rows, metric);
  const maxScore = Math.max(...rankedRows.map((row) => rankingMetricScore(row, metric, rows)), 1);
  const labels = rankingLabels(language);
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
      <p className="overview-evaluation-note">{rankingDescription(metric, language)}</p>
      <div className="panel-body interactive-bars">
        {rankedRows.map((row, index) => {
          const value = rankingMetricScore(row, metric, rows);
          const width = `${Math.max(3, (value / maxScore) * 100)}%`;
          return (
            <button
              className={`interactive-bar-row leaderboard-rank-row rank-${Math.min(index + 1, 4)}${selectedAlgorithmId === row.algorithmId ? " active" : ""}`}
              key={row.algorithmId}
              onClick={() => onPick(row)}
              type="button"
            >
              <span><b>{index + 1}</b> {displayAlgorithm(row.algorithmId, algorithmNames)}</span>
              <i style={{ width }} />
              <strong>{formatRankingValue(row, metric, rows)}</strong>
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

function buildAlgorithmHistories(rows: BenchmarkLeaderboardRow[]): AlgorithmHistory[] {
  const grouped = new Map<string, BenchmarkLeaderboardRow[]>();
  for (const row of rows) {
    grouped.set(row.algorithmId, [...(grouped.get(row.algorithmId) ?? []), row]);
  }
  return Array.from(grouped.entries())
    .map(([algorithmId, historyRows]) => {
      const runs = [...historyRows].sort(compareHistoryRows);
      return { algorithmId, primary: runs[0], runs };
    })
    .sort((left, right) => left.algorithmId.localeCompare(right.algorithmId));
}

function compareHistoryRows(left: BenchmarkLeaderboardRow, right: BenchmarkLeaderboardRow): number {
  if (left.officialEligible !== right.officialEligible) {
    return left.officialEligible ? -1 : 1;
  }
  const updatedDifference = Date.parse(right.updatedAt ?? "") - Date.parse(left.updatedAt ?? "");
  if (Number.isFinite(updatedDifference) && updatedDifference !== 0) {
    return updatedDifference;
  }
  return (right.wrs ?? -1) - (left.wrs ?? -1) || (right.runId ?? "").localeCompare(left.runId ?? "");
}

function formatRunTimestamp(value: string | undefined, language: string): string {
  if (!value) {
    return language === "zh" ? "时间未知" : "Unknown time";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat(language === "zh" ? "zh-CN" : "en-US", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  }).format(parsed);
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
    robustness: language === "zh" ? "官方 WRS" : "Official WRS",
    complexity: language === "zh" ? "算法复杂度" : "Algorithm complexity",
    fidelity: language === "zh" ? "自身保真度" : "Clean fidelity",
    composite: language === "zh" ? "产品综合分" : "Product composite"
  };
}

function rankingDescription(metric: RankingMetric, language: string): string {
  if (metric === "robustness") {
    return language === "zh"
      ? "官方口径：直接按 WRS-v2 排序，用于正式鲁棒性结论；临时结果仍会标记为 provisional。"
      : "Official scope: ranked directly by WRS-v2 for formal robustness claims; provisional results remain labeled.";
  }
  if (metric === "complexity") {
    return language === "zh"
      ? "按相对运行效率排序，分数越高表示算法越轻量、运行越快。"
      : "Ranked by relative runtime efficiency; higher means lighter and faster.";
  }
  if (metric === "fidelity") {
    return language === "zh"
      ? "按无攻击条件下的图像保真度排序，分数越高表示水印引入的失真越小。"
      : "Ranked by clean fidelity; higher means less visible distortion.";
  }
  return language === "zh"
    ? "产品展示口径（非官方 WRS）：50% 鲁棒性 + 30% 自身保真度 + 20% 算法效率；缺失维度按可用项重新归一化。"
    : "Product display scope (not official WRS): 50% robustness + 30% fidelity + 20% efficiency; missing dimensions are renormalized.";
}

function rankRowsByMetric(rows: BenchmarkLeaderboardRow[], metric: RankingMetric): BenchmarkLeaderboardRow[] {
  return [...rows].sort((left, right) => {
    const leftScore = rankingMetricScore(left, metric, rows);
    const rightScore = rankingMetricScore(right, metric, rows);
    return rightScore - leftScore || left.algorithmId.localeCompare(right.algorithmId);
  });
}

function rankingMetricScore(row: BenchmarkLeaderboardRow, metric: RankingMetric, rows: BenchmarkLeaderboardRow[]): number {
  if (metric === "robustness") {
    return row.wrs ?? 0;
  }
  if (metric === "fidelity") {
    return (normalizedFidelityScore(row, rows) ?? 0) * 100;
  }
  if (metric === "complexity") {
    return (algorithmSimplicityScore(row, rows) ?? 0) * 100;
  }
  return (compositeScore(row, rows) ?? 0) * 100;
}

function formatRankingValue(row: BenchmarkLeaderboardRow, metric: RankingMetric, rows: BenchmarkLeaderboardRow[]): string {
  if (metric === "robustness") {
    return row.wrs == null ? "n/a" : row.wrs.toFixed(1);
  }
  if (metric === "fidelity") {
    return formatPercent(normalizedFidelityScore(row, rows));
  }
  if (metric === "complexity") {
    return formatPercent(algorithmSimplicityScore(row, rows));
  }
  return formatPercent(compositeScore(row, rows));
}

function compositeScore(row: BenchmarkLeaderboardRow, rows: BenchmarkLeaderboardRow[]): number | null {
  const components: Array<{ value: number; weight: number }> = [];
  const robustness = robustnessScore(row);
  if (robustness != null) {
    components.push({ value: robustness, weight: COMPOSITE_WEIGHTS.robustness });
  }
  const fidelity = normalizedFidelityScore(row, rows);
  if (fidelity != null) {
    components.push({ value: fidelity, weight: COMPOSITE_WEIGHTS.fidelity });
  }
  const complexity = algorithmSimplicityScore(row, rows);
  if (complexity != null) {
    components.push({ value: complexity, weight: COMPOSITE_WEIGHTS.complexity });
  }
  if (components.length === 0) {
    return null;
  }
  const weightSum = components.reduce((total, item) => total + item.weight, 0);
  return components.reduce((total, item) => total + item.value * item.weight, 0) / weightSum;
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
