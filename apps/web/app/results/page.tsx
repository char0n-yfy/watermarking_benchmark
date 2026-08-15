"use client";

import Link from "next/link";
import { memo, useCallback, useEffect, useMemo, useRef, useState, type MemoExoticComponent } from "react";
import {
  BarChart3,
  Download,
  Filter,
  Gauge,
  Info,
  LoaderCircle,
  PlayCircle,
  RefreshCw,
  SlidersHorizontal,
  Trophy
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { BenchmarkRadar } from "@/components/BenchmarkRadar";
import { useLanguage } from "@/components/LanguageProvider";
import { PageState } from "@/components/PageState";
import {
  CurveLegendGlyph,
  RobustnessCurve,
  curveDomainColor,
  curveDomainShape,
  curveSeriesColor
} from "@/components/RobustnessCurve";
import {
  buildMainOverviewRadarSeries,
  buildMainOverviewRadarTemplate,
  buildOverviewDetailRadars
} from "@/lib/overview-radar";
import {
  fetchAlgorithms,
  fetchAttacks,
  fetchDatasetCatalog,
  fetchRunScore,
  fetchRuns,
  runResultsCsvUrl
} from "@/lib/api";
import {
  countBenchmarkAttackTypes,
  countBenchmarkAttackTypesFromMethods,
  countSelectedBenchmarkAttackTypes,
  isHiddenBenchmarkAttack
} from "@/lib/benchmark-attack-catalog";
import { formatMetric, rankAggregates } from "@/lib/insights";
import { resolveWatermarkDisplayName } from "@/lib/watermark-display";
import type {
  BenchmarkCategoryScore,
  BenchmarkAttackLeaderboardRow,
  BenchmarkCurvePoint,
  BenchmarkLeaderboardRow,
  BenchmarkScore,
  AlgorithmVersion,
  AttackPreset,
  DemoRunRecord,
  RunAggregate,
  RunResultUnit,
  RunResults
} from "@/lib/types";

type ResultsTab = "overview" | "attack" | "quality";

type AttackSelectorKey = "dataset" | "attack";
type AttackHeatmapMetric = "tpr" | "nqd";
type AttackHeatmapRowMode = "category" | "attack";
type QualitySelectorKey = "dataset" | "algorithm" | "attack";
type SummaryIconKind = "experiment" | "status" | "dataset" | "watermark" | "attack";
type StringArraySetter = (value: string[] | ((current: string[]) => string[])) => void;
type AttackSelectorSetter = (
  value: AttackSelectorKey | null | ((current: AttackSelectorKey | null) => AttackSelectorKey | null)
) => void;
type QualitySelectorSetter = (
  value: QualitySelectorKey | null | ((current: QualitySelectorKey | null) => QualitySelectorKey | null)
) => void;

type ChartInsight = {
  kind: "run" | "algorithm" | "category" | "attack" | "curve";
  title: string;
  body: string;
  details?: Array<{ label: string; value: string }>;
  meta?: string;
  key?: string;
};

type QualityAttackSummary = {
  key: string;
  attackPresetId: string;
  attackMethod: string;
  attackCategory: string;
  algorithmCount: number;
  pointCount: number;
  strengthName: string;
  strengthMin: number;
  strengthMax: number;
  avgTpr: number | null;
  avgNqd: number | null;
  weakestPoint: BenchmarkCurvePoint | null;
};

type QualityAttackOption = {
  attackPresetId: string;
  attackMethod: string;
  attackCategory: string;
  pointCount: number;
  variantCount: number;
};

type QualitySelectorOption = {
  id: string;
  label: string;
  meta: string;
  count: number;
};

type QualityComboSummary = {
  key: string;
  datasetId: string;
  algorithmId: string;
  attackPresetId: string;
  attackMethod: string;
  attackCategory: string;
  variantLabel: string;
  pointCount: number;
  strengthName: string;
  strengthMin: number;
  strengthMax: number;
  avgTpr: number | null;
  avgNqd: number | null;
  weakestPoint: BenchmarkCurvePoint | null;
};

type AttackVisualSummary = {
  attackPresetId: string;
  attackMethod: string;
  attackCategory: string;
  label: string;
  pointCount: number;
  algorithmCount: number;
  avgTpr: number | null;
  avgNqd: number | null;
  minTpr: number | null;
  qAtP95: number | "inf" | "-inf" | null;
  qAtP70: number | "inf" | "-inf" | null;
  avgP: number | null;
  avgQ: number | null;
  auc: number | null;
  riskScore: number;
  points: BenchmarkCurvePoint[];
};

type AttackCategoryDistribution = {
  key: string;
  label: string;
  pointCount: number;
  attackCount: number;
  avgNqd: number | null;
  avgTpr: number | null;
  points: BenchmarkCurvePoint[];
};

type AttackHeatmapCell = {
  rowKey: string;
  rowLabel: string;
  rowMode: AttackHeatmapRowMode;
  algorithmId: string;
  algorithmLabel: string;
  attackPresetIds: string[];
  avgTpr: number | null;
  avgNqd: number | null;
  pointCount: number;
};

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

const RESULT_TABS: ResultsTab[] = ["overview", "attack", "quality"];
const EMPTY_SCORE_ROWS: BenchmarkLeaderboardRow[] = [];
const MemoizedRobustnessCurve = memo(RobustnessCurve);
const MemoizedAttackViolinPlot = memo(AttackViolinPlot);
const MemoizedAttackHeatmapMatrix = memo(AttackHeatmapMatrix);

export default function ResultsPage() {
  const { language, t } = useLanguage();
  const uiText = (zh: string, en: string) => (language === "zh" ? zh : en);
  const [runs, setRuns] = useState<DemoRunRecord[]>([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [runRefreshKey, setRunRefreshKey] = useState(0);
  const [results, setResults] = useState<RunResults | null>(null);
  const [score, setScore] = useState<BenchmarkScore | null>(null);
  const [resourceAlgorithmNames, setResourceAlgorithmNames] = useState<Record<string, string>>({});
  const [resourceAttackNames, setResourceAttackNames] = useState<Record<string, string>>({});
  const [resourceAttacks, setResourceAttacks] = useState<AttackPreset[]>([]);
  const [resourceCatalogCounts, setResourceCatalogCounts] = useState({
    datasets: 0,
    algorithms: 0,
    attacks: 0
  });
  const [activeInsight, setActiveInsight] = useState<ChartInsight | null>(null);
  const [notice, setNotice] = useState("");
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<ResultsTab>("overview");
  const [selectedAlgorithmIds, setSelectedAlgorithmIds] = useState<string[]>([]);
  const [attackDatasetIds, setAttackDatasetIds] = useState<string[]>([]);
  const [attackAttackIds, setAttackAttackIds] = useState<string[]>([]);
  const [activeAttackSelectorKey, setActiveAttackSelectorKey] = useState<AttackSelectorKey | null>("attack");
  const [attackHeatmapMetric, setAttackHeatmapMetric] = useState<AttackHeatmapMetric>("tpr");
  const [attackHeatmapRowMode, setAttackHeatmapRowMode] = useState<AttackHeatmapRowMode>("category");
  const [selectedAttackHeatmapCell, setSelectedAttackHeatmapCell] = useState<AttackHeatmapCell | null>(null);
  const [qualityAttackFilter, setQualityAttackFilter] = useState("all");
  const [qualityDatasetIds, setQualityDatasetIds] = useState<string[]>([]);
  const [qualityAlgorithmIds, setQualityAlgorithmIds] = useState<string[]>([]);
  const [qualityAttackIds, setQualityAttackIds] = useState<string[]>([]);
  const [activeQualitySelectorKey, setActiveQualitySelectorKey] = useState<QualitySelectorKey | null>("attack");

  const legacyRanking = useMemo(() => rankAggregates(results?.aggregates ?? []), [results]);
  const scoreRows = score?.leaderboardRows ?? EMPTY_SCORE_ROWS;
  const algorithmIds = useMemo(() => collectAlgorithmIds(results, scoreRows, legacyRanking), [legacyRanking, results, scoreRows]);
  const qualityFallbackDatasetIds = useMemo(
    () => Array.from(new Set((results?.resultUnits ?? []).map((unit) => unit.datasetId).filter(Boolean))),
    [results]
  );
  const qualityFallbackDatasetId = qualityFallbackDatasetIds.length === 1 ? qualityFallbackDatasetIds[0] : "unknown";
  const qualityAvailableCurvePoints = useMemo(
    () =>
      (score?.curvePoints ?? [])
        .map((point) => ({ ...point, datasetId: point.datasetId || qualityFallbackDatasetId }))
        .sort(qualityCurvePointSort),
    [qualityFallbackDatasetId, score]
  );
  const qualityDatasetOptionIds = useMemo(
    () => buildQualityDatasetOptions(qualityAvailableCurvePoints).map((item) => item.id),
    [qualityAvailableCurvePoints]
  );
  const qualityAlgorithmOptionIds = useMemo(
    () => buildQualityAlgorithmOptions(qualityAvailableCurvePoints).map((item) => item.id),
    [qualityAvailableCurvePoints]
  );
  const qualityAttackOptionIds = useMemo(
    () =>
      buildQualityAttackOptions(qualityAvailableCurvePoints)
        .filter((item) => !isIdentityAttackOption(item))
        .map((item) => item.attackPresetId),
    [qualityAvailableCurvePoints]
  );
  const attackDatasetOptionIds = useMemo(
    () => buildQualityDatasetOptions(qualityAvailableCurvePoints).map((item) => item.id),
    [qualityAvailableCurvePoints]
  );
  const attackAttackOptionIds = useMemo(
    () =>
      buildQualityAttackOptions(qualityAvailableCurvePoints)
        .filter((item) => !isIdentityAttackOption(item))
        .map((item) => item.attackPresetId),
    [qualityAvailableCurvePoints]
  );

  useEffect(() => {
    setSelectedAlgorithmIds((current) => {
      const next = current.filter((id) => algorithmIds.includes(id));
      const normalized = next.length > 0 ? next : algorithmIds.slice(0, 3);
      return sameStringArray(current, normalized) ? current : normalized;
    });
  }, [algorithmIds]);

  useEffect(() => {
    let cancelled = false;
    const loadResourceNames = async () => {
      const [algorithms, attacks, datasetCatalog] = await Promise.all([
        fetchAlgorithms().catch(() => [] as AlgorithmVersion[]),
        fetchAttacks().catch(() => [] as AttackPreset[]),
        fetchDatasetCatalog().catch(() => ({ items: [] }))
      ]);
      if (cancelled) {
        return;
      }
      const nextAlgorithmNames: Record<string, string> = {};
      for (const algorithm of algorithms) {
        const label = localizedAlgorithmResourceName(language, algorithm);
        nextAlgorithmNames[algorithm.id] = label;
        if (algorithm.method) {
          nextAlgorithmNames[algorithm.method] = label;
        }
      }
      const nextAttackNames: Record<string, string> = {};
      for (const attack of attacks) {
        const label = localizedAttackResourceName(language, attack);
        nextAttackNames[attack.id] = label;
        nextAttackNames[attack.method] = label;
      }
      setResourceAlgorithmNames(nextAlgorithmNames);
      setResourceAttackNames(nextAttackNames);
      setResourceAttacks(attacks);
      setResourceCatalogCounts({
        datasets: datasetCatalog.items?.length ?? 0,
        algorithms: algorithms.length,
        attacks: countBenchmarkAttackTypes(attacks)
      });
    };
    void loadResourceNames();
    return () => {
      cancelled = true;
    };
  }, [language]);

  useEffect(() => {
    setQualityDatasetIds((current) => reconcileQualitySelection(current, qualityDatasetOptionIds, qualityDatasetOptionIds));
  }, [qualityDatasetOptionIds]);

  useEffect(() => {
    setAttackDatasetIds((current) => reconcileQualitySelection(current, attackDatasetOptionIds, attackDatasetOptionIds));
  }, [attackDatasetOptionIds]);

  useEffect(() => {
    setAttackAttackIds((current) => reconcileQualitySelection(current, attackAttackOptionIds, attackAttackOptionIds));
  }, [attackAttackOptionIds]);

  useEffect(() => {
    const seeded = selectedAlgorithmIds.filter((id) => qualityAlgorithmOptionIds.includes(id));
    const fallback = seeded.length ? seeded : qualityAlgorithmOptionIds.slice(0, Math.min(3, qualityAlgorithmOptionIds.length));
    setQualityAlgorithmIds((current) => reconcileQualitySelection(current, qualityAlgorithmOptionIds, fallback));
  }, [qualityAlgorithmOptionIds, selectedAlgorithmIds]);

  useEffect(() => {
    const fallback =
      qualityAttackFilter !== "all" && qualityAttackOptionIds.includes(qualityAttackFilter)
        ? [qualityAttackFilter]
        : qualityAttackOptionIds;
    setQualityAttackIds((current) => reconcileQualitySelection(current, qualityAttackOptionIds, fallback));
  }, [qualityAttackFilter, qualityAttackOptionIds]);

  useEffect(() => {
    let cancelled = false;
    fetchRuns()
      .then((nextRuns) => {
        if (cancelled) {
          return;
        }
        setRuns(nextRuns);
        setSelectedRunId((current) => {
          if (current && nextRuns.some((run) => run.id === current)) {
            return current;
          }
          const latest =
            nextRuns.find((run) => run.status === "succeeded" || run.status === "partially_failed") ??
            nextRuns.find((run) => run.status !== "queued") ??
            nextRuns[0];
          return latest?.id ?? "";
        });
        if (nextRuns.length === 0) {
          setNotice(t.results.noRealResults);
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setNotice(t.results.apiUnavailable);
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [runRefreshKey, t.results.apiUnavailable, t.results.noRealResults]);

  useEffect(() => {
    if (!selectedRunId) {
      setResults(null);
      setScore(null);
      return;
    }
    let cancelled = false;
    setNotice("");
    const selectedRun = runs.find((run) => run.id === selectedRunId);
    if (selectedRun) {
      setResults((current) =>
        current?.run.id === selectedRun.id ? { ...current, run: selectedRun } : makeRunResultsShell(selectedRun)
      );
      setActiveInsight({
        kind: "run",
        title: selectedRun.taskName || selectedRun.configName || selectedRun.id,
        body: `${selectedRun.completedProgress}/${selectedRun.cells} result units, status ${selectedRun.status}`,
        meta: selectedRun.artifactRoot
      });
      setLoading(false);
    } else {
      setLoading(true);
    }

    fetchRunScore(selectedRunId)
      .then((scoreResponse) => {
        if (!cancelled) {
          setScore(scoreResponse.score);
          setResults((current) => ({
            ...(current?.run.id === scoreResponse.run.id ? current : makeRunResultsShell(scoreResponse.run)),
            run: scoreResponse.run,
            score: scoreResponse.score,
            summaryPath: scoreResponse.summaryPath,
            summaryExists: scoreResponse.summaryExists,
            summary: scoreResponse.summary,
            aggregates: scoreResponse.aggregates ?? []
          }));
          setActiveInsight({
            kind: "run",
            title: scoreResponse.run.taskName || scoreResponse.run.configName || scoreResponse.run.id,
            body: `${scoreResponse.score.curvePoints.length} curve points, status ${scoreResponse.run.status}`,
            meta: scoreResponse.run.artifactRoot
          });
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setResults(null);
          setScore(null);
          setNotice(t.results.apiUnavailable);
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [runs, selectedRunId, t.results.apiUnavailable]);

  const selectedSet = useMemo(() => new Set(selectedAlgorithmIds), [selectedAlgorithmIds]);
  const selectedScoreRows = useMemo(
    () => scoreRows.filter((row) => selectedSet.has(row.algorithmId)),
    [scoreRows, selectedSet]
  );
  const selectedLegacyRows = useMemo(
    () => legacyRanking.filter((row) => selectedSet.has(row.algorithmId)),
    [legacyRanking, selectedSet]
  );
  const summary = useMemo(
    () => buildRunSummary(results, resourceCatalogCounts, resourceAttacks, t.common.status as Record<string, string>, language),
    [language, resourceAttacks, resourceCatalogCounts, results, t.common.status]
  );
  const aggregateRows = useMemo(
    () =>
      (results?.aggregates ?? [])
        .filter((item) => selectedSet.size === 0 || selectedSet.has(item.algorithmId))
        .map((item) => ({ aggregate: item, point: findScorePoint(score, item) })),
    [results, score, selectedSet]
  );
  const overviewMainRadarCategories = useMemo(
    () => buildMainOverviewRadarTemplate(language),
    [language]
  );
  const overviewMainRadarSeries = useMemo(
    () =>
      buildMainOverviewRadarSeries(
        selectedScoreRows,
        scoreRows,
        score?.attackLeaderboard ?? [],
        overviewMainRadarCategories,
        algorithmSimplicityScore,
        Object.fromEntries(
          selectedScoreRows.map((row) => [row.algorithmId, displayAlgorithm(row.algorithmId, resourceAlgorithmNames)])
        )
      ),
    [overviewMainRadarCategories, resourceAlgorithmNames, score?.attackLeaderboard, scoreRows, selectedScoreRows]
  );
  const overviewDetailRadars = useMemo(
    () =>
      buildOverviewDetailRadars(
        selectedScoreRows,
        score?.attackLeaderboard ?? [],
        Object.fromEntries(
          selectedScoreRows.map((row) => [row.algorithmId, displayAlgorithm(row.algorithmId, resourceAlgorithmNames)])
        ),
        language
      ),
    [language, resourceAlgorithmNames, score?.attackLeaderboard, selectedScoreRows]
  );

  const emptyCopy =
    language === "zh"
      ? {
          loadingTitle: "正在读取运行结果",
          loadingBody: "如果 worker 正在执行，结果会在运行完成后出现。",
          title: "暂无可展示的真实结果",
          body: "请先创建测评配置，在运行页提交并启动 worker。完成后这里会显示 WRS、覆盖率、曲线和调试单元。",
          openRuns: "去运行页",
          openConfigs: "去配置页"
        }
      : {
          loadingTitle: "Loading run results",
          loadingBody: "If a worker is running, results will appear here after completion.",
          title: "No real results to show yet",
          body: "Create a config, submit it on Runs, and start a worker. WRS, coverage, and curves appear after completion.",
          openRuns: "Open Runs",
          openConfigs: "Open Configs"
        };
  const insightKindLabel = (kind: ChartInsight["kind"]) =>
    ({
      algorithm: uiText("算法详情", "Algorithm detail"),
      attack: uiText("攻击详情", "Attack detail"),
      category: uiText("类别详情", "Category detail"),
      curve: uiText("选中数据点", "Selected point"),
      run: uiText("运行详情", "Run detail")
    })[kind] ?? kind;
  const makeCurvePointInsight = (point: BenchmarkCurvePoint, key = qualityPointKey(point)): ChartInsight => {
    const strengthName = point.attackParamStrengthName ?? "strength";
    const strengthValue = formatStrength(point.attackParamStrength ?? point.attackStrength);
    const algorithmName = displayAlgorithm(point.algorithmId, resourceAlgorithmNames);
    const attackName = displayAttackPoint(point, resourceAttackNames);
    return {
      kind: "curve",
      key,
      title: `${algorithmName} × ${attackName}`,
      body: `${point.datasetId || "unknown"} / ${variantLabel(point)} / ${strengthName} ${strengthValue}`,
      details: [
        { label: uiText("数据集", "Dataset"), value: point.datasetId || "unknown" },
        { label: uiText("水印算法", "Watermark"), value: algorithmName },
        { label: uiText("攻击方法", "Attack"), value: attackName },
        { label: uiText("攻击预设", "Preset"), value: point.attackPresetId ?? "n/a" },
        { label: uiText("变体", "Variant"), value: variantLabel(point) },
        { label: strengthName, value: strengthValue },
        { label: "TPR", value: formatMetric(point.yTprAtFpr) },
        { label: "NQD", value: formatMetric(point.xNqd) },
        { label: uiText("样本数", "Samples"), value: formatCount(point.sampleCount) },
        { label: uiText("类别", "Category"), value: point.attackCategory ?? "n/a" }
      ],
      meta: `${point.attackCategory ?? "n/a"} / ${point.attackPresetId ?? "n/a"}`
    };
  };

  const tabComponentCache = useRef<{
    language: typeof language;
    resourceAlgorithmNames: typeof resourceAlgorithmNames;
    resourceAttackNames: typeof resourceAttackNames;
    AttackAnalysisTab: MemoExoticComponent<typeof AttackAnalysisTab>;
    QualityWorkbenchTab: MemoExoticComponent<typeof QualityWorkbenchTab>;
  } | null>(null);
  if (
    !tabComponentCache.current ||
    tabComponentCache.current.language !== language ||
    tabComponentCache.current.resourceAlgorithmNames !== resourceAlgorithmNames ||
    tabComponentCache.current.resourceAttackNames !== resourceAttackNames
  ) {
    tabComponentCache.current = {
      language,
      resourceAlgorithmNames,
      resourceAttackNames,
      AttackAnalysisTab: memo(AttackAnalysisTab),
      QualityWorkbenchTab: memo(QualityWorkbenchTab)
    };
  }
  const StableAttackAnalysisTab = tabComponentCache.current.AttackAnalysisTab;
  const StableQualityWorkbenchTab = tabComponentCache.current.QualityWorkbenchTab;

  return (
    <AppShell active="results">
      <div className="topbar run-picker-topbar">
        <div className="title-block">
          <h1>{t.results.title}</h1>
        </div>
        <div className="toolbar run-picker-toolbar">
          <select
            aria-label={uiText("选择测评结果", "Select run result")}
            className="run-result-select"
            disabled={runs.length === 0}
            onChange={(event) => setSelectedRunId(event.target.value)}
            value={selectedRunId}
          >
            {runs.length === 0 ? <option value="">{loading ? uiText("正在加载…", "Loading…") : uiText("暂无运行", "No runs")}</option> : null}
            {runs.map((run) => (
              <option key={run.id} value={run.id}>
                {friendlyRunRecordName(run, language).slice(0, 72)} / {t.common.status[run.status]}
              </option>
            ))}
          </select>
          <button
            aria-label={t.common.refresh}
            className="button icon-button result-refresh-button"
            onClick={() => setRunRefreshKey((value) => value + 1)}
            title={t.common.refresh}
            type="button"
          >
            <RefreshCw size={16} />
          </button>
          <button
            className="button result-export-button"
            disabled={!results || !selectedRunId}
            onClick={() => exportResultsCsv(selectedRunId)}
            type="button"
          >
            <Download size={16} />
            {t.results.exportCsv}
          </button>
        </div>
      </div>

      {notice ? <div className={`risk ${notice === t.results.apiUnavailable ? "error" : "warn"}`} role={notice === t.results.apiUnavailable ? "alert" : "status"}>{notice}</div> : null}

      {!results ? (
        <PageState
          actions={!loading ? (
            <>
              <Link className="button primary" href="/runs">
                <PlayCircle size={16} />
                {emptyCopy.openRuns}
              </Link>
              <Link className="button" href="/configs">
                <SlidersHorizontal size={16} />
                {emptyCopy.openConfigs}
              </Link>
            </>
          ) : undefined}
          description={loading ? emptyCopy.loadingBody : emptyCopy.body}
          icon={loading ? LoaderCircle : Info}
          title={loading ? emptyCopy.loadingTitle : emptyCopy.title}
          tone={loading ? "loading" : notice === t.results.apiUnavailable ? "error" : "empty"}
        />
      ) : (
        <div className="results-workspace">
      <section className="results-summary-grid">
        <SummaryCard
          iconKind="experiment"
          iconTitle={uiText("可追踪的测评运行实例", "Traceable experiment run")}
          label={language === "zh" ? "测评名称" : "Experiment"}
          meta={summary.experimentMeta}
          value={summary.experimentName}
        />
        <SummaryCard
          iconKind="status"
          iconTitle={uiText("测评运行状态与完成进度", "Run status and completion progress")}
          label={language === "zh" ? "测评状态" : "Status"}
          meta={summary.statusMeta}
          value={summary.statusLabel}
        />
        <SummaryCard
          iconKind="dataset"
          iconTitle={uiText("标准化图像样本集合", "Standardized image sample collection")}
          label={language === "zh" ? "数据集" : "Datasets"}
          meta={summary.datasetMeta}
          showMeta={false}
          value={summary.datasetValue}
        />
        <SummaryCard
          iconKind="watermark"
          iconTitle={uiText("隐藏嵌入与可检测标识", "Hidden embedding and detectable mark")}
          label={language === "zh" ? "水印算法" : "Watermarks"}
          meta={summary.watermarkMeta}
          showMeta={false}
          value={summary.watermarkValue}
        />
        <SummaryCard
          iconKind="attack"
          iconTitle={uiText("鲁棒性压力测试与失真模拟", "Robustness stress tests and distortion simulation")}
          label={language === "zh" ? "攻击算法" : "Attacks"}
          meta={summary.attackMeta}
          showMeta={false}
          value={summary.attackValue}
        />
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

      {activeInsight && activeInsight.kind !== "run" && activeTab !== "quality" && activeTab !== "overview" ? (
        <InsightStrip insight={activeInsight} />
      ) : null}

      {activeTab === "overview" ? (
        <OverviewTab
          algorithmIds={algorithmIds}
          allScoreRows={scoreRows}
          legacyRows={selectedLegacyRows}
          qualityAvailableCurvePoints={qualityAvailableCurvePoints}
          resourceAlgorithmNames={resourceAlgorithmNames}
          overviewDetailRadars={overviewDetailRadars}
          overviewMainRadarCategories={overviewMainRadarCategories}
          overviewMainRadarSeries={overviewMainRadarSeries}
          results={results}
          score={score}
          scoreRows={selectedScoreRows}
          selectedAlgorithmIds={selectedAlgorithmIds}
          setSelectedAlgorithmIds={setSelectedAlgorithmIds}
        />
      ) : null}

      {activeTab === "attack" ? (
        <StableAttackAnalysisTab
          activeSelectorKey={activeAttackSelectorKey}
          attackAttackIds={attackAttackIds}
          attackDatasetIds={attackDatasetIds}
          attackHeatmapMetric={attackHeatmapMetric}
          attackHeatmapRowMode={attackHeatmapRowMode}
          qualityAvailableCurvePoints={qualityAvailableCurvePoints}
          resourceAlgorithmNames={resourceAlgorithmNames}
          resourceAttackNames={resourceAttackNames}
          score={score}
          selectedAttackHeatmapCell={selectedAttackHeatmapCell}
          setActiveSelectorKey={setActiveAttackSelectorKey}
          setAttackAttackIds={setAttackAttackIds}
          setAttackDatasetIds={setAttackDatasetIds}
          setAttackHeatmapMetric={setAttackHeatmapMetric}
          setAttackHeatmapRowMode={setAttackHeatmapRowMode}
          setSelectedAttackHeatmapCell={setSelectedAttackHeatmapCell}
        />
      ) : null}

      {activeTab === "quality" ? (
        <StableQualityWorkbenchTab
          qualityAttackFilter={qualityAttackFilter}
          qualityAlgorithmIds={qualityAlgorithmIds}
          qualityAttackIds={qualityAttackIds}
          qualityDatasetIds={qualityDatasetIds}
          qualityAvailableCurvePoints={qualityAvailableCurvePoints}
          resourceAlgorithmNames={resourceAlgorithmNames}
          resourceAttackNames={resourceAttackNames}
          results={results}
          score={score}
          scoreRows={selectedScoreRows}
          selectedAlgorithmIds={selectedAlgorithmIds}
          activeSelectorKey={activeQualitySelectorKey}
          setActiveSelectorKey={setActiveQualitySelectorKey}
          setQualityAlgorithmIds={setQualityAlgorithmIds}
          setQualityAttackFilter={setQualityAttackFilter}
          setQualityAttackIds={setQualityAttackIds}
          setQualityDatasetIds={setQualityDatasetIds}
        />
      ) : null}

        </div>
      )}
    </AppShell>
  );

  function InsightStrip({ insight }: { insight: ChartInsight }) {
    const hasDetails = Boolean(insight.details?.length);
    return (
      <section className={hasDetails ? "insight-strip dense" : "insight-strip"}>
        <div className="insight-heading">
          <span>{insightKindLabel(insight.kind)}</span>
          <strong>{insight.title}</strong>
        </div>
        {hasDetails ? (
          <div className="insight-detail-grid">
            {insight.details?.map((item) => (
              <div className="insight-detail-item" key={`${item.label}-${item.value}`}>
                <span>{item.label}</span>
                <strong>{item.value}</strong>
              </div>
            ))}
          </div>
        ) : (
          <div className="insight-copy">
            <p>{insight.body}</p>
            {insight.meta ? <code>{insight.meta}</code> : null}
          </div>
        )}
      </section>
    );
  }

  function OverviewTab({
    algorithmIds,
    allScoreRows,
    legacyRows,
    qualityAvailableCurvePoints,
    resourceAlgorithmNames,
    overviewDetailRadars,
    overviewMainRadarCategories,
    overviewMainRadarSeries,
    results,
    score,
    scoreRows,
    selectedAlgorithmIds,
    setSelectedAlgorithmIds
  }: {
    algorithmIds: string[];
    allScoreRows: BenchmarkLeaderboardRow[];
    legacyRows: ReturnType<typeof rankAggregates>;
    qualityAvailableCurvePoints: BenchmarkCurvePoint[];
    resourceAlgorithmNames: Record<string, string>;
    overviewDetailRadars: ReturnType<typeof buildOverviewDetailRadars>;
    overviewMainRadarCategories: ReturnType<typeof buildMainOverviewRadarTemplate>;
    overviewMainRadarSeries: ReturnType<typeof buildMainOverviewRadarSeries>;
    results: RunResults | null;
    score: BenchmarkScore | null;
    scoreRows: BenchmarkLeaderboardRow[];
    selectedAlgorithmIds: string[];
    setSelectedAlgorithmIds: (value: string[] | ((current: string[]) => string[])) => void;
  }) {
    const algorithmOptions =
      qualityAvailableCurvePoints.length > 0
        ? buildQualityAlgorithmOptions(qualityAvailableCurvePoints, resourceAlgorithmNames)
        : algorithmIds.map((algorithmId) => {
            const row = allScoreRows.find((item) => item.algorithmId === algorithmId);
            return {
              id: algorithmId,
              label: displayAlgorithm(algorithmId, resourceAlgorithmNames),
              meta: `${algorithmId} / ${row?.cellCount ?? 0} ${uiText("个单元", "cells")}`,
              count: row?.cellCount ?? 0
            };
          });

    const algorithmColorDomain = useMemo(
      () => [...new Set(scoreRows.map((row) => row.algorithmId))].sort((left, right) => left.localeCompare(right)),
      [scoreRows]
    );

    return (
      <div className="overview-stack">
        <section className="panel quality-selector-panel overview-algorithm-panel">
          <div className="panel-header">
            <h2>{uiText("水印算法评估", "Watermark algorithm evaluation")}</h2>
            <Filter size={16} />
          </div>
          <div className="panel-body">
            <div className="quality-selector-grid overview-algorithm-grid">
              <div className="quality-selector-trigger active overview-algorithm-trigger">
                <span>{uiText("水印算法", "Watermark algorithms")}</span>
                <strong>
                  {selectedAlgorithmIds.length}/{algorithmOptions.length}
                </strong>
              </div>
            </div>
            <div className="quality-selector-drawer overview-algorithm-drawer">
              <div className="quality-selector-actions">
                <button onClick={() => setSelectedAlgorithmIds(algorithmOptions.map((item) => item.id))} type="button">
                  {uiText("全选", "All")}
                </button>
                <button onClick={() => setSelectedAlgorithmIds([])} type="button">
                  {uiText("清空", "Clear")}
                </button>
              </div>
              <div className="quality-selector-list overview-algorithm-list">
                {algorithmOptions.map((item) => (
                  <label className="quality-selector-option" key={item.id}>
                    <input
                      checked={selectedAlgorithmIds.includes(item.id)}
                      onChange={() =>
                        setSelectedAlgorithmIds((current) =>
                          current.includes(item.id) ? current.filter((id) => id !== item.id) : [...current, item.id]
                        )
                      }
                      type="checkbox"
                    />
                    <span>
                      <strong>{item.label}</strong>
                    </span>
                    <b>{item.count}</b>
                  </label>
                ))}
              </div>
            </div>
            <div className="quality-selection-summary">
              <span>
                {selectedAlgorithmIds.length} {uiText("个已选算法", "algorithms selected")} / {scoreRows.length}{" "}
                {uiText("个参与鲁棒性对比", "in robustness comparison")}
              </span>
            </div>
          </div>
        </section>

        <section className="panel overview-radar-panel">
          <div className="panel-header">
            <h2>{uiText("算法综合雷达图", "Algorithm overview radar")}</h2>
            <BarChart3 size={16} />
          </div>
          <div className="panel-body overview-radar-body">
            <BenchmarkRadar
              categories={overviewMainRadarCategories}
              emptyText={t.common.noData}
              onSelectCategory={(category) =>
                setActiveInsight({
                  kind: "category",
                  key: category.key,
                  title: category.label,
                  body: `${uiText("得分", "Score")} ${formatMetric(category.score)}`,
                  meta: category.key
                })
              }
              selectedCategoryKey={activeInsight?.kind === "category" ? activeInsight.key : undefined}
              series={overviewMainRadarSeries}
              colorDomain={algorithmColorDomain}
              scoreLabel={uiText("得分", "Score")}
              variant="hero"
            />
          </div>
        </section>

        <section className="overview-radar-detail-grid">
          {overviewDetailRadars.map((detail) => (
            <div className="panel overview-radar-detail-panel" key={detail.categoryKey}>
              <div className="panel-header">
                <div className="overview-radar-detail-heading">
                  <h2>{detail.title}</h2>
                </div>
                <BarChart3 size={16} />
              </div>
              <div className="panel-body overview-radar-detail-body">
                {detail.categories.length > 0 ? (
                  <BenchmarkRadar
                    categories={detail.categories}
                    colorDomain={algorithmColorDomain}
                    emptyText={t.common.noData}
                    onSelectCategory={(category) =>
                      setActiveInsight({
                        kind: "category",
                        key: `${detail.categoryKey}:${category.key}`,
                        title: `${detail.title} / ${category.label}`,
                        body: `${uiText("得分", "Score")} ${formatMetric(category.score)}`,
                        meta: category.key
                      })
                    }
                    selectedCategoryKey={
                      activeInsight?.kind === "category" && activeInsight.key?.startsWith(`${detail.categoryKey}:`)
                        ? activeInsight.key.slice(detail.categoryKey.length + 1)
                        : undefined
                    }
                    scoreLabel={uiText("得分", "Score")}
                    series={detail.series}
                  />
                ) : (
                  <div className="empty compact-empty">{t.common.noData}</div>
                )}
              </div>
            </div>
          ))}
        </section>

        {scoreRows.length === 0 && results ? (
          <section className="panel">
            <div className="panel-header">
              <h2>{t.results.benchmarkScore}</h2>
              <Trophy size={16} />
            </div>
            <div className="panel-body table-scroll">
              <LegacyRowsTable rows={legacyRows} />
            </div>
          </section>
        ) : null}
      </div>
    );
  }

  function AttackAnalysisTab({
    activeSelectorKey,
    attackAttackIds,
    attackDatasetIds,
    attackHeatmapMetric,
    attackHeatmapRowMode,
    qualityAvailableCurvePoints,
    resourceAlgorithmNames,
    resourceAttackNames,
    score,
    selectedAttackHeatmapCell,
    setActiveSelectorKey,
    setAttackAttackIds,
    setAttackDatasetIds,
    setAttackHeatmapMetric,
    setAttackHeatmapRowMode,
    setSelectedAttackHeatmapCell
  }: {
    activeSelectorKey: AttackSelectorKey | null;
    attackAttackIds: string[];
    attackDatasetIds: string[];
    attackHeatmapMetric: AttackHeatmapMetric;
    attackHeatmapRowMode: AttackHeatmapRowMode;
    qualityAvailableCurvePoints: BenchmarkCurvePoint[];
    resourceAlgorithmNames: Record<string, string>;
    resourceAttackNames: Record<string, string>;
    score: BenchmarkScore | null;
    selectedAttackHeatmapCell: AttackHeatmapCell | null;
    setActiveSelectorKey: AttackSelectorSetter;
    setAttackAttackIds: StringArraySetter;
    setAttackDatasetIds: StringArraySetter;
    setAttackHeatmapMetric: (value: AttackHeatmapMetric) => void;
    setAttackHeatmapRowMode: (value: AttackHeatmapRowMode) => void;
    setSelectedAttackHeatmapCell: (value: AttackHeatmapCell | null) => void;
  }) {
    const allPoints = qualityAvailableCurvePoints;
    const datasetOptions = useMemo(() => buildQualityDatasetOptions(allPoints), [allPoints]);
    const attackOptions = useMemo(
      () => buildQualityAttackOptions(allPoints).filter((item) => !isIdentityAttackOption(item)),
      [allPoints]
    );
    const filteredPoints = useMemo(() => {
      const selectedDatasetSet = new Set(attackDatasetIds);
      const selectedAttackSet = new Set(attackAttackIds);
      return allPoints.filter(
        (point) =>
          selectedDatasetSet.has(point.datasetId || "unknown") &&
          selectedAttackSet.has(point.attackPresetId)
      );
    }, [allPoints, attackAttackIds, attackDatasetIds]);
    const categorySummaries = useMemo(
      () => buildAttackCategoryDistributions(filteredPoints, resourceAttackNames),
      [filteredPoints, resourceAttackNames]
    );
    const attackCount = useMemo(
      () => new Set(filteredPoints.map((point) => point.attackPresetId)).size,
      [filteredPoints]
    );
    const filteredAlgorithmCount = useMemo(
      () => new Set(filteredPoints.map((point) => point.algorithmId)).size,
      [filteredPoints]
    );
    const selectorConfigs: Array<{
      key: AttackSelectorKey;
      options: QualitySelectorOption[];
      selectedIds: string[];
      setSelectedIds: StringArraySetter;
      title: string;
    }> = [
      {
        key: "dataset",
        options: datasetOptions,
        selectedIds: attackDatasetIds,
        setSelectedIds: setAttackDatasetIds,
        title: uiText("数据集", "Datasets")
      },
      {
        key: "attack",
        options: attackOptions.map((item) => ({
          id: item.attackPresetId,
          label: displayAttackByIds(item.attackPresetId, item.attackMethod, resourceAttackNames),
          meta: `${displayAttackCategory(normalizeAttackCategory(item.attackCategory, item.attackPresetId, item.attackMethod))} / ${
            item.variantCount
          } ${uiText("个变体", "variants")}`,
          count: item.pointCount
        })),
        selectedIds: attackAttackIds,
        setSelectedIds: setAttackAttackIds,
        title: uiText("攻击算法", "Attack algorithms")
      }
    ];
    const activeSelector = activeSelectorKey ? selectorConfigs.find((item) => item.key === activeSelectorKey) ?? null : null;
    const handlePickHeatmapCell = useCallback((cell: AttackHeatmapCell) => {
      setSelectedAttackHeatmapCell(cell);
      setActiveInsight({
        kind: "curve",
        key: `${cell.rowMode}:${cell.rowKey}:${cell.algorithmId}`,
        title: `${cell.algorithmLabel} × ${cell.rowLabel}`,
        body: `TPR ${formatMetric(cell.avgTpr)}, NQD ${formatMetric(cell.avgNqd)}`,
        meta: `${cell.attackPresetIds.length} ${uiText("个攻击", "attacks")} / ${cell.pointCount} ${uiText("个点", "points")}`
      });
    }, [setActiveInsight, setSelectedAttackHeatmapCell, uiText]);
    const handlePickViolinPoint = useCallback((point: BenchmarkCurvePoint) => {
      setSelectedAttackHeatmapCell(null);
      setActiveInsight(makeCurvePointInsight(point));
    }, [makeCurvePointInsight, setActiveInsight, setSelectedAttackHeatmapCell]);

    return (
      <>
        <section className="panel attack-selector-panel">
          <div className="panel-header">
            <h2>{uiText("攻击分析筛选", "Attack analysis filters")}</h2>
            <Filter size={16} />
          </div>
          <div className="panel-body">
            <div className="quality-selector-grid attack-selector-grid">
              {selectorConfigs.map((config) => (
                <button
                  aria-expanded={activeSelectorKey === config.key}
                  className={activeSelectorKey === config.key ? "quality-selector-trigger active" : "quality-selector-trigger"}
                  key={config.key}
                  onClick={() => setActiveSelectorKey((current) => (current === config.key ? null : config.key))}
                  type="button"
                >
                  <span>{config.title}</span>
                  <strong>
                    {config.selectedIds.length}/{config.options.length}
                  </strong>
                </button>
              ))}
            </div>
            {activeSelector ? (
              <div className="quality-selector-drawer">
                <div className="quality-selector-actions">
                  <button onClick={() => activeSelector.setSelectedIds(activeSelector.options.map((item) => item.id))} type="button">
                    {uiText("全选", "All")}
                  </button>
                  <button onClick={() => activeSelector.setSelectedIds([])} type="button">
                    {uiText("清空", "Clear")}
                  </button>
                </div>
                <div className="quality-selector-list attack-selector-list">
                  {activeSelector.options.map((item) => (
                    <label className="quality-selector-option" key={item.id}>
                      <input
                        checked={activeSelector.selectedIds.includes(item.id)}
                        onChange={() =>
                          activeSelector.setSelectedIds((current) =>
                            current.includes(item.id) ? current.filter((id) => id !== item.id) : [...current, item.id]
                          )
                        }
                        type="checkbox"
                      />
                      <span>
                        <strong>{item.label}</strong>
                      </span>
                      <b>{item.count}</b>
                    </label>
                  ))}
                </div>
              </div>
            ) : null}
            <div className="quality-selection-summary">
              <span>
                {filteredPoints.length} {uiText("个曲线点", "curve points")} / {attackCount}{" "}
                {uiText("种攻击", "attacks")}
              </span>
              <span>
                {filteredAlgorithmCount} {uiText("个水印算法全部纳入", "watermark algorithms included")}
              </span>
              <span>
                {categorySummaries.length} {uiText("个攻击类别", "attack categories")}
              </span>
            </div>
          </div>
        </section>

        <section className="panel attack-visual-panel">
          <div className="panel-header">
            <h2>{uiText("攻击类别分布", "Attack category distribution")}</h2>
            <Gauge size={16} />
          </div>
          <div className="panel-body">
            <MemoizedAttackViolinPlot
              categories={categorySummaries}
              onPickPoint={handlePickViolinPoint}
            />
            <div className="attack-heatmap-section">
              <div className="attack-subheading">
                <div>
                  <strong>{uiText("水印弱点热力图", "Watermark weakness heatmap")}</strong>
                  <span>
                    {uiText(
                      "行是水印算法，列是攻击类别或攻击方法；点击格子查看该组合摘要。",
                      "Rows are watermark algorithms, columns are attack categories or methods; click a cell for a summary."
                    )}
                  </span>
                </div>
                <div className="attack-heatmap-controls">
                  <button
                    className={attackHeatmapRowMode === "category" ? "active" : ""}
                    onClick={() => {
                      setAttackHeatmapRowMode("category");
                      setSelectedAttackHeatmapCell(null);
                    }}
                    type="button"
                  >
                    {uiText("按类别", "Categories")}
                  </button>
                  <button
                    className={attackHeatmapRowMode === "attack" ? "active" : ""}
                    onClick={() => {
                      setAttackHeatmapRowMode("attack");
                      setSelectedAttackHeatmapCell(null);
                    }}
                    type="button"
                  >
                    {uiText("按攻击", "Attacks")}
                  </button>
                  <button
                    className={attackHeatmapMetric === "tpr" ? "active" : ""}
                    onClick={() => setAttackHeatmapMetric("tpr")}
                    type="button"
                  >
                    TPR
                  </button>
                  <button
                    className={attackHeatmapMetric === "nqd" ? "active" : ""}
                    onClick={() => setAttackHeatmapMetric("nqd")}
                    type="button"
                  >
                    NQD
                  </button>
                </div>
              </div>
              <MemoizedAttackHeatmapMatrix
                algorithmNames={resourceAlgorithmNames}
                attackNames={resourceAttackNames}
                metric={attackHeatmapMetric}
                onPickCell={handlePickHeatmapCell}
                points={filteredPoints}
                rowMode={attackHeatmapRowMode}
                selectedCell={selectedAttackHeatmapCell}
                uiText={uiText}
              />
            </div>
          </div>
        </section>
      </>
    );
  }

  function QualityWorkbenchTab({
    qualityAlgorithmIds,
    qualityAttackFilter,
    qualityAttackIds,
    qualityDatasetIds,
    qualityAvailableCurvePoints,
    resourceAlgorithmNames,
    resourceAttackNames,
    results,
    score,
    selectedAlgorithmIds,
    scoreRows,
    activeSelectorKey,
    setActiveSelectorKey,
    setQualityAlgorithmIds,
    setQualityAttackFilter,
    setQualityAttackIds,
    setQualityDatasetIds
  }: {
    qualityAlgorithmIds: string[];
    qualityAttackFilter: string;
    qualityAttackIds: string[];
    qualityDatasetIds: string[];
    qualityAvailableCurvePoints: BenchmarkCurvePoint[];
    resourceAlgorithmNames: Record<string, string>;
    resourceAttackNames: Record<string, string>;
    results: RunResults | null;
    score: BenchmarkScore | null;
    selectedAlgorithmIds: string[];
    scoreRows: BenchmarkLeaderboardRow[];
    activeSelectorKey: QualitySelectorKey | null;
    setActiveSelectorKey: QualitySelectorSetter;
    setQualityAlgorithmIds: StringArraySetter;
    setQualityAttackFilter: (value: string) => void;
    setQualityAttackIds: StringArraySetter;
    setQualityDatasetIds: StringArraySetter;
  }) {
    const allCurvePoints = qualityAvailableCurvePoints;
    const datasetOptions = useMemo(() => buildQualityDatasetOptions(allCurvePoints), [allCurvePoints]);
    const algorithmOptions = useMemo(
      () => buildQualityAlgorithmOptions(allCurvePoints, resourceAlgorithmNames),
      [allCurvePoints, resourceAlgorithmNames]
    );
    const attackOptions = useMemo(
      () => buildQualityAttackOptions(allCurvePoints).filter((item) => !isIdentityAttackOption(item)),
      [allCurvePoints]
    );
    const selectedDatasetSet = useMemo(() => new Set(qualityDatasetIds), [qualityDatasetIds]);
    const selectedAlgorithmSet = useMemo(() => new Set(qualityAlgorithmIds), [qualityAlgorithmIds]);
    const selectedAttackSet = useMemo(() => new Set(qualityAttackIds), [qualityAttackIds]);
    const filteredCurvePoints = useMemo(
      () =>
        allCurvePoints.filter(
          (point) =>
            selectedDatasetSet.has(point.datasetId) &&
            selectedAlgorithmSet.has(point.algorithmId) &&
            selectedAttackSet.has(point.attackPresetId)
        ),
      [allCurvePoints, selectedAlgorithmSet, selectedAttackSet, selectedDatasetSet]
    );
    const attackSummaries = useMemo(
      () =>
        buildQualityAttackSummaries(
          allCurvePoints.filter(
            (point) => selectedDatasetSet.has(point.datasetId) && selectedAlgorithmSet.has(point.algorithmId)
          )
        ),
      [allCurvePoints, selectedAlgorithmSet, selectedDatasetSet]
    );
    const comboSummaries = useMemo(() => buildQualityComboSummaries(filteredCurvePoints), [filteredCurvePoints]);
    const qualitySeriesDomain = useMemo(() => comboSummaries.map((combo) => combo.key), [comboSummaries]);
    const weakestPoint = useMemo(
      () =>
        filteredCurvePoints.reduce<BenchmarkCurvePoint | null>(
          (current, point) => (current == null || point.yTprAtFpr < current.yTprAtFpr ? point : current),
          null
        ),
      [filteredCurvePoints]
    );
    const selectedAttackCount = useMemo(
      () => new Set(filteredCurvePoints.map((point) => point.attackPresetId)).size,
      [filteredCurvePoints]
    );
    const selectedStrengthCount = useMemo(
      () =>
        new Set(
          filteredCurvePoints.map((point) => `${point.attackPresetId}:${point.attackParamStrength ?? point.attackStrength}`)
        ).size,
      [filteredCurvePoints]
    );
    const selectedVariantCount = useMemo(
      () => new Set(filteredCurvePoints.map((point) => point.attackVariantKey ?? "default")).size,
      [filteredCurvePoints]
    );
    const handleQualityPointSelect = useCallback(
      (point: BenchmarkCurvePoint) => setActiveInsight(makeCurvePointInsight(point)),
      [makeCurvePointInsight, setActiveInsight]
    );
    const selectorConfigs: Array<{
      key: QualitySelectorKey;
      options: QualitySelectorOption[];
      selectedIds: string[];
      setSelectedIds: StringArraySetter;
      title: string;
    }> = [
      {
        key: "dataset",
        options: datasetOptions,
        selectedIds: qualityDatasetIds,
        setSelectedIds: setQualityDatasetIds,
        title: uiText("数据集", "Datasets")
      },
      {
        key: "algorithm",
        options: algorithmOptions,
        selectedIds: qualityAlgorithmIds,
        setSelectedIds: setQualityAlgorithmIds,
        title: uiText("水印算法", "Watermark algorithms")
      },
      {
        key: "attack",
        options: attackOptions.map((item) => ({
          id: item.attackPresetId,
          label: displayAttackByIds(item.attackPresetId, item.attackMethod, resourceAttackNames),
          meta: `${item.attackPresetId} / ${item.attackCategory} / ${item.variantCount} ${uiText("个变体", "variants")}`,
          count: item.pointCount
        })),
        selectedIds: qualityAttackIds,
        setSelectedIds: (value) => {
          setQualityAttackIds(value);
          if (typeof value !== "function") {
            setQualityAttackFilter(value.length === 1 ? value[0] : "all");
          }
        },
        title: uiText("攻击方法", "Attack methods")
      }
    ];
    const activeSelector = activeSelectorKey ? selectorConfigs.find((item) => item.key === activeSelectorKey) ?? null : null;

    return (
      <>
        <section className="panel quality-selector-panel">
          <div className="panel-header">
            <h2>{uiText("质量-鲁棒性筛选", "Quality-robustness filters")}</h2>
            <Filter size={16} />
          </div>
          <div className="panel-body">
            <div className="quality-selector-grid">
              {selectorConfigs.map((config) => (
                <button
                  aria-expanded={activeSelectorKey === config.key}
                  className={activeSelectorKey === config.key ? "quality-selector-trigger active" : "quality-selector-trigger"}
                  key={config.key}
                  onClick={() => setActiveSelectorKey((current) => (current === config.key ? null : config.key))}
                  type="button"
                >
                  <span>{config.title}</span>
                  <strong>
                    {config.selectedIds.length}/{config.options.length}
                  </strong>
                </button>
              ))}
            </div>
            {activeSelector ? (
            <div className="quality-selector-drawer">
              <div className="quality-selector-actions">
                <button onClick={() => activeSelector.setSelectedIds(activeSelector.options.map((item) => item.id))} type="button">
                  {uiText("全选", "All")}
                </button>
                <button onClick={() => activeSelector.setSelectedIds([])} type="button">
                  {uiText("清空", "Clear")}
                </button>
              </div>
              <div className="quality-selector-list">
                {activeSelector.options.map((item) => (
                  <label className="quality-selector-option" key={item.id}>
                    <input
                      checked={activeSelector.selectedIds.includes(item.id)}
                      onChange={() =>
                        activeSelector.setSelectedIds((current) =>
                          current.includes(item.id) ? current.filter((id) => id !== item.id) : [...current, item.id]
                        )
                      }
                      type="checkbox"
                    />
                    <span>
                      <strong>{item.label}</strong>
                    </span>
                    <b>{item.count}</b>
                  </label>
                ))}
              </div>
            </div>
            ) : null}
            <div className="quality-selection-summary">
              <span>
                {filteredCurvePoints.length} {uiText("个曲线点", "curve points")} / {comboSummaries.length}{" "}
                {uiText("个组合", "combinations")}
              </span>
              <span>
                {selectedAttackCount} {uiText("种攻击", "attacks")} / {selectedVariantCount} {uiText("个变体", "variants")} /{" "}
                {selectedStrengthCount} {uiText("个强度点", "strength points")}
              </span>
              <span>
                {scoreRows.length} {uiText("个排名算法", "ranked algorithms")}
                {qualityAttackFilter !== "all" ? ` / ${displayAttack(qualityAttackFilter, resourceAttackNames)}` : ""}
              </span>
            </div>
          </div>
        </section>

        <section className="quality-analysis-grid">
          <div className="panel quality-chart-panel">
            <div className="panel-header">
              <h2>{t.results.qualityRobustness}</h2>
              <BarChart3 size={16} />
            </div>
            <div className="panel-body">
              <div className="quality-chart-guide">
                <span>{uiText("横轴 NQD: 质量损失", "X: NQD quality loss")}</span>
                <span>{uiText("纵轴 TPR: 检出率", "Y: TPR detection rate")}</span>
                <span>{uiText("图例: 每个组合独立颜色和形状", "Legend: unique color and shape per combination")}</span>
                <span>{uiText("点: 强度或变体", "Point: strength or variant")}</span>
                <span>{uiText("虚线: Q@P95 / Q@P70", "Dashed: Q@P95 / Q@P70")}</span>
              </div>
              <MemoizedRobustnessCurve
                algorithmLabels={resourceAlgorithmNames}
                attackLabels={resourceAttackNames}
                colorDomain={qualitySeriesDomain}
                curvePoints={filteredCurvePoints}
                emptyText={t.console.needMultipleStrengths}
                groupMode="combination"
                onSelectPoint={handleQualityPointSelect}
                performanceThresholds={score?.performanceThresholds}
                pointCaption={uiText("点: 攻击变体 / 强度", "Point: attack variant / strength")}
                results={results}
                score={score}
                selectedAlgorithmIds={qualityAlgorithmIds}
                selectedAttackPresetIds={qualityAttackIds}
                selectedDatasetIds={qualityDatasetIds}
                seriesCaption={uiText("曲线: 数据集 × 水印算法 × 攻击方法", "Series: dataset × watermark × attack")}
                shapeDomain={qualitySeriesDomain}
                showCaption={false}
                showLegend={false}
              />
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <h2>{uiText("组合清单与图例", "Combination list and legend")}</h2>
              <Gauge size={16} />
            </div>
            <div className="panel-body quality-combo-list">
              {comboSummaries.map((combo) => (
                <button
                  className="quality-combo-row"
                  key={combo.key}
                  onClick={() => {
                    if (!combo.weakestPoint) {
                      return;
                    }
                    setActiveInsight({
                      kind: "attack",
                      key: combo.key,
                      title: `${displayAlgorithm(combo.algorithmId, resourceAlgorithmNames)} / ${displayAttackByIds(combo.attackPresetId, combo.attackMethod, resourceAttackNames)}`,
                      body: `${combo.datasetId}, ${combo.attackPresetId}, ${formatParamRange(combo.strengthName, combo.strengthMin, combo.strengthMax)}`,
                      details: [
                        { label: uiText("数据集", "Dataset"), value: combo.datasetId },
                        { label: uiText("水印算法", "Watermark"), value: displayAlgorithm(combo.algorithmId, resourceAlgorithmNames) },
                        { label: uiText("攻击方法", "Attack"), value: displayAttackByIds(combo.attackPresetId, combo.attackMethod, resourceAttackNames) },
                        { label: uiText("攻击预设", "Preset"), value: combo.attackPresetId },
                        { label: uiText("变体", "Variant"), value: combo.variantLabel },
                        { label: uiText("强度范围", "Strength"), value: formatParamRange(combo.strengthName, combo.strengthMin, combo.strengthMax) },
                        { label: "TPR", value: formatMetric(combo.avgTpr) },
                        { label: "NQD", value: formatMetric(combo.avgNqd) },
                        { label: uiText("点数", "Points"), value: String(combo.pointCount) }
                      ],
                      meta: `TPR ${formatMetric(combo.avgTpr)}, NQD ${formatMetric(combo.avgNqd)}, ${combo.variantLabel}`
                    });
                  }}
                  type="button"
                >
                  <CurveLegendGlyph
                    color={curveDomainColor(qualitySeriesDomain, combo.key)}
                    shape={curveDomainShape(qualitySeriesDomain, combo.key)}
                  />
                  <span className="combo-main">
                    <strong>
                      {displayAlgorithm(combo.algorithmId, resourceAlgorithmNames)} -{" "}
                      {displayAttackByIds(combo.attackPresetId, combo.attackMethod, resourceAttackNames)} -{" "}
                      {combo.variantLabel}
                    </strong>
                  </span>
                </button>
              ))}
              {comboSummaries.length === 0 ? <div className="empty compact-empty">{t.common.noData}</div> : null}
            </div>
          </div>
        </section>

      </>
    );
  }

  function QualityTab({
    qualityAttackFilter,
    results,
    score,
    selectedAlgorithmIds,
    scoreRows,
    setQualityAttackFilter
  }: {
    qualityAttackFilter: string;
    results: RunResults | null;
    score: BenchmarkScore | null;
    selectedAlgorithmIds: string[];
    scoreRows: BenchmarkLeaderboardRow[];
    setQualityAttackFilter: (value: string) => void;
  }) {
    const visibleCurvePoints = (score?.curvePoints ?? [])
      .filter((point) => selectedAlgorithmIds.length === 0 || selectedAlgorithmIds.includes(point.algorithmId))
      .sort((left, right) =>
        left.algorithmId.localeCompare(right.algorithmId) ||
        left.attackCategory.localeCompare(right.attackCategory) ||
        left.attackMethod.localeCompare(right.attackMethod) ||
        left.attackPresetId.localeCompare(right.attackPresetId) ||
        (left.attackVariantLabel ?? "default").localeCompare(right.attackVariantLabel ?? "default") ||
        (left.attackParamStrength ?? left.attackStrength) - (right.attackParamStrength ?? right.attackStrength) ||
        left.attackStrength - right.attackStrength
      );
    const attackOptions = buildQualityAttackOptions(visibleCurvePoints);
    const activeAttackFilter = attackOptions.some((item) => item.attackPresetId === qualityAttackFilter)
      ? qualityAttackFilter
      : "all";
    const filteredCurvePoints = visibleCurvePoints.filter(
      (point) => activeAttackFilter === "all" || point.attackPresetId === activeAttackFilter
    );
    const attackSummaries = buildQualityAttackSummaries(visibleCurvePoints);
    const weakestPoint = filteredCurvePoints.reduce<BenchmarkCurvePoint | null>(
      (current, point) => (current == null || point.yTprAtFpr < current.yTprAtFpr ? point : current),
      null
    );
    const selectedAttackCount = new Set(filteredCurvePoints.map((point) => point.attackPresetId)).size;
    const selectedStrengthCount = new Set(
      filteredCurvePoints.map((point) => `${point.attackPresetId}:${point.attackParamStrength ?? point.attackStrength}`)
    ).size;
    const selectedVariantCount = new Set(filteredCurvePoints.map((point) => point.attackVariantKey ?? "default")).size;

    return (
      <>
        <section className="quality-focus-grid">
          <div className="quality-focus-card">
            <span>{uiText("水印算法", "Watermark algorithms")}</span>
            <strong>{scoreRows.length}</strong>
            <small>{scoreRows.map((row) => displayAlgorithm(row.algorithmId)).slice(0, 4).join(" / ") || "n/a"}</small>
          </div>
          <div className="quality-focus-card">
            <span>{uiText("当前攻击方法", "Current attack method")}</span>
            <strong>{selectedAttackCount}</strong>
            <small>
              {selectedVariantCount} {uiText("个变体", "variants")} / {selectedStrengthCount} {uiText("个强度点", "strength points")} / {filteredCurvePoints.length} {uiText("个曲线点", "curve points")}
            </small>
          </div>
          <button
            className="quality-focus-card interactive"
            disabled={!weakestPoint}
            onClick={() =>
              weakestPoint
                ? setActiveInsight({
                    kind: "curve",
                    key: qualityPointKey(weakestPoint),
                    title: `${displayAlgorithm(weakestPoint.algorithmId)} / ${displayAttack(weakestPoint.attackMethod)}`,
                    body: `${weakestPoint.attackPresetId}, ${weakestPoint.attackParamStrengthName ?? "strength"} ${formatStrength(weakestPoint.attackParamStrength ?? weakestPoint.attackStrength)}, TPR ${formatMetric(weakestPoint.yTprAtFpr)}`,
                    meta: `${weakestPoint.attackCategory}, ${variantLabel(weakestPoint)}, NQD ${formatMetric(weakestPoint.xNqd)}`
                  })
                : undefined
            }
            type="button"
          >
            <span>{uiText("最弱鲁棒点", "Weakest robustness point")}</span>
            <strong>{weakestPoint ? formatMetric(weakestPoint.yTprAtFpr) : "n/a"}</strong>
            <small>
              {weakestPoint
                ? `${displayAlgorithm(weakestPoint.algorithmId)} / ${displayAttack(weakestPoint.attackMethod)}`
                : "n/a"}
            </small>
          </button>
        </section>

        <section className="panel quality-filter-panel">
          <div className="panel-header">
            <h2>{uiText("攻击方法筛选", "Attack method filter")}</h2>
            <Filter size={16} />
          </div>
          <div className="panel-body quality-filter-body">
            <select
              className="run-result-select"
              onChange={(event) => setQualityAttackFilter(event.target.value)}
              value={activeAttackFilter}
            >
              <option value="all">{uiText("全部攻击方法", "All attack methods")}</option>
              {attackOptions.map((item) => (
                <option key={item.attackPresetId} value={item.attackPresetId}>
                  {displayAttack(item.attackMethod)} / {item.attackPresetId} / {item.variantCount} {uiText("个变体", "variants")} / {item.pointCount} {uiText("个点", "points")}
                </option>
              ))}
            </select>
            <div className="quality-filter-meta">
              <strong>
                {activeAttackFilter === "all"
                  ? uiText("曲线按水印算法分组", "Curves grouped by watermark algorithm")
                  : uiText("曲线按水印算法 × 攻击变体分组", "Curves grouped by watermark algorithm × attack variant")}
              </strong>
              <span>
                {activeAttackFilter === "all"
                  ? uiText("选择一个攻击方法后，可以把权重、模型和校正选项拆成不同曲线。", "Pick one attack method to split weights, models, and correction options into separate colors.")
                  : `${displayAttack(attackOptions.find((item) => item.attackPresetId === activeAttackFilter)?.attackMethod ?? activeAttackFilter)}: ${selectedVariantCount} ${uiText("个变体", "variants")}, ${filteredCurvePoints.length} ${uiText("个点", "points")}`}
              </span>
            </div>
          </div>
        </section>

        <section className="results-grid">
          <div className="panel">
            <div className="panel-header">
              <h2>{t.results.qualityRobustness}</h2>
              <BarChart3 size={16} />
            </div>
            <div className="panel-body">
              <RobustnessCurve
                emptyText={t.console.needMultipleStrengths}
                onSelectPoint={(point) =>
                  setActiveInsight({
                    kind: "curve",
                    key: `${point.algorithmId}-${point.attackPresetId}-${point.attackStrength}`,
                    title: `${displayAlgorithm(point.algorithmId)} / ${displayAttack(point.attackMethod ?? point.attackPresetId ?? "")}`,
                    body: `${point.attackPresetId ?? "n/a"}, ${point.attackParamStrengthName ?? "strength"} ${formatStrength(point.attackParamStrength ?? point.attackStrength)}, TPR ${formatMetric(point.yTprAtFpr)}`,
                    meta: `${point.attackCategory ?? "n/a"}, ${variantLabel(point)}, NQD ${formatMetric(point.xNqd)}`
                  })
                }
                pointCaption={uiText("点: 攻击变体 / 强度", "Point: attack variant / strength")}
                results={results}
                score={score}
                selectedAttackPresetId={activeAttackFilter}
                selectedAlgorithmIds={selectedAlgorithmIds}
                seriesCaption={uiText("曲线: 水印算法", "Series: watermark algorithm")}
              />
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <h2>{t.results.auxiliaryMetrics}</h2>
              <Gauge size={16} />
            </div>
            <div className="panel-body quality-method-list">
              {scoreRows.map((row) => (
                <button
                  className="quality-method-card"
                  key={row.algorithmId}
                  onClick={() => {
                    setSelectedAlgorithmIds([row.algorithmId]);
                    setActiveInsight({
                      kind: "algorithm",
                      key: row.algorithmId,
                      title: displayAlgorithm(row.algorithmId),
                      body: `WRS-v2 ${row.wrs == null ? "n/a" : row.wrs.toFixed(1)}, ${uiText("无攻击保真度", "clean fidelity")} ${formatMetric(row.cleanFidelity)}`,
                      meta: `${uiText("平均 NQD", "Avg NQD")} ${formatMetric(row.avgNqd)}, ${uiText("物理信道", "physical")} ${formatMetric(row.physicalScore)}`
                    });
                  }}
                  type="button"
                >
                  <span>{displayAlgorithm(row.algorithmId)}</span>
                  <strong>{row.wrs == null ? "n/a" : row.wrs.toFixed(1)}</strong>
                  <small>
                    {row.algorithmId}
                  </small>
                  <em>
                    Clean {formatMetric(row.cleanFidelity)} / NQD {formatMetric(row.avgNqd)}
                  </em>
                </button>
              ))}
              {scoreRows.length === 0 ? <div className="empty compact-empty">{t.common.noData}</div> : null}
            </div>
          </div>
        </section>

        <section className="panel results-detail-panel">
          <div className="panel-header">
            <h2>{uiText("攻击方法拆解", "Attack method breakdown")}</h2>
            <Gauge size={16} />
          </div>
          <div className="panel-note">
            {attackSummaries.length} {uiText("个有测评结果的攻击方法", "attack methods with experiment results")}
            {selectedAlgorithmIds.length ? ` / ${scoreRows.length} ${uiText("个已选水印算法", "selected watermark algorithms")}` : ""}
          </div>
          <div className="panel-body quality-attack-grid">
            {attackSummaries.map((item) => (
              <button
                className="quality-attack-card"
                key={item.key}
                onClick={() => {
                  setQualityAttackFilter(item.attackPresetId);
                  if (item.weakestPoint) {
                    setActiveInsight({
                        kind: "curve",
                        key: qualityPointKey(item.weakestPoint),
                        title: `${displayAttack(item.attackMethod)} / ${displayAlgorithm(item.weakestPoint.algorithmId)}`,
                        body: `${item.attackPresetId}, ${item.weakestPoint.attackParamStrengthName ?? "strength"} ${formatStrength(item.weakestPoint.attackParamStrength ?? item.weakestPoint.attackStrength)}, TPR ${formatMetric(item.weakestPoint.yTprAtFpr)}`,
                        meta: `${item.attackCategory}, ${variantLabel(item.weakestPoint)}, NQD ${formatMetric(item.weakestPoint.xNqd)}`
                    });
                  }
                }}
                type="button"
              >
                <span>{item.attackCategory}</span>
                <strong>{displayAttack(item.attackMethod)}</strong>
                <small>{item.attackPresetId}</small>
                <div className="quality-card-metrics">
                  <b>TPR {formatMetric(item.avgTpr)}</b>
                  <b>NQD {formatMetric(item.avgNqd)}</b>
                  <b>{formatParamRange(item.strengthName, item.strengthMin, item.strengthMax)}</b>
                </div>
                <em>
                  {item.algorithmCount} {uiText("个算法", "algorithms")} / {item.pointCount} {uiText("个点", "points")}
                  {item.weakestPoint ? ` / ${uiText("最弱", "weakest")} ${displayAlgorithm(item.weakestPoint.algorithmId)}` : ""}
                </em>
              </button>
            ))}
            {attackSummaries.length === 0 ? <div className="empty compact-empty">{t.common.noData}</div> : null}
          </div>
        </section>

        <section className="panel results-detail-panel">
          <div className="panel-header">
            <h2>{t.results.qualityPoints}</h2>
          </div>
          <div className="panel-body table-scroll quality-points-scroll">
            <table className="table quality-points-table">
              <thead>
                <tr>
                  <th>{uiText("水印算法", "Watermark algorithm")}</th>
                  <th>{uiText("攻击方法", "Attack method")}</th>
                  <th>{uiText("预设 / 类别", "Preset / family")}</th>
                  <th>{uiText("变体参数", "Variant")}</th>
                  <th>{t.results.strength}</th>
                  <th>{t.results.tprAtFpr}</th>
                  <th>{t.results.nqd}</th>
                </tr>
              </thead>
              <tbody>
                {filteredCurvePoints.map((point, index) => (
                  <tr
                    className="clickable-row"
                    key={qualityPointKey(point, index)}
                    onClick={() =>
                      setActiveInsight({
                        kind: "curve",
                        key: qualityPointKey(point, index),
                        title: `${displayAlgorithm(point.algorithmId)} / ${displayAttack(point.attackMethod)}`,
                        body: `${point.attackPresetId}, ${point.attackParamStrengthName ?? "strength"} ${formatStrength(point.attackParamStrength ?? point.attackStrength)}, TPR ${formatMetric(point.yTprAtFpr)}`,
                        meta: `${point.attackCategory}, ${variantLabel(point)}, NQD ${formatMetric(point.xNqd)}`
                      })
                    }
                  >
                    <td>
                      <strong>{displayAlgorithm(point.algorithmId)}</strong>
                      <span>{point.algorithmId}</span>
                    </td>
                    <td>
                      <strong>{displayAttack(point.attackMethod)}</strong>
                      <span>{point.attackMethod}</span>
                    </td>
                    <td>
                      <strong>{point.attackPresetId}</strong>
                      <span>{point.attackCategory}</span>
                    </td>
                    <td>
                      <strong>{variantLabel(point)}</strong>
                      <span>{formatAttackParams(point.attackParams)}</span>
                    </td>
                    <td>
                      <b className="metric-pill">
                        {point.attackParamStrengthName ?? "strength"} {formatStrength(point.attackParamStrength ?? point.attackStrength)}
                      </b>
                    </td>
                    <td>
                      <b className={point.yTprAtFpr < 0.7 ? "metric-pill risk" : "metric-pill ok"}>
                        {formatMetric(point.yTprAtFpr)}
                      </b>
                    </td>
                    <td>{formatMetric(point.xNqd)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!filteredCurvePoints.length ? <div className="empty compact-empty">{t.common.noData}</div> : null}
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
            <th>{uiText("物理信道", "Physical")}</th>
            <th>{t.results.cleanFidelity}</th>
            <th>{t.results.nqd}</th>
            <th>{t.common.coverage}</th>
            <th>{uiText("特性标签", "Profile")}</th>
            <th>{t.runs.cells}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              className="clickable-row"
              key={row.algorithmId}
              onClick={() => {
                setSelectedAlgorithmIds([row.algorithmId]);
                setActiveInsight({
                  kind: "algorithm",
                  key: row.algorithmId,
                  title: displayAlgorithm(row.algorithmId, resourceAlgorithmNames),
                  body: `WRS-v2 ${row.wrs == null ? "n/a" : row.wrs.toFixed(1)}, ${uiText("物理信道", "physical")} ${formatMetric(row.physicalScore)}`,
                  meta: displayProfileTags(row.profileTags, language) || row.worstCategory?.label
                });
              }}
            >
              <td>{row.rank}</td>
              <td>{displayAlgorithm(row.algorithmId, resourceAlgorithmNames)}</td>
              <td>{row.wrs == null ? "n/a" : row.wrs.toFixed(1)}</td>
              <td>{formatMetric(row.physicalScore)}</td>
              <td>{formatMetric(row.cleanFidelity)}</td>
              <td>{formatMetric(row.avgNqd)}</td>
              <td>
                {row.coverage.coveredCategoryCount}/{row.coverage.requiredCategoryCount}
              </td>
              <td>{displayProfileTags(row.profileTags, language) || (row.worstCategory?.label ?? "n/a")}</td>
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

  function SummaryCard({
    iconKind,
    iconTitle,
    label,
    meta,
    showMeta = true,
    value
  }: {
    iconKind?: SummaryIconKind;
    iconTitle?: string;
    label: string;
    meta: string;
    showMeta?: boolean;
    value: string;
  }) {
    const summaryInsight = iconKind
      ? {
          kind: "run" as const,
          key: `summary-card:${iconKind}`,
          title: `${label}: ${value}`,
          body: iconTitle ?? meta,
          details: [
            { label: uiText("当前值", "Current"), value },
            { label: uiText("说明", "Meaning"), value: iconTitle ?? label }
          ],
          meta
        }
      : null;

    return (
      <div
        className={iconKind ? `result-summary-card has-summary-icon ${iconKind}` : "result-summary-card"}
        data-active={activeInsight?.key === summaryInsight?.key ? "true" : undefined}
        onClick={() => summaryInsight && setActiveInsight(summaryInsight)}
        onKeyDown={(event) => {
          if (!summaryInsight) {
            return;
          }
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            setActiveInsight(summaryInsight);
          }
        }}
        role={summaryInsight ? "button" : undefined}
        tabIndex={summaryInsight ? 0 : undefined}
        title={iconTitle ?? label}
      >
        {iconKind ? (
          <span
            aria-hidden="true"
            className="result-summary-icon-button"
          >
            <img alt="" draggable={false} src={`/result-icons/${iconKind}.png`} />
          </span>
        ) : null}
        <div className="result-summary-copy">
          <span>{label}</span>
          <strong title={value}>{value}</strong>
          {showMeta ? <small title={meta}>{meta}</small> : null}
        </div>
      </div>
    );
  }
}

function reconcileQualitySelection(current: string[], availableIds: string[], fallbackIds: string[]) {
  const available = new Set(availableIds);
  const next = current.filter((id) => available.has(id));
  const normalized = next.length ? next : fallbackIds.filter((id) => available.has(id));
  return sameStringArray(current, normalized) ? current : normalized;
}

function appendGrouped<K, V>(grouped: Map<K, V[]>, key: K, value: V) {
  const current = grouped.get(key);
  if (current) {
    current.push(value);
  } else {
    grouped.set(key, [value]);
  }
}

function qualityCurvePointSort(left: BenchmarkCurvePoint, right: BenchmarkCurvePoint) {
  return (
    (left.datasetId || "").localeCompare(right.datasetId || "") ||
    left.algorithmId.localeCompare(right.algorithmId) ||
    (left.attackCategory ?? "").localeCompare(right.attackCategory ?? "") ||
    left.attackMethod.localeCompare(right.attackMethod) ||
    left.attackPresetId.localeCompare(right.attackPresetId) ||
    (left.attackVariantLabel ?? "default").localeCompare(right.attackVariantLabel ?? "default") ||
    (left.attackParamStrength ?? left.attackStrength) - (right.attackParamStrength ?? right.attackStrength) ||
    left.attackStrength - right.attackStrength
  );
}

function buildQualityDatasetOptions(points: BenchmarkCurvePoint[]): QualitySelectorOption[] {
  const grouped = new Map<string, BenchmarkCurvePoint[]>();
  for (const point of points) {
    const datasetId = point.datasetId || "unknown";
    appendGrouped(grouped, datasetId, point);
  }
  return Array.from(grouped.entries())
    .map(([datasetId, items]) => ({
      id: datasetId,
      label: datasetId,
      meta: `${new Set(items.map((point) => point.algorithmId)).size} algorithms / ${
        new Set(items.map((point) => point.attackPresetId)).size
      } attacks`,
      count: items.length
    }))
    .sort((left, right) => left.label.localeCompare(right.label));
}

function buildQualityAlgorithmOptions(
  points: BenchmarkCurvePoint[],
  algorithmNames: Record<string, string> = {}
): QualitySelectorOption[] {
  const grouped = new Map<string, BenchmarkCurvePoint[]>();
  for (const point of points) {
    appendGrouped(grouped, point.algorithmId, point);
  }
  return Array.from(grouped.entries())
    .map(([algorithmId, items]) => ({
      id: algorithmId,
      label: displayAlgorithm(algorithmId, algorithmNames),
      meta: `${algorithmId} / ${new Set(items.map((point) => point.attackPresetId)).size} attacks`,
      count: items.length
    }))
    .sort((left, right) => left.label.localeCompare(right.label));
}

function buildQualityAttackSummaries(points: BenchmarkCurvePoint[]): QualityAttackSummary[] {
  const grouped = new Map<string, BenchmarkCurvePoint[]>();
  for (const point of points) {
    const key = `${point.attackCategory}:${point.attackPresetId}:${point.attackMethod}`;
    appendGrouped(grouped, key, point);
  }
  return Array.from(grouped.entries())
    .map(([key, items]) => {
      const strengths = items.map((point) => point.attackParamStrength ?? point.attackStrength);
      const strengthName = items.find((point) => point.attackParamStrengthName)?.attackParamStrengthName ?? "strength";
      const weakestPoint = items.reduce<BenchmarkCurvePoint | null>(
        (current, point) => (current == null || point.yTprAtFpr < current.yTprAtFpr ? point : current),
        null
      );
      return {
        key,
        attackPresetId: items[0]?.attackPresetId ?? "n/a",
        attackMethod: items[0]?.attackMethod ?? "n/a",
        attackCategory: items[0]?.attackCategory ?? "n/a",
        algorithmCount: new Set(items.map((point) => point.algorithmId)).size,
        pointCount: items.length,
        strengthName,
        strengthMin: Math.min(...strengths),
        strengthMax: Math.max(...strengths),
        avgTpr: meanNumber(items.map((point) => point.yTprAtFpr)),
        avgNqd: meanNumber(items.map((point) => point.xNqd)),
        weakestPoint
      };
    })
    .sort((left, right) =>
      (left.avgTpr ?? 1) - (right.avgTpr ?? 1) ||
      (right.avgNqd ?? 0) - (left.avgNqd ?? 0) ||
      left.attackMethod.localeCompare(right.attackMethod)
    );
}

function buildQualityComboSummaries(points: BenchmarkCurvePoint[]): QualityComboSummary[] {
  const grouped = new Map<string, BenchmarkCurvePoint[]>();
  for (const point of points) {
    const key = qualityComboKey(point);
    appendGrouped(grouped, key, point);
  }
  return Array.from(grouped.entries()).map(([key, items]) => {
    const strengths = items.map((point) => point.attackParamStrength ?? point.attackStrength);
    const weakestPoint = items.reduce<BenchmarkCurvePoint | null>(
      (current, point) => (current == null || point.yTprAtFpr < current.yTprAtFpr ? point : current),
      null
    );
    const first = items[0];
    const strengthName = items.find((point) => point.attackParamStrengthName)?.attackParamStrengthName ?? "strength";
    return {
      key,
      datasetId: first?.datasetId || "unknown",
      algorithmId: first?.algorithmId ?? "n/a",
      attackPresetId: first?.attackPresetId ?? "n/a",
      attackMethod: first?.attackMethod ?? "n/a",
      attackCategory: first?.attackCategory ?? "n/a",
      variantLabel: first ? variantLabel(first) : "default",
      pointCount: items.length,
      strengthName,
      strengthMin: Math.min(...strengths),
      strengthMax: Math.max(...strengths),
      avgTpr: meanNumber(items.map((point) => point.yTprAtFpr)),
      avgNqd: meanNumber(items.map((point) => point.xNqd)),
      weakestPoint
    };
  });
}

function buildQualityAttackOptions(points: BenchmarkCurvePoint[]): QualityAttackOption[] {
  const grouped = new Map<string, BenchmarkCurvePoint[]>();
  for (const point of points) {
    appendGrouped(grouped, point.attackPresetId, point);
  }
  return Array.from(grouped.entries())
    .map(([attackPresetId, items]) => ({
      attackPresetId,
      attackMethod: items[0]?.attackMethod ?? attackPresetId,
      attackCategory: items[0]?.attackCategory ?? "unknown",
      pointCount: items.length,
      variantCount: new Set(items.map((point) => point.attackVariantKey ?? "default")).size
    }))
    .sort(
      (left, right) =>
        attackCategoryRank(normalizeAttackCategory(left.attackCategory, left.attackPresetId, left.attackMethod)) -
          attackCategoryRank(normalizeAttackCategory(right.attackCategory, right.attackPresetId, right.attackMethod)) ||
        left.attackMethod.localeCompare(right.attackMethod) ||
        left.attackPresetId.localeCompare(right.attackPresetId)
    );
}

function isIdentityAttackOption(option: Pick<QualityAttackOption, "attackPresetId" | "attackMethod">): boolean {
  const preset = option.attackPresetId.toLowerCase().replace(/[_\s]+/g, "-");
  const method = option.attackMethod.toLowerCase().replace(/[_\s]+/g, "-");
  return preset === "identity" || preset === "atk-identity" || method === "identity";
}

function buildAttackVisualSummaries(
  points: BenchmarkCurvePoint[],
  leaderboardRows: BenchmarkAttackLeaderboardRow[],
  attackNames: Record<string, string> = {}
): AttackVisualSummary[] {
  const pointsByAttack = new Map<string, BenchmarkCurvePoint[]>();
  for (const point of points) {
    appendGrouped(pointsByAttack, point.attackPresetId, point);
  }
  const rowsByAttack = new Map<string, BenchmarkAttackLeaderboardRow[]>();
  for (const row of leaderboardRows) {
    appendGrouped(rowsByAttack, row.attackPresetId, row);
  }
  const rawSummaries = Array.from(pointsByAttack.entries()).map(([attackPresetId, attackPoints]) => {
    const rows = rowsByAttack.get(attackPresetId) ?? [];
    const first = attackPoints[0];
    const avgTpr = meanNumber(attackPoints.map((point) => point.yTprAtFpr));
    const avgNqd = meanNumber(attackPoints.map((point) => point.xNqd));
    const minTpr = finiteMin(attackPoints.map((point) => point.yTprAtFpr));
    const attackMethod = first?.attackMethod ?? rows[0]?.attackMethod ?? attackPresetId;
    const attackCategory = normalizeAttackCategory(
      first?.attackCategory ?? rows[0]?.attackCategory ?? "unknown",
      attackPresetId,
      attackMethod
    );
    return {
      attackPresetId,
      attackMethod,
      attackCategory,
      label: displayAttackByIds(attackPresetId, attackMethod, attackNames),
      pointCount: attackPoints.length,
      algorithmCount: new Set(attackPoints.map((point) => point.algorithmId)).size,
      avgTpr,
      avgNqd,
      minTpr,
      qAtP95: aggregateThreshold(rows.map((row) => row.qAtP95)),
      qAtP70: aggregateThreshold(rows.map((row) => row.qAtP70)),
      avgP: meanNumber(rows.map((row) => row.avgPerformance)),
      avgQ: meanNumber(rows.map((row) => row.avgNqd)),
      auc: meanNumber(rows.map((row) => row.auc)),
      riskScore: 0,
      points: attackPoints
    };
  });
  const maxNqd = Math.max(0.001, ...rawSummaries.map((summary) => summary.avgNqd ?? 0));
  return rawSummaries
    .map((summary) => ({
      ...summary,
      riskScore: clamp01((1 - (summary.avgTpr ?? 1)) * 0.72 + ((summary.avgNqd ?? 0) / maxNqd) * 0.28)
    }))
    .sort(
      (left, right) =>
        right.riskScore - left.riskScore ||
        (left.avgTpr ?? 1) - (right.avgTpr ?? 1) ||
        (right.avgNqd ?? 0) - (left.avgNqd ?? 0) ||
        left.label.localeCompare(right.label)
    );
}

function buildAttackCategoryDistributions(
  points: BenchmarkCurvePoint[],
  attackNames: Record<string, string> = {}
): AttackCategoryDistribution[] {
  const grouped = new Map<string, BenchmarkCurvePoint[]>();
  for (const point of points) {
    const key = normalizeAttackCategory(point.attackCategory, point.attackPresetId, point.attackMethod);
    appendGrouped(grouped, key, point);
  }
  return Array.from(grouped.entries())
    .map(([key, items]) => ({
      key,
      label: displayAttackCategory(key),
      pointCount: items.length,
      attackCount: new Set(items.map((point) => displayAttackByIds(point.attackPresetId, point.attackMethod, attackNames))).size,
      avgNqd: meanNumber(items.map((point) => point.xNqd)),
      avgTpr: meanNumber(items.map((point) => point.yTprAtFpr)),
      points: items
    }))
    .sort((left, right) => attackCategoryRank(left.key) - attackCategoryRank(right.key) || left.label.localeCompare(right.label));
}

function AttackViolinPlot({
  categories,
  onPickPoint
}: {
  categories: AttackCategoryDistribution[];
  onPickPoint: (point: BenchmarkCurvePoint) => void;
}) {
  if (!categories.length) {
    return <div className="empty compact-empty">No attack distribution data</div>;
  }
  const width = 760;
  const rowHeight = 70;
  const leftPad = 176;
  const rightPad = 42;
  const topPad = 28;
  const bottomPad = 44;
  const height = topPad + bottomPad + categories.length * rowHeight;
  const plotWidth = width - leftPad - rightPad;
  const values = categories.flatMap((category) => category.points.map((point) => point.xNqd)).filter(Number.isFinite);
  const domain = numericDomain(values, 0.1);
  const ticks = visualAxisTicks(domain.min, domain.max, 6);
  const xFor = (value: number) =>
    leftPad + ((Math.max(domain.min, Math.min(domain.max, value)) - domain.min) / Math.max(0.0001, domain.max - domain.min)) * plotWidth;
  return (
    <div className="attack-violin-wrap">
      <svg className="attack-violin-chart" role="img" viewBox={`0 0 ${width} ${height}`}>
        {ticks.map((tick) => (
          <g key={tick}>
            <line className="attack-chart-grid" x1={xFor(tick)} x2={xFor(tick)} y1={topPad - 10} y2={height - bottomPad} />
            <text className="chart-label" textAnchor="middle" x={xFor(tick)} y={height - 14}>
              {formatVisualTick(tick, ticks)}
            </text>
          </g>
        ))}
        {categories.map((category, index) => {
          const color = curveSeriesColor(index);
          const centerY = topPad + index * rowHeight + rowHeight / 2;
          const density = violinDensity(category.points.map((point) => point.xNqd), domain.min, domain.max);
          const maxDensity = Math.max(1, ...density.map((item) => item.value));
          const halfHeight = 19;
          const upper = density.map((item) => `${xFor(item.x)},${centerY - (item.value / maxDensity) * halfHeight}`).join(" ");
          const lower = density
            .slice()
            .reverse()
            .map((item) => `${xFor(item.x)},${centerY + (item.value / maxDensity) * halfHeight}`)
            .join(" ");
          const nqds = category.points.map((point) => point.xNqd).filter(Number.isFinite).sort((a, b) => a - b);
          const q1 = quantile(nqds, 0.25);
          const median = quantile(nqds, 0.5);
          const q3 = quantile(nqds, 0.75);
          const dots = sampledPoints(category.points, 90);
          return (
            <g key={category.key}>
              <text className="attack-row-label" textAnchor="end" x={leftPad - 14} y={centerY + 5}>
                {category.label}
              </text>
              <path className="attack-violin-shape" d={`M ${upper} L ${lower} Z`} fill={color} />
              {q1 != null && q3 != null ? (
                <line className="attack-violin-box" x1={xFor(q1)} x2={xFor(q3)} y1={centerY} y2={centerY} />
              ) : null}
              {median != null ? <circle className="attack-violin-median" cx={xFor(median)} cy={centerY} r={2.4} /> : null}
              {dots.map((point, dotIndex) => (
                <circle
                  className="attack-violin-dot"
                  cx={xFor(point.xNqd)}
                  cy={centerY + deterministicJitter(`${point.algorithmId}:${point.attackPresetId}:${dotIndex}`, halfHeight - 3)}
                  fill={color}
                  key={`${category.key}-${point.algorithmId}-${point.attackPresetId}-${dotIndex}`}
                  onClick={() => onPickPoint(point)}
                  r={2.5}
                >
                  <title>{`${point.algorithmId} / ${point.attackPresetId}\nNQD ${formatMetric(point.xNqd)} / TPR ${formatMetric(point.yTprAtFpr)}`}</title>
                </circle>
              ))}
              <text className="attack-row-meta" x={width - rightPad} y={centerY + 5} textAnchor="end">
                {category.attackCount} attacks / {category.pointCount} pts
              </text>
            </g>
          );
        })}
        <text className="chart-axis-title" textAnchor="middle" x={leftPad + plotWidth / 2} y={height - 4}>
          Normalized Quality Degradation
        </text>
      </svg>
    </div>
  );
}

function AttackHeatmapMatrix({
  algorithmNames,
  attackNames,
  metric,
  onPickCell,
  points,
  rowMode,
  selectedCell,
  uiText
}: {
  algorithmNames: Record<string, string>;
  attackNames: Record<string, string>;
  metric: AttackHeatmapMetric;
  onPickCell: (cell: AttackHeatmapCell) => void;
  points: BenchmarkCurvePoint[];
  rowMode: AttackHeatmapRowMode;
  selectedCell: AttackHeatmapCell | null;
  uiText: (zh: string, en: string) => string;
}) {
  const algorithmRows = Array.from(new Set(points.map((point) => point.algorithmId)))
    .map((algorithmId) => ({
      id: algorithmId,
      label: displayAlgorithm(algorithmId, algorithmNames)
    }))
    .sort((left, right) => left.label.localeCompare(right.label));
  const attackColumnMap = new Map<string, { key: string; label: string; sortKey: string; points: BenchmarkCurvePoint[] }>();
  for (const point of points) {
    const key = heatmapRowKeyForPoint(point, rowMode);
    const label = heatmapRowLabelForPoint(point, rowMode, attackNames);
    const categoryRank = attackCategoryRank(normalizeAttackCategory(point.attackCategory, point.attackPresetId, point.attackMethod));
    const sortKey = rowMode === "category" ? String(categoryRank).padStart(2, "0") : `${String(categoryRank).padStart(2, "0")}:${label}`;
    const current = attackColumnMap.get(key) ?? { key, label, sortKey, points: [] };
    current.points.push(point);
    attackColumnMap.set(key, current);
  }
  const attackColumns = Array.from(attackColumnMap.values()).sort((left, right) => left.sortKey.localeCompare(right.sortKey));
  if (!algorithmRows.length || !attackColumns.length) {
    return <div className="empty compact-empty">No heatmap data</div>;
  }
  const cellMap = new Map<string, AttackHeatmapCell>();
  for (const algorithm of algorithmRows) {
    for (const attackColumn of attackColumns) {
      const cellPoints = attackColumn.points.filter((point) => point.algorithmId === algorithm.id);
      if (!cellPoints.length) {
        continue;
      }
      cellMap.set(`${algorithm.id}:${attackColumn.key}`, {
        rowKey: attackColumn.key,
        rowLabel: attackColumn.label,
        rowMode,
        algorithmId: algorithm.id,
        algorithmLabel: algorithm.label,
        attackPresetIds: Array.from(new Set(cellPoints.map((point) => point.attackPresetId))),
        avgTpr: meanNumber(cellPoints.map((point) => point.yTprAtFpr)),
        avgNqd: meanNumber(cellPoints.map((point) => point.xNqd)),
        pointCount: cellPoints.length
      });
    }
  }
  const nqdValues = Array.from(cellMap.values())
    .map((cell) => cell.avgNqd)
    .filter((value): value is number => value != null && Number.isFinite(value));
  const nqdDomain = numericDomain(nqdValues, 0.05);
  const gridTemplateColumns = `minmax(180px, 220px) repeat(${attackColumns.length}, minmax(118px, 1fr))`;
  return (
    <div className="attack-heatmap-scroll">
      <div className="attack-heatmap-grid" style={{ gridTemplateColumns }}>
        <div className="attack-heatmap-corner">
          <strong>{uiText("水印算法", "Watermark algorithm")}</strong>
          <span>{metric === "tpr" ? uiText("颜色: 平均 TPR", "Color: Avg TPR") : uiText("颜色: NQD 损失", "Color: NQD loss")}</span>
        </div>
        {attackColumns.map((column) => (
          <div className="attack-heatmap-column" key={column.key} title={column.key}>
            {column.label}
          </div>
        ))}
        {algorithmRows.map((algorithm) => {
          const algorithmPoints = points.filter((point) => point.algorithmId === algorithm.id);
          const algorithmAttackCount = new Set(algorithmPoints.map((point) => point.attackPresetId)).size;
          return (
            <div className="attack-heatmap-row-fragment" key={algorithm.id} style={{ display: "contents" }}>
              <div className="attack-heatmap-row-label">
                <strong>{algorithm.label}</strong>
                <span>
                  {algorithmAttackCount} {uiText("个攻击", "attacks")} / {algorithmPoints.length} {uiText("点", "pts")}
                </span>
              </div>
              {attackColumns.map((column) => {
                const cell = cellMap.get(`${algorithm.id}:${column.key}`) ?? null;
                const active =
                  selectedCell?.rowKey === column.key &&
                  selectedCell.algorithmId === algorithm.id &&
                  selectedCell.rowMode === rowMode;
                if (!cell) {
                  return <div className="attack-heatmap-cell empty-cell" key={`${algorithm.id}:${column.key}`}>-</div>;
                }
                const value = metric === "tpr" ? cell.avgTpr : cell.avgNqd;
                return (
                  <button
                    className={active ? "attack-heatmap-cell active" : "attack-heatmap-cell"}
                    key={`${algorithm.id}:${column.key}`}
                    onClick={() => onPickCell(cell)}
                    style={heatmapCellStyle(value, metric, nqdDomain)}
                    title={`${cell.algorithmLabel} × ${cell.rowLabel}\nTPR ${formatMetric(cell.avgTpr)} / NQD ${formatMetric(cell.avgNqd)}\n${cell.attackPresetIds.length} attacks / ${cell.pointCount} points`}
                    type="button"
                  >
                    <strong>{metric === "tpr" ? `TPR ${formatMetric(cell.avgTpr)}` : `NQD ${formatMetric(cell.avgNqd)}`}</strong>
                    <span>{cell.attackPresetIds.length} atk / {cell.pointCount} pts</span>
                  </button>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function AttackDumbbellPlot({
  onPickAttack,
  rows
}: {
  onPickAttack: (summary: AttackVisualSummary) => void;
  rows: AttackVisualSummary[];
}) {
  const visibleRows = rows.filter((row) => row.qAtP95 != null || row.qAtP70 != null || row.avgNqd != null);
  if (!visibleRows.length) {
    return <div className="empty compact-empty">No Q@P data</div>;
  }
  const width = 760;
  const rowHeight = 38;
  const leftPad = 188;
  const rightPad = 44;
  const topPad = 26;
  const bottomPad = 42;
  const height = topPad + bottomPad + visibleRows.length * rowHeight;
  const values = visibleRows.flatMap((row) => [
    finiteThresholdNumber(row.qAtP95),
    finiteThresholdNumber(row.qAtP70),
    row.avgNqd
  ]).filter((value): value is number => value != null && Number.isFinite(value));
  const domain = numericDomain(values, 0.12);
  const ticks = visualAxisTicks(domain.min, domain.max, 6);
  const plotWidth = width - leftPad - rightPad;
  const xFor = (value: number) =>
    leftPad + ((Math.max(domain.min, Math.min(domain.max, value)) - domain.min) / Math.max(0.0001, domain.max - domain.min)) * plotWidth;
  return (
    <div className="attack-dumbbell-scroll">
      <svg className="attack-dumbbell-chart" role="img" viewBox={`0 0 ${width} ${height}`}>
        {ticks.map((tick) => (
          <g key={tick}>
            <line className="attack-chart-grid" x1={xFor(tick)} x2={xFor(tick)} y1={topPad - 8} y2={height - bottomPad} />
            <text className="chart-label" textAnchor="middle" x={xFor(tick)} y={height - 12}>
              {formatVisualTick(tick, ticks)}
            </text>
          </g>
        ))}
        {visibleRows.map((row, index) => {
          const y = topPad + index * rowHeight + rowHeight / 2;
          const q95 = finiteThresholdNumber(row.qAtP95);
          const q70 = finiteThresholdNumber(row.qAtP70);
          const color = curveSeriesColor(index);
          const left = q95 == null || q70 == null ? null : Math.min(xFor(q95), xFor(q70));
          const right = q95 == null || q70 == null ? null : Math.max(xFor(q95), xFor(q70));
          return (
            <g className="attack-dumbbell-row" key={row.attackPresetId} onClick={() => onPickAttack(row)}>
              <text className="attack-row-label" textAnchor="end" x={leftPad - 12} y={y + 5}>
                {row.label}
              </text>
              {left != null && right != null ? <line className="attack-dumbbell-link" x1={left} x2={right} y1={y} y2={y} /> : null}
              {q95 != null ? <circle className="attack-dumbbell-dot q95" cx={xFor(q95)} cy={y} fill={color} r={5.5} /> : null}
              {q70 != null ? <rect className="attack-dumbbell-dot q70" fill={color} height={10} width={10} x={xFor(q70) - 5} y={y - 5} /> : null}
              {row.avgNqd != null ? <circle className="attack-dumbbell-avg" cx={xFor(row.avgNqd)} cy={y} r={3.2} /> : null}
              <title>{`${row.label}\nQ@P95 ${formatThreshold(row.qAtP95)} / Q@P70 ${formatThreshold(row.qAtP70)}\nAvg NQD ${formatMetric(row.avgNqd)}`}</title>
            </g>
          );
        })}
        <text className="chart-axis-title" textAnchor="middle" x={leftPad + plotWidth / 2} y={height - 3}>
          Normalized Quality Degradation
        </text>
      </svg>
    </div>
  );
}

function heatmapRowKeyForPoint(point: BenchmarkCurvePoint, rowMode: AttackHeatmapRowMode): string {
  if (rowMode === "attack") {
    return point.attackPresetId;
  }
  return normalizeAttackCategory(point.attackCategory, point.attackPresetId, point.attackMethod);
}

function heatmapRowLabelForPoint(
  point: BenchmarkCurvePoint,
  rowMode: AttackHeatmapRowMode,
  attackNames: Record<string, string> = {}
): string {
  if (rowMode === "attack") {
    return displayAttackByIds(point.attackPresetId, point.attackMethod, attackNames);
  }
  return displayAttackCategory(normalizeAttackCategory(point.attackCategory, point.attackPresetId, point.attackMethod));
}

function heatmapCellStyle(
  value: number | null,
  metric: AttackHeatmapMetric,
  nqdDomain: { min: number; max: number }
): { background: string; borderColor: string; color: string } {
  if (value == null || !Number.isFinite(value)) {
    return {
      background: "rgb(248 250 247)",
      borderColor: "rgb(217 226 218)",
      color: "rgb(99 111 125)"
    };
  }
  const intensity =
    metric === "tpr"
      ? 1 - clamp01(value)
      : clamp01((value - nqdDomain.min) / Math.max(0.0001, nqdDomain.max - nqdDomain.min));
  const hue = 154 - intensity * 145;
  const saturation = 48 + intensity * 14;
  const lightness = 93 - intensity * 30;
  return {
    background: `hsl(${hue} ${saturation}% ${lightness}%)`,
    borderColor: `hsl(${hue} ${Math.min(72, saturation + 8)}% ${Math.max(35, lightness - 18)}%)`,
    color: intensity > 0.72 ? "rgb(255 255 255)" : "rgb(23 31 28)"
  };
}

function aggregateThreshold(values: Array<number | "inf" | "-inf" | null | undefined>): number | "inf" | "-inf" | null {
  const finite = values
    .map(finiteThresholdNumber)
    .filter((value): value is number => value != null && Number.isFinite(value));
  if (finite.length) {
    return finite.reduce((total, value) => total + value, 0) / finite.length;
  }
  if (values.includes("inf")) {
    return "inf";
  }
  if (values.includes("-inf")) {
    return "-inf";
  }
  return null;
}

function finiteThresholdNumber(value: number | "inf" | "-inf" | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function finiteMin(values: Array<number | null | undefined>): number | null {
  const finite = values.filter((value): value is number => value != null && Number.isFinite(value));
  return finite.length ? Math.min(...finite) : null;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function numericDomain(values: number[], padRatio: number) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (!finite.length) {
    return { min: 0, max: 1 };
  }
  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (max <= min) {
    min -= 0.05;
    max += 0.05;
  }
  const padding = Math.max((max - min) * padRatio, 0.04);
  return { min: Math.max(0, min - padding), max: max + padding };
}

function visualAxisTicks(min: number, max: number, targetCount: number): number[] {
  const span = Math.max(0.0001, max - min);
  const step = visualTickStep(span / Math.max(1, targetCount));
  const first = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let value = first; value <= max + step * 0.5; value += step) {
    const rounded = Number(value.toFixed(6));
    if (rounded >= min - 0.000001 && rounded <= max + 0.000001) {
      ticks.push(rounded);
    }
  }
  return ticks.length ? ticks : [Number(min.toFixed(3)), Number(max.toFixed(3))];
}

function visualTickStep(rawStep: number): number {
  const exponent = Math.floor(Math.log10(Math.max(rawStep, 0.000001)));
  const base = 10 ** exponent;
  const fraction = rawStep / base;
  if (fraction <= 1) {
    return base;
  }
  if (fraction <= 2) {
    return base * 2;
  }
  if (fraction <= 2.5) {
    return base * 2.5;
  }
  if (fraction <= 5) {
    return base * 5;
  }
  return base * 10;
}

function formatVisualTick(value: number, ticks: number[]): string {
  const intervals = ticks.slice(1).map((tick, index) => Math.abs(tick - ticks[index]));
  const smallest = intervals.length ? Math.min(...intervals) : 1;
  return value.toFixed(smallest < 0.1 ? 2 : 1);
}

function violinDensity(values: number[], min: number, max: number) {
  const binCount = 24;
  const span = Math.max(0.0001, max - min);
  const bins = Array.from({ length: binCount }, (_, index) => ({
    x: min + (span * index) / Math.max(1, binCount - 1),
    value: 0
  }));
  for (const value of values) {
    if (!Number.isFinite(value)) {
      continue;
    }
    const position = clamp01((value - min) / span) * (binCount - 1);
    const center = Math.round(position);
    for (let offset = -2; offset <= 2; offset += 1) {
      const index = center + offset;
      if (index >= 0 && index < bins.length) {
        bins[index].value += Math.exp(-(offset * offset) / 2.2);
      }
    }
  }
  return bins;
}

function quantile(sortedValues: number[], q: number): number | null {
  if (!sortedValues.length) {
    return null;
  }
  const position = (sortedValues.length - 1) * q;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) {
    return sortedValues[lower];
  }
  const ratio = position - lower;
  return sortedValues[lower] * (1 - ratio) + sortedValues[upper] * ratio;
}

function sampledPoints(points: BenchmarkCurvePoint[], limit: number): BenchmarkCurvePoint[] {
  if (points.length <= limit) {
    return points;
  }
  const step = Math.ceil(points.length / limit);
  return points.filter((_, index) => index % step === 0).slice(0, limit);
}

function deterministicJitter(seed: string, radius: number): number {
  let hash = 0;
  for (let index = 0; index < seed.length; index += 1) {
    hash = (hash * 31 + seed.charCodeAt(index)) % 9973;
  }
  return ((hash / 9973) * 2 - 1) * radius;
}

function trimSvgLabel(value: string, maxLength: number): string {
  return value.length > maxLength ? `${value.slice(0, Math.max(1, maxLength - 1))}…` : value;
}

const ATTACK_CATEGORY_ORDER = [
  "classical-distortion",
  "physical-channel",
  "3d-viewpoint-rerendering",
  "regeneration",
  "consumer-enhancement"
] as const;

const ATTACK_CATEGORY_LABELS: Record<string, string> = {
  "classical-distortion": "经典失真",
  "physical-channel": "物理信道",
  "3d-viewpoint-rerendering": "3D 视角重渲染",
  regeneration: "再生成",
  "consumer-enhancement": "消费级增强"
};

function normalizeAttackCategory(category: string, attackPresetId = "", attackMethod = ""): string {
  const text = `${category} ${attackPresetId} ${attackMethod}`.toLowerCase();
  if (
    text.includes("distortion_attacks") ||
    text.includes("distortion-single") ||
    text.includes("distortion-combination") ||
    text.includes("content-preserving") ||
    text.includes("adversarial")
  ) {
    return "classical-distortion";
  }
  if (
    text.includes("physical_channel_attacks") ||
    text.includes("physical") ||
    text.includes("screen_shoot") ||
    text.includes("screen-shoot") ||
    text.includes("print_camera") ||
    text.includes("print-camera") ||
    text.includes("combined_physical") ||
    text.includes("combined-physical")
  ) {
    return "physical-channel";
  }
  if (
    text.includes("3d_viewpoint_rerendering") ||
    text.includes("3d") ||
    text.includes("viewpoint") ||
    text.includes("rerender")
  ) {
    return "3d-viewpoint-rerendering";
  }
  if (
    text.includes("consumer_enhancement_workflow_attacks") ||
    text.includes("consumer-enhancement") ||
    text.includes("cew_") ||
    text.includes("atk-cew")
  ) {
    return "consumer-enhancement";
  }
  if (
    text.includes("regeneration_attacks") ||
    text.includes("regeneration") ||
    text.includes("regen") ||
    text.includes("diffusion") ||
    text.includes("vae") ||
    text.includes("noise_to_image") ||
    text.includes("noise-to-image") ||
    text.includes("image_to_vedio") ||
    text.includes("image-to-vedio") ||
    text.includes("image_to_video") ||
    text.includes("image-to-video")
  ) {
    return "regeneration";
  }
  return "classical-distortion";
}

function attackCategoryRank(category: string): number {
  const normalized = normalizeAttackCategory(category);
  const index = ATTACK_CATEGORY_ORDER.indexOf(normalized as (typeof ATTACK_CATEGORY_ORDER)[number]);
  return index === -1 ? ATTACK_CATEGORY_ORDER.length : index;
}

function displayAttackCategory(category: string): string {
  const normalized = normalizeAttackCategory(category);
  return ATTACK_CATEGORY_LABELS[normalized] ?? displayTokenLabel(category);
}

const RESULT_ALGORITHM_LABELS: Record<string, string> = {
  "invisible-watermark-dwtdct": "DWT-DCT",
  "invisible-watermark-dwtdctsvd": "DWT-DCT-SVD",
  "invisible-watermark-rivagan": "RivaGAN",
  "traditional-spread-dct": "DCT",
  chunkyseal: "ChunkySeal",
  cin: "CIN",
  dwsf: "DWSF",
  hidden: "HiDDeN",
  invismark: "InvisMark",
  maskwm: "MaskWM",
  mbrs: "MBRS",
  pimog: "PIMoG",
  pixelseal: "PixelSeal",
  stegastamp: "StegaStamp",
  trustmark: "TrustMark",
  videoseal: "VideoSeal",
  vine: "VINE",
  wam: "WAM"
};

const RESULT_ATTACK_LABELS: Record<string, string> = {
  brightness: "亮度调整",
  contrast: "对比度调整",
  gaussian_blur: "高斯模糊",
  gaussian_noise: "高斯噪声",
  jpeg: "JPEG 压缩",
  resize: "缩放",
  resized_crop: "缩放裁剪",
  rotation: "旋转",
  erasing: "区域擦除",
  screen_shoot: "屏幕拍摄信道",
  print_camera: "打印拍摄信道",
  combined_physical: "组合物理信道",
  "2x_regen": "2 轮扩散再生成",
  "4x_regen": "4 轮扩散再生成",
  regen_diffusion: "扩散再生成",
  regen_vae: "VAE 重建",
  noise_to_image: "噪声到图像再生成",
  image_to_vedio: "图像到视频再生成",
  cew_e1: "自动色调增强",
  cew_e2: "暖色鲜艳增强",
  cew_e3: "胶片褪色增强",
  cew_e4: "局部清晰 HDR",
  cew_c1: "自动修复超分",
  cew_c2: "色彩修饰超分",
  cew_c3: "细节增强超分",
  cew_c4: "完整增强链",
  cew_d1: "自动补光",
  cew_d2: "自动白平衡",
  cew_d3: "自适应 AI 色彩",
  cew_d4: "低光细节增强",
  cew_d5: "AI 去噪",
  cew_s1: "Real-ESRGAN 超分",
  cew_s2: "SwinIR 超分",
  cew_s3: "BSRGAN 超分"
};

const RESULT_PROFILE_TAG_LABELS: Record<string, string> = {
  "physical-robust": "物理鲁棒",
  "screen-specialist": "屏摄专长",
  "print-specialist": "打印专长",
  "combined-fragile": "组合攻击脆弱",
  "quality-first": "保真优先",
  "fast-lightweight": "轻量快速",
  "geometry-fragile": "几何脆弱"
};

const RESULT_ATTACK_ENGLISH_LABELS: Record<string, string> = {
  brightness: "Brightness",
  contrast: "Contrast",
  gaussian_blur: "Gaussian Blur",
  gaussian_noise: "Gaussian Noise",
  jpeg: "JPEG Compression",
  resize: "Resize",
  resized_crop: "Resized Crop",
  rotation: "Rotation",
  erasing: "Random Erasing",
  screen_shoot: "PIMoG-style Screen-Camera",
  print_camera: "CamMark-style Print-Camera",
  combined_physical: "Combined Physical Channel",
  "2x_regen": "2-pass Diffusion Regeneration",
  "4x_regen": "4-pass Diffusion Regeneration",
  regen_diffusion: "WAVES Diffusion Regeneration",
  regen_vae: "CompressAI VAE Reconstruction",
  noise_to_image: "CtrlRegen Noise-to-Image",
  image_to_vedio: "NFPA Image-to-Video",
  cew_e1: "Auto-Tone",
  cew_e2: "Warm-Vivid",
  cew_e3: "Film-Faded",
  cew_e4: "Local-Clarity HDR",
  cew_c1: "Basic Auto-Fix SR",
  cew_c2: "Color Retouch SR",
  cew_c3: "Detail Enhance SR",
  cew_c4: "Full Enhancement Chain",
  cew_d1: "Zero-DCE++ Auto-Light",
  cew_d2: "DeepWB Auto-WhiteBalance",
  cew_d3: "Image-Adaptive 3D LUT",
  cew_d4: "Retinexformer Low-Light Enhance",
  cew_d5: "NAFNet/Restormer AI-Denoise",
  cew_s1: "Real-ESRGAN",
  cew_s2: "SwinIR",
  cew_s3: "BSRGAN"
};

function displayAlgorithm(id: string, resourceNames: Record<string, string> = {}): string {
  const normalized = id.replace(/^alg[-_]/, "");
  const fallback = RESULT_ALGORITHM_LABELS[normalized] ?? displayTokenLabel(normalized);
  return resourceNames[id] ?? resourceNames[normalized] ?? resolveWatermarkDisplayName(normalized, fallback);
}

function displayAttack(value: string, resourceNames: Record<string, string> = {}): string {
  const normalized = value.replace(/^atk[-_]/, "").replace(/-/g, "_");
  return fixMojibakeLabel(
    resourceNames[value] ??
    resourceNames[normalized] ??
    RESULT_ATTACK_LABELS[normalized] ??
    displayParameterizedAttack(normalized) ??
    displayTokenLabel(normalized)
  );
}

function displayAttackByIds(presetId: string, method: string, resourceNames: Record<string, string> = {}): string {
  return fixMojibakeLabel(resourceNames[presetId] ?? displayAttack(method || presetId, resourceNames));
}

function displayAttackPoint(point: BenchmarkCurvePoint, resourceNames: Record<string, string> = {}): string {
  return displayAttackByIds(point.attackPresetId, point.attackMethod ?? point.attackPresetId, resourceNames);
}

function displayAttackSubtitle(method: string, strengthName: string): string {
  const normalized = method.replace(/^atk[-_]/, "").replace(/-/g, "_");
  const english = RESULT_ATTACK_ENGLISH_LABELS[normalized] ?? displayTokenLabel(normalized);
  const mapping =
    strengthName === "strength" || strengthName === "step"
      ? "0-1 强度映射"
      : strengthName === "scale"
        ? "倍率参数"
        : strengthName === "xy"
          ? "固定/离散参数"
          : `${strengthName} 参数`;
  return `${english} · ${mapping}`;
}

function displayProfileTags(tags: string[] | undefined, language: string): string {
  if (!tags?.length) {
    return "";
  }
  return tags
    .map((tag) => (language === "zh" ? RESULT_PROFILE_TAG_LABELS[tag] ?? displayTokenLabel(tag) : displayTokenLabel(tag)))
    .join(language === "zh" ? "，" : ", ");
}

function localizedAlgorithmResourceName(language: string, algorithm: AlgorithmVersion): string {
  const method = algorithm.method ?? algorithm.id.replace(/^alg[-_]/, "");
  return resolveWatermarkDisplayName(method, algorithm.name || RESULT_ALGORITHM_LABELS[method] || displayAlgorithm(algorithm.id));
}

function localizedAttackResourceName(language: string, attack: AttackPreset): string {
  const fallback = RESULT_ATTACK_LABELS[attack.method] ?? displayParameterizedAttack(attack.method) ?? attack.name;
  if (language === "zh") {
    return fixMojibakeLabel(fallback);
  }
  return fixMojibakeLabel(attack.name || fallback || displayAttack(attack.method));
}

function displayParameterizedAttack(method: string): string | null {
  const normalized = method.replace(/-/g, "_");
  const viewpoint = normalized.match(/^3d_viewpoint_rerendering_(swipe|shake|rotate|rotate_forward)_(point|ahead)$/);
  if (viewpoint) {
    const motion: Record<string, string> = {
      rotate: "旋转",
      rotate_forward: "前向旋转",
      shake: "晃动",
      swipe: "平移"
    };
    const target = viewpoint[2] === "point" ? "定点" : "前视";
    return `3D 视角重渲染-${motion[viewpoint[1]] ?? viewpoint[1]}-${target}`;
  }
  return null;
}

function fixMojibakeLabel(value: string): string {
  return value
    .replaceAll("鏃嬭浆", "旋转")
    .replaceAll("鍓嶅悜鏃嬭浆", "前向旋转")
    .replaceAll("鏅冨姩", "晃动")
    .replaceAll("骞崇Щ", "平移")
    .replaceAll("瀹氱偣", "定点")
    .replaceAll("鍓嶈", "前视")
    .replaceAll("瑙嗚", "视角")
    .replaceAll("閲嶆覆鏌?", "重渲染")
    .replaceAll("3D 视角重渲染?", "3D 视角重渲染");
}

function displayTokenLabel(value: string): string {
  const special: Record<string, string> = {
    cin: "CIN",
    dct: "DCT",
    dwsf: "DWSF",
    dwt: "DWT",
    dwtdct: "DWT-DCT",
    dwtdctsvd: "DWT-DCT-SVD",
    mbrs: "MBRS",
    nsn: "NSN",
    pimog: "PIMoG",
    riva: "Riva",
    svd: "SVD",
    wam: "WAM"
  };
  return value
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((part) => special[part.toLowerCase()] ?? part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatStrength(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) {
    return "n/a";
  }
  return value.toFixed(2).replace(/\.00$/, "").replace(/0$/, "");
}

function formatCount(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) {
    return "n/a";
  }
  return Math.round(value).toLocaleString();
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

function formatParamRange(name: string, min: number, max: number): string {
  const label = name || "strength";
  return min === max ? `${label} ${formatStrength(min)}` : `${label} ${formatStrength(min)}-${formatStrength(max)}`;
}

function variantLabel(point: BenchmarkCurvePoint): string {
  const label = point.attackVariantLabel?.trim();
  return label && label !== "default" ? label : "default";
}

function formatAttackParams(params: Record<string, unknown> | undefined): string {
  if (!params) {
    return "default";
  }
  const entries = Object.entries(params)
    .filter(([, value]) => value != null)
    .sort(([left], [right]) => left.localeCompare(right));
  if (!entries.length) {
    return "default";
  }
  return entries.map(([key, value]) => `${key}=${String(value)}`).join(", ");
}

function qualityComboKey(point: BenchmarkCurvePoint): string {
  return [point.datasetId || "unknown", point.algorithmId, point.attackPresetId, point.attackVariantKey ?? "default"].join(":");
}

function qualityPointKey(point: BenchmarkCurvePoint, index?: number): string {
  return [
    point.datasetId || "unknown",
    point.algorithmId,
    point.attackCategory,
    point.attackMethod,
    point.attackPresetId,
    point.attackVariantKey ?? "default",
    point.attackParamStrength ?? point.attackStrength,
    point.attackStrength,
    point.xNqd,
    point.yTprAtFpr,
    index ?? ""
  ].join(":");
}

function makeRunResultsShell(run: DemoRunRecord): RunResults {
  return {
    run,
    resultUnits: [],
    summaryPath: `${run.artifactRoot}/run_summary.json`,
    summaryExists: false,
    summary: null,
    aggregates: []
  };
}

type ResourceCatalogCounts = {
  datasets: number;
  algorithms: number;
  attacks: number;
};

function selectionIdList(selection: Record<string, unknown> | null | undefined, field: string) {
  return Array.isArray(selection?.[field]) ? selection[field].map((item) => String(item)) : [];
}

function formatCatalogRatio(used: number, total: number) {
  const safeUsed = Math.max(0, used);
  const safeTotal = Math.max(safeUsed, total);
  return `${safeUsed}/${safeTotal}`;
}

function friendlyDatasetName(datasetLabel: string): string {
  return datasetLabel
    .split(",")
    .map((item) => {
      const trimmed = item.trim();
      return trimmed.toLowerCase() === "imagenet" ? "ImageNet" : trimmed;
    })
    .filter(Boolean)
    .join(", ");
}

function friendlyExperimentName(rawName: string, datasetLabel: string, maxSamples: number, language: string): string {
  const isImportedPlaceholder = /^Imported run\b/i.test(rawName);
  if (!isImportedPlaceholder || maxSamples <= 0) {
    return rawName;
  }
  const datasetName = friendlyDatasetName(datasetLabel);
  if (language === "zh") {
    return `${datasetName} ${maxSamples.toLocaleString()} 张图片测评`;
  }
  return `${datasetName} ${maxSamples.toLocaleString()}-image experiment`;
}

function friendlyRunRecordName(run: DemoRunRecord, language: string): string {
  const rawName = run.taskName?.trim() || run.configName?.trim() || run.id;
  if (!/^Imported run\b/i.test(rawName)) {
    return rawName;
  }
  const selectedDatasetIds = run.selection?.datasetIds ?? [];
  const selectedDataset = selectedDatasetIds.length === 1 ? selectedDatasetIds[0] : "dataset";
  const selectedSampleCount = Number(run.selection?.maxSamples ?? 0);
  if (selectedSampleCount > 0) {
    return friendlyExperimentName(rawName, selectedDataset, selectedSampleCount, language);
  }
  const canonicalPhase = run.phases?.find((phase) => phase.key === "canonical");
  const currentItem = canonicalPhase?.currentItem;
  const datasetId = typeof currentItem?.datasetId === "string" ? currentItem.datasetId : "dataset";
  const sampleCount = Number(currentItem?.sampleCount ?? canonicalPhase?.total ?? 0);
  return friendlyExperimentName(rawName, datasetId, sampleCount, language);
}

function isBaselineAttackId(attackId: string) {
  return isHiddenBenchmarkAttack({ id: attackId, method: attackId.replace(/^atk-/, "").replace(/-/g, "_") });
}

function selectionAttackPresetIds(selection: Record<string, unknown> | null | undefined) {
  if (!Array.isArray(selection?.attackPresetIds)) {
    return [];
  }
  return selection.attackPresetIds.map((item) => String(item)).filter((attackId) => !isBaselineAttackId(attackId));
}

function countUsedBenchmarkAttacks(
  results: RunResults | null,
  selection: Record<string, unknown> | null,
  attacks: AttackPreset[]
) {
  const presetIdsFromSelection = selectionAttackPresetIds(selection);
  if (presetIdsFromSelection.length > 0) {
    if (attacks.length === 0) {
      return countBenchmarkAttackTypesFromMethods(presetIdsFromSelection.map(attackPresetIdToMethod));
    }
    return countSelectedBenchmarkAttackTypes(attacks, presetIdsFromSelection);
  }

  const units = results?.resultUnits ?? [];
  const methods = Array.from(
    new Set(
      units
        .map((unit) => unit.attackMethod || unit.attackPresetId)
        .filter((method) => method && !isBaselineAttackId(method))
    )
  );
  if (methods.length > 0) {
    return countBenchmarkAttackTypesFromMethods(methods);
  }

  const presetIdsFromUnits = Array.from(
    new Set(units.map((unit) => unit.attackPresetId).filter((attackId) => attackId && !isBaselineAttackId(attackId)))
  );
  if (attacks.length === 0) {
    return countBenchmarkAttackTypesFromMethods(presetIdsFromUnits.map(attackPresetIdToMethod));
  }
  return countSelectedBenchmarkAttackTypes(attacks, presetIdsFromUnits);
}

function attackPresetIdToMethod(presetId: string) {
  return presetId.replace(/^atk-/, "").replaceAll("-", "_");
}

function buildRunSummary(
  results: RunResults | null,
  catalog: ResourceCatalogCounts,
  attacks: AttackPreset[],
  statusLabels: Record<string, string>,
  language: "zh" | "en"
) {
  const units = results?.resultUnits ?? [];
  const selection =
    results?.summary && typeof results.summary.selection === "object" && results.summary.selection !== null
      ? (results.summary.selection as Record<string, unknown>)
      : null;

  const datasetIdsFromSelection = selectionIdList(selection, "datasetIds");
  const datasetIdsFromUnits = Array.from(new Set(units.map((unit) => unit.datasetId).filter(Boolean)));
  const usedDatasetIds = datasetIdsFromSelection.length ? datasetIdsFromSelection : datasetIdsFromUnits;
  const usedDatasetCount = usedDatasetIds.length;

  const algorithmIdsFromSelection = selectionIdList(selection, "algorithmIds");
  const algorithmIdsFromUnits = Array.from(new Set(units.map((unit) => unit.algorithmId).filter(Boolean)));
  const usedAlgorithmIds = algorithmIdsFromSelection.length ? algorithmIdsFromSelection : algorithmIdsFromUnits;
  const usedAlgorithmCount = usedAlgorithmIds.length;

  const usedAttackCount = countUsedBenchmarkAttacks(results, selection, attacks);
  const totalAttackCount = catalog.attacks || countBenchmarkAttackTypes(attacks);

  const maxSamplesFromSelection =
    typeof selection?.maxSamples === "number" ? selection.maxSamples : Number(selection?.maxSamples ?? 0);
  const maxSamplesFromUnits = units.length ? Math.max(...units.map((unit) => unit.sampleCount)) : 0;
  const maxSamples = maxSamplesFromSelection > 0 ? maxSamplesFromSelection : maxSamplesFromUnits;

  const statusKey = results?.run.status ?? "n/a";
  const statusLabel = statusLabels[statusKey] ?? statusKey;
  const progress = Math.round(results?.run.progress ?? 0);

  const datasetLabel =
    usedDatasetIds.length === 1
      ? usedDatasetIds[0]
      : usedDatasetIds.length > 1
        ? usedDatasetIds.join(", ")
        : language === "zh"
          ? "未指定"
          : "n/a";

  const rawExperimentName =
    results?.run.taskName?.trim() || results?.run.configName?.trim() || results?.run.id || "n/a";
  const experimentName = friendlyExperimentName(rawExperimentName, datasetLabel, maxSamples, language);

  return {
    experimentName,
    experimentMeta: results?.run.id ?? "n/a",
    statusLabel,
    statusMeta: `${progress}% ${language === "zh" ? "进度" : "progress"}`,
    datasetValue: formatCatalogRatio(usedDatasetCount, catalog.datasets),
    datasetMeta:
      maxSamples > 0
        ? language === "zh"
          ? `${datasetLabel} · ${maxSamples.toLocaleString()} 张/数据集`
          : `${datasetLabel} · ${maxSamples.toLocaleString()} samples/dataset`
        : datasetLabel,
    watermarkValue: formatCatalogRatio(usedAlgorithmCount, catalog.algorithms),
    watermarkMeta:
      language === "zh" ? "本测评选用的水印算法数" : "Watermark algorithms selected in this run",
    attackValue: formatCatalogRatio(usedAttackCount, totalAttackCount),
    attackMeta: language === "zh" ? "本测评选用的攻击算法数" : "Attack algorithms selected in this run"
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
  results?.resultUnits.forEach((unit) => ids.add(unit.algorithmId));
  return Array.from(ids).sort();
}

function sameStringArray(left: string[], right: string[]) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function resultUnitScoring(unit: RunResultUnit): ScoringSummary | undefined {
  return unit.summary?.scoring as ScoringSummary | undefined;
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
  return <BarChart3 size={15} />;
}

function tabLabel(tab: ResultsTab, t: ReturnType<typeof useLanguage>["t"]) {
  if (tab === "overview") {
    return t.results.overview;
  }
  if (tab === "attack") {
    return t.results.attackAnalysis;
  }
  return t.results.qualityRobustness;
}

function formatThreshold(value: number | "inf" | "-inf" | null | undefined) {
  if (value === "inf") {
    return "inf";
  }
  if (value === "-inf") {
    return "-inf";
  }
  return formatMetric(value ?? null);
}

function exportResultsCsv(runId: string) {
  if (!runId) {
    return;
  }
  const anchor = document.createElement("a");
  anchor.href = runResultsCsvUrl(runId);
  anchor.download = `${runId}-results.csv`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}
