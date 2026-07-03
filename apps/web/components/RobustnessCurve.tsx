import { buildCurveSeries } from "@/lib/insights";
import type { BenchmarkCurvePoint, BenchmarkScore, RunResults } from "@/lib/types";

export const SCORE_CURVE_COLORS = [
  "#2563eb",
  "#0f766e",
  "#a16207",
  "#b42318",
  "#7c3aed",
  "#0891b2",
  "#be185d",
  "#4d7c0f",
  "#c2410c",
  "#4338ca",
  "#047857",
  "#9f1239",
  "#e11d48",
  "#65a30d",
  "#0e7490",
  "#ca8a04",
  "#9333ea",
  "#dc2626",
  "#059669",
  "#475569"
] as const;

export const SCORE_CURVE_SHAPES = [
  "circle",
  "square",
  "triangle",
  "downTriangle",
  "diamond",
  "cross",
  "plus",
  "star",
  "hexagon",
  "pentagon"
] as const;

export type ScoreCurveShape = (typeof SCORE_CURVE_SHAPES)[number];
export type RobustnessCurveGroupMode = "algorithm" | "variant" | "combination";
export type RobustnessCurveEncodingMode = "series" | "semantic";

export function curveSeriesColor(index: number) {
  return SCORE_CURVE_COLORS[index % SCORE_CURVE_COLORS.length];
}

export function curveSeriesShape(index: number): ScoreCurveShape {
  return SCORE_CURVE_SHAPES[index % SCORE_CURVE_SHAPES.length];
}

export function curveDomainIndex(domain: string[], key: string): number {
  const index = domain.indexOf(key);
  return index >= 0 ? index : domain.length;
}

export function curveDomainColor(domain: string[], key: string): string {
  return curveSeriesColor(curveDomainIndex(domain, key));
}

export function curveDomainShape(domain: string[], key: string): ScoreCurveShape {
  return curveSeriesShape(curveDomainIndex(domain, key));
}

export function CurveLegendGlyph({ color, shape }: { color: string; shape: ScoreCurveShape }) {
  return (
    <svg aria-hidden="true" className="legend-glyph" viewBox="0 0 16 16">
      <PointGlyph color={color} shape={shape} x={8} y={8} />
    </svg>
  );
}

export function RobustnessCurve({
  curvePoints,
  results,
  score,
  emptyText,
  groupMode = "algorithm",
  selectedAlgorithmIds = [],
  selectedAttackPresetIds,
  selectedAttackPresetId = "all",
  selectedDatasetIds = [],
  algorithmLabels = {},
  attackLabels = {},
  colorDomain,
  encodingMode = "series",
  shapeDomain,
  performanceThresholds,
  scalePointSize = false,
  showCaption = true,
  showLegend = true,
  onSelectPoint,
  pointCaption = "Point: attack method / strength",
  seriesCaption = "Series: watermark algorithm"
}: {
  curvePoints?: BenchmarkCurvePoint[];
  results: RunResults | null;
  score?: BenchmarkScore | null;
  emptyText: string;
  groupMode?: RobustnessCurveGroupMode;
  selectedAlgorithmIds?: string[];
  selectedAttackPresetIds?: string[];
  selectedAttackPresetId?: string;
  selectedDatasetIds?: string[];
  algorithmLabels?: Record<string, string>;
  attackLabels?: Record<string, string>;
  colorDomain?: string[];
  encodingMode?: RobustnessCurveEncodingMode;
  shapeDomain?: string[];
  performanceThresholds?: number[];
  scalePointSize?: boolean;
  showCaption?: boolean;
  showLegend?: boolean;
  onSelectPoint?: (point: BenchmarkCurvePoint) => void;
  pointCaption?: string;
  seriesCaption?: string;
}) {
  const scoreCurvePoints = curvePoints ?? score?.curvePoints ?? [];
  if (scoreCurvePoints.length) {
    const selected = new Set(selectedAlgorithmIds);
    const selectedDatasets = new Set(selectedDatasetIds);
    const selectedAttacks = new Set(selectedAttackPresetIds ?? []);
    const selectedAttack = selectedAttackPresetId !== "all" ? selectedAttackPresetId : "";
    const labelAlgorithm = (algorithmId: string) => labelFromMap(algorithmLabels, algorithmId) ?? cleanLabel(algorithmId);
    const labelAttack = (attackPresetId: string, attackMethod: string) =>
      labelFromMap(attackLabels, attackPresetId) ?? labelFromMap(attackLabels, attackMethod) ?? cleanLabel(attackMethod || attackPresetId);
    const grouped = new Map<
      string,
      {
        colorKey: string;
        label: string;
        points: Array<{ x: number; y: number; raw: BenchmarkCurvePoint }>;
        shapeKey: string;
      }
    >();
    for (const point of scoreCurvePoints) {
      const datasetId = point.datasetId || "unknown";
      if (selectedDatasets.size > 0 && !selectedDatasets.has(datasetId)) {
        continue;
      }
      if (selected.size > 0 && !selected.has(point.algorithmId)) {
        continue;
      }
      if (selectedAttacks.size > 0 && !selectedAttacks.has(point.attackPresetId)) {
        continue;
      }
      if (selectedAttack && point.attackPresetId !== selectedAttack) {
        continue;
      }
      const variantKey = point.attackVariantKey || "default";
      const variantLabel = variantDisplayLabel(point);
      const key =
        groupMode === "combination"
          ? `${datasetId}:${point.algorithmId}:${point.attackPresetId}:${variantKey}`
          : groupMode === "variant" || selectedAttack
            ? `${point.algorithmId}:${point.attackPresetId}:${variantKey}`
            : point.algorithmId;
      const current =
        grouped.get(key) ??
        {
          colorKey: encodingMode === "semantic" ? point.algorithmId : key,
          label:
            groupMode === "combination"
              ? `${cleanLabel(datasetId)} / ${labelAlgorithm(point.algorithmId)} / ${labelAttack(point.attackPresetId, point.attackMethod)}${
                  variantLabel === "default" ? "" : ` / ${variantLabel}`
                }`
              : selectedAttack
                ? `${labelAlgorithm(point.algorithmId)} / ${variantLabel}`
                : labelAlgorithm(point.algorithmId),
          points: [],
          shapeKey: encodingMode === "semantic" ? point.attackPresetId || point.attackMethod : key
        };
      current.points.push({ x: point.xNqd, y: point.yTprAtFpr, raw: point });
      grouped.set(key, current);
    }
    const scoreSeries = Array.from(grouped.entries())
      .map(([id, item]) => ({
        id,
        colorKey: item.colorKey,
        label: item.label,
        shapeKey: item.shapeKey,
        points: item.points.sort((a, b) =>
          selectedAttack || groupMode !== "algorithm"
            ? (a.raw.attackParamStrength ?? a.raw.attackStrength) - (b.raw.attackParamStrength ?? b.raw.attackStrength) ||
              a.x - b.x
            : a.x - b.x
        )
      }))
      .filter((item) => item.points.length >= 1);
    if (scoreSeries.length > 0) {
      return (
        <ScoreCurve
          algorithmLabels={algorithmLabels}
          attackLabels={attackLabels}
          colorDomain={colorDomain}
          onSelectPoint={onSelectPoint}
          performanceThresholds={performanceThresholds ?? score?.performanceThresholds}
          pointCaption={pointCaption}
          scalePointSize={scalePointSize}
          series={scoreSeries}
          seriesCaption={seriesCaption}
          shapeDomain={shapeDomain}
          showCaption={showCaption}
          showLegend={showLegend}
        />
      );
    }
  }

  const selected = new Set(selectedAlgorithmIds);
  const series = buildCurveSeries(results).filter((item) => selected.size === 0 || selected.has(item.algorithmId));
  if (series.length === 0) {
    return <div className="empty compact-empty">{emptyText}</div>;
  }

  const width = 540;
  const height = 220;
  const pad = 34;
  const colors = ["#2563eb", "#0f766e", "#a16207", "#b42318", "#7c3aed"];
  const strengths = series.flatMap((item) => item.points.map((point) => point.strength));
  const minStrength = Math.min(...strengths);
  const maxStrength = Math.max(...strengths);
  const xFor = (strength: number) =>
    pad + ((strength - minStrength) / Math.max(0.001, maxStrength - minStrength)) * (width - pad * 2);
  const yFor = (accuracy: number) => height - pad - accuracy * (height - pad * 2);

  return (
    <div className="curve-wrap">
      <svg className="curve-chart" role="img" viewBox={`0 0 ${width} ${height}`}>
        <line className="chart-axis" x1={pad} x2={pad} y1={pad} y2={height - pad} />
        <line className="chart-axis" x1={pad} x2={width - pad} y1={height - pad} y2={height - pad} />
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line
              className="chart-grid"
              x1={pad}
              x2={width - pad}
              y1={yFor(tick)}
              y2={yFor(tick)}
            />
            <text className="chart-label" x={8} y={yFor(tick) + 4}>
              {Math.round(tick * 100)}
            </text>
          </g>
        ))}
        {series.map((item, index) => {
          const color = colors[index % colors.length];
          const points = item.points.map((point) => `${xFor(point.strength)},${yFor(point.accuracy)}`).join(" ");
          return (
            <g key={item.algorithmId}>
              <polyline fill="none" points={points} stroke={color} strokeWidth="2.2" />
              {item.points.map((point) => (
                <circle
                  cx={xFor(point.strength)}
                  cy={yFor(point.accuracy)}
                  fill={color}
                  key={`${item.algorithmId}-${point.strength}`}
                  r="3.4"
                />
              ))}
            </g>
          );
        })}
      </svg>
      <div className="curve-legend">
        {series.map((item, index) => (
          <span key={item.algorithmId}>
            <i style={{ background: colors[index % colors.length] }} />
            {cleanLabel(item.algorithmId)}
          </span>
        ))}
      </div>
    </div>
  );
}

function ScoreCurve({
  series,
  algorithmLabels,
  attackLabels,
  colorDomain,
  shapeDomain,
  performanceThresholds,
  scalePointSize,
  onSelectPoint,
  pointCaption,
  seriesCaption,
  showCaption,
  showLegend
}: {
  series: Array<{
    id: string;
    colorKey: string;
    label: string;
    shapeKey: string;
    points: Array<{ x: number; y: number; raw: BenchmarkCurvePoint }>;
  }>;
  algorithmLabels: Record<string, string>;
  attackLabels: Record<string, string>;
  colorDomain?: string[];
  shapeDomain?: string[];
  performanceThresholds?: number[];
  scalePointSize: boolean;
  onSelectPoint?: (point: BenchmarkCurvePoint) => void;
  pointCaption: string;
  seriesCaption: string;
  showCaption: boolean;
  showLegend: boolean;
}) {
  const width = 540;
  const height = 260;
  const leftPad = 48;
  const rightPad = 30;
  const topPad = 28;
  const bottomPad = 52;
  const plotWidth = width - leftPad - rightPad;
  const plotHeight = height - topPad - bottomPad;
  const axisY = height - bottomPad;
  const plotCenterY = topPad + plotHeight / 2;
  const colorKeys = colorDomain?.length ? colorDomain : Array.from(new Set(series.map((item) => item.colorKey)));
  const shapeKeys = shapeDomain?.length ? shapeDomain : Array.from(new Set(series.map((item) => item.shapeKey)));
  const colorFor = (colorKey: string) => curveDomainColor(colorKeys, colorKey);
  const shapeFor = (shapeKey: string) => curveDomainShape(shapeKeys, shapeKey);
  const pointSizeStats = collectPointSizeStats(series);
  const referenceThresholds = normalizePerformanceThresholds(performanceThresholds);
  const domain = buildScoreCurveDomain(series, referenceThresholds);
  const xTicks = buildAxisTicks(domain.xMin, domain.xMax, 4);
  const yTicks = buildAxisTicks(domain.yMin, domain.yMax, 4);
  const xFor = (nqd: number) =>
    leftPad + ((Math.max(domain.xMin, Math.min(domain.xMax, nqd)) - domain.xMin) / Math.max(0.0001, domain.xMax - domain.xMin)) * plotWidth;
  const yFor = (tpr: number) =>
    axisY - ((Math.max(domain.yMin, Math.min(domain.yMax, tpr)) - domain.yMin) / Math.max(0.0001, domain.yMax - domain.yMin)) * plotHeight;
  const labelAlgorithm = (algorithmId: string) => labelFromMap(algorithmLabels, algorithmId) ?? cleanLabel(algorithmId);
  const labelAttack = (point: BenchmarkCurvePoint) =>
    labelFromMap(attackLabels, point.attackPresetId) ??
    labelFromMap(attackLabels, point.attackMethod) ??
    cleanLabel(point.attackMethod || point.attackPresetId);

  return (
    <div className="curve-wrap">
      {showCaption ? (
        <div className="curve-caption">
          <span>{seriesCaption}</span>
          <span>{pointCaption}</span>
        </div>
      ) : null}
      <svg className="curve-chart" role="img" viewBox={`0 0 ${width} ${height}`}>
        <line className="chart-axis" x1={leftPad} x2={leftPad} y1={topPad} y2={axisY} />
        <line className="chart-axis" x1={leftPad} x2={width - rightPad} y1={axisY} y2={axisY} />
        {yTicks.map((tick) => (
          <g key={tick}>
            <line className="chart-grid" x1={leftPad} x2={width - rightPad} y1={yFor(tick)} y2={yFor(tick)} />
            <text className="chart-label" textAnchor="end" x={leftPad - 8} y={yFor(tick) + 4}>
              {Math.round(tick * 100)}
            </text>
          </g>
        ))}
        {xTicks.map((tick) => (
          <text className="chart-label" key={tick} textAnchor="middle" x={xFor(tick)} y={height - 30}>
            {formatAxisTick(tick, xTicks)}
          </text>
        ))}
        {referenceThresholds.map((threshold) => (
          <g key={threshold}>
            <line
              className="chart-reference-line"
              x1={leftPad}
              x2={width - rightPad}
              y1={yFor(threshold)}
              y2={yFor(threshold)}
            />
            <text className="chart-reference-label" textAnchor="end" x={width - rightPad - 4} y={yFor(threshold) - 5}>
              Q@P{Math.round(threshold * 100)}
            </text>
          </g>
        ))}
        <text className="chart-axis-title" textAnchor="middle" x={leftPad + plotWidth / 2} y={height - 8}>
          Normalized Quality Degradation
        </text>
        <text className="chart-axis-title" textAnchor="middle" transform={`rotate(-90 14 ${plotCenterY})`} x={14} y={plotCenterY}>
          TPR
        </text>
        {series.map((item) => {
          const color = colorFor(item.colorKey);
          const shape = shapeFor(item.shapeKey);
          const points = item.points.map((point) => `${xFor(point.x)},${yFor(point.y)}`).join(" ");
          return (
            <g key={item.id}>
              <polyline fill="none" points={points} stroke={color} strokeWidth="2.2" />
              {item.points.map((point, pointIndex) => (
                <g key={`${item.id}-${point.raw.attackPresetId}-${point.raw.attackParamStrength ?? point.raw.attackStrength}-${point.x}-${pointIndex}`}>
                  <title>
                    {`Algorithm: ${labelAlgorithm(point.raw.algorithmId)}
Attack: ${labelAttack(point.raw)}
Variant: ${variantDisplayLabel(point.raw)}
Strength: ${point.raw.attackParamStrengthName ?? "strength"} ${formatStrength(point.raw.attackParamStrength ?? point.raw.attackStrength)}
NQD: ${formatMetric(point.raw.xNqd)}
TPR: ${formatMetric(point.raw.yTprAtFpr)}
Samples: ${formatSampleCount(point.raw.sampleCount)}`}
                  </title>
                  <PointGlyph
                    className="clickable-chart-point"
                    color={color}
                    onClick={() => onSelectPoint?.(point.raw)}
                    shape={shape}
                    size={scalePointSize ? pointRadius(point.raw, pointSizeStats) : undefined}
                    x={xFor(point.x)}
                    y={yFor(point.y)}
                  />
                </g>
              ))}
            </g>
          );
        })}
      </svg>
      {showLegend ? (
        <div className="curve-legend">
          {series.map((item) => (
            <span key={item.id}>
              <CurveLegendGlyph color={colorFor(item.colorKey)} shape={shapeFor(item.shapeKey)} />
              {item.label}
              <small>{item.points.length}</small>
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function PointGlyph({
  className,
  color,
  onClick,
  shape,
  size = 3.6,
  x,
  y
}: {
  className?: string;
  color: string;
  onClick?: () => void;
  shape: ScoreCurveShape;
  size?: number;
  x: number;
  y: number;
}) {
  if (shape === "square") {
    return (
      <rect
        className={className}
        fill={color}
        height={size * 2}
        onClick={onClick}
        rx="1.2"
        width={size * 2}
        x={x - size}
        y={y - size}
      />
    );
  }
  if (shape === "triangle") {
    return (
      <path
        className={className}
        d={`M ${x} ${y - size - 1} L ${x + size + 1} ${y + size} L ${x - size - 1} ${y + size} Z`}
        fill={color}
        onClick={onClick}
      />
    );
  }
  if (shape === "downTriangle") {
    return (
      <path
        className={className}
        d={`M ${x} ${y + size + 1} L ${x + size + 1} ${y - size} L ${x - size - 1} ${y - size} Z`}
        fill={color}
        onClick={onClick}
      />
    );
  }
  if (shape === "diamond") {
    return (
      <path
        className={className}
        d={`M ${x} ${y - size - 1} L ${x + size + 1} ${y} L ${x} ${y + size + 1} L ${x - size - 1} ${y} Z`}
        fill={color}
        onClick={onClick}
      />
    );
  }
  if (shape === "cross" || shape === "plus") {
    const rotate = shape === "cross" ? `rotate(45 ${x} ${y})` : undefined;
    return (
      <g className={className} onClick={onClick} transform={rotate}>
        <line stroke={color} strokeLinecap="round" strokeWidth="1.9" x1={x - size} x2={x + size} y1={y} y2={y} />
        <line stroke={color} strokeLinecap="round" strokeWidth="1.9" x1={x} x2={x} y1={y - size} y2={y + size} />
      </g>
    );
  }
  if (shape === "star") {
    return <path className={className} d={starPath(x, y, size + 1.4, size * 0.48)} fill={color} onClick={onClick} />;
  }
  if (shape === "hexagon") {
    return <path className={className} d={polygonPath(x, y, size + 1.1, 6)} fill={color} onClick={onClick} />;
  }
  if (shape === "pentagon") {
    return <path className={className} d={polygonPath(x, y, size + 1.2, 5)} fill={color} onClick={onClick} />;
  }
  return <circle className={className} cx={x} cy={y} fill={color} onClick={onClick} r={size} />;
}

function polygonPath(x: number, y: number, radius: number, sides: number): string {
  return Array.from({ length: sides }, (_, index) => {
    const angle = -Math.PI / 2 + (index * Math.PI * 2) / sides;
    const prefix = index === 0 ? "M" : "L";
    return `${prefix} ${x + Math.cos(angle) * radius} ${y + Math.sin(angle) * radius}`;
  }).join(" ") + " Z";
}

function starPath(x: number, y: number, outerRadius: number, innerRadius: number): string {
  return Array.from({ length: 10 }, (_, index) => {
    const angle = -Math.PI / 2 + (index * Math.PI) / 5;
    const radius = index % 2 === 0 ? outerRadius : innerRadius;
    const prefix = index === 0 ? "M" : "L";
    return `${prefix} ${x + Math.cos(angle) * radius} ${y + Math.sin(angle) * radius}`;
  }).join(" ") + " Z";
}

function variantDisplayLabel(point: BenchmarkCurvePoint): string {
  const label = point.attackVariantLabel?.trim();
  if (label && label !== "default") {
    return label;
  }
  return "default";
}

function cleanLabel(value: string): string {
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
    svd: "SVD",
    wam: "WAM"
  };
  return value
    .replace(/^alg[-_]/, "")
    .replace(/^atk[-_]/, "")
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((part) => special[part.toLowerCase()] ?? part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatStrength(value: number): string {
  if (!Number.isFinite(value)) {
    return "n/a";
  }
  return value.toFixed(2).replace(/\.00$/, "").replace(/0$/, "");
}

function formatMetric(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) {
    return "n/a";
  }
  return value.toFixed(3);
}

function labelFromMap(labels: Record<string, string>, key: string | undefined): string | undefined {
  if (!key) {
    return undefined;
  }
  const normalized = key.replace(/^alg[-_]/, "").replace(/^atk[-_]/, "");
  const underscore = key.replace(/-/g, "_");
  const normalizedUnderscore = normalized.replace(/-/g, "_");
  return labels[key] ?? labels[underscore] ?? labels[normalized] ?? labels[normalizedUnderscore];
}

function normalizePerformanceThresholds(thresholds: number[] | undefined): number[] {
  const source = thresholds?.length ? thresholds : [0.95, 0.7];
  return Array.from(
    new Set(
      source
        .filter((threshold) => Number.isFinite(threshold) && threshold > 0 && threshold < 1)
        .map((threshold) => Number(threshold.toFixed(4)))
    )
  ).sort((left, right) => right - left);
}

function buildScoreCurveDomain(
  series: Array<{
    points: Array<{ x: number; y: number }>;
  }>,
  referenceThresholds: number[]
) {
  const xValues = series
    .flatMap((item) => item.points.map((point) => point.x))
    .filter((value): value is number => Number.isFinite(value));
  const yValues = [
    ...series.flatMap((item) => item.points.map((point) => point.y)),
    ...referenceThresholds
  ].filter((value): value is number => Number.isFinite(value));
  const xDomain = paddedDomain(xValues, { floor: 0, fallbackMax: 1.2, minSpan: 0.12, padRatio: 0.08 });
  const yDomain = paddedDomain(yValues, { floor: 0, ceiling: 1, fallbackMax: 1, minSpan: 0.16, padRatio: 0.1 });
  return {
    xMin: xDomain.min,
    xMax: xDomain.max,
    yMin: yDomain.min,
    yMax: yDomain.max
  };
}

function paddedDomain(
  values: number[],
  {
    ceiling,
    fallbackMax,
    floor,
    minSpan,
    padRatio
  }: { ceiling?: number; fallbackMax: number; floor: number; minSpan: number; padRatio: number }
) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (!finite.length) {
    return { min: floor, max: fallbackMax };
  }
  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (max - min < minSpan) {
    const center = (min + max) / 2;
    min = center - minSpan / 2;
    max = center + minSpan / 2;
  }
  const padding = Math.max((max - min) * padRatio, minSpan * 0.25);
  min -= padding;
  max += padding;
  min = Math.max(floor, min);
  if (ceiling != null) {
    max = Math.min(ceiling, max);
    if (max - min < minSpan) {
      min = Math.max(floor, max - minSpan);
    }
  }
  if (max - min < minSpan) {
    max = min + minSpan;
  }
  return { min, max };
}

function buildAxisTicks(min: number, max: number, targetCount: number): number[] {
  const span = Math.max(0.0001, max - min);
  const step = niceTickStep(span / Math.max(1, targetCount));
  const first = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let value = first; value <= max + step * 0.5; value += step) {
    const rounded = Number(value.toFixed(6));
    if (rounded >= min - 0.000001 && rounded <= max + 0.000001) {
      ticks.push(rounded);
    }
  }
  if (ticks.length < 2) {
    ticks.push(Number(min.toFixed(6)), Number(max.toFixed(6)));
  }
  return Array.from(new Set(ticks)).sort((left, right) => left - right);
}

function formatAxisTick(value: number, ticks: number[]): string {
  const intervals = ticks.slice(1).map((tick, index) => Math.abs(tick - ticks[index]));
  const smallestInterval = intervals.length ? Math.min(...intervals) : 1;
  const decimals = smallestInterval < 0.1 ? 2 : 1;
  return value.toFixed(decimals);
}

function niceTickStep(rawStep: number): number {
  const exponent = Math.floor(Math.log10(Math.max(rawStep, 0.000001)));
  const base = 10 ** exponent;
  const fraction = rawStep / base;
  if (fraction <= 1) {
    return base;
  }
  if (fraction <= 2) {
    return base * 2;
  }
  if (fraction <= 5) {
    return base * 5;
  }
  return base * 10;
}

function collectPointSizeStats(
  series: Array<{
    points: Array<{ raw: BenchmarkCurvePoint }>;
  }>
) {
  const sampleCounts = series
    .flatMap((item) => item.points.map((point) => point.raw.sampleCount))
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value) && value > 0);
  const strengths = series
    .flatMap((item) => item.points.map((point) => point.raw.attackParamStrength ?? point.raw.attackStrength))
    .filter((value): value is number => Number.isFinite(value));
  return {
    sampleMin: sampleCounts.length ? Math.min(...sampleCounts) : null,
    sampleMax: sampleCounts.length ? Math.max(...sampleCounts) : null,
    strengthMin: strengths.length ? Math.min(...strengths) : null,
    strengthMax: strengths.length ? Math.max(...strengths) : null
  };
}

function pointRadius(
  point: BenchmarkCurvePoint,
  stats: {
    sampleMin: number | null;
    sampleMax: number | null;
    strengthMin: number | null;
    strengthMax: number | null;
  }
): number {
  const sampleCount = point.sampleCount;
  if (
    typeof sampleCount === "number" &&
    Number.isFinite(sampleCount) &&
    sampleCount > 0 &&
    stats.sampleMin != null &&
    stats.sampleMax != null &&
    stats.sampleMax > stats.sampleMin
  ) {
    return scaleRadius(sampleCount, stats.sampleMin, stats.sampleMax, 3.4, 5.6);
  }
  const strength = point.attackParamStrength ?? point.attackStrength;
  if (Number.isFinite(strength) && stats.strengthMin != null && stats.strengthMax != null) {
    return scaleRadius(strength, stats.strengthMin, stats.strengthMax, 3.2, 5.4);
  }
  return 3.6;
}

function scaleRadius(value: number, min: number, max: number, minRadius: number, maxRadius: number): number {
  if (max <= min) {
    return (minRadius + maxRadius) / 2;
  }
  const ratio = Math.max(0, Math.min(1, (value - min) / (max - min)));
  return minRadius + ratio * (maxRadius - minRadius);
}

function formatSampleCount(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) {
    return "n/a";
  }
  return Math.round(value).toLocaleString();
}
