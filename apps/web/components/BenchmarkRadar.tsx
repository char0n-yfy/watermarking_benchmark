import { chartAreaFill, chartColorByIndex, chartColorForDomain, chartStrokeColor } from "@/lib/chart-colors";
import type { BenchmarkCategoryScore } from "@/lib/types";
import type { CSSProperties } from "react";
const RING_LEVELS = [0.2, 0.4, 0.6, 0.8, 1];

export function BenchmarkRadar({
  categories,
  emptyText,
  series,
  selectedCategoryKey,
  onSelectCategory,
  colorDomain,
  variant = "default"
}: {
  categories: BenchmarkCategoryScore[];
  emptyText: string;
  series?: Array<{ id: string; label: string; categories: BenchmarkCategoryScore[] }>;
  selectedCategoryKey?: string;
  onSelectCategory?: (category: BenchmarkCategoryScore) => void;
  colorDomain?: string[];
  variant?: "default" | "hero";
}) {
  const visibleSeries = (series ?? [])
    .map((item) => ({
      ...item,
      categories: categories.map((category) => item.categories.find((candidate) => candidate.key === category.key) ?? category)
    }))
    .filter((item) => item.categories.some((category) => category.score != null));

  if (categories.length === 0 || (visibleSeries.length === 0 && categories.every((item) => item.score == null))) {
    return <div className="empty compact-empty">{emptyText}</div>;
  }

  const isHero = variant === "hero";
  const layout = radarLayout(categories, isHero);
  const { padding, size, center, radius, labelRadius, maxLabelWidth } = layout;
  const isDense = categories.length >= 12;
  const isLongLabels = maxLabelWidth >= 72;

  const axisCoverage = categories.map((category) =>
    visibleSeries.some((item) => {
      const match = item.categories.find((candidate) => candidate.key === category.key);
      return match?.covered === true;
    })
  );

  const axisPoints = categories.map((category, index) => {
    const angle = axisAngle(index, categories.length);
    const rawLabelX = center + Math.cos(angle) * labelRadius;
    const rawLabelY = center + Math.sin(angle) * labelRadius;
    return {
      category,
      covered: axisCoverage[index],
      angle,
      labelX: rawLabelX,
      labelY: Math.max(padding * 0.18, Math.min(size - padding * 0.18, rawLabelY)),
      axisX: center + Math.cos(angle) * radius,
      axisY: center + Math.sin(angle) * radius
    };
  });

  const fallbackSeries = [{ id: "score", label: "score", categories }];
  const drawableSeries = visibleSeries.length > 0 ? visibleSeries : fallbackSeries;
  const renderSeries = drawableSeries.slice().sort((left, right) => seriesArea(right.categories) - seriesArea(left.categories));

  return (
    <div
      className={["radar-wrap", isHero ? "hero" : "", isDense ? "dense" : "", isLongLabels ? "long-labels" : ""]
        .filter(Boolean)
        .join(" ")}
    >
      <svg
        className={isHero ? "radar-chart radar-chart-hero" : "radar-chart"}
        role="img"
        viewBox={`0 0 ${size} ${size}`}
      >
        <circle className="radar-boundary" cx={center} cy={center} r={radius} />
        {RING_LEVELS.map((level) => (
          <circle className="radar-ring" cx={center} cy={center} key={level} r={radius * level} />
        ))}
        {axisPoints.map((point) => (
          <line
            className={[
              "radar-axis",
              point.covered ? "covered" : "uncovered",
              selectedCategoryKey === point.category.key ? "active" : ""
            ]
              .filter(Boolean)
              .join(" ")}
            key={point.category.key}
            x1={center}
            x2={point.axisX}
            y1={center}
            y2={point.axisY}
          />
        ))}
        {isHero
          ? RING_LEVELS.map((level) => (
              <text
                className="radar-scale-label"
                key={`scale-${level}`}
                textAnchor="start"
                x={center + radius * level + 5}
                y={center + 4}
              >
                {level === 1 ? "1" : level.toFixed(1)}
              </text>
            ))
          : null}
        {renderSeries.map((item, index) => {
          const color = seriesColor(item.id, index, colorDomain);
          const points = item.categories.map((category, categoryIndex) => {
            const angle = axisAngle(categoryIndex, categories.length);
            const scoreRadius = categoryScoreRadius(category);
            return {
              category,
              scoreRadius,
              x: scoreRadius == null ? center : center + Math.cos(angle) * radius * scoreRadius,
              y: scoreRadius == null ? center : center + Math.sin(angle) * radius * scoreRadius
            };
          });
          const areaPath = buildRadarAreaPath(points);
          return (
            <g key={item.id}>
              {areaPath ? (
                <path
                  className="radar-area"
                  d={areaPath}
                  style={
                    {
                      "--radar-color": chartStrokeColor(color),
                      "--radar-fill": chartAreaFill(color),
                      "--radar-dot-color": color
                    } as CSSProperties
                  }
                />
              ) : null}
              {points
                .filter((point) => point.scoreRadius != null)
                .map((point) => (
                  <circle
                    className={[
                      "radar-dot",
                      "covered",
                      selectedCategoryKey === point.category.key ? "active" : ""
                    ]
                      .filter(Boolean)
                      .join(" ")}
                    cx={point.x}
                    cy={point.y}
                    fill={color}
                    key={`${item.id}-${point.category.key}-dot`}
                    onClick={() => onSelectCategory?.(point.category)}
                    r={isHero ? "5" : "4"}
                    style={{ "--radar-color": color } as CSSProperties}
                  />
                ))}
            </g>
          );
        })}
        {axisPoints.map((point) => (
          <text
            className={[
              "radar-label",
              point.covered ? "covered" : "uncovered",
              selectedCategoryKey === point.category.key ? "active" : ""
            ]
              .filter(Boolean)
              .join(" ")}
            key={`${point.category.key}-label`}
            onClick={() => onSelectCategory?.(point.category)}
            textAnchor={point.labelX < center - 8 ? "end" : point.labelX > center + 8 ? "start" : "middle"}
            dominantBaseline="middle"
            x={point.labelX}
            y={point.labelY}
          >
            {isHero ? point.category.label : shortLabel(point.category.label)}
          </text>
        ))}
      </svg>
      <div className={isHero ? "radar-score-list hero" : "radar-score-list"}>
        {drawableSeries.map((item, index) => (
          <span key={item.id}>
            <i className="covered" style={{ background: seriesColor(item.id, index, colorDomain) }} />
            {item.label}
          </span>
        ))}
      </div>
    </div>
  );
}

function axisAngle(index: number, count: number): number {
  return (Math.PI * 2 * index) / count - Math.PI / 2;
}

function estimateLabelWidth(label: string, isHero: boolean): number {
  let width = 0;
  for (const char of label) {
    width += char.charCodeAt(0) > 0xff ? (isHero ? 12 : 10.5) : (isHero ? 6.8 : 6);
  }
  return width;
}

function radarLayout(categories: BenchmarkCategoryScore[], isHero: boolean) {
  const axisCount = categories.length;
  const maxLabelWidth = Math.max(...categories.map((item) => estimateLabelWidth(item.label, isHero)), 28);
  const plotSize = isHero ? 420 : Math.round(Math.max(210, 280 - Math.max(0, axisCount - 9) * 7));
  const radius = isHero ? 168 : Math.round(plotSize * (axisCount >= 12 ? 0.34 : axisCount <= 6 ? 0.35 : 0.39));
  const labelGap = isHero ? 52 : axisCount >= 12 ? 20 : axisCount <= 6 ? 38 : 32;
  const labelRadius = radius + labelGap;
  const edgeMargin = 8;
  const padding = Math.max(
    isHero ? 84 : 56,
    Math.ceil(labelRadius + maxLabelWidth + edgeMargin - plotSize / 2)
  );
  const size = plotSize + padding * 2;

  return {
    padding,
    plotSize,
    size,
    center: size / 2,
    radius,
    labelRadius,
    maxLabelWidth
  };
}

function seriesColor(id: string, index: number, colorDomain?: string[]): string {
  if (colorDomain && colorDomain.length > 0) {
    return chartColorForDomain(colorDomain, id);
  }
  return chartColorByIndex(index);
}

function seriesArea(categories: BenchmarkCategoryScore[]): number {
  return categories.reduce((sum, category) => sum + (categoryScoreRadius(category) ?? 0), 0);
}

function categoryScoreRadius(category: BenchmarkCategoryScore): number | null {
  if (!category.covered || category.score == null) {
    return null;
  }
  return Math.max(0, Math.min(1, category.score));
}

type RadarPlotPoint = {
  category: BenchmarkCategoryScore;
  scoreRadius: number | null;
  x: number;
  y: number;
};

function buildRadarAreaPath(points: RadarPlotPoint[]): string | null {
  const segments: string[] = [];
  let ringStart: RadarPlotPoint | null = null;
  let previous: RadarPlotPoint | null = null;

  const closeRing = () => {
    if (ringStart && previous && (ringStart !== previous || segments.length > 0)) {
      segments.push(`L ${ringStart.x},${ringStart.y} Z`);
    }
    ringStart = null;
    previous = null;
  };

  for (const point of points) {
    if (point.scoreRadius == null) {
      closeRing();
      continue;
    }
    if (!ringStart) {
      segments.push(`M ${point.x},${point.y}`);
      ringStart = point;
    } else {
      segments.push(`L ${point.x},${point.y}`);
    }
    previous = point;
  }

  closeRing();
  return segments.length > 0 ? segments.join(" ") : null;
}

function shortLabel(label: string): string {
  return label
    .replace("Distortion Combination", "Dist. Combo")
    .replace("Distortion Single", "Dist. Single")
    .replace("Regeneration Single", "Regen. Single")
    .replace("Regeneration Rinsing", "Regen. Rinsing")
    .replace("Adv Embedding Grey-box", "Emb. Grey-box")
    .replace("Adv Embedding Black-box", "Emb. Black-box")
    .replace("Adv Surrogate Detector", "Surrogate");
}
