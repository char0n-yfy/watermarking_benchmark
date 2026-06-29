import type { BenchmarkCategoryScore } from "@/lib/types";
import type { CSSProperties } from "react";

export function BenchmarkRadar({
  categories,
  emptyText,
  series
}: {
  categories: BenchmarkCategoryScore[];
  emptyText: string;
  series?: Array<{ id: string; label: string; categories: BenchmarkCategoryScore[] }>;
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

  const size = 280;
  const center = size / 2;
  const radius = 96;
  const rings = [0.25, 0.5, 0.75, 1];
  const colors = ["#2563eb", "#0f766e", "#a16207", "#b42318", "#7c3aed"];
  const axisPoints = categories.map((category, index) => {
    const angle = (Math.PI * 2 * index) / categories.length - Math.PI / 2;
    return {
      category,
      angle,
      labelX: center + Math.cos(angle) * (radius + 34),
      labelY: center + Math.sin(angle) * (radius + 34),
      axisX: center + Math.cos(angle) * radius,
      axisY: center + Math.sin(angle) * radius
    };
  });
  const fallbackSeries = [
    {
      id: "score",
      label: "score",
      categories
    }
  ];
  const drawableSeries = visibleSeries.length > 0 ? visibleSeries : fallbackSeries;

  return (
    <div className="radar-wrap">
      <svg className="radar-chart" role="img" viewBox={`0 0 ${size} ${size}`}>
        {rings.map((ring) => {
          const ringPoints = categories
            .map((_, index) => {
              const angle = (Math.PI * 2 * index) / categories.length - Math.PI / 2;
              return `${center + Math.cos(angle) * radius * ring},${center + Math.sin(angle) * radius * ring}`;
            })
            .join(" ");
          return <polygon className="radar-ring" key={ring} points={ringPoints} />;
        })}
        {axisPoints.map((point) => (
          <line
            className="radar-axis"
            key={point.category.key}
            x1={center}
            x2={point.axisX}
            y1={center}
            y2={point.axisY}
          />
        ))}
        {drawableSeries.map((item, index) => {
          const color = colors[index % colors.length];
          const points = item.categories.map((category, categoryIndex) => {
            const angle = (Math.PI * 2 * categoryIndex) / categories.length - Math.PI / 2;
            const score = Math.max(0, Math.min(1, category.score ?? 0));
            return {
              category,
              x: center + Math.cos(angle) * radius * score,
              y: center + Math.sin(angle) * radius * score
            };
          });
          return (
            <g key={item.id}>
              <polygon
                className="radar-area"
                points={points.map((point) => `${point.x},${point.y}`).join(" ")}
                style={{
                  "--radar-color": color,
                  "--radar-fill": `${color}24`
                } as CSSProperties}
              />
              {points.map((point) => (
                <circle
                  className={point.category.covered ? "radar-dot covered" : "radar-dot"}
                  cx={point.x}
                  cy={point.y}
                  fill={color}
                  key={`${item.id}-${point.category.key}-dot`}
                  r="4"
                />
              ))}
            </g>
          );
        })}
        {axisPoints.map((point) => (
          <text
            className="radar-label"
            key={`${point.category.key}-label`}
            textAnchor={point.labelX < center - 8 ? "end" : point.labelX > center + 8 ? "start" : "middle"}
            x={point.labelX}
            y={point.labelY}
          >
            {shortLabel(point.category.label)}
          </text>
        ))}
      </svg>
      <div className="radar-score-list">
        {drawableSeries.map((item, index) => (
          <span key={item.id}>
            <i className="covered" style={{ background: colors[index % colors.length] }} />
            {item.label}
          </span>
        ))}
      </div>
    </div>
  );
}

function shortLabel(label: string): string {
  return label
    .replace("Distortion ", "Dist. ")
    .replace("Regeneration ", "Regen. ")
    .replace("Adv Embedding ", "Adv. Emb. ")
    .replace("Adv Surrogate Detector", "Adv. Surrogate");
}
