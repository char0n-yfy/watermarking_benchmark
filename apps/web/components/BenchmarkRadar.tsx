import type { BenchmarkCategoryScore } from "@/lib/types";

export function BenchmarkRadar({
  categories,
  emptyText
}: {
  categories: BenchmarkCategoryScore[];
  emptyText: string;
}) {
  if (categories.length === 0 || categories.every((item) => item.score == null)) {
    return <div className="empty compact-empty">{emptyText}</div>;
  }

  const size = 280;
  const center = size / 2;
  const radius = 96;
  const rings = [0.25, 0.5, 0.75, 1];
  const points = categories.map((category, index) => {
    const angle = (Math.PI * 2 * index) / categories.length - Math.PI / 2;
    const score = Math.max(0, Math.min(1, category.score ?? 0));
    return {
      category,
      angle,
      x: center + Math.cos(angle) * radius * score,
      y: center + Math.sin(angle) * radius * score,
      labelX: center + Math.cos(angle) * (radius + 34),
      labelY: center + Math.sin(angle) * (radius + 34),
      axisX: center + Math.cos(angle) * radius,
      axisY: center + Math.sin(angle) * radius
    };
  });
  const polygon = points.map((point) => `${point.x},${point.y}`).join(" ");

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
        {points.map((point) => (
          <line
            className="radar-axis"
            key={point.category.key}
            x1={center}
            x2={point.axisX}
            y1={center}
            y2={point.axisY}
          />
        ))}
        <polygon className="radar-area" points={polygon} />
        {points.map((point) => (
          <circle
            className={point.category.covered ? "radar-dot covered" : "radar-dot"}
            cx={point.x}
            cy={point.y}
            key={`${point.category.key}-dot`}
            r="4"
          />
        ))}
        {points.map((point) => (
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
        {categories.map((category) => (
          <span key={category.key}>
            <i className={category.covered ? "covered" : ""} />
            {shortLabel(category.label)} {category.score == null ? "n/a" : category.score.toFixed(2)}
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
