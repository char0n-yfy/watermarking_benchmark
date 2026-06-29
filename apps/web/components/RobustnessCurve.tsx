import { buildCurveSeries } from "@/lib/insights";
import type { BenchmarkScore, RunResults } from "@/lib/types";

export function RobustnessCurve({
  results,
  score,
  emptyText,
  selectedAlgorithmIds = []
}: {
  results: RunResults | null;
  score?: BenchmarkScore | null;
  emptyText: string;
  selectedAlgorithmIds?: string[];
}) {
  if (score?.curvePoints?.length) {
    const selected = new Set(selectedAlgorithmIds);
    const grouped = new Map<string, Array<{ x: number; y: number; attack: string }>>();
    for (const point of score.curvePoints) {
      if (selected.size > 0 && !selected.has(point.algorithmId)) {
        continue;
      }
      const key = point.algorithmId;
      const current = grouped.get(key) ?? [];
      current.push({ x: point.xNqd, y: point.yTprAtFpr, attack: point.attackMethod });
      grouped.set(key, current);
    }
    const scoreSeries = Array.from(grouped.entries())
      .map(([algorithmId, points]) => ({
        algorithmId,
        points: points.sort((a, b) => a.x - b.x)
      }))
      .filter((item) => item.points.length >= 2);
    if (scoreSeries.length > 0) {
      return <ScoreCurve series={scoreSeries} />;
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
              <polyline fill="none" points={points} stroke={color} strokeWidth="3" />
              {item.points.map((point) => (
                <circle
                  cx={xFor(point.strength)}
                  cy={yFor(point.accuracy)}
                  fill={color}
                  key={`${item.algorithmId}-${point.strength}`}
                  r="4"
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
            {item.algorithmId}
          </span>
        ))}
      </div>
    </div>
  );
}

function ScoreCurve({
  series
}: {
  series: Array<{ algorithmId: string; points: Array<{ x: number; y: number; attack: string }> }>;
}) {
  const width = 540;
  const height = 220;
  const pad = 34;
  const colors = ["#2563eb", "#0f766e", "#a16207", "#b42318", "#7c3aed"];
  const xFor = (nqd: number) => pad + Math.max(0, Math.min(1.2, nqd)) / 1.2 * (width - pad * 2);
  const yFor = (tpr: number) => height - pad - Math.max(0, Math.min(1, tpr)) * (height - pad * 2);

  return (
    <div className="curve-wrap">
      <svg className="curve-chart" role="img" viewBox={`0 0 ${width} ${height}`}>
        <line className="chart-axis" x1={pad} x2={pad} y1={pad} y2={height - pad} />
        <line className="chart-axis" x1={pad} x2={width - pad} y1={height - pad} y2={height - pad} />
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line className="chart-grid" x1={pad} x2={width - pad} y1={yFor(tick)} y2={yFor(tick)} />
            <text className="chart-label" x={8} y={yFor(tick) + 4}>
              {Math.round(tick * 100)}
            </text>
          </g>
        ))}
        {[0, 0.4, 0.8, 1.2].map((tick) => (
          <text className="chart-label" key={tick} x={xFor(tick) - 8} y={height - 8}>
            {tick.toFixed(1)}
          </text>
        ))}
        <text className="chart-label" x={width - 88} y={height - 8}>
          NQD
        </text>
        <text className="chart-label" x={6} y={24}>
          TPR
        </text>
        {series.map((item, index) => {
          const color = colors[index % colors.length];
          const points = item.points.map((point) => `${xFor(point.x)},${yFor(point.y)}`).join(" ");
          return (
            <g key={item.algorithmId}>
              <polyline fill="none" points={points} stroke={color} strokeWidth="3" />
              {item.points.map((point) => (
                <circle
                  cx={xFor(point.x)}
                  cy={yFor(point.y)}
                  fill={color}
                  key={`${item.algorithmId}-${point.attack}-${point.x}`}
                  r="4"
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
            {item.algorithmId}
          </span>
        ))}
      </div>
    </div>
  );
}
