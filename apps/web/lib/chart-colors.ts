/**
 * Matplotlib / seaborn "tab10" palette used in the Benchmark Watermarks paper figures.
 * Order: blue, orange, green, red, purple, brown, pink, gray, olive, cyan.
 */
export const TAB10_COLORS = [
  "#1f77b4",
  "#ff7f0e",
  "#2ca02c",
  "#d62728",
  "#9467bd",
  "#8c564b",
  "#e377c2",
  "#7f7f7f",
  "#bcbd22",
  "#17becf"
] as const;

/** tab20 light variants for additional watermark algorithms. */
const TAB20_LIGHT_COLORS = [
  "#aec7e8",
  "#ffbb78",
  "#98df8a",
  "#ff9896",
  "#c5b0d5",
  "#c49c94",
  "#f7b6d2",
  "#c7c7c7",
  "#dbdb8d",
  "#9edae5",
  "#393b79"
] as const;

export const ALGORITHM_CHART_COLORS = [...TAB10_COLORS, ...TAB20_LIGHT_COLORS] as readonly string[];

export function chartColorForDomain(domain: string[], key: string): string {
  const sorted = [...domain].sort((left, right) => left.localeCompare(right));
  const index = sorted.indexOf(key);
  return chartColorByIndex(index >= 0 ? index : domain.length);
}

export function chartColorByIndex(index: number): string {
  return ALGORITHM_CHART_COLORS[Math.max(0, index) % ALGORITHM_CHART_COLORS.length];
}

/** Solid fills like the reference violin / bar plots. */
export function chartBarFill(color: string): string {
  return color;
}

/** Light translucent radar fill like the reference radar polygons. */
export function chartAreaFill(color: string): string {
  return `color-mix(in srgb, ${color} 22%, transparent)`;
}

export function chartStrokeColor(color: string): string {
  return color;
}
