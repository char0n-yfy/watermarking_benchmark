import {
  buildResourcePageDetailAxes,
  type AttackResourceCategoryKey,
  type ResourcePageDetailAxisSpec
} from "@/lib/attack-resource-taxonomy";
import type {
  BenchmarkAttackLeaderboardRow,
  BenchmarkCategoryScore,
  BenchmarkLeaderboardRow
} from "@/lib/types";

export const OVERVIEW_ATTACK_RESOURCE_CATEGORIES = [
  { key: "distortion_attacks", labelZh: "经典失真", labelEn: "Classic distortion" },
  { key: "physical_channel_attacks", labelZh: "物理信道", labelEn: "Physical channel" },
  { key: "3d_viewpoint_rerendering", labelZh: "3D 视角重渲染", labelEn: "3D viewpoint re-rendering" },
  { key: "regeneration_attacks", labelZh: "再生成", labelEn: "Regeneration" },
  { key: "consumer_enhancement_workflow_attacks", labelZh: "消费级增强", labelEn: "Consumer enhancement" }
] as const;

export type OverviewAttackResourceCategoryKey = (typeof OVERVIEW_ATTACK_RESOURCE_CATEGORIES)[number]["key"];

const MAIN_RADAR_EXTRA_AXES = [
  { key: "fidelity", labelZh: "自身保真度", labelEn: "Clean fidelity" },
  { key: "complexity", labelZh: "算法复杂度", labelEn: "Algorithm complexity" }
] as const;

export type OverviewDetailRadar = {
  categoryKey: OverviewAttackResourceCategoryKey;
  title: string;
  subtitle: string;
  categories: BenchmarkCategoryScore[];
  series: Array<{ id: string; label: string; categories: BenchmarkCategoryScore[] }>;
};

export function attackResourceCategoryKey(attackMethod: string): OverviewAttackResourceCategoryKey {
  if (attackMethod.startsWith("3d_viewpoint_rerendering")) {
    return "3d_viewpoint_rerendering";
  }
  if (attackMethod === "screen_shoot" || attackMethod === "print_camera" || attackMethod === "combined_physical") {
    return "physical_channel_attacks";
  }
  if (attackMethod.startsWith("cew_")) {
    return "consumer_enhancement_workflow_attacks";
  }
  if (
    attackMethod.startsWith("regen_") ||
    attackMethod === "2x_regen" ||
    attackMethod === "4x_regen" ||
    attackMethod === "noise_to_image" ||
    attackMethod === "image_to_vedio"
  ) {
    return "regeneration_attacks";
  }
  return "distortion_attacks";
}

function categoryLabel(categoryKey: string, fallback: string, language: string): string {
  const attack = OVERVIEW_ATTACK_RESOURCE_CATEGORIES.find((item) => item.key === categoryKey);
  if (attack) {
    return language === "zh" ? attack.labelZh : attack.labelEn;
  }
  const extra = MAIN_RADAR_EXTRA_AXES.find((item) => item.key === categoryKey);
  if (extra) {
    return language === "zh" ? extra.labelZh : extra.labelEn;
  }
  return fallback;
}

function detailAxisLabel(spec: ResourcePageDetailAxisSpec, language: string): string {
  return language === "zh" ? spec.labelZh : spec.labelEn;
}

function detailRadarSubtitle(categoryKey: OverviewAttackResourceCategoryKey, language: string): string {
  const subtitles: Record<OverviewAttackResourceCategoryKey, { zh: string; en: string }> = {
    distortion_attacks: {
      zh: "9 种经典失真资源（与资源页一致）的检测鲁棒性（AUC）",
      en: "Detection robustness (AUC) across 9 classic distortion resources"
    },
    physical_channel_attacks: {
      zh: "3 种物理信道资源（与资源页一致）的检测鲁棒性（AUC）",
      en: "Detection robustness (AUC) across 3 physical-channel resources"
    },
    "3d_viewpoint_rerendering": {
      zh: "4 种 3D 视角运动资源（合并 point/ahead 底层 preset）的检测鲁棒性（AUC）",
      en: "Detection robustness (AUC) across 4 3D motion resources (point/ahead merged)"
    },
    regeneration_attacks: {
      zh: "6 种再生成资源（与资源页一致）的检测鲁棒性（AUC）",
      en: "Detection robustness (AUC) across 6 regeneration resources (aligned with Resources page)"
    },
    consumer_enhancement_workflow_attacks: {
      zh: "16 种消费级增强资源（与资源页一致）的检测鲁棒性（AUC）",
      en: "Detection robustness (AUC) across 16 consumer-enhancement resources"
    }
  };
  const copy = subtitles[categoryKey];
  return language === "zh" ? copy.zh : copy.en;
}

function meanFinite(values: Array<number | null | undefined>): number | null {
  const finite = values.filter((value): value is number => value != null && Number.isFinite(value));
  if (finite.length === 0) {
    return null;
  }
  return finite.reduce((total, value) => total + value, 0) / finite.length;
}

function categoryAttackScore(
  algorithmId: string,
  categoryKey: OverviewAttackResourceCategoryKey,
  attackLeaderboard: BenchmarkAttackLeaderboardRow[]
): number | null {
  return meanFinite(
    attackLeaderboard
      .filter(
        (row) =>
          row.algorithmId === algorithmId &&
          attackResourceCategoryKey(row.attackMethod) === categoryKey &&
          row.auc != null
      )
      .map((row) => row.auc)
  );
}

function detailAxisScore(
  algorithmId: string,
  categoryKey: OverviewAttackResourceCategoryKey,
  axis: ResourcePageDetailAxisSpec,
  attackLeaderboard: BenchmarkAttackLeaderboardRow[]
): number | null {
  return meanFinite(
    attackLeaderboard
      .filter(
        (row) =>
          row.algorithmId === algorithmId &&
          attackResourceCategoryKey(row.attackMethod) === categoryKey &&
          axis.matchMethod(row.attackMethod) &&
          row.auc != null
      )
      .map((row) => row.auc)
  );
}

function emptyAxis(key: string, label: string): BenchmarkCategoryScore {
  return {
    key,
    label,
    score: null,
    meanNqd: null,
    cellCount: 0,
    covered: false
  };
}

function filledAxis(key: string, label: string, score: number | null): BenchmarkCategoryScore {
  return {
    key,
    label,
    score,
    meanNqd: null,
    cellCount: score == null ? 0 : 1,
    covered: score != null
  };
}

export function buildMainOverviewRadarTemplate(language: string): BenchmarkCategoryScore[] {
  const attackAxes = OVERVIEW_ATTACK_RESOURCE_CATEGORIES.map((category) =>
    emptyAxis(category.key, categoryLabel(category.key, category.key, language))
  );
  const extraAxes = MAIN_RADAR_EXTRA_AXES.map((axis) => emptyAxis(axis.key, categoryLabel(axis.key, axis.key, language)));
  return [...attackAxes, ...extraAxes];
}

export function buildMainOverviewRadarSeries(
  rows: BenchmarkLeaderboardRow[],
  allRows: BenchmarkLeaderboardRow[],
  attackLeaderboard: BenchmarkAttackLeaderboardRow[],
  template: BenchmarkCategoryScore[],
  complexityScore: (row: BenchmarkLeaderboardRow, peers: BenchmarkLeaderboardRow[]) => number | null,
  algorithmLabels: Record<string, string>
): Array<{ id: string; label: string; categories: BenchmarkCategoryScore[] }> {
  return rows.map((row) => ({
    id: row.algorithmId,
    label: algorithmLabels[row.algorithmId] ?? row.algorithmId,
    categories: template.map((axis) => {
      if (axis.key === "fidelity") {
        return filledAxis(axis.key, axis.label, row.cleanFidelity);
      }
      if (axis.key === "complexity") {
        return filledAxis(axis.key, axis.label, complexityScore(row, allRows));
      }
      const attackCategory = axis.key as OverviewAttackResourceCategoryKey;
      return filledAxis(
        axis.key,
        axis.label,
        categoryAttackScore(row.algorithmId, attackCategory, attackLeaderboard)
      );
    })
  }));
}

export function buildOverviewDetailRadars(
  rows: BenchmarkLeaderboardRow[],
  attackLeaderboard: BenchmarkAttackLeaderboardRow[],
  algorithmLabels: Record<string, string>,
  language: string
): OverviewDetailRadar[] {
  return OVERVIEW_ATTACK_RESOURCE_CATEGORIES.map((category) => {
    const axisSpecs = buildResourcePageDetailAxes(category.key as AttackResourceCategoryKey);

    const template = axisSpecs.map((axis) => emptyAxis(axis.key, detailAxisLabel(axis, language)));
    const series = rows.map((row) => ({
      id: row.algorithmId,
      label: algorithmLabels[row.algorithmId] ?? row.algorithmId,
      categories: axisSpecs.map((axis) =>
        filledAxis(
          axis.key,
          detailAxisLabel(axis, language),
          detailAxisScore(row.algorithmId, category.key, axis, attackLeaderboard)
        )
      )
    }));

    return {
      categoryKey: category.key,
      title: language === "zh" ? category.labelZh : category.labelEn,
      subtitle: detailRadarSubtitle(category.key, language),
      categories: template,
      series
    };
  });
}
