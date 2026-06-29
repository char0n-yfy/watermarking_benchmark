import type { AttackPreset, DatasetVersion, ExperimentSelection } from "./types";

export interface MatrixEstimate {
  datasetCount: number;
  algorithmCount: number;
  attackCount: number;
  strengthCount: number;
  seedCount: number;
  cellCount: number;
  sampleCount: number;
  imageOperationCount: number;
  level: "ok" | "warn";
}

export function estimateMatrix(
  selection: ExperimentSelection,
  datasets: DatasetVersion[],
  attacks: AttackPreset[]
): MatrixEstimate {
  const selectedDatasets = datasets.filter((dataset) => selection.datasetIds.includes(dataset.id));
  const selectedAttacks = attacks.filter((attack) => selection.attackPresetIds.includes(attack.id));
  const datasetCount = selectedDatasets.length;
  const algorithmCount = selection.algorithmIds.length;
  const attackCount = selectedAttacks.length;
  const strengthOverrides = selection.attackStrengthOverrides ?? {};
  const paramOverrides = selection.attackParamOverrides ?? {};
  const strengthCount = selectedAttacks.reduce((sum, attack) => {
    const paramOverrideCount = paramOverrides[attack.id]?.length ?? 0;
    if (paramOverrideCount > 0) {
      return sum + paramOverrideCount;
    }
    const overrideCount = strengthOverrides[attack.id]?.filter((value) => Number.isFinite(value)).length ?? 0;
    return sum + Math.max(1, overrideCount || attack.strengths.length);
  }, 0);
  const seedCount = selection.seeds.length;
  const cellCount = datasetCount * algorithmCount * Math.max(1, strengthCount) * seedCount;
  const sampleCount = selectedDatasets.reduce(
    (sum, dataset) => sum + Math.min(dataset.sampleCount, selection.maxSamples),
    0
  );
  const imageOperationCount = cellCount * Math.max(1, sampleCount);

  return {
    datasetCount,
    algorithmCount,
    attackCount,
    strengthCount,
    seedCount,
    cellCount,
    sampleCount,
    imageOperationCount,
    level: cellCount > 64 || imageOperationCount > 10000 ? "warn" : "ok"
  };
}
