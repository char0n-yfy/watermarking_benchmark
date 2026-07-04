import type { AttackPreset } from "@/lib/types";

const VIEWPOINT_METHOD_PATTERN = /^3d_viewpoint_rerendering_(swipe|shake|rotate|rotate_forward)_(point|ahead)$/;

export function parseViewpointAttackMethod(method: string) {
  const match = VIEWPOINT_METHOD_PATTERN.exec(method);
  if (!match) {
    return null;
  }
  return {
    motion: match[1],
    lookatMode: match[2] as "point" | "ahead"
  };
}

export function isHiddenBenchmarkAttack(attack: Pick<AttackPreset, "id" | "method">) {
  return attack.method === "identity" || attack.id === "atk-identity";
}

export function benchmarkAttackMethodKey(
  attack: Pick<AttackPreset, "method" | "executionMethod">
) {
  const method = attack.executionMethod || attack.method;
  const parsed = parseViewpointAttackMethod(method);
  return parsed ? parsed.motion : method;
}

export function countBenchmarkAttackTypes(attacks: AttackPreset[]) {
  const keys = new Set<string>();
  for (const attack of attacks) {
    if (isHiddenBenchmarkAttack(attack)) {
      continue;
    }
    keys.add(benchmarkAttackMethodKey(attack));
  }
  return keys.size;
}

export function countSelectedBenchmarkAttackTypes(attacks: AttackPreset[], selectedPresetIds: string[]) {
  const attackById = new Map(attacks.map((attack) => [attack.id, attack]));
  const keys = new Set<string>();
  for (const presetId of selectedPresetIds) {
    const attack = attackById.get(presetId);
    if (!attack) {
      continue;
    }
    if (isHiddenBenchmarkAttack(attack)) {
      continue;
    }
    keys.add(benchmarkAttackMethodKey(attack));
  }
  return keys.size;
}

export function countBenchmarkAttackTypesFromMethods(methods: string[]) {
  const keys = new Set<string>();
  for (const method of methods) {
    if (!method || method === "identity") {
      continue;
    }
    const parsed = parseViewpointAttackMethod(method);
    keys.add(parsed ? parsed.motion : method);
  }
  return keys.size;
}
