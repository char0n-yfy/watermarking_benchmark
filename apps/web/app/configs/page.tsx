"use client";

import { useEffect, useMemo, useState } from "react";
import { Archive, Braces, Check, Database, Edit3, Gauge, Plus, Save, Search, Shield, Trash2, X } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { useLanguage } from "@/components/LanguageProvider";
import {
  createSavedConfig,
  deleteSavedConfig,
  fetchAlgorithms,
  fetchAttacks,
  fetchDatasets,
  fetchSavedConfigs,
  renameSavedConfig
} from "@/lib/api";
import { localizedName } from "@/lib/i18n";
import { estimateMatrix } from "@/lib/matrix";
import type {
  AlgorithmVersion,
  AttackPreset,
  DatasetVersion,
  ExperimentSelection,
  SavedExperimentConfig
} from "@/lib/types";

const emptySelection: ExperimentSelection = {
  datasetIds: [],
  algorithmIds: [],
  attackPresetIds: [],
  attackStrengthOverrides: {},
  attackParamOverrides: {},
  seeds: [42],
  maxSamples: 100
};

function toggle(values: string[], value: string) {
  return values.includes(value) ? values.filter((item) => item !== value) : [...values, value];
}

function addIds(values: string[], ids: string[]) {
  const next = [...values];
  const seen = new Set(next);
  for (const id of ids) {
    if (!seen.has(id)) {
      next.push(id);
      seen.add(id);
    }
  }
  return next;
}

function removeIds(values: string[], ids: string[]) {
  const removed = new Set(ids);
  return values.filter((value) => !removed.has(value));
}

function matchesResource(
  query: string,
  resource: { id: string; name: string; method?: string; category?: string; description?: string }
) {
  const normalizedQuery = query.trim().toLowerCase();
  if (!normalizedQuery) {
    return true;
  }
  return [resource.id, resource.name, resource.method, resource.category, resource.description]
    .filter(Boolean)
    .join(" ")
    .toLowerCase()
    .includes(normalizedQuery);
}

function categoryLabel(language: string, category: string) {
  const zh: Record<string, string> = {
    "3d_viewpoint_rerendering": "3D 视角重渲染",
    "3d-viewpoint-rerendering": "3D 视角重渲染",
    consumer_enhancement_workflow_attacks: "消费级增强",
    "consumer-enhancement": "消费级增强",
    distortion: "经典失真",
    distortion_attacks: "经典失真",
    geometric: "几何变换",
    identity: "无攻击",
    physical_channel_attacks: "物理信道",
    "physical-channel": "物理信道",
    regeneration: "再生成",
    regeneration_attacks: "再生成",
    removal: "移除攻击"
  };
  const en: Record<string, string> = {
    "3d_viewpoint_rerendering": "3D viewpoint re-rendering",
    "3d-viewpoint-rerendering": "3D viewpoint re-rendering",
    consumer_enhancement_workflow_attacks: "Consumer enhancement",
    "consumer-enhancement": "Consumer enhancement",
    distortion: "Distortion",
    distortion_attacks: "Distortion",
    geometric: "Geometric",
    identity: "Identity",
    physical_channel_attacks: "Physical channel",
    "physical-channel": "Physical channel",
    regeneration: "Regeneration",
    regeneration_attacks: "Regeneration",
    removal: "Removal"
  };
  const labels = language === "zh" ? zh : en;
  return labels[category] ?? category;
}

const VIEWPOINT_CATEGORY = "3d_viewpoint_rerendering";
const DISTORTION_CATEGORY = "distortion_attacks";
const REGENERATION_CATEGORY = "regeneration_attacks";
const PHYSICAL_CATEGORY = "physical_channel_attacks";
const PHYSICAL_CORRECTION_OPTIONS = [true, false];
const HIDDEN_IDENTITY_ATTACK_METHOD = "identity";
const REGENERATION_UNIT_METHODS = ["2x_regen", "4x_regen", "regen_diffusion", "noise_to_image"] as const;
const REGENERATION_VAE_METHOD = "regen_vae";
const REGENERATION_IMAGE_TO_VIDEO_METHOD = "image_to_vedio";
const REGENERATION_IMAGE_TO_VIDEO_XY = [0, 10, 20, 30, 40, 60];
const REGENERATION_VAE_QUALITIES = [1, 2, 3, 4, 5, 6];
const REGENERATION_VAE_QUALITY_BY_MODEL: Record<string, number[]> = {
  "bmshj2018-factorized": REGENERATION_VAE_QUALITIES,
  "cheng2020-anchor": [1, 2, 3, 4, 5, 6],
  "bmshj2018-hyperprior": REGENERATION_VAE_QUALITIES,
  "mbt2018-mean": REGENERATION_VAE_QUALITIES
};
const REGENERATION_VAE_MODEL_NAMES = Object.keys(REGENERATION_VAE_QUALITY_BY_MODEL);
const VIEWPOINT_METHOD_PATTERN = /^3d_viewpoint_rerendering_phase(\d+)_(point|ahead)$/;
const VIEWPOINT_MOTION_ORDER = ["swipe", "shake", "rotate", "rotate_forward"] as const;
const VIEWPOINT_LOOKAT_MODES = ["point", "ahead"] as const;
const VIEWPOINT_PHASES = [0, 1, 2, 3, 4, 5, 6, 7];

type ViewpointMotion = (typeof VIEWPOINT_MOTION_ORDER)[number];
type ViewpointLookatMode = (typeof VIEWPOINT_LOOKAT_MODES)[number];
type StrengthRangeSettings = {
  strengthMin: number;
  strengthMax: number;
  strengthLevelCount: number;
};
type ViewpointSettings = StrengthRangeSettings & {
  enabled: boolean;
  motions: ViewpointMotion[];
  lookatModes: ViewpointLookatMode[];
  phases: number[];
};
type DistortionSettings = StrengthRangeSettings;
type PhysicalSettings = StrengthRangeSettings & {
  correctPerspectiveOptions: boolean[];
};
type RegenerationSettings = StrengthRangeSettings & {
  vaeModelNames: string[];
  vaeQualities: number[];
  imageToVideoXy: number[];
};

const defaultViewpointSettings: ViewpointSettings = {
  enabled: false,
  motions: [],
  lookatModes: ["point", "ahead"],
  phases: VIEWPOINT_PHASES,
  strengthMin: 0,
  strengthMax: 1,
  strengthLevelCount: 5
};
const defaultDistortionSettings: DistortionSettings = {
  strengthMin: 0,
  strengthMax: 1,
  strengthLevelCount: 5
};
const defaultPhysicalSettings: PhysicalSettings = {
  strengthMin: 0,
  strengthMax: 1,
  strengthLevelCount: 5,
  correctPerspectiveOptions: [true]
};
const defaultRegenerationSettings: RegenerationSettings = {
  strengthMin: 0,
  strengthMax: 1,
  strengthLevelCount: 5,
  vaeModelNames: ["cheng2020-anchor"],
  vaeQualities: [3],
  imageToVideoXy: [20, 40, 60]
};

function parseViewpointMethod(method: string):
  | {
      phaseIndex: number;
      motion: ViewpointMotion;
      lookatMode: ViewpointLookatMode;
    }
  | null {
  const match = VIEWPOINT_METHOD_PATTERN.exec(method);
  if (match === null) {
    return null;
  }
  const phaseIndex = Number(match[1]);
  const motion = VIEWPOINT_MOTION_ORDER[Math.floor(phaseIndex / 2)] ?? "rotate_forward";
  return {
    phaseIndex,
    motion,
    lookatMode: match[2] as ViewpointLookatMode
  };
}

function viewpointMotionLabel(language: string, motion: ViewpointMotion) {
  const zh: Record<ViewpointMotion, string> = {
    swipe: "Swipe 横向扫动",
    shake: "Shake 抖动",
    rotate: "Rotate 环绕",
    rotate_forward: "Rotate forward 前向环绕"
  };
  const en: Record<ViewpointMotion, string> = {
    swipe: "Swipe",
    shake: "Shake",
    rotate: "Rotate",
    rotate_forward: "Rotate forward"
  };
  return (language === "zh" ? zh : en)[motion];
}

function viewpointLookatLabel(language: string, lookatMode: ViewpointLookatMode) {
  const zh: Record<ViewpointLookatMode, string> = {
    point: "Look-at point",
    ahead: "Look ahead"
  };
  const en: Record<ViewpointLookatMode, string> = {
    point: "Look-at point",
    ahead: "Look ahead"
  };
  return (language === "zh" ? zh : en)[lookatMode];
}

function clampUnit(value: number) {
  return Math.max(0, Math.min(1, value));
}

function normalizeLevelCount(value: number) {
  if (!Number.isFinite(value)) {
    return 1;
  }
  return Math.max(1, Math.min(101, Math.round(value)));
}

function formatStrength(value: number) {
  return Number(value.toFixed(4)).toString();
}

function strengthLevelsForSettings(settings: StrengthRangeSettings) {
  const lower = Math.min(settings.strengthMin, settings.strengthMax);
  const upper = Math.max(settings.strengthMin, settings.strengthMax);
  const count = normalizeLevelCount(settings.strengthLevelCount);
  if (count === 1) {
    return [Number(((lower + upper) / 2).toFixed(6))];
  }
  return Array.from({ length: count }, (_, index) =>
    Number((lower + ((upper - lower) * index) / (count - 1)).toFixed(6))
  );
}

type StrengthRangeAxisProps = {
  id: string;
  language: string;
  settings: StrengthRangeSettings;
  onChange: (settings: StrengthRangeSettings) => void;
};

function StrengthRangeAxis({ id, language, settings, onChange }: StrengthRangeAxisProps) {
  const lower = Math.min(settings.strengthMin, settings.strengthMax);
  const upper = Math.max(settings.strengthMin, settings.strengthMax);
  const levelCount = normalizeLevelCount(settings.strengthLevelCount);
  const [levelText, setLevelText] = useState(String(levelCount));
  const ticks = strengthLevelsForSettings(settings);
  const selectedLeft = `${lower * 100}%`;
  const selectedWidth = `${Math.max(0, upper - lower) * 100}%`;
  const badgeLeft = `${Math.min(94, Math.max(6, ((lower + upper) / 2) * 100))}%`;
  const rangeLabel = language === "zh" ? "强度范围" : "Strength range";
  const levelLabel = language === "zh" ? "档位数" : "Levels";
  const levelSuffix = language === "zh" ? "档" : "levels";

  useEffect(() => {
    setLevelText(String(levelCount));
  }, [levelCount]);

  return (
    <div className="strength-range-control">
      <div className="strength-range-main">
        <div className="strength-range-heading">
          <span>{rangeLabel}</span>
          <strong>
            {formatStrength(lower)} - {formatStrength(upper)}
          </strong>
        </div>
        <div className="strength-axis">
          <div className="strength-axis-track" />
          <div className="strength-axis-selection" style={{ left: selectedLeft, width: selectedWidth }} />
          <div className="strength-axis-ticks">
            {ticks.map((strength, index) => (
              <span
                className="strength-axis-tick"
                key={`${strength}-${index}`}
                style={{ left: `${strength * 100}%` }}
              />
            ))}
          </div>
          <span className="strength-axis-badge" style={{ left: badgeLeft }}>
            {levelCount} {levelSuffix}
          </span>
          <input
            aria-label={language === "zh" ? "强度范围下界" : "Strength range minimum"}
            className="strength-range-input strength-range-input-min"
            max={1}
            min={0}
            onChange={(event) => {
              const strengthMin = Math.min(clampUnit(Number(event.target.value)), settings.strengthMax);
              onChange({ ...settings, strengthMin });
            }}
            step={0.01}
            type="range"
            value={settings.strengthMin}
          />
          <input
            aria-label={language === "zh" ? "强度范围上界" : "Strength range maximum"}
            className="strength-range-input strength-range-input-max"
            max={1}
            min={0}
            onChange={(event) => {
              const strengthMax = Math.max(clampUnit(Number(event.target.value)), settings.strengthMin);
              onChange({ ...settings, strengthMax });
            }}
            step={0.01}
            type="range"
            value={settings.strengthMax}
          />
        </div>
        <div className="strength-axis-endpoints">
          <span>0</span>
          <span>1</span>
        </div>
      </div>
      <label className="strength-level-field" htmlFor={`${id}-strength-level-count`}>
        <span>{levelLabel}</span>
        <input
          id={`${id}-strength-level-count`}
          max={101}
          min={1}
          onBlur={() => {
            const normalized = normalizeLevelCount(Number(levelText));
            setLevelText(String(normalized));
            onChange({ ...settings, strengthLevelCount: normalized });
          }}
          onChange={(event) => {
            const nextValue = event.target.value;
            if (!/^\d{0,3}$/.test(nextValue)) {
              return;
            }
            setLevelText(nextValue);
            if (nextValue !== "") {
              onChange({ ...settings, strengthLevelCount: normalizeLevelCount(Number(nextValue)) });
            }
          }}
          type="number"
          value={levelText}
        />
      </label>
    </div>
  );
}

function attackSupportsStrengthOverride(attack: AttackPreset) {
  return attack.strengthParam === "strength";
}

function attackSupportsUnitRangeOverride(attack: AttackPreset) {
  return attack.strengthParam === "strength" || attack.strengthParam === "step";
}

function isHiddenIdentityAttack(attack: AttackPreset) {
  return attack.method === HIDDEN_IDENTITY_ATTACK_METHOD;
}

function viewpointAttackIdsForSettings(attacks: AttackPreset[], settings: ViewpointSettings) {
  if (
    !settings.enabled ||
    settings.motions.length === 0 ||
    settings.lookatModes.length === 0 ||
    settings.phases.length === 0
  ) {
    return [];
  }
  const motions = new Set(settings.motions);
  const lookatModes = new Set(settings.lookatModes);
  const phases = new Set(settings.phases);
  return attacks
    .filter((attack) => {
      const parsed = parseViewpointMethod(attack.method);
      return (
        parsed !== null &&
        motions.has(parsed.motion) &&
        lookatModes.has(parsed.lookatMode) &&
        phases.has(parsed.phaseIndex)
      );
    })
    .map((attack) => attack.id);
}

function applyViewpointSettings(
  current: ExperimentSelection,
  attacks: AttackPreset[],
  settings: ViewpointSettings
): ExperimentSelection {
  const allViewpointIds = attacks.map((attack) => attack.id);
  const nextIds = viewpointAttackIdsForSettings(attacks, settings);
  const attackPresetIds = addIds(removeIds(current.attackPresetIds, allViewpointIds), nextIds);
  const attackStrengthOverrides = { ...(current.attackStrengthOverrides ?? {}) };
  for (const attackId of allViewpointIds) {
    delete attackStrengthOverrides[attackId];
  }
  if (settings.enabled) {
    const strengths = strengthLevelsForSettings(settings);
    for (const attackId of nextIds) {
      attackStrengthOverrides[attackId] = strengths;
    }
  }
  return {
    ...current,
    attackPresetIds,
    attackStrengthOverrides: cleanAttackStrengthOverrides(attackStrengthOverrides, new Set(attackPresetIds)),
    attackParamOverrides: cleanAttackParamOverrides(current.attackParamOverrides, new Set(attackPresetIds))
  };
}

function applyStrengthRangeOverrides(
  current: ExperimentSelection,
  attacks: AttackPreset[],
  settings: StrengthRangeSettings,
  supportsOverride: (attack: AttackPreset) => boolean = attackSupportsStrengthOverride
): ExperimentSelection {
  const attackIds = attacks.map((attack) => attack.id);
  const selectedIds = new Set(current.attackPresetIds);
  const strengths = strengthLevelsForSettings(settings);
  const attackStrengthOverrides = { ...(current.attackStrengthOverrides ?? {}) };
  for (const attackId of attackIds) {
    delete attackStrengthOverrides[attackId];
  }
  for (const attack of attacks) {
    if (selectedIds.has(attack.id) && supportsOverride(attack)) {
      attackStrengthOverrides[attack.id] = strengths;
    }
  }
  return {
    ...current,
    attackStrengthOverrides: cleanAttackStrengthOverrides(attackStrengthOverrides, selectedIds),
    attackParamOverrides: cleanAttackParamOverrides(current.attackParamOverrides, selectedIds)
  };
}

function sortedSelectedNumbers(values: number[], allowedValues: number[]) {
  if (allowedValues.length === 0) {
    return [];
  }
  const allowed = new Set(allowedValues);
  const selected = sortStrengths(values).filter((value) => allowed.has(value));
  return selected.length > 0 ? selected : [allowedValues[0]];
}

function physicalCorrectionOptionsForSettings(settings: PhysicalSettings) {
  const selected = new Set(settings.correctPerspectiveOptions);
  const options = PHYSICAL_CORRECTION_OPTIONS.filter((option) => selected.has(option));
  return options.length > 0 ? options : [true];
}

function applyPhysicalSettings(
  current: ExperimentSelection,
  attacks: AttackPreset[],
  settings: PhysicalSettings
): ExperimentSelection {
  const attackIds = attacks.map((attack) => attack.id);
  const selectedIds = new Set(current.attackPresetIds);
  const strengthLevels = strengthLevelsForSettings(settings);
  const attackStrengthOverrides = { ...(current.attackStrengthOverrides ?? {}) };
  const attackParamOverrides = { ...(current.attackParamOverrides ?? {}) };
  for (const attackId of attackIds) {
    delete attackStrengthOverrides[attackId];
    delete attackParamOverrides[attackId];
  }

  for (const attack of attacks) {
    if (selectedIds.has(attack.id) && attackSupportsStrengthOverride(attack)) {
      const correctionOptions = physicalCorrectionOptionsForSettings(settings);
      attackParamOverrides[attack.id] = strengthLevels.flatMap((strength) =>
        correctionOptions.map((correctPerspective) => ({
          strength,
          correct_perspective: correctPerspective
        }))
      );
    }
  }

  return {
    ...current,
    attackStrengthOverrides: cleanAttackStrengthOverrides(attackStrengthOverrides, selectedIds),
    attackParamOverrides: cleanAttackParamOverrides(attackParamOverrides, selectedIds)
  };
}

function regenerationVaeModelNamesForSettings(settings: RegenerationSettings) {
  const selected = new Set(settings.vaeModelNames);
  const modelNames = REGENERATION_VAE_MODEL_NAMES.filter((modelName) => selected.has(modelName));
  return modelNames.length > 0 ? modelNames : [REGENERATION_VAE_MODEL_NAMES[0]];
}

function regenerationVaeSharedQualities(modelNames: string[]) {
  if (modelNames.length === 0) {
    return [];
  }
  const qualitySets = modelNames.map((modelName) => new Set(REGENERATION_VAE_QUALITY_BY_MODEL[modelName] ?? []));
  return sortStrengths(REGENERATION_VAE_QUALITY_BY_MODEL[modelNames[0]] ?? []).filter((quality) =>
    qualitySets.every((qualitySet) => qualitySet.has(quality))
  );
}

function isRegenerationUnitAttack(attack: AttackPreset) {
  return (REGENERATION_UNIT_METHODS as readonly string[]).includes(attack.method);
}

function applyRegenerationSettings(
  current: ExperimentSelection,
  attacks: AttackPreset[],
  settings: RegenerationSettings
): ExperimentSelection {
  const attackIds = attacks.map((attack) => attack.id);
  const selectedIds = new Set(current.attackPresetIds);
  const strengthLevels = strengthLevelsForSettings(settings);
  const attackStrengthOverrides = { ...(current.attackStrengthOverrides ?? {}) };
  const attackParamOverrides = { ...(current.attackParamOverrides ?? {}) };
  for (const attackId of attackIds) {
    delete attackStrengthOverrides[attackId];
    delete attackParamOverrides[attackId];
  }

  for (const attack of attacks) {
    if (!selectedIds.has(attack.id)) {
      continue;
    }
    if (isRegenerationUnitAttack(attack) && attackSupportsUnitRangeOverride(attack)) {
      attackStrengthOverrides[attack.id] = strengthLevels;
    } else if (attack.method === REGENERATION_IMAGE_TO_VIDEO_METHOD) {
      attackStrengthOverrides[attack.id] = sortedSelectedNumbers(
        settings.imageToVideoXy,
        REGENERATION_IMAGE_TO_VIDEO_XY
      );
    } else if (attack.method === REGENERATION_VAE_METHOD) {
      const variants: Array<Record<string, string | number>> = [];
      const modelNames = regenerationVaeModelNamesForSettings(settings);
      const qualities = sortedSelectedNumbers(settings.vaeQualities, regenerationVaeSharedQualities(modelNames));
      for (const modelName of modelNames) {
        for (const quality of qualities) {
          variants.push({ vae_model_name: modelName, quality });
        }
      }
      if (variants.length > 0) {
        attackParamOverrides[attack.id] = variants;
      }
    }
  }

  return {
    ...current,
    attackStrengthOverrides: cleanAttackStrengthOverrides(attackStrengthOverrides, selectedIds),
    attackParamOverrides: cleanAttackParamOverrides(attackParamOverrides, selectedIds)
  };
}

function sortStrengths(values: number[]) {
  return [...new Set(values.filter((value) => Number.isFinite(value)))].sort((left, right) => left - right);
}

function cleanAttackStrengthOverrides(overrides: Record<string, number[]> | undefined, validIds: Set<string>) {
  const cleaned: Record<string, number[]> = {};
  for (const [attackId, strengths] of Object.entries(overrides ?? {})) {
    if (!validIds.has(attackId) || !Array.isArray(strengths)) {
      continue;
    }
    const normalized = sortStrengths(strengths.map((strength) => Number(strength)));
    if (normalized.length > 0) {
      cleaned[attackId] = normalized;
    }
  }
  return cleaned;
}

function cleanAttackParamOverrides(
  overrides: Record<string, Array<Record<string, unknown>>> | undefined,
  validIds: Set<string>
) {
  const cleaned: Record<string, Array<Record<string, unknown>>> = {};
  for (const [attackId, variants] of Object.entries(overrides ?? {})) {
    if (!validIds.has(attackId) || !Array.isArray(variants)) {
      continue;
    }
    const normalized = variants
      .filter((variant) => variant && typeof variant === "object" && !Array.isArray(variant))
      .map((variant) => ({ ...variant }));
    if (normalized.length > 0) {
      cleaned[attackId] = normalized;
    }
  }
  return cleaned;
}

function hiddenIdentityIds(attacks: AttackPreset[]) {
  return new Set(attacks.filter(isHiddenIdentityAttack).map((attack) => attack.id));
}

function removeHiddenIdentityFromSelection(selection: ExperimentSelection, attacks: AttackPreset[]): ExperimentSelection {
  const hiddenIds = hiddenIdentityIds(attacks);
  if (hiddenIds.size === 0) {
    return selection;
  }
  const attackPresetIds = selection.attackPresetIds.filter((attackId) => !hiddenIds.has(attackId));
  return {
    ...selection,
    attackPresetIds,
    attackStrengthOverrides: cleanAttackStrengthOverrides(selection.attackStrengthOverrides, new Set(attackPresetIds)),
    attackParamOverrides: cleanAttackParamOverrides(selection.attackParamOverrides, new Set(attackPresetIds))
  };
}

function withHiddenIdentityAttack(selection: ExperimentSelection, attacks: AttackPreset[]): ExperimentSelection {
  const identityAttack = attacks.find(isHiddenIdentityAttack);
  if (identityAttack === undefined) {
    return selection;
  }
  const attackPresetIds = addIds(
    selection.attackPresetIds.filter((attackId) => attackId !== identityAttack.id),
    [identityAttack.id]
  );
  return {
    ...selection,
    attackPresetIds,
    attackStrengthOverrides: cleanAttackStrengthOverrides(selection.attackStrengthOverrides, new Set(attackPresetIds)),
    attackParamOverrides: cleanAttackParamOverrides(selection.attackParamOverrides, new Set(attackPresetIds))
  };
}

function addAttackIdsToSelection(current: ExperimentSelection, attackIds: string[]): ExperimentSelection {
  const attackPresetIds = addIds(current.attackPresetIds, attackIds);
  return {
    ...current,
    attackPresetIds,
    attackStrengthOverrides: cleanAttackStrengthOverrides(current.attackStrengthOverrides, new Set(attackPresetIds)),
    attackParamOverrides: cleanAttackParamOverrides(current.attackParamOverrides, new Set(attackPresetIds))
  };
}

function removeAttackIdsFromSelection(current: ExperimentSelection, attackIds: string[]): ExperimentSelection {
  const attackPresetIds = removeIds(current.attackPresetIds, attackIds);
  return {
    ...current,
    attackPresetIds,
    attackStrengthOverrides: cleanAttackStrengthOverrides(current.attackStrengthOverrides, new Set(attackPresetIds)),
    attackParamOverrides: cleanAttackParamOverrides(current.attackParamOverrides, new Set(attackPresetIds))
  };
}

function toggleAttackIdInSelection(current: ExperimentSelection, attackId: string): ExperimentSelection {
  return current.attackPresetIds.includes(attackId)
    ? removeAttackIdsFromSelection(current, [attackId])
    : addAttackIdsToSelection(current, [attackId]);
}

function attackCategory(attack: AttackPreset) {
  return attack.category || attack.method.split("_")[0] || "other";
}

function parseSeeds(value: string) {
  return value
    .split(",")
    .map((seed) => Number(seed.trim()))
    .filter((seed) => Number.isFinite(seed));
}

function idsFromQuery(paramName: string, validIds: Set<string>): string[] | null {
  if (typeof window === "undefined") {
    return null;
  }
  const raw = new URLSearchParams(window.location.search).get(paramName);
  if (raw == null) {
    return null;
  }
  return raw
    .split(",")
    .map((value) => value.trim())
    .filter((value) => value && validIds.has(value));
}

export default function ConfigsPage() {
  const { language, t } = useLanguage();
  const [selection, setSelection] = useState<ExperimentSelection>(emptySelection);
  const [configName, setConfigName] = useState("");
  const [seedText, setSeedText] = useState(emptySelection.seeds.join(","));
  const [savedConfigs, setSavedConfigs] = useState<SavedExperimentConfig[]>([]);
  const [datasets, setDatasets] = useState<DatasetVersion[]>([]);
  const [algorithms, setAlgorithms] = useState<AlgorithmVersion[]>([]);
  const [attacks, setAttacks] = useState<AttackPreset[]>([]);
  const [algorithmFilter, setAlgorithmFilter] = useState("");
  const [attackFilter, setAttackFilter] = useState("");
  const [viewpointSettings, setViewpointSettings] = useState<ViewpointSettings>(defaultViewpointSettings);
  const [distortionSettings, setDistortionSettings] = useState<DistortionSettings>(defaultDistortionSettings);
  const [physicalSettings, setPhysicalSettings] = useState<PhysicalSettings>(defaultPhysicalSettings);
  const [regenerationSettings, setRegenerationSettings] =
    useState<RegenerationSettings>(defaultRegenerationSettings);
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [renameTarget, setRenameTarget] = useState<SavedExperimentConfig | null>(null);
  const [renameName, setRenameName] = useState("");
  const [message, setMessage] = useState("");

  const configurableAttacks = useMemo(() => attacks.filter((attack) => !isHiddenIdentityAttack(attack)), [attacks]);
  const effectiveSelection = useMemo(() => withHiddenIdentityAttack(selection, attacks), [selection, attacks]);
  const selectedVisibleAttackCount = useMemo(() => {
    const hiddenIds = hiddenIdentityIds(attacks);
    return selection.attackPresetIds.filter((attackId) => !hiddenIds.has(attackId)).length;
  }, [attacks, selection.attackPresetIds]);
  const estimate = useMemo(
    () => estimateMatrix(effectiveSelection, datasets, attacks),
    [effectiveSelection, datasets, attacks]
  );
  const filteredAlgorithms = useMemo(
    () => algorithms.filter((algorithm) => matchesResource(algorithmFilter, algorithm)),
    [algorithmFilter, algorithms]
  );
  const filteredAttacks = useMemo(
    () => configurableAttacks.filter((attack) => matchesResource(attackFilter, attack)),
    [attackFilter, configurableAttacks]
  );
  const visibleDatasetIds = useMemo(() => datasets.map((dataset) => dataset.id), [datasets]);
  const visibleAlgorithmIds = useMemo(
    () =>
      filteredAlgorithms
        .filter((algorithm) => algorithm.status === "enabled" && algorithm.available !== false)
        .map((algorithm) => algorithm.id),
    [filteredAlgorithms]
  );
  const visibleAttackIds = useMemo(
    () => filteredAttacks.filter((attack) => attack.available !== false).map((attack) => attack.id),
    [filteredAttacks]
  );
  const attackGroups = useMemo(() => {
    const groups = new Map<string, AttackPreset[]>();
    for (const attack of filteredAttacks) {
      const category = attackCategory(attack);
      groups.set(category, [...(groups.get(category) ?? []), attack]);
    }
    return [...groups.entries()].sort(([left], [right]) => left.localeCompare(right));
  }, [filteredAttacks]);

  const renderAttackTile = (attack: AttackPreset, onToggle?: (attack: AttackPreset) => void) => (
    <label className="check-tile resource-check-tile" key={attack.id}>
      <input
        checked={selection.attackPresetIds.includes(attack.id)}
        disabled={attack.available === false}
        onChange={() =>
          onToggle
            ? onToggle(attack)
            : setSelection((current) => toggleAttackIdInSelection(current, attack.id))
        }
        type="checkbox"
      />
      <span className="tile-copy">
        <strong>{localizedName(language, attack.id, attack.name)}</strong>
        <small>
          {attack.method}
          {attack.strengths.length > 1 ? ` · ${attack.strengths.length} strengths` : ""}
        </small>
      </span>
      {attack.requiresGpu ? <span className="badge warn">{t.common.gpu}</span> : null}
      {attack.available === false ? <span className="badge error">Missing</span> : null}
    </label>
  );

  const canSave =
    configName.trim().length > 0 &&
    effectiveSelection.datasetIds.length > 0 &&
    effectiveSelection.algorithmIds.length > 0 &&
    effectiveSelection.attackPresetIds.length > 0 &&
    effectiveSelection.seeds.length > 0 &&
    effectiveSelection.maxSamples > 0;

  const copy =
    language === "zh"
      ? {
          addConfig: "新增配置",
          cancel: "取消",
          createTitle: "新增实验配置",
          datasetSettings: "数据集配置",
          draftEmpty: "当前没有正在编辑的实验配置。点击新增配置后，在弹窗中选择数据集、水印算法和攻击算法。",
          filterAlgorithms: "筛选水印算法",
          filterAttacks: "筛选攻击方法",
          selectAll: "全选",
          selectAllVisible: "全选可见",
          clear: "清空",
          deleteConfig: "删除",
          deleteConfirm: "确定删除这个实验配置？已有运行记录不会被删除。",
          renameConfig: "重命名",
          renameTitle: "重命名配置",
          save: "保存配置",
          modalHint: "保存后，该配置会进入运行页，可提交给 worker 执行。"
        }
      : {
          addConfig: "New config",
          cancel: "Cancel",
          createTitle: "New experiment config",
          datasetSettings: "Dataset settings",
          draftEmpty: "No experiment config is being edited. Create one in the dialog.",
          filterAlgorithms: "Filter watermark algorithms",
          filterAttacks: "Filter attacks",
          selectAll: "Select all",
          selectAllVisible: "Select visible",
          clear: "Clear",
          deleteConfig: "Delete",
          deleteConfirm: "Delete this experiment config? Existing runs will be kept.",
          renameConfig: "Rename",
          renameTitle: "Rename config",
          save: "Save config",
          modalHint: "After saving, launch this config from the Runs page."
        };

  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchDatasets(), fetchAlgorithms(), fetchAttacks(), fetchSavedConfigs()])
      .then(([apiDatasets, apiAlgorithms, apiAttacks, apiConfigs]) => {
        if (cancelled) {
          return;
        }
        const validDatasetIds = new Set(apiDatasets.map((dataset) => dataset.id));
        const validAlgorithmIds = new Set(apiAlgorithms.map((algorithm) => algorithm.id));
        const validAttackIds = new Set(apiAttacks.map((attack) => attack.id));
        const hiddenAttackIds = hiddenIdentityIds(apiAttacks);
        const queryDatasetIds = idsFromQuery("datasetIds", validDatasetIds);
        const queryAlgorithmIds = idsFromQuery("algorithmIds", validAlgorithmIds);
        const queryAttackIds = idsFromQuery("attackPresetIds", validAttackIds)?.filter(
          (attackId) => !hiddenAttackIds.has(attackId)
        );
        const hasQuerySelection =
          Boolean(queryDatasetIds && queryDatasetIds.length > 0) ||
          Boolean(queryAlgorithmIds && queryAlgorithmIds.length > 0) ||
          Boolean(queryAttackIds && queryAttackIds.length > 0);

        setDatasets(apiDatasets);
        setAlgorithms(apiAlgorithms);
        setAttacks(apiAttacks);
        setSavedConfigs(apiConfigs);

        if (hasQuerySelection) {
          setSelection((current) => {
            const visibleCurrent = removeHiddenIdentityFromSelection(current, apiAttacks);
            const attackSelection = addAttackIdsToSelection(
              {
                ...visibleCurrent,
                attackPresetIds: [],
                attackStrengthOverrides: {},
                attackParamOverrides: {}
              },
              queryAttackIds ??
                visibleCurrent.attackPresetIds.filter((id) => validAttackIds.has(id) && !hiddenAttackIds.has(id))
            );
            return {
              ...attackSelection,
              datasetIds: queryDatasetIds ?? visibleCurrent.datasetIds.filter((id) => validDatasetIds.has(id)),
              algorithmIds: queryAlgorithmIds ?? visibleCurrent.algorithmIds.filter((id) => validAlgorithmIds.has(id))
            };
          });
          setIsCreateOpen(true);
          setMessage(t.configs.prefilledFromResources);
          return;
        }

        if (apiDatasets.length === 0) {
          setMessage("resources/datasets 下还没有可用图片，请先解压数据集。");
        }
      })
      .catch(() => {
        if (!cancelled) {
          setMessage("API 未启动或资源接口不可访问，暂时无法创建配置。");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const openCreateModal = () => {
    setSelection(emptySelection);
    setSeedText(emptySelection.seeds.join(","));
    setConfigName("");
    setAlgorithmFilter("");
    setAttackFilter("");
    setViewpointSettings(defaultViewpointSettings);
    setDistortionSettings(defaultDistortionSettings);
    setPhysicalSettings(defaultPhysicalSettings);
    setRegenerationSettings(defaultRegenerationSettings);
    setMessage("");
    setIsCreateOpen(true);
  };

  const saveConfig = async () => {
    if (!canSave) {
      setMessage("请至少选择一个数据集、一个水印算法，并填写配置名称。");
      return;
    }
    try {
      const config = await createSavedConfig(configName.trim(), effectiveSelection);
      setSavedConfigs([config, ...savedConfigs]);
      setIsCreateOpen(false);
      setMessage(t.configs.savedToast);
    } catch {
      setMessage("API 保存失败，请先启动 FastAPI 服务后再保存。");
    }
  };

  const openRenameModal = (config: SavedExperimentConfig) => {
    setRenameTarget(config);
    setRenameName(config.name);
    setMessage("");
  };

  const renameConfig = async () => {
    if (!renameTarget) {
      return;
    }
    const nextName = renameName.trim();
    if (!nextName) {
      setMessage("配置名称不能为空。");
      return;
    }
    try {
      const updated = await renameSavedConfig(renameTarget.id, nextName);
      setSavedConfigs((configs) => configs.map((config) => (config.id === updated.id ? updated : config)));
      setRenameTarget(null);
      setRenameName("");
      setMessage(language === "zh" ? "配置已重命名。" : "Config renamed.");
    } catch {
      setMessage("重命名失败，请确认 API 服务可用。");
    }
  };

  const deleteConfig = async (config: SavedExperimentConfig) => {
    if (!window.confirm(copy.deleteConfirm)) {
      return;
    }
    try {
      await deleteSavedConfig(config.id);
      setSavedConfigs((configs) => configs.filter((item) => item.id !== config.id));
      setMessage(language === "zh" ? "配置已删除。" : "Config deleted.");
    } catch {
      setMessage("删除失败，请确认 API 服务可用。");
    }
  };

  return (
    <AppShell active="configs">
      <div className="topbar">
        <div className="title-block">
          <h1>{t.configs.title}</h1>
          <p>{t.configs.subtitle}</p>
        </div>
        <button className="button primary" onClick={openCreateModal} type="button">
          <Plus size={16} />
          {copy.addConfig}
        </button>
      </div>

      <section className="configs-home-grid">
        <div className="panel">
          <div className="panel-header">
            <h2>{t.configs.savedConfigs}</h2>
            <Archive size={16} />
          </div>
          <div className="panel-body resource-list">
            {savedConfigs.length === 0 ? (
              <div className="config-empty-state">
                <Braces size={34} />
                <h2>{t.configs.empty}</h2>
                <p>{copy.draftEmpty}</p>
                <button className="button primary" onClick={openCreateModal} type="button">
                  <Plus size={16} />
                  {copy.addConfig}
                </button>
              </div>
            ) : null}
            {savedConfigs.map((config) => (
              <div className="resource-item config-list-item" key={config.id}>
                <div>
                  <strong>{config.name}</strong>
                  <span>
                    {config.cellCount} {t.console.cells} · {config.sampleCount.toLocaleString()}{" "}
                    {t.common.samples}
                  </span>
                </div>
                <div className="config-actions">
                  <button
                    aria-label={`${copy.renameConfig}: ${config.name}`}
                    className="icon-button"
                    onClick={() => openRenameModal(config)}
                    title={copy.renameConfig}
                    type="button"
                  >
                    <Edit3 size={15} />
                  </button>
                  <button
                    aria-label={`${copy.deleteConfig}: ${config.name}`}
                    className="icon-button danger"
                    onClick={() => deleteConfig(config)}
                    title={copy.deleteConfig}
                    type="button"
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {message ? <div className="risk ok config-page-message">{message}</div> : null}

      {isCreateOpen ? (
        <div aria-modal="true" className="modal-backdrop" role="dialog">
          <div className="config-modal">
            <div className="modal-header">
              <div>
                <h2>{copy.createTitle}</h2>
                <p>{copy.modalHint}</p>
              </div>
              <button aria-label="Close" className="icon-button" onClick={() => setIsCreateOpen(false)} type="button">
                <X size={16} />
              </button>
            </div>

            <div className="modal-body config-modal-body">
              <div className="field config-name-field">
                <label htmlFor="config-name">{t.configs.nameLabel}</label>
                <input
                  id="config-name"
                  onChange={(event) => setConfigName(event.target.value)}
                  placeholder={t.configs.namePlaceholder}
                  value={configName}
                />
              </div>

              <section className="modal-section">
                <div className="modal-section-heading">
                  <Database size={16} />
                  <h3>{copy.datasetSettings}</h3>
                  <span className="count-pill">
                    {selection.datasetIds.length}/{datasets.length}
                  </span>
                  <div className="bulk-actions">
                    <button
                      className="button compact"
                      disabled={visibleDatasetIds.length === 0}
                      onClick={() =>
                        setSelection((current) => ({
                          ...current,
                          datasetIds: addIds(current.datasetIds, visibleDatasetIds)
                        }))
                      }
                      type="button"
                    >
                      <Check size={14} />
                      {copy.selectAll}
                    </button>
                    <button
                      className="button compact"
                      disabled={selection.datasetIds.length === 0}
                      onClick={() =>
                        setSelection((current) => ({
                          ...current,
                          datasetIds: removeIds(current.datasetIds, visibleDatasetIds)
                        }))
                      }
                      type="button"
                    >
                      <X size={14} />
                      {copy.clear}
                    </button>
                  </div>
                </div>
                <div className="field-grid">
                  <div className="field">
                    <label htmlFor="seeds">{t.console.seeds}</label>
                    <input
                      id="seeds"
                      onChange={(event) => {
                        setSeedText(event.target.value);
                        setSelection((current) => ({ ...current, seeds: parseSeeds(event.target.value) }));
                      }}
                      placeholder="42, 123, 2026"
                      value={seedText}
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="max-samples">{t.console.maxSamples}</label>
                    <input
                      id="max-samples"
                      min={1}
                      onChange={(event) =>
                        setSelection((current) => ({
                          ...current,
                          maxSamples: Number(event.target.value)
                        }))
                      }
                      type="number"
                      value={selection.maxSamples}
                    />
                  </div>
                </div>
                <div className="option-grid dense-options">
                  {datasets.map((dataset) => (
                    <label className="check-tile resource-check-tile" key={dataset.id}>
                      <input
                        checked={selection.datasetIds.includes(dataset.id)}
                        onChange={() =>
                          setSelection((current) => ({
                            ...current,
                            datasetIds: toggle(current.datasetIds, dataset.id)
                          }))
                        }
                        type="checkbox"
                      />
                      <span className="tile-copy">
                        <strong>{localizedName(language, dataset.id, dataset.name)}</strong>
                        <small>
                          {dataset.sampleCount.toLocaleString()} {t.common.samples}
                        </small>
                      </span>
                    </label>
                  ))}
                </div>
                {datasets.length === 0 ? <div className="empty compact-empty">{t.common.noData}</div> : null}
              </section>

              <section className="modal-section">
                <div className="modal-section-heading">
                  <Shield size={16} />
                  <h3>{t.console.algorithms}</h3>
                  <span className="count-pill">
                    {selection.algorithmIds.length}/{algorithms.length}
                  </span>
                </div>
                <div className="selector-tools">
                  <div className="field-icon-input">
                    <Search size={15} />
                    <input
                      aria-label="Filter watermark algorithms"
                      onChange={(event) => setAlgorithmFilter(event.target.value)}
                      placeholder={copy.filterAlgorithms}
                      value={algorithmFilter}
                    />
                  </div>
                  <div className="bulk-actions">
                    <button
                      className="button compact"
                      disabled={visibleAlgorithmIds.length === 0}
                      onClick={() =>
                        setSelection((current) => ({
                          ...current,
                          algorithmIds: addIds(current.algorithmIds, visibleAlgorithmIds)
                        }))
                      }
                      type="button"
                    >
                      <Check size={14} />
                      {copy.selectAllVisible}
                    </button>
                    <button
                      className="button compact"
                      disabled={visibleAlgorithmIds.length === 0}
                      onClick={() =>
                        setSelection((current) => ({
                          ...current,
                          algorithmIds: removeIds(current.algorithmIds, visibleAlgorithmIds)
                        }))
                      }
                      type="button"
                    >
                      <X size={14} />
                      {copy.clear}
                    </button>
                  </div>
                </div>
                <div className="option-grid dense-options">
                  {filteredAlgorithms.map((algorithm) => (
                    <label className="check-tile resource-check-tile" key={algorithm.id}>
                      <input
                        checked={selection.algorithmIds.includes(algorithm.id)}
                        disabled={algorithm.status !== "enabled" || algorithm.available === false}
                        onChange={() =>
                          setSelection((current) => ({
                            ...current,
                            algorithmIds: toggle(current.algorithmIds, algorithm.id)
                          }))
                        }
                        type="checkbox"
                      />
                      <span className="tile-copy">
                        <strong>{algorithm.name}</strong>
                        <small>
                          {algorithm.method ?? algorithm.id}
                          {algorithm.category ? ` · ${algorithm.category}` : ""}
                        </small>
                      </span>
                      {algorithm.requiresGpu ? <span className="badge warn">{t.common.gpu}</span> : null}
                      {algorithm.available === false ? <span className="badge error">Missing</span> : null}
                    </label>
                  ))}
                </div>
                {filteredAlgorithms.length === 0 ? <div className="empty compact-empty">{t.common.noData}</div> : null}
              </section>

              <section className="modal-section">
                <div className="modal-section-heading">
                  <Gauge size={16} />
                  <h3>{t.console.attacks}</h3>
                  <span className="count-pill">
                    {selectedVisibleAttackCount}/{configurableAttacks.length}
                  </span>
                </div>
                <div className="selector-tools">
                  <div className="field-icon-input">
                    <Search size={15} />
                    <input
                      aria-label="Filter attacks"
                      onChange={(event) => setAttackFilter(event.target.value)}
                      placeholder={copy.filterAttacks}
                      value={attackFilter}
                    />
                  </div>
                  <div className="bulk-actions">
                    <button
                      className="button compact"
                      disabled={visibleAttackIds.length === 0}
                      onClick={() =>
                        setSelection((current) => {
                          let nextSelection = addAttackIdsToSelection(current, visibleAttackIds);
                          const visibleDistortionAttacks = filteredAttacks.filter(
                            (attack) => attackCategory(attack) === DISTORTION_CATEGORY
                          );
                          if (visibleDistortionAttacks.length > 0) {
                            nextSelection = applyStrengthRangeOverrides(
                              nextSelection,
                              visibleDistortionAttacks,
                              distortionSettings
                            );
                          }
                          const visiblePhysicalAttacks = filteredAttacks.filter(
                            (attack) => attackCategory(attack) === PHYSICAL_CATEGORY
                          );
                          if (visiblePhysicalAttacks.length > 0) {
                            nextSelection = applyPhysicalSettings(nextSelection, visiblePhysicalAttacks, physicalSettings);
                          }
                          const visibleRegenerationAttacks = filteredAttacks.filter(
                            (attack) => attackCategory(attack) === REGENERATION_CATEGORY
                          );
                          if (visibleRegenerationAttacks.length > 0) {
                            nextSelection = applyRegenerationSettings(
                              nextSelection,
                              visibleRegenerationAttacks,
                              regenerationSettings
                            );
                          }
                          return nextSelection;
                        })
                      }
                      type="button"
                    >
                      <Check size={14} />
                      {copy.selectAllVisible}
                    </button>
                    <button
                      className="button compact"
                      disabled={visibleAttackIds.length === 0}
                      onClick={() =>
                        setSelection((current) => removeAttackIdsFromSelection(current, visibleAttackIds))
                      }
                      type="button"
                    >
                      <X size={14} />
                      {copy.clear}
                    </button>
                  </div>
                </div>
                <div className="attack-group-list">
                  {attackGroups.map(([category, categoryAttacks]) => {
                    const isViewpointCategory = category === VIEWPOINT_CATEGORY;
                    const isDistortionCategory = category === DISTORTION_CATEGORY;
                    const isRegenerationCategory = category === REGENERATION_CATEGORY;
                    const isPhysicalCategory = category === PHYSICAL_CATEGORY;
                    const regenerationUnitAttacks = REGENERATION_UNIT_METHODS.map((method) =>
                      categoryAttacks.find((attack) => attack.method === method)
                    ).filter((attack): attack is AttackPreset => attack !== undefined);
                    const regenerationVaeAttack = categoryAttacks.find(
                      (attack) => attack.method === REGENERATION_VAE_METHOD
                    );
                    const regenerationImageToVideoAttack = categoryAttacks.find(
                      (attack) => attack.method === REGENERATION_IMAGE_TO_VIDEO_METHOD
                    );
                    const remainingRegenerationAttacks = categoryAttacks.filter(
                      (attack) =>
                        !isRegenerationUnitAttack(attack) &&
                        attack.method !== REGENERATION_VAE_METHOD &&
                        attack.method !== REGENERATION_IMAGE_TO_VIDEO_METHOD
                    );
                    const selectedViewpointIds = categoryAttacks
                      .map((attack) => attack.id)
                      .filter((attackId) => selection.attackPresetIds.includes(attackId));
                    const renderedViewpointSettings =
                      isViewpointCategory && selectedViewpointIds.length > 0 && !viewpointSettings.enabled
                        ? { ...viewpointSettings, enabled: true, motions: [...VIEWPOINT_MOTION_ORDER] }
                        : viewpointSettings;
                    const updateViewpointSettings = (nextSettings: ViewpointSettings) => {
                      setViewpointSettings(nextSettings);
                      setSelection((current) => applyViewpointSettings(current, categoryAttacks, nextSettings));
                    };
                    const updateDistortionSettings = (nextSettings: DistortionSettings) => {
                      setDistortionSettings(nextSettings);
                      setSelection((current) => applyStrengthRangeOverrides(current, categoryAttacks, nextSettings));
                    };
                    const updatePhysicalSettings = (nextSettings: PhysicalSettings) => {
                      setPhysicalSettings(nextSettings);
                      setSelection((current) => applyPhysicalSettings(current, categoryAttacks, nextSettings));
                    };
                    const updateRegenerationSettings = (nextSettings: RegenerationSettings) => {
                      setRegenerationSettings(nextSettings);
                      setSelection((current) => applyRegenerationSettings(current, categoryAttacks, nextSettings));
                    };
                    const toggleDistortionAttack = (attack: AttackPreset) => {
                      setSelection((current) =>
                        applyStrengthRangeOverrides(
                          toggleAttackIdInSelection(current, attack.id),
                          categoryAttacks,
                          distortionSettings
                        )
                      );
                    };
                    const togglePhysicalAttack = (attack: AttackPreset) => {
                      setSelection((current) =>
                        applyPhysicalSettings(toggleAttackIdInSelection(current, attack.id), categoryAttacks, physicalSettings)
                      );
                    };
                    const selectedPhysicalCorrectionOptions = physicalCorrectionOptionsForSettings(physicalSettings);
                    const togglePhysicalCorrectionOption = (correctPerspective: boolean) => {
                      const selected = new Set(selectedPhysicalCorrectionOptions);
                      if (selected.has(correctPerspective)) {
                        if (selected.size === 1) {
                          return;
                        }
                        selected.delete(correctPerspective);
                      } else {
                        selected.add(correctPerspective);
                      }
                      updatePhysicalSettings({
                        ...physicalSettings,
                        correctPerspectiveOptions: PHYSICAL_CORRECTION_OPTIONS.filter((option) => selected.has(option))
                      });
                    };
                    const toggleRegenerationAttack = (attack: AttackPreset) => {
                      setSelection((current) =>
                        applyRegenerationSettings(
                          toggleAttackIdInSelection(current, attack.id),
                          categoryAttacks,
                          regenerationSettings
                        )
                      );
                    };
                    const selectedVaeModelNames = regenerationVaeModelNamesForSettings(regenerationSettings);
                    const toggleRegenerationVaeModel = (vaeModelName: string) => {
                      const selected = new Set(selectedVaeModelNames);
                      if (selected.has(vaeModelName)) {
                        if (selected.size === 1) {
                          return;
                        }
                        selected.delete(vaeModelName);
                      } else {
                        selected.add(vaeModelName);
                      }
                      const vaeModelNames = REGENERATION_VAE_MODEL_NAMES.filter((modelName) =>
                        selected.has(modelName)
                      );
                      const validQualities = regenerationVaeSharedQualities(vaeModelNames);
                      const vaeQualities = sortStrengths(regenerationSettings.vaeQualities).filter((quality) =>
                        validQualities.includes(quality)
                      );
                      updateRegenerationSettings({
                        ...regenerationSettings,
                        vaeModelNames,
                        vaeQualities: vaeQualities.length > 0 ? vaeQualities : [validQualities[0] ?? 1]
                      });
                    };
                    const toggleRegenerationVaeQuality = (quality: number) => {
                      const selected = regenerationSettings.vaeQualities.includes(quality)
                        ? regenerationSettings.vaeQualities.filter((value) => value !== quality)
                        : sortStrengths([...regenerationSettings.vaeQualities, quality]);
                      if (selected.length === 0) {
                        return;
                      }
                      updateRegenerationSettings({ ...regenerationSettings, vaeQualities: selected });
                    };
                    const toggleRegenerationXy = (xy: number) => {
                      const selected = regenerationSettings.imageToVideoXy.includes(xy)
                        ? regenerationSettings.imageToVideoXy.filter((value) => value !== xy)
                        : sortStrengths([...regenerationSettings.imageToVideoXy, xy]);
                      if (selected.length === 0) {
                        return;
                      }
                      updateRegenerationSettings({ ...regenerationSettings, imageToVideoXy: selected });
                    };
                    const toggleViewpointMotion = (motion: ViewpointMotion) => {
                      const motions = renderedViewpointSettings.motions.includes(motion)
                        ? renderedViewpointSettings.motions.filter((value) => value !== motion)
                        : VIEWPOINT_MOTION_ORDER.filter((value) =>
                            [...renderedViewpointSettings.motions, motion].includes(value)
                          );
                      updateViewpointSettings({ ...renderedViewpointSettings, enabled: motions.length > 0, motions });
                    };
                    const toggleViewpointLookat = (lookatMode: ViewpointLookatMode) => {
                      const lookatModes = renderedViewpointSettings.lookatModes.includes(lookatMode)
                        ? renderedViewpointSettings.lookatModes.filter((value) => value !== lookatMode)
                        : [...renderedViewpointSettings.lookatModes, lookatMode];
                      updateViewpointSettings({ ...renderedViewpointSettings, lookatModes });
                    };
                    const toggleViewpointPhase = (phase: number) => {
                      const phases = renderedViewpointSettings.phases.includes(phase)
                        ? renderedViewpointSettings.phases.filter((value) => value !== phase)
                        : sortStrengths([...renderedViewpointSettings.phases, phase]);
                      updateViewpointSettings({ ...renderedViewpointSettings, phases });
                    };

                    return (
                      <div className="attack-group" key={category}>
                        <div className="attack-group-title">
                          <strong>{categoryLabel(language, category)}</strong>
                          <span>{categoryAttacks.length}</span>
                          <div className="bulk-actions group-bulk-actions">
                            <button
                              className="button compact"
                              onClick={() => {
                                if (isViewpointCategory) {
                                  const nextSettings = {
                                    ...renderedViewpointSettings,
                                    enabled: true,
                                    motions: [...VIEWPOINT_MOTION_ORDER],
                                    lookatModes: [...VIEWPOINT_LOOKAT_MODES],
                                    phases: [...VIEWPOINT_PHASES]
                                  };
                                  updateViewpointSettings(nextSettings);
                                  return;
                                }
                                if (isDistortionCategory) {
                                  setSelection((current) =>
                                    applyStrengthRangeOverrides(
                                      addAttackIdsToSelection(
                                        current,
                                        categoryAttacks
                                          .filter((attack) => attack.available !== false)
                                          .map((attack) => attack.id)
                                      ),
                                      categoryAttacks,
                                      distortionSettings
                                    )
                                  );
                                  return;
                                }
                                if (isPhysicalCategory) {
                                  setSelection((current) =>
                                    applyPhysicalSettings(
                                      addAttackIdsToSelection(
                                        current,
                                        categoryAttacks
                                          .filter((attack) => attack.available !== false)
                                          .map((attack) => attack.id)
                                      ),
                                      categoryAttacks,
                                      physicalSettings
                                    )
                                  );
                                  return;
                                }
                                if (isRegenerationCategory) {
                                  setSelection((current) =>
                                    applyRegenerationSettings(
                                      addAttackIdsToSelection(
                                        current,
                                        categoryAttacks
                                          .filter((attack) => attack.available !== false)
                                          .map((attack) => attack.id)
                                      ),
                                      categoryAttacks,
                                      regenerationSettings
                                    )
                                  );
                                  return;
                                }
                                setSelection((current) =>
                                  addAttackIdsToSelection(
                                    current,
                                    categoryAttacks
                                      .filter((attack) => attack.available !== false)
                                      .map((attack) => attack.id)
                                  )
                                );
                              }}
                              type="button"
                            >
                              <Check size={14} />
                              {copy.selectAll}
                            </button>
                            <button
                              className="button compact"
                              onClick={() => {
                                if (isViewpointCategory) {
                                  updateViewpointSettings({
                                    ...renderedViewpointSettings,
                                    enabled: false,
                                    motions: []
                                  });
                                  return;
                                }
                                if (isDistortionCategory) {
                                  setSelection((current) =>
                                    applyStrengthRangeOverrides(
                                      removeAttackIdsFromSelection(
                                        current,
                                        categoryAttacks.map((attack) => attack.id)
                                      ),
                                      categoryAttacks,
                                      distortionSettings
                                    )
                                  );
                                  return;
                                }
                                if (isPhysicalCategory) {
                                  setSelection((current) =>
                                    applyPhysicalSettings(
                                      removeAttackIdsFromSelection(
                                        current,
                                        categoryAttacks.map((attack) => attack.id)
                                      ),
                                      categoryAttacks,
                                      physicalSettings
                                    )
                                  );
                                  return;
                                }
                                if (isRegenerationCategory) {
                                  setSelection((current) =>
                                    applyRegenerationSettings(
                                      removeAttackIdsFromSelection(
                                        current,
                                        categoryAttacks.map((attack) => attack.id)
                                      ),
                                      categoryAttacks,
                                      regenerationSettings
                                    )
                                  );
                                  return;
                                }
                                setSelection((current) =>
                                  removeAttackIdsFromSelection(
                                    current,
                                    categoryAttacks.map((attack) => attack.id)
                                  )
                                );
                              }}
                              type="button"
                            >
                              <X size={14} />
                              {copy.clear}
                            </button>
                          </div>
                        </div>
                        {isViewpointCategory ? (
                          <>
                            <div className="option-grid dense-options viewpoint-variant-grid">
                              {VIEWPOINT_MOTION_ORDER.map((motion) => {
                                const motionAttacks = categoryAttacks.filter((attack) => {
                                  const parsed = parseViewpointMethod(attack.method);
                                  return parsed?.motion === motion;
                                });
                                const motionIds = motionAttacks
                                  .filter((attack) => attack.available !== false)
                                  .map((attack) => attack.id);
                                const checked = renderedViewpointSettings.motions.includes(motion);

                                return (
                                  <label className="check-tile resource-check-tile" key={motion}>
                                    <input
                                      checked={checked}
                                      disabled={motionIds.length === 0}
                                      onChange={() => toggleViewpointMotion(motion)}
                                      type="checkbox"
                                    />
                                    <span className="tile-copy">
                                      <strong>{viewpointMotionLabel(language, motion)}</strong>
                                      <small>
                                        {motionAttacks.length} {language === "zh" ? "变体" : "variants"}
                                      </small>
                                    </span>
                                    {motionAttacks.some((attack) => attack.requiresGpu) ? (
                                      <span className="badge warn">{t.common.gpu}</span>
                                    ) : null}
                                    {motionIds.length === 0 ? <span className="badge error">Missing</span> : null}
                                  </label>
                                );
                              })}
                            </div>

                            <div className="viewpoint-settings-panel">
                              <div className="viewpoint-settings-row">
                                <span className="viewpoint-settings-label">Lookat mode</span>
                                <div className="viewpoint-toggle-grid two-options">
                                  {VIEWPOINT_LOOKAT_MODES.map((lookatMode) => (
                                    <label className="viewpoint-toggle" key={lookatMode}>
                                      <input
                                        checked={renderedViewpointSettings.lookatModes.includes(lookatMode)}
                                        onChange={() => toggleViewpointLookat(lookatMode)}
                                        type="checkbox"
                                      />
                                      <span>{viewpointLookatLabel(language, lookatMode)}</span>
                                    </label>
                                  ))}
                                </div>
                              </div>

                              <div className="viewpoint-settings-row">
                                <span className="viewpoint-settings-label">Phase</span>
                                <div className="viewpoint-toggle-grid phase-options">
                                  {VIEWPOINT_PHASES.map((phase) => (
                                    <label className="viewpoint-toggle" key={phase}>
                                      <input
                                        checked={renderedViewpointSettings.phases.includes(phase)}
                                        onChange={() => toggleViewpointPhase(phase)}
                                        type="checkbox"
                                      />
                                      <span>{phase}</span>
                                    </label>
                                  ))}
                                </div>
                              </div>

                              <StrengthRangeAxis
                                id="viewpoint"
                                language={language}
                                onChange={(nextSettings) =>
                                  updateViewpointSettings({ ...renderedViewpointSettings, ...nextSettings })
                                }
                                settings={renderedViewpointSettings}
                              />
                            </div>
                          </>
                        ) : isDistortionCategory ? (
                          <>
                            <div className="option-grid dense-options">
                              {categoryAttacks.map((attack) => renderAttackTile(attack, toggleDistortionAttack))}
                            </div>

                            <div className="viewpoint-settings-panel">
                              <StrengthRangeAxis
                                id="distortion"
                                language={language}
                                onChange={(nextSettings) =>
                                  updateDistortionSettings({ ...distortionSettings, ...nextSettings })
                                }
                                settings={distortionSettings}
                              />
                            </div>
                          </>
                        ) : isPhysicalCategory ? (
                          <>
                            <div className="option-grid dense-options">
                              {categoryAttacks.map((attack) => renderAttackTile(attack, togglePhysicalAttack))}
                            </div>

                            <div className="viewpoint-settings-panel">
                              <StrengthRangeAxis
                                id="physical"
                                language={language}
                                onChange={(nextSettings) =>
                                  updatePhysicalSettings({ ...physicalSettings, ...nextSettings })
                                }
                                settings={physicalSettings}
                              />
                              <div className="viewpoint-settings-row">
                                <span className="viewpoint-settings-label">
                                  {language === "zh" ? "透视矫正" : "Correction"}
                                </span>
                                <div className="viewpoint-toggle-grid two-options">
                                  {PHYSICAL_CORRECTION_OPTIONS.map((correctPerspective) => (
                                    <label className="viewpoint-toggle" key={String(correctPerspective)}>
                                      <input
                                        checked={selectedPhysicalCorrectionOptions.includes(correctPerspective)}
                                        onChange={() => togglePhysicalCorrectionOption(correctPerspective)}
                                        type="checkbox"
                                      />
                                      <span>
                                        {correctPerspective
                                          ? language === "zh"
                                            ? "开启矫正"
                                            : "Correct perspective"
                                          : language === "zh"
                                            ? "不开启矫正"
                                            : "No correction"}
                                      </span>
                                    </label>
                                  ))}
                                </div>
                              </div>
                            </div>
                          </>
                        ) : isRegenerationCategory ? (
                          <>
                            <div className="attack-subgroup">
                              <div className="attack-subgroup-title">
                                <strong>{language === "zh" ? "0-1 强度组" : "0-1 strength group"}</strong>
                                <span>{regenerationUnitAttacks.length}</span>
                              </div>
                              <div className="option-grid dense-options viewpoint-variant-grid">
                                {regenerationUnitAttacks.map((attack) =>
                                  renderAttackTile(attack, toggleRegenerationAttack)
                                )}
                              </div>

                              <div className="viewpoint-settings-panel">
                                <StrengthRangeAxis
                                  id="regeneration"
                                  language={language}
                                  onChange={(nextSettings) =>
                                    updateRegenerationSettings({ ...regenerationSettings, ...nextSettings })
                                  }
                                  settings={regenerationSettings}
                                />
                              </div>
                            </div>

                            {regenerationVaeAttack ? (
                              <div className="attack-subgroup">
                                <div className="attack-subgroup-title">
                                  <strong>regen_vae</strong>
                                  <span>{selection.attackPresetIds.includes(regenerationVaeAttack.id) ? 1 : 0}</span>
                                </div>
                                <div className="option-grid dense-options">
                                  {renderAttackTile(regenerationVaeAttack, toggleRegenerationAttack)}
                                </div>
                                <div className="viewpoint-settings-panel">
                                  <div className="viewpoint-settings-row">
                                    <span className="viewpoint-settings-label">
                                      {language === "zh" ? "权重类型" : "Weight type"}
                                    </span>
                                    <div className="viewpoint-toggle-grid vae-model-options">
                                      {REGENERATION_VAE_MODEL_NAMES.map((modelName) => (
                                        <label className="viewpoint-toggle" key={modelName}>
                                          <input
                                            checked={selectedVaeModelNames.includes(modelName)}
                                            onChange={() => toggleRegenerationVaeModel(modelName)}
                                            type="checkbox"
                                          />
                                          <span>{modelName}</span>
                                        </label>
                                      ))}
                                    </div>
                                  </div>

                                  <div className="viewpoint-settings-row">
                                    <span className="viewpoint-settings-label">Quality</span>
                                    <div className="viewpoint-toggle-grid phase-options">
                                      {REGENERATION_VAE_QUALITIES.map((quality) => {
                                        const enabled = selectedVaeModelNames.every((modelName) =>
                                          (REGENERATION_VAE_QUALITY_BY_MODEL[modelName] ?? []).includes(quality)
                                        );
                                        return (
                                          <label className="viewpoint-toggle" key={quality}>
                                            <input
                                              checked={regenerationSettings.vaeQualities.includes(quality)}
                                              disabled={!enabled}
                                              onChange={() => toggleRegenerationVaeQuality(quality)}
                                              type="checkbox"
                                            />
                                            <span>{quality}</span>
                                          </label>
                                        );
                                      })}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            ) : null}

                            {regenerationImageToVideoAttack ? (
                              <div className="attack-subgroup">
                                <div className="attack-subgroup-title">
                                  <strong>image_to_vedio</strong>
                                  <span>
                                    {selection.attackPresetIds.includes(regenerationImageToVideoAttack.id) ? 1 : 0}
                                  </span>
                                </div>
                                <div className="option-grid dense-options">
                                  {renderAttackTile(regenerationImageToVideoAttack, toggleRegenerationAttack)}
                                </div>
                                <div className="viewpoint-settings-panel">
                                  <div className="viewpoint-settings-row">
                                    <span className="viewpoint-settings-label">XY</span>
                                    <div className="viewpoint-toggle-grid xy-options">
                                      {REGENERATION_IMAGE_TO_VIDEO_XY.map((xy) => (
                                        <label className="viewpoint-toggle" key={xy}>
                                          <input
                                            checked={regenerationSettings.imageToVideoXy.includes(xy)}
                                            onChange={() => toggleRegenerationXy(xy)}
                                            type="checkbox"
                                          />
                                          <span>{xy}</span>
                                        </label>
                                      ))}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            ) : null}

                            {remainingRegenerationAttacks.length > 0 ? (
                              <div className="option-grid dense-options">
                                {remainingRegenerationAttacks.map((attack) =>
                                  renderAttackTile(attack, toggleRegenerationAttack)
                                )}
                              </div>
                            ) : null}
                          </>
                        ) : (
                          <div className="option-grid dense-options">
                            {categoryAttacks.map((attack) => renderAttackTile(attack))}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
                {filteredAttacks.length === 0 ? <div className="empty compact-empty">{t.common.noData}</div> : null}
              </section>
            </div>

            <div className="modal-footer">
              <div className="config-estimate">
                <span>
                  {estimate.cellCount} {t.console.cells}
                </span>
                <span>
                  {estimate.sampleCount.toLocaleString()} {t.common.samples}
                </span>
                <span>
                  {estimate.imageOperationCount.toLocaleString()} {t.console.ops}
                </span>
              </div>
              <div className="toolbar">
                <button className="button" onClick={() => setIsCreateOpen(false)} type="button">
                  {copy.cancel}
                </button>
                <button className="button primary" disabled={!canSave} onClick={saveConfig} type="button">
                  <Save size={16} />
                  {copy.save}
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {renameTarget ? (
        <div aria-modal="true" className="modal-backdrop compact-modal-backdrop" role="dialog">
          <div className="rename-modal">
            <div className="modal-header">
              <div>
                <h2>{copy.renameTitle}</h2>
                <p>{renameTarget.id}</p>
              </div>
              <button aria-label="Close" className="icon-button" onClick={() => setRenameTarget(null)} type="button">
                <X size={16} />
              </button>
            </div>
            <div className="modal-body">
              <div className="field">
                <label htmlFor="rename-config-name">{t.configs.nameLabel}</label>
                <input
                  autoFocus
                  id="rename-config-name"
                  onChange={(event) => setRenameName(event.target.value)}
                  value={renameName}
                />
              </div>
            </div>
            <div className="modal-footer">
              <span />
              <div className="toolbar">
                <button className="button" onClick={() => setRenameTarget(null)} type="button">
                  {copy.cancel}
                </button>
                <button className="button primary" disabled={!renameName.trim()} onClick={renameConfig} type="button">
                  <Save size={16} />
                  {copy.save}
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </AppShell>
  );
}
