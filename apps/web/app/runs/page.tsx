"use client";

import { useEffect, useMemo, useState } from "react";
import {
  BarChart3,
  Check,
  CheckCircle2,
  FolderOpen,
  LineChart,
  PauseCircle,
  PlayCircle,
  RefreshCw,
  RotateCcw,
  Save,
  Search,
  SlidersHorizontal,
  Square,
  X,
  XCircle,
  Zap
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { useLanguage } from "@/components/LanguageProvider";
import {
  cancelParallelTuning,
  cancelRun,
  createRun,
  fetchAlgorithms,
  fetchAttacks,
  fetchDatasetCatalog,
  fetchGpuTelemetry,
  fetchLatestParallelTuning,
  fetchParallelTuning,
  fetchRun,
  fetchManageableRuns,
  fetchRunState,
  fetchSavedConfigs,
  pauseParallelTuning,
  pauseRun,
  resumeParallelTuning,
  resumeRun,
  saveParallelTuning,
  startParallelTuning
} from "@/lib/api";
import { localizedDate, localizedName } from "@/lib/i18n";
import type { Language, Translation } from "@/lib/i18n";
import { resolveWatermarkDisplayName } from "@/lib/watermark-display";
import type {
  AlgorithmVersion,
  AttackPreset,
  DemoRunRecord,
  GpuTelemetry,
  GpuTelemetryDevice,
  GpuTelemetrySample,
  ParallelTuningEvent,
  ParallelTuningJob,
  RunPhaseState,
  RunShardProgress,
  RunState,
  SavedExperimentConfig
} from "@/lib/types";

type StartMode = "new" | "resume";
type TuningMode = "quick" | "full";

type ProgressStep = {
  key: string;
  label: string;
  current: number;
  total: number;
  percent: number;
  meta: string;
};

type ExperimentStageKey = "canonical" | "watermark" | "attack" | "extract" | "quality" | "summary";

type ExperimentStageTab = {
  key: ExperimentStageKey;
  label: string;
  progressKey: string;
  step: ProgressStep;
  phase?: RunPhaseState;
  reached: boolean;
  completed: boolean;
  active: boolean;
};

type MaterializedStats = {
  root: string;
  cacheHits: number;
  latestDir: string;
};

type StageCellProgress = {
  current: number;
  total: number;
  percent: number;
  unit: string;
};

type ExecutionSummary = {
  taskName: string;
  runId: string;
  status: DemoRunRecord["status"];
  progress: number;
  resultUnits: number;
  succeededResultUnits: number;
  failedResultUnits: number;
  remainingResultUnits: number;
  configName: string;
  workerId?: string | null;
  artifactRoot?: string;
  createdAt?: string;
  startedAt?: string | null;
  finishedAt?: string | null;
  updatedAt?: string;
  durationMs: number | null;
  selection?: {
    datasets: number;
    watermarks: number;
    attacks: number;
    seeds: number;
    maxSamples: number;
    sampleCount: number;
    imageOperationCount: number;
  };
  materialized?: MaterializedStats | null;
  note: string;
};

type TuningForm = {
  mode: TuningMode;
  sampleCount: number;
  warmupCount: number;
  candidateBatchCount: number;
  probeSampleCount: number;
  finalistCount: number;
  minBatchSize: number;
  maxBatchSize: number;
  minWorkerCount: number;
  maxWorkerCount: number;
  repeatCount: number;
  minImprovementRatio: number;
  tuneWatermarks: boolean;
  tuneAttacks: boolean;
  includeViewpoint3dAttacks: boolean;
  tuneQuality: boolean;
  selectedQualityMetrics: string[];
  selectedWatermarkMethods: string[];
  selectedAttackMethods: string[];
};

type TuningPoint = {
  key: string;
  eventIndex: number;
  eventTime: number;
  stage: string;
  stageLabel: string;
  scopeLabel: string;
  method: string;
  label: string;
  candidate: number;
  throughput: number;
  kind: "batch" | "workers";
  ok: boolean;
  groupKey: string;
  groupLabel: string;
};

type TuningProcessGroup = {
  key: string;
  label: string;
  description: string;
  pointCount: number;
  successCount: number;
  failedCount: number;
  bestThroughput: number;
  latestThroughput: number;
  latestEventIndex: number;
  latestEventTime: number;
};

type TuningCatalogStats = {
  watermarkMethods: number;
  attackMethods: number;
  weightedAttackVariants: number;
};

type TuningMethodOption = {
  method: string;
  label: string;
  subtitle: string;
  category: string;
  count: number;
  requiresGpu: boolean;
  available: boolean;
  weighted?: boolean;
  viewpoint?: boolean;
};

type TuningStageDetail = {
  key: "watermarks" | "attacks" | "quality";
  title: string;
  badge: string;
  checked: boolean;
  disabled?: boolean;
  description: string;
};

const emptyTuningCatalogStats: TuningCatalogStats = {
  watermarkMethods: 0,
  attackMethods: 0,
  weightedAttackVariants: 0
};

const terminalStatuses = new Set<DemoRunRecord["status"]>(["succeeded", "failed", "paused", "cancelled", "partially_failed"]);
const resumableStatuses = new Set<DemoRunRecord["status"]>(["paused", "failed", "partially_failed"]);
const rawArtifactFiles = [
  "run_plan.json",
  "run_state.json",
  "phase_state.json",
  "artifact_tree.json",
  "result_units.jsonl",
  "image_quality.jsonl",
  "image_detection.jsonl",
  "runtime_profile.jsonl",
  "run_status.json"
];
const experimentStageOrder: ExperimentStageKey[] = ["canonical", "watermark", "attack", "extract", "quality", "summary"];

const tuningQualityMetricOptions = [
  { id: "psnr", label: "PSNR", kind: "workers" },
  { id: "ssim", label: "SSIM", kind: "workers" },
  { id: "ms_ssim", label: "MS-SSIM", kind: "workers" },
  { id: "nmi", label: "NMI", kind: "workers" },
  { id: "lpips", label: "LPIPS", kind: "batch" },
  { id: "dists", label: "DISTS", kind: "batch" }
];
const allTuningQualityMetrics = tuningQualityMetricOptions.map((option) => option.id);

const quickTuningDefaults: TuningForm = {
  mode: "quick",
  sampleCount: 16,
  warmupCount: 2,
  candidateBatchCount: 3,
  probeSampleCount: 8,
  finalistCount: 2,
  minBatchSize: 1,
  maxBatchSize: 16,
  minWorkerCount: 1,
  maxWorkerCount: 32,
  repeatCount: 1,
  minImprovementRatio: 0.03,
  tuneWatermarks: true,
  tuneAttacks: true,
  includeViewpoint3dAttacks: false,
  tuneQuality: true,
  selectedQualityMetrics: allTuningQualityMetrics,
  selectedWatermarkMethods: [],
  selectedAttackMethods: []
};

const fullTuningDefaults: TuningForm = {
  mode: "full",
  sampleCount: 64,
  warmupCount: 4,
  candidateBatchCount: 3,
  probeSampleCount: 16,
  finalistCount: 3,
  minBatchSize: 1,
  maxBatchSize: 64,
  minWorkerCount: 1,
  maxWorkerCount: 64,
  repeatCount: 3,
  minImprovementRatio: 0.03,
  tuneWatermarks: true,
  tuneAttacks: true,
  includeViewpoint3dAttacks: false,
  tuneQuality: true,
  selectedQualityMetrics: allTuningQualityMetrics,
  selectedWatermarkMethods: [],
  selectedAttackMethods: []
};

const tuningDraftStorageKey = "wm-bench-tuning-form-draft";

function badgeClass(status: DemoRunRecord["status"]) {
  const normalized = status.replaceAll("_", "-");
  if (status === "running") {
    return "badge status-badge status-running ok";
  }
  if (status === "queued") {
    return "badge status-badge status-queued warn";
  }
  if (status === "succeeded") {
    return "badge status-badge status-succeeded ok";
  }
  if (status === "failed" || status === "partially_failed" || status === "cancelled") {
    return `badge status-badge status-${normalized} error`;
  }
  return `badge status-badge status-${normalized} warn`;
}

function isActiveRun(status: DemoRunRecord["status"]) {
  return status === "running" || status === "paused";
}

function isResumableRun(status: DemoRunRecord["status"]) {
  return resumableStatuses.has(status);
}

function isRestartableTerminalRun(status: DemoRunRecord["status"]) {
  return isResumableRun(status);
}

function isTerminalRun(status: DemoRunRecord["status"]) {
  return terminalStatuses.has(status);
}

function isPausableRun(run: DemoRunRecord) {
  return isActiveRun(run.status) && !run.cancelRequested;
}

function isCancellableRun(run: DemoRunRecord) {
  return isActiveRun(run.status) && run.stopIntent !== "cancel";
}

function stopIntentNotice(
  run: DemoRunRecord,
  labels: {
    pauseRequestedNotice: string;
    cancelRequestedNotice: string;
  }
) {
  if (!run.cancelRequested) {
    return null;
  }
  return run.stopIntent === "cancel" ? labels.cancelRequestedNotice : labels.pauseRequestedNotice;
}

function taskName(run: DemoRunRecord) {
  return run.taskName || run.configName || run.id;
}

function progressWidth(progress: number) {
  return `${Math.max(0, Math.min(100, progress))}%`;
}

function positiveInteger(value: number, fallback: number) {
  return Number.isFinite(value) && value > 0 ? Math.round(value) : fallback;
}

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

function matchesTuningOption(query: string, option: TuningMethodOption) {
  const normalized = query.trim().toLowerCase();
  if (!normalized) {
    return true;
  }
  return [option.method, option.label, option.subtitle, option.category].join(" ").toLowerCase().includes(normalized);
}

function watermarkMethod(algorithm: AlgorithmVersion) {
  return algorithm.method || algorithm.id.replace(/^alg-/, "");
}

function attackMethod(attack: AttackPreset) {
  return attack.executionMethod || attack.method;
}

function displayMethodToken(method: string) {
  if (method === "image_to_vedio") {
    return "image_to_video";
  }
  return method;
}

function isViewpointTuningMethod(method: string) {
  return method.startsWith("3d_viewpoint_rerendering_");
}

const VIEWPOINT_RERENDERING_PRIMARY_METHOD = "3d_viewpoint_rerendering_rotate_point";
const DIFFUSION_REGENERATION_PRIMARY_METHOD = "regen_diffusion";
const DIFFUSION_REGENERATION_METHODS = new Set(["regen_diffusion", "2x_regen", "4x_regen"]);
const NON_TUNABLE_ATTACK_METHODS = new Set(["identity", "atk-identity"]);

const ATTACK_DISPLAY_NAMES: Record<string, { en: string; zh: string }> = {
  brightness: { en: "Brightness", zh: "亮度调整" },
  contrast: { en: "Contrast", zh: "对比度调整" },
  gaussian_blur: { en: "Gaussian Blur", zh: "高斯模糊" },
  gaussian_noise: { en: "Gaussian Noise", zh: "高斯噪声" },
  jpeg: { en: "JPEG Compression", zh: "JPEG 压缩" },
  resize: { en: "Resize", zh: "缩放" },
  resized_crop: { en: "Resized Crop", zh: "缩放裁剪" },
  rotation: { en: "Rotation", zh: "旋转" },
  erasing: { en: "Random Erasing", zh: "区域擦除" },
  screen_shoot: { en: "PIMoG-style Screen-Camera", zh: "屏幕-拍摄信道" },
  print_camera: { en: "CamMark-style Print-Camera", zh: "打印-拍摄信道" },
  combined_physical: { en: "Combined Physical Channel", zh: "组合物理信道" },
  "2x_regen": { en: "2-pass Diffusion Regeneration", zh: "2轮扩散再生成" },
  "4x_regen": { en: "4-pass Diffusion Regeneration", zh: "4轮扩散再生成" },
  regen_diffusion: { en: "WAVES Diffusion Regeneration", zh: "扩散再生成" },
  noise_to_image: { en: "CtrlRegen Noise-to-Image", zh: "噪声到图像再生成" },
  regen_vae: { en: "CompressAI VAE Reconstruction", zh: "VAE 再生成" },
  image_to_vedio: { en: "NFPA Image-to-Video", zh: "图像到视频再生成" },
  cew_e1: { en: "Auto-Tone", zh: "自动色调" },
  cew_e2: { en: "Warm-Vivid", zh: "暖色鲜艳" },
  cew_e3: { en: "Film-Faded", zh: "胶片褪色" },
  cew_e4: { en: "Local-Clarity HDR", zh: "局部清晰 HDR" },
  cew_c1: { en: "Basic Auto-Fix SR", zh: "自动修复+超分" },
  cew_c2: { en: "Color Retouch SR", zh: "色彩修饰+超分" },
  cew_c3: { en: "Detail Enhance SR", zh: "细节增强+超分" },
  cew_c4: { en: "Full Enhancement Chain", zh: "完整增强链" },
  cew_d1: { en: "Zero-DCE++ Auto-Light", zh: "自动补光" },
  cew_d2: { en: "DeepWB Auto-WhiteBalance", zh: "自动白平衡" },
  cew_d3: { en: "Image-Adaptive 3D LUT", zh: "自适应 AI 色彩" },
  cew_d4: { en: "Retinexformer Detail Low-Light Enhance", zh: "低光细节增强" },
  cew_d5: { en: "NAFNet/Restormer AI-Denoise", zh: "AI 去噪" },
  cew_s1: { en: "Real-ESRGAN", zh: "Real-ESRGAN" },
  cew_s2: { en: "SwinIR", zh: "SwinIR" },
  cew_s3: { en: "BSRGAN", zh: "BSRGAN" }
};

const VIEWPOINT_METHOD_PATTERN = /^3d_viewpoint_rerendering_(swipe|shake|rotate|rotate_forward)_(point|ahead)$/;
const VIEWPOINT_MOTION_LABELS: Record<string, string> = {
  swipe: "横向扫动",
  shake: "抖动",
  rotate: "环绕旋转",
  rotate_forward: "前向环绕"
};

function watermarkCategoryLabel(category: string | undefined) {
  if (category === "classical" || category === "traditional_watermark") {
    return "传统水印";
  }
  return "深度水印";
}

function attackCategoryLabel(category: string | undefined) {
  const labels: Record<string, string> = {
    distortion: "失真攻击",
    distortion_attacks: "经典失真",
    physical: "物理信道",
    physical_channel_attacks: "物理信道",
    regeneration: "再生成",
    regeneration_attacks: "再生成",
    consumer_enhancement: "消费级增强",
    consumer_enhancement_workflow_attacks: "消费级增强",
    "3d_viewpoint_rerendering": "3D 视角重渲染",
    adversarial: "对抗攻击"
  };
  return labels[category || ""] || category || "其他攻击";
}

function viewpointDisplayName(method: string) {
  const parsed = VIEWPOINT_METHOD_PATTERN.exec(method);
  if (!parsed) {
    return null;
  }
  const motion = VIEWPOINT_MOTION_LABELS[parsed[1]] ?? parsed[1];
  const mode = parsed[2] === "point" ? "point" : "ahead";
  return `3D 视角 ${motion} (${mode})`;
}

function attackTuningDisplayName(attack: AttackPreset | undefined, method: string) {
  if (method === VIEWPOINT_RERENDERING_PRIMARY_METHOD) {
    return "3D 视角重渲染";
  }
  if (method === DIFFUSION_REGENERATION_PRIMARY_METHOD) {
    return ATTACK_DISPLAY_NAMES[DIFFUSION_REGENERATION_PRIMARY_METHOD].zh;
  }
  const viewpointName = viewpointDisplayName(method);
  if (viewpointName) {
    return viewpointName;
  }
  const display = ATTACK_DISPLAY_NAMES[method];
  if (display) {
    return display.zh;
  }
  if (attack?.method && ATTACK_DISPLAY_NAMES[attack.method]) {
    return ATTACK_DISPLAY_NAMES[attack.method].zh;
  }
  return attack?.name || method;
}

function attackCatalogDisplayName(language: Language, attack: AttackPreset) {
  const display = ATTACK_DISPLAY_NAMES[attack.method];
  if (display) {
    return language === "zh" ? display.zh : display.en;
  }
  return localizedName(language, attack.id, attack.name);
}

function attackMethodDisplayName(language: Language, method: string) {
  const viewpointName = viewpointDisplayName(method);
  if (viewpointName) {
    return viewpointName;
  }
  const display = ATTACK_DISPLAY_NAMES[method];
  if (display) {
    return language === "zh" ? display.zh : display.en;
  }
  return displayMethodToken(method);
}

function attackTuningRepresentativeMethod(method: string) {
  if (DIFFUSION_REGENERATION_METHODS.has(method)) {
    return DIFFUSION_REGENERATION_PRIMARY_METHOD;
  }
  if (isViewpointTuningMethod(method)) {
    return VIEWPOINT_RERENDERING_PRIMARY_METHOD;
  }
  return method;
}

function normalizeTuningAttackMethods(methods: string[]) {
  return [
    ...new Set(
      methods
        .map(attackTuningRepresentativeMethod)
        .filter((method) => method && !NON_TUNABLE_ATTACK_METHODS.has(method))
    )
  ].sort();
}

function buildWatermarkTuningOptions(algorithms: AlgorithmVersion[]): TuningMethodOption[] {
  const groups = new Map<string, AlgorithmVersion[]>();
  for (const algorithm of algorithms) {
    const method = watermarkMethod(algorithm);
    groups.set(method, [...(groups.get(method) ?? []), algorithm]);
  }
  return [...groups.entries()]
    .map(([method, items]) => {
      const primary = items.find((item) => item.status === "enabled" && item.available !== false) ?? items[0];
      const category = watermarkCategoryLabel(primary?.category);
      return {
        method,
        label: primary?.name || method,
        subtitle: `${category} · ${method}${items.length > 1 ? ` · ${items.length} 个版本` : ""}`,
        category,
        count: items.length,
        requiresGpu: items.some((item) => item.requiresGpu),
        available: items.some((item) => item.status === "enabled" && item.available !== false),
        weighted: items.some((item) => item.weightsPackId || item.weightsDir || item.weightsPath)
      };
    })
    .sort((left, right) => `${left.category} ${left.label}`.localeCompare(`${right.category} ${right.label}`, undefined, { numeric: true }));
}

function buildAttackTuningOptions(attacks: AttackPreset[]): TuningMethodOption[] {
  const groups = new Map<string, AttackPreset[]>();
  for (const attack of attacks) {
    const method = attackTuningRepresentativeMethod(attackMethod(attack));
    if (NON_TUNABLE_ATTACK_METHODS.has(method) || NON_TUNABLE_ATTACK_METHODS.has(attack.id)) {
      continue;
    }
    groups.set(method, [...(groups.get(method) ?? []), attack]);
  }
  return [...groups.entries()]
    .map(([method, items]) => {
      const primary =
        items.find((item) => attackMethod(item) === method && item.available !== false) ??
        items.find((item) => item.available !== false) ??
        items[0];
      const category = attackCategoryLabel(primary?.category);
      const label = attackTuningDisplayName(primary, method);
      const rawMethods = [...new Set(items.map((item) => attackMethod(item)))].sort();
      const inheritedCount = rawMethods.filter((item) => item !== method).length;
      const policyNote =
        method === DIFFUSION_REGENERATION_PRIMARY_METHOD
          ? " · 2/4 轮继承同一模型测量"
          : method === VIEWPOINT_RERENDERING_PRIMARY_METHOD
            ? " · 3D 变体继承同一代表测量"
            : "";
      return {
        method,
        label,
        subtitle: `${category} · ${method}${items.length > 1 ? ` · ${items.length} 个配置` : ""}${
          inheritedCount > 0 ? ` · ${inheritedCount} 个继承项` : ""
        }${policyNote}`,
        category,
        count: items.length,
        requiresGpu: items.some((item) => item.requiresGpu),
        available: items.some((item) => item.available !== false),
        weighted: items.some((item) => item.weightsPackId || item.weightsDir || item.weightsPath),
        viewpoint: items.some((item) => isViewpointTuningMethod(attackMethod(item)))
      };
    })
    .sort((left, right) => `${left.category} ${left.label}`.localeCompare(`${right.category} ${right.label}`, undefined, { numeric: true }));
}

function candidateRange(minValue: number, maxValue: number, extras: number[] = []) {
  const minCandidate = positiveInteger(Math.min(minValue, maxValue), 1);
  const maxCandidate = positiveInteger(Math.max(minValue, maxValue), minCandidate);
  const values = new Set<number>(extras.filter((value) => value >= minCandidate && value <= maxCandidate));
  let value = 1;
  while (value <= maxCandidate) {
    if (value >= minCandidate) {
      values.add(value);
    }
    value *= 2;
  }
  values.add(minCandidate);
  values.add(maxCandidate);
  return [...values].sort((left, right) => left - right);
}

function buildTuningPayload(form: TuningForm) {
  const maxBatchSize = positiveInteger(form.maxBatchSize, form.mode === "full" ? 64 : 16);
  const maxWorkerCount = positiveInteger(form.maxWorkerCount, form.mode === "full" ? 64 : 32);
  const batchCandidates = candidateRange(form.minBatchSize, maxBatchSize);
  const workerCandidates = candidateRange(form.minWorkerCount, maxWorkerCount, [24, 32, 48, 64, 96, 128]);
  const candidateBatchCount = positiveInteger(form.candidateBatchCount, 3);
  const sampleCount = Math.max(
    positiveInteger(form.sampleCount, maxBatchSize * candidateBatchCount),
    maxBatchSize * candidateBatchCount,
    Math.max(...batchCandidates) * candidateBatchCount
  );
  const watermarkMethods = [...new Set(form.selectedWatermarkMethods)].sort();
  const attackMethods = normalizeTuningAttackMethods(form.selectedAttackMethods);
  const qualityMetrics = allTuningQualityMetrics.filter((metric) => form.selectedQualityMetrics.includes(metric));
  const tuneWatermarks = form.tuneWatermarks && watermarkMethods.length > 0;
  const tuneAttacks = form.tuneAttacks && attackMethods.length > 0;
  const tuneQuality = form.tuneQuality && qualityMetrics.length > 0;
  return {
    mode: form.mode,
    sampleCount,
    warmupCount: positiveInteger(form.warmupCount, 2),
    candidateBatchCount,
    searchStrategy: "single_pass",
    probeSampleCount: Math.min(sampleCount, positiveInteger(form.probeSampleCount, form.mode === "full" ? 16 : 8)),
    finalistCount: 1,
    batchCandidates,
    workerCandidates,
    repeatCount: 1,
    maxBatchSize,
    maxWorkerCount,
    autoExpandCandidates: form.mode === "full",
    minImprovementRatio: Math.max(0, form.minImprovementRatio),
    boundaryPatience: form.mode === "full" ? 2 : 1,
    tuneWatermarks,
    tuneAttacks,
    watermarkMethods,
    attackMethods,
    includeViewpoint3dAttacks: tuneAttacks && attackMethods.some(isViewpointTuningMethod),
    tuneQuality,
    qualityMetrics
  };
}

function buildTuningCatalogStats(
  algorithms: Array<{ id: string; method?: string | null }>,
  attacks: Array<{
    id: string;
    method: string;
    executionMethod?: string | null;
    displayMethod?: string | null;
    displayGroup?: string | null;
    category?: string | null;
    categoryPath?: string | null;
    viewpointMotion?: string | null;
    viewpointPhase?: number | null;
    weightsDir?: string | null;
    weightsPackId?: string | null;
    weightsPath?: string | null;
  }>
): TuningCatalogStats {
  const watermarkMethods = new Set(algorithms.map((algorithm) => algorithm.method || algorithm.id).filter(Boolean));
  const attackMethods = new Set(
    attacks
      .map((attack) => attackTuningRepresentativeMethod(attack.executionMethod || attack.method))
      .filter((method) => method && !NON_TUNABLE_ATTACK_METHODS.has(method))
  );
  const weightedAttackVariants = attacks.filter((attack) => attack.weightsPackId || attack.weightsDir || attack.weightsPath).length;
  return {
    watermarkMethods: watermarkMethods.size,
    attackMethods: attackMethods.size,
    weightedAttackVariants
  };
}

function tuningStageCount(payload: ReturnType<typeof buildTuningPayload>) {
  return [payload.tuneWatermarks, payload.tuneAttacks, payload.tuneQuality].filter(Boolean).length;
}

function tuningParameterSegmentCount(payload: ReturnType<typeof buildTuningPayload>) {
  let total = 0;
  if (payload.tuneWatermarks) {
    total += 3;
  }
  if (payload.tuneAttacks) {
    total += 2;
  }
  if (payload.tuneQuality) {
    total += payload.qualityMetrics.length;
  }
  return total;
}

function tuningCombinationCount(payload: ReturnType<typeof buildTuningPayload>) {
  const batchSearchCount = payload.batchCandidates.length;
  const workerSearchCount = payload.workerCandidates.length;
  let total = 0;
  if (payload.tuneWatermarks) {
    total += payload.watermarkMethods.length * (batchSearchCount + workerSearchCount);
  }
  if (payload.tuneAttacks) {
    total += payload.attackMethods.length * batchSearchCount;
  }
  if (payload.tuneQuality) {
    const selectedCpuMetrics = payload.qualityMetrics.filter((metric) => ["psnr", "ssim", "ms_ssim", "nmi"].includes(metric)).length;
    const selectedPerceptualMetrics = payload.qualityMetrics.filter((metric) => ["lpips", "dists"].includes(metric)).length;
    total += selectedCpuMetrics * workerSearchCount + selectedPerceptualMetrics * batchSearchCount;
  }
  return Math.max(0, total);
}

function tuningEstimateText(payload: ReturnType<typeof buildTuningPayload>) {
  const combinations = tuningCombinationCount(payload);
  if (payload.mode === "full") {
    return combinations >= 49 ? "约 10-20 分钟" : "约 6-12 分钟";
  }
  return combinations >= 35 ? "约 3-5 分钟" : "约 1-3 分钟";
}

function buildTuningStageDetails(
  payload: ReturnType<typeof buildTuningPayload>,
  stats: TuningCatalogStats
): TuningStageDetail[] {
  return [
    {
      key: "watermarks",
      title: "水印算法",
      badge: payload.tuneWatermarks ? `${payload.watermarkMethods.length}/${stats.watermarkMethods || payload.watermarkMethods.length} 项` : "跳过",
      checked: payload.tuneWatermarks,
      description: "选择需要搜索 embed、extract 与 CPU workers 参数的水印算法。"
    },
    {
      key: "attacks",
      title: "攻击算法",
      badge: payload.tuneAttacks ? `${payload.attackMethods.length}/${stats.attackMethods || payload.attackMethods.length} 项` : "跳过",
      checked: payload.tuneAttacks,
      description: `选择需要搜索 batch 或 CPU workers 参数的攻击算法；同模型变体会继承代表项结果，避免重复测量。${
        stats.weightedAttackVariants > 0 ? ` ${stats.weightedAttackVariants} 个攻击配置需要先在资源页安装权重。` : ""
      }`
    },
    {
      key: "quality",
      title: "quality 指标",
      badge: payload.tuneQuality ? `${payload.qualityMetrics.length}/${allTuningQualityMetrics.length} 项` : "跳过",
      checked: payload.tuneQuality,
      description: "选择需要搜索的视觉质量指标；CPU 指标写入 workers，感知指标写入 batch。"
    }
  ];
}

function percent(current: number, total: number) {
  if (total <= 0) {
    return 0;
  }
  return Math.min(100, Math.round((Math.max(0, current) / total) * 100));
}

function materializedStatsFromRunState(runState: RunState | null, phase?: RunPhaseState): MaterializedStats | null {
  if (!runState) {
    return null;
  }
  const item = phase?.currentItem ?? {};
  const counters = phase?.counters ?? {};
  const root = runState.materializedRoot ?? "";
  const latestDir = typeof item.materializedDir === "string" ? item.materializedDir : root;
  const cacheHits = typeof counters.cacheHits === "number" ? counters.cacheHits : 0;
  if (!root && !latestDir && cacheHits === 0) {
    return null;
  }
  return {
    root: root || "n/a",
    latestDir: latestDir || "n/a",
    cacheHits
  };
}

function formatNumber(value: number) {
  return Number.isInteger(value) ? value.toString() : Number(value.toFixed(4)).toString();
}

function durationMsBetween(start?: string | null, end?: string | null) {
  if (!start || !end) {
    return null;
  }
  const startMs = Date.parse(start);
  const endMs = Date.parse(end);
  if (Number.isNaN(startMs) || Number.isNaN(endMs)) {
    return null;
  }
  return Math.max(0, endMs - startMs);
}

function formatDurationMs(durationMs: number | null, language: "zh" | "en") {
  if (durationMs === null) {
    return "n/a";
  }
  const totalSeconds = Math.max(0, Math.round(durationMs / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (language === "zh") {
    if (hours > 0) {
      return `${hours}小时 ${minutes}分钟 ${seconds}秒`;
    }
    if (minutes > 0) {
      return `${minutes}分钟 ${seconds}秒`;
    }
    return `${seconds}秒`;
  }
  if (hours > 0) {
    return `${hours}h ${minutes}m ${seconds}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
}

function formatParamValue(value: unknown): string {
  if (typeof value === "number") {
    return formatNumber(value);
  }
  if (typeof value === "string" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => formatParamValue(item)).join(", ")}]`;
  }
  if (value && typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function paramsLabel(params: unknown) {
  if (!params || typeof params !== "object" || Array.isArray(params)) {
    return null;
  }
  const entries = Object.entries(params as Record<string, unknown>).filter(([, value]) => value !== null && value !== undefined);
  if (!entries.length) {
    return null;
  }
  return entries.map(([key, value]) => `${key}=${formatParamValue(value)}`).join(", ");
}

function humanizeId(id: string) {
  return id.replace(/^(atk|alg|ds)-/, "").replace(/[_-]+/g, " ");
}

function displayName(id: string, names: Record<string, string>) {
  if (!id || id === "n/a") {
    return "n/a";
  }
  return names[id] ?? humanizeId(id);
}

function eventNumber(event: { [key: string]: unknown }, key: string) {
  const value = event[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function eventText(event: ParallelTuningEvent, key: string) {
  const value = event[key];
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function tuningStageLabel(stage: string) {
  const labels: Record<string, string> = {
    watermark_embed: "水印嵌入 batch",
    watermark_extract: "水印提取 batch",
    watermark_cpu: "水印 CPU workers",
    attack_batch: "攻击 batch",
    attack_cpu: "攻击 CPU workers",
    quality_cpu: "quality CPU workers",
    quality_perceptual: "感知质量 batch",
    quality_lpips: "LPIPS batch",
    quality_dists: "DISTS batch",
    quality_psnr: "PSNR workers",
    quality_ssim: "SSIM workers",
    quality_ms_ssim: "MS-SSIM workers",
    quality_nmi: "NMI workers"
  };
  return labels[stage] ?? humanizeId(stage);
}

function tuningScopeLabel(stage: string) {
  if (stage.startsWith("watermark_")) {
    return "水印算法";
  }
  if (stage.startsWith("attack_")) {
    return "攻击算法";
  }
  if (stage.startsWith("quality_")) {
    return "quality 指标";
  }
  return "调参过程";
}

function tuningMethodLabel(method: string, stage: string, names: Record<string, string>) {
  if (!method || method === "n/a") {
    return tuningStageLabel(stage);
  }
  if (names[method]) {
    return names[method];
  }
  if (stage.startsWith("watermark_")) {
    return resolveWatermarkDisplayName(method, humanizeId(method));
  }
  if (stage.startsWith("attack_")) {
    return attackTuningDisplayName(undefined, method);
  }
  return humanizeId(method);
}

function tuningPoints(job: ParallelTuningJob | null, names: Record<string, string>): TuningPoint[] {
  const events = job?.events ?? [];
  return events
    .map((event, index) => {
      const throughput = eventNumber(event, "imagesPerSecond");
      const batchSize = eventNumber(event, "batchSize");
      const workers = eventNumber(event, "workers");
      if (throughput == null || (batchSize == null && workers == null)) {
        return null;
      }
      const stage = String(event.stage ?? "step");
      const method = eventText(event, "method") ?? String(event.message ?? stage);
      const kind = batchSize != null ? "batch" : "workers";
      const candidate = batchSize ?? workers ?? 1;
      const stageLabel = tuningStageLabel(stage);
      const scopeLabel = tuningScopeLabel(stage);
      const label = tuningMethodLabel(method, stage, names);
      const groupKey = `${stage}:${method}`;
      const parsedTime =
        typeof event.timestamp === "number"
          ? event.timestamp
          : typeof event.timestamp === "string"
            ? Date.parse(event.timestamp)
            : NaN;
      return {
        key: `${event.timestamp ?? "point"}-${index}`,
        eventIndex: index,
        eventTime: Number.isFinite(parsedTime) ? parsedTime : index,
        stage,
        stageLabel,
        scopeLabel,
        method,
        label,
        candidate,
        throughput,
        kind,
        ok: event.ok !== false,
        groupKey,
        groupLabel: `${label} · ${stageLabel}`
      } satisfies TuningPoint;
    })
    .filter((point): point is TuningPoint => Boolean(point));
}

function buildTuningProcessGroups(points: TuningPoint[]): TuningProcessGroup[] {
  const groups = new Map<string, TuningPoint[]>();
  points.forEach((point) => {
    groups.set(point.groupKey, [...(groups.get(point.groupKey) ?? []), point]);
  });
  return [...groups.entries()]
    .map(([key, items]) => {
      const latest = items[items.length - 1];
      const successful = items.filter((point) => point.ok);
      const bestThroughput = Math.max(0, ...successful.map((point) => point.throughput));
      return {
        key,
        label: latest.groupLabel,
        description: `${latest.scopeLabel} · ${items.length} 个候选 · ${
          latest.kind === "batch" ? "batch 搜索" : "workers 搜索"
        }`,
        pointCount: items.length,
        successCount: successful.length,
        failedCount: items.length - successful.length,
        bestThroughput,
        latestThroughput: latest.throughput,
        latestEventIndex: latest.eventIndex,
        latestEventTime: latest.eventTime
      } satisfies TuningProcessGroup;
    })
    .sort(
      (left, right) =>
        right.latestEventTime - left.latestEventTime ||
        right.latestEventIndex - left.latestEventIndex ||
        right.bestThroughput - left.bestThroughput
    );
}

function variantCountForAttack(config: SavedExperimentConfig | undefined, attackId: string) {
  const params = config?.selection.attackParamOverrides?.[attackId];
  if (params?.length) {
    return params.length;
  }
  const strengths = config?.selection.attackStrengthOverrides?.[attackId];
  if (strengths?.length) {
    return strengths.length;
  }
  return 1;
}

function selectionSummary(config: SavedExperimentConfig | undefined): ExecutionSummary["selection"] {
  if (!config) {
    return undefined;
  }
  return {
    datasets: config.selection.datasetIds.length,
    watermarks: config.selection.algorithmIds.length,
    attacks: config.selection.attackPresetIds.length,
    seeds: config.selection.seeds.length,
    maxSamples: config.selection.maxSamples,
    sampleCount: config.sampleCount,
    imageOperationCount: config.imageOperationCount
  };
}

function phaseKeyForStage(stageKey: ExperimentStageKey): RunPhaseState["key"] {
  if (stageKey === "watermark") {
    return "watermark_embed";
  }
  if (stageKey === "extract") {
    return "watermark_extract";
  }
  return stageKey;
}

function progressStepFromPhase(phase: RunPhaseState, label: string): ProgressStep {
  const rawCurrent = Number(phase.current ?? 0);
  const total = Math.max(0, Number(phase.total ?? 0));
  const current = total > 0 ? Math.min(Math.max(0, rawCurrent), total) : Math.max(0, rawCurrent);
  const rawPercent = Number(phase.percent ?? percent(current, total));
  const displayPercent = Number.isFinite(rawPercent) ? Math.min(100, Math.max(0, rawPercent)) : percent(current, total);
  return {
    key: phase.key,
    label,
    current,
    total,
    percent: displayPercent,
    meta: total ? `${current}/${total}` : phase.status
  };
}

function stageCellProgress(
  stage: ExperimentStageTab | undefined,
  shards: RunShardProgress[]
): StageCellProgress {
  const stageKey = stage?.key ?? "canonical";
  const phaseKey = phaseKeyForStage(stageKey);
  const unit = stageKey === "canonical" ? "images" : "cells";
  if (!shards.length) {
    const phaseProgress = stage?.phase?.cellProgress;
    const fallbackCurrent = Number(phaseProgress?.current ?? 0);
    const fallbackTotal = Number(phaseProgress?.total ?? 0);
    return {
      current: fallbackCurrent,
      total: Math.max(fallbackCurrent, fallbackTotal),
      percent: percent(fallbackCurrent, Math.max(fallbackCurrent, fallbackTotal)),
      unit
    };
  }
  const totals = shards.reduce(
    (acc, shard) => {
      const progressDoc = shardPhaseCellProgress(shard, phaseKey, stageKey);
      acc.current += progressDoc.current;
      acc.total += progressDoc.total;
      return acc;
    },
    { current: 0, total: 0 }
  );
  const total = Math.max(totals.current, totals.total);
  return {
    current: totals.current,
    total,
    percent: percent(totals.current, total),
    unit
  };
}

function shardPhaseCellProgress(
  shard: RunShardProgress,
  phaseKey: RunPhaseState["key"],
  stageKey: ExperimentStageKey
) {
  const phase = shard.phases?.find((item) => item.key === phaseKey);
  const phaseProgress = phase?.cellProgress;
  if (phaseProgress) {
    const current = Math.max(0, Number(phaseProgress.current ?? 0));
    const total = Math.max(current, Number(phaseProgress.total ?? 0));
    return { current, total };
  }
  const counters = phase?.counters ?? {};
  if (stageKey === "canonical") {
    const current = Math.max(0, Number(counters.imagesDone ?? phase?.current ?? 0));
    const total = Math.max(current, Number(shard.sampleCount ?? phase?.total ?? 0));
    return { current: phase?.status === "succeeded" ? total : current, total };
  }
  const total = Math.max(0, Number(counters.phaseCellsTotal ?? shard.expectedCells ?? 0));
  const sampleCount = Math.max(1, Number(shard.sampleCount ?? 0));
  if (!counters.phaseCellsDone && !counters.cellsDone && !counters.resultUnitsDone) {
    if (stageKey === "attack") {
      const current = Math.floor(Number(counters.positiveImagesDone ?? 0) / sampleCount);
      return { current: phase?.status === "succeeded" ? total : Math.min(current, total), total };
    }
    if (stageKey === "extract") {
      const positive = Math.floor(Number(counters.positiveImagesDone ?? 0) / sampleCount);
      const negative = Math.floor(Number(counters.negativeImagesDone ?? 0) / sampleCount);
      const current = Math.min(positive, negative);
      return { current: phase?.status === "succeeded" ? total : Math.min(current, total), total };
    }
    if (stageKey === "quality") {
      const current = Math.floor(Number(counters.pairsDone ?? 0) / (sampleCount * 2));
      return { current: phase?.status === "succeeded" ? total : Math.min(current, total), total };
    }
    if (stageKey === "watermark" && phase?.total) {
      const ratio = Number(phase.current ?? 0) / Math.max(1, Number(phase.total));
      const current = Math.floor(ratio * total);
      return { current: phase?.status === "succeeded" ? total : Math.min(current, total), total };
    }
  }
  const rawCurrent = Number(
    counters.phaseCellsDone ??
      counters.cellsDone ??
      counters.resultUnitsDone ??
      (phaseKey === shard.currentPhase ? shard.cellProgress?.current : 0) ??
      0
  );
  const current = phase?.status === "succeeded" ? total : Math.max(0, rawCurrent);
  return { current: Math.min(current, total), total };
}

function buildExperimentStagesFromRunState(
  runState: RunState,
  labels: Record<ExperimentStageKey, string>
): ExperimentStageTab[] {
  const phaseByKey = new Map(runState.phases.map((phase) => [phase.key, phase]));
  return experimentStageOrder.map((key) => {
    const phase = phaseByKey.get(phaseKeyForStage(key));
    const status = phase?.status ?? "pending";
    const reached = key === "canonical" || status !== "pending";
    const completed = status === "succeeded";
    const active = runState.currentPhase === phaseKeyForStage(key) || status === "running";
    return {
      key,
      label: labels[key],
      progressKey: phase?.key ?? key,
      step: phase ? progressStepFromPhase(phase, labels[key]) : { key, label: labels[key], current: 0, total: 0, percent: 0, meta: "pending" },
      phase,
      reached,
      completed,
      active
    };
  });
}

function makeSummary(
  run: DemoRunRecord,
  note: string,
  options: {
    statusOverride?: DemoRunRecord["status"];
    config?: SavedExperimentConfig;
    runState?: RunState | null;
  } = {}
): ExecutionSummary {
  const summaryPhase = options.runState?.phases.find((phase) => phase.key === "summary");
  const counters = summaryPhase?.counters ?? {};
  const resultUnits = Number(counters.resultUnitsDone ?? options.runState?.expectedResultUnits ?? run.cells ?? 0);
  const failedResultUnits = Number(counters.failedUnits ?? 0);
  const totalResultUnits = Number(options.runState?.expectedResultUnits ?? run.cells ?? resultUnits);
  return {
    taskName: taskName(run),
    runId: run.id,
    status: options.statusOverride ?? run.status,
    progress: run.progress,
    resultUnits,
    succeededResultUnits: Math.max(0, resultUnits - failedResultUnits),
    failedResultUnits,
    remainingResultUnits: Math.max(0, totalResultUnits - resultUnits),
    configName: run.configName,
    workerId: run.workerId,
    artifactRoot: run.artifactRoot,
    createdAt: run.createdAt,
    startedAt: run.startedAt,
    finishedAt: run.finishedAt,
    updatedAt: run.updatedAt,
    durationMs: durationMsBetween(run.startedAt, run.finishedAt ?? run.updatedAt),
    selection: selectionSummary(options.config),
    materialized: materializedStatsFromRunState(options.runState ?? null, summaryPhase),
    note
  };
}

function terminalRunNote(
  status: DemoRunRecord["status"],
  labels: {
    runFinishedNotice: string;
    stopSavedNotice: string;
    runCancelledNotice: string;
    runFailedNotice: string;
  }
) {
  if (status === "succeeded") {
    return labels.runFinishedNotice;
  }
  if (status === "paused") {
    return labels.stopSavedNotice;
  }
  if (status === "cancelled") {
    return labels.runCancelledNotice;
  }
  return labels.runFailedNotice;
}

function runStatusLabel(
  status: DemoRunRecord["status"],
  statusLabels: Record<string, string>
) {
  return statusLabels[status] ?? status;
}

export default function RunsPage() {
  const { language, t } = useLanguage();
  const [configs, setConfigs] = useState<SavedExperimentConfig[]>([]);
  const [activeRuns, setActiveRuns] = useState<DemoRunRecord[]>([]);
  const [selectedConfigId, setSelectedConfigId] = useState("");
  const [selectedResumeRunId, setSelectedResumeRunId] = useState("");
  const [taskNameInput, setTaskNameInput] = useState("");
  const [startMode, setStartMode] = useState<StartMode>("new");
  const [startDialogOpen, setStartDialogOpen] = useState(false);
  const [monitorRunId, setMonitorRunId] = useState("");
  const [monitorRun, setMonitorRun] = useState<DemoRunRecord | null>(null);
  const [runState, setRunState] = useState<RunState | null>(null);
  const [selectedStageKey, setSelectedStageKey] = useState<ExperimentStageKey>("canonical");
  const [stageSelectionPinned, setStageSelectionPinned] = useState(false);
  const [lastSummary, setLastSummary] = useState<ExecutionSummary | null>(null);
  const [cancelConfirmOpen, setCancelConfirmOpen] = useState(false);
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [autoRefreshSeconds, setAutoRefreshSeconds] = useState(10);
  const [resourceNames, setResourceNames] = useState<Record<string, string>>({});
  const [gpuTelemetry, setGpuTelemetry] = useState<GpuTelemetry | null>(null);
  const [expandedShardIds, setExpandedShardIds] = useState<string[]>([]);
  const [tuningJob, setTuningJob] = useState<ParallelTuningJob | null>(null);
  const [tuningDialogOpen, setTuningDialogOpen] = useState(false);
  const [tuningWorkspaceOpen, setTuningWorkspaceOpen] = useState(false);
  const [tuningForm, setTuningForm] = useState<TuningForm>(quickTuningDefaults);
  const [tuningBusy, setTuningBusy] = useState(false);
  const [tuningNotice, setTuningNotice] = useState("");
  const [tuningDraftNotice, setTuningDraftNotice] = useState("");
  const [tuningCatalogStats, setTuningCatalogStats] = useState<TuningCatalogStats>(emptyTuningCatalogStats);
  const [tuningWatermarkOptions, setTuningWatermarkOptions] = useState<TuningMethodOption[]>([]);
  const [tuningAttackOptions, setTuningAttackOptions] = useState<TuningMethodOption[]>([]);
  const [tuningWatermarkFilter, setTuningWatermarkFilter] = useState("");
  const [tuningAttackFilter, setTuningAttackFilter] = useState("");
  const [selectedTuningProcessKey, setSelectedTuningProcessKey] = useState("all");

  const selectedConfig = useMemo(
    () => configs.find((config) => config.id === selectedConfigId),
    [configs, selectedConfigId]
  );
  const selectedResumeRun = useMemo(
    () => activeRuns.find((run) => run.id === selectedResumeRunId),
    [activeRuns, selectedResumeRunId]
  );
  const monitoredConfig = useMemo(
    () => configs.find((config) => config.id === monitorRun?.configId),
    [configs, monitorRun?.configId]
  );
  const experimentStages = useMemo(
    () => {
      const labels = {
        canonical: t.runs.canonicalStage,
        watermark: t.runs.watermarkStage,
        attack: t.runs.attackStage,
        extract: t.runs.extractStage,
        quality: t.runs.qualityStage,
        summary: t.runs.summaryStage
      };
      return buildExperimentStagesFromRunState(
        runState ?? {
          runId: monitorRun?.id ?? "",
          status: monitorRun?.status ?? "queued",
          currentPhase: "canonical",
          overallProgress: monitorRun?.progress ?? 0,
          progress: monitorRun?.progress ?? 0,
          progressKind: "phaseOperations",
          expectedResultUnits: monitorRun?.cells ?? 0,
          phases: []
        },
        labels
      );
    },
    [
      monitorRun,
      runState,
      t.runs.attackStage,
      t.runs.canonicalStage,
      t.runs.extractStage,
      t.runs.qualityStage,
      t.runs.summaryStage,
      t.runs.watermarkStage
    ]
  );
  const selectedStage =
    experimentStages.find((stage) => stage.key === selectedStageKey && stage.reached) ??
    experimentStages.find((stage) => stage.active) ??
    experimentStages[0];
  const selectedStageShards = useMemo(() => {
    const phaseKey = phaseKeyForStage(selectedStage?.key ?? "canonical");
    return (runState?.shards ?? []).filter((shard) => shard.currentPhase === phaseKey);
  }, [runState?.shards, selectedStage?.key]);
  const selectedStageProgress = useMemo(
    () => stageCellProgress(selectedStage, runState?.shards ?? []),
    [runState?.shards, selectedStage]
  );
  const tuningRunning = tuningJob?.status === "running";
  const tuningPausing = tuningJob?.status === "pausing";
  const tuningPaused = tuningJob?.status === "paused";
  const tuningStopping = tuningJob?.status === "cancelling";
  const tuningActive = tuningRunning || tuningPausing || tuningStopping;
  const tuningEnvEntries = Object.entries(tuningJob?.summary?.envUpdates ?? {});
  const tuningChartPoints = useMemo(() => tuningPoints(tuningJob, resourceNames), [resourceNames, tuningJob]);
  const tuningProcessGroups = useMemo(() => buildTuningProcessGroups(tuningChartPoints), [tuningChartPoints]);
  const selectedTuningProcess = useMemo(
    () => tuningProcessGroups.find((group) => group.key === selectedTuningProcessKey) ?? null,
    [selectedTuningProcessKey, tuningProcessGroups]
  );
  const visibleTuningChartPoints = useMemo(
    () =>
      selectedTuningProcessKey === "all"
        ? tuningChartPoints
        : tuningChartPoints.filter((point) => point.groupKey === selectedTuningProcessKey),
    [selectedTuningProcessKey, tuningChartPoints]
  );
  const tuningChartScope =
    selectedTuningProcessKey === "all"
      ? `全部候选点 · ${tuningChartPoints.length} 次测量`
      : selectedTuningProcess?.label ?? "当前过程";
  const effectiveTuningPayload = useMemo(() => buildTuningPayload(tuningForm), [tuningForm]);
  const effectiveTuningCombinations = tuningCombinationCount(effectiveTuningPayload);
  const effectiveTuningStages = tuningStageCount(effectiveTuningPayload);
  const effectiveTuningSegments = tuningParameterSegmentCount(effectiveTuningPayload);
  const filteredTuningWatermarkOptions = useMemo(
    () => tuningWatermarkOptions.filter((option) => matchesTuningOption(tuningWatermarkFilter, option)),
    [tuningWatermarkFilter, tuningWatermarkOptions]
  );
  const filteredTuningAttackOptions = useMemo(
    () => tuningAttackOptions.filter((option) => matchesTuningOption(tuningAttackFilter, option)),
    [tuningAttackFilter, tuningAttackOptions]
  );
  const visibleTuningWatermarkMethods = useMemo(
    () => filteredTuningWatermarkOptions.filter((option) => option.available).map((option) => option.method),
    [filteredTuningWatermarkOptions]
  );
  const visibleTuningAttackMethods = useMemo(
    () => filteredTuningAttackOptions.filter((option) => option.available).map((option) => option.method),
    [filteredTuningAttackOptions]
  );
  const selectedTuningWatermarkCount = effectiveTuningPayload.watermarkMethods.length;
  const selectedTuningAttackCount = effectiveTuningPayload.attackMethods.length;
  const tuningStageDetails = useMemo(
    () => buildTuningStageDetails(effectiveTuningPayload, tuningCatalogStats),
    [effectiveTuningPayload, tuningCatalogStats]
  );
  const experimentActive = Boolean(monitorRun);
  const tuningWorkspaceVisible = !experimentActive && Boolean(tuningJob?.id) && (tuningWorkspaceOpen || tuningActive);
  const showEntryCards = !experimentActive && !tuningWorkspaceVisible;
  const showTuningSection = tuningWorkspaceVisible;
  const showExperimentSection = experimentActive;
  const tuningStartDisabled = tuningBusy || tuningActive || effectiveTuningStages === 0;
  const toggleExpandedShard = (shardId: string) => {
    setExpandedShardIds((current) =>
      current.includes(shardId) ? current.filter((item) => item !== shardId) : [...current, shardId]
    );
  };

  useEffect(() => {
    const activeStage = experimentStages.find((stage) => stage.active) ?? experimentStages.find((stage) => stage.reached);
    const selected = experimentStages.find((stage) => stage.key === selectedStageKey);
    if (!activeStage) {
      return;
    }
    if (!selected?.reached || !stageSelectionPinned) {
      setSelectedStageKey(activeStage.key);
    }
  }, [experimentStages, selectedStageKey, stageSelectionPinned]);

  useEffect(() => {
    if (!showExperimentSection || !monitorRun) {
      setGpuTelemetry(null);
      setExpandedShardIds([]);
      return;
    }
    let cancelled = false;
    const loadTelemetry = async () => {
      try {
        const nextTelemetry = await fetchGpuTelemetry();
        if (!cancelled) {
          setGpuTelemetry(nextTelemetry);
        }
      } catch {
        if (!cancelled) {
          setGpuTelemetry(null);
        }
      }
    };
    loadTelemetry();
    const timer = window.setInterval(loadTelemetry, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [monitorRun?.id, showExperimentSection]);

  useEffect(() => {
    try {
      const rawDraft = window.localStorage.getItem(tuningDraftStorageKey);
      if (!rawDraft) {
        return;
      }
      const draft = JSON.parse(rawDraft) as Partial<TuningForm>;
      if (draft.mode !== "quick" && draft.mode !== "full") {
        return;
      }
      const defaults = draft.mode === "full" ? fullTuningDefaults : quickTuningDefaults;
      setTuningForm({ ...defaults, ...draft, mode: draft.mode });
    } catch {
      // Ignore stale local UI drafts.
    }
  }, []);

  const refreshBase = async () => {
    let loadedConfigs: SavedExperimentConfig[] = [];
    let loadedRuns: DemoRunRecord[] = [];
    let latestTuning: ParallelTuningJob | null = null;
    let configsFailed = false;

    try {
      loadedConfigs = await fetchSavedConfigs();
    } catch {
      configsFailed = true;
    }

    try {
      loadedRuns = await fetchManageableRuns();
    } catch {
      loadedRuns = [];
    }

    try {
      latestTuning = await fetchLatestParallelTuning();
    } catch {
      latestTuning = null;
    }

    const manageableRuns = loadedRuns.filter((run) => isActiveRun(run.status));
    setConfigs(loadedConfigs);
    setActiveRuns(manageableRuns);
    if (latestTuning) {
      setTuningJob(latestTuning);
    }
    setSelectedConfigId((current) => {
      if (current && loadedConfigs.some((config) => config.id === current)) {
        return current;
      }
      return loadedConfigs[0]?.id ?? "";
    });
    setSelectedResumeRunId((current) => {
      if (current && manageableRuns.some((run) => run.id === current)) {
        return current;
      }
      return manageableRuns[0]?.id ?? "";
    });

    if (configsFailed) {
      throw new Error("experiment-configs unavailable");
    }
    return { loadedConfigs, manageableRuns };
  };

  useEffect(() => {
    let cancelled = false;
    const loadResourceNames = async () => {
      const [datasetCatalog, algorithms, attacks] = await Promise.all([
        fetchDatasetCatalog().catch(() => null),
        fetchAlgorithms().catch(() => []),
        fetchAttacks().catch(() => [])
      ]);
      if (cancelled) {
        return;
      }
      const nextResourceNames: Record<string, string> = {};
      datasetCatalog?.items.forEach((dataset) => {
        nextResourceNames[dataset.id] = language === "zh" ? dataset.nameZh || dataset.name : dataset.name || dataset.nameZh;
      });
      algorithms.forEach((algorithm) => {
        nextResourceNames[algorithm.id] = resolveWatermarkDisplayName(
          algorithm.method ?? algorithm.id,
          algorithm.name
        );
      });
      attacks.forEach((attack) => {
        const presetLabel = attackCatalogDisplayName(language, attack);
        const methodLabel = attackMethodDisplayName(language, attack.method);
        nextResourceNames[attack.id] = presetLabel;
        nextResourceNames[attack.method] = methodLabel;
        if (attack.executionMethod) {
          nextResourceNames[attack.executionMethod] = attackMethodDisplayName(language, attack.executionMethod);
        }
      });
      const watermarkOptions = buildWatermarkTuningOptions(algorithms);
      const attackOptions = buildAttackTuningOptions(attacks);
      const availableWatermarkMethods = watermarkOptions.filter((option) => option.available).map((option) => option.method);
      const availableAttackMethods = attackOptions.filter((option) => option.available).map((option) => option.method);
      const defaultAttackMethods = attackOptions
        .filter((option) => option.available && !option.viewpoint && !NON_TUNABLE_ATTACK_METHODS.has(option.method))
        .map((option) => option.method);
      setResourceNames(nextResourceNames);
      setTuningCatalogStats(buildTuningCatalogStats(algorithms, attacks));
      setTuningWatermarkOptions(watermarkOptions);
      setTuningAttackOptions(attackOptions);
      setTuningForm((current) => {
        const selectedWatermarkMethods = current.selectedWatermarkMethods.filter((method) =>
          availableWatermarkMethods.includes(method)
        );
        const selectedAttackMethods = normalizeTuningAttackMethods(current.selectedAttackMethods).filter((method) =>
          availableAttackMethods.includes(method)
        );
        const nextWatermarkMethods =
          selectedWatermarkMethods.length > 0 || !current.tuneWatermarks ? selectedWatermarkMethods : availableWatermarkMethods;
        const nextAttackMethods =
          selectedAttackMethods.length > 0 || !current.tuneAttacks ? selectedAttackMethods : defaultAttackMethods;
        return {
          ...current,
          selectedWatermarkMethods: nextWatermarkMethods,
          selectedAttackMethods: nextAttackMethods,
          tuneWatermarks: current.tuneWatermarks && nextWatermarkMethods.length > 0,
          tuneAttacks: current.tuneAttacks && nextAttackMethods.length > 0,
          includeViewpoint3dAttacks: current.includeViewpoint3dAttacks && nextAttackMethods.some(isViewpointTuningMethod)
        };
      });
    };
    loadResourceNames().catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [language]);

  useEffect(() => {
    if (monitorRunId || startDialogOpen || lastSummary) {
      return;
    }
    const runningRun = activeRuns.find((run) => run.status === "running" && !run.cancelRequested);
    if (runningRun) {
      setMonitorRunId(runningRun.id);
      setMonitorRun(runningRun);
      setNotice("");
    }
  }, [activeRuns, lastSummary, monitorRunId, startDialogOpen]);

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      refreshBase()
        .then(() => {
          if (!cancelled) {
            setNotice("");
          }
        })
        .catch(() => {
          if (!cancelled) {
            setNotice(t.runs.apiUnavailable);
          }
        });
    };
    load();
    const timer = window.setInterval(load, autoRefreshSeconds * 1000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [autoRefreshSeconds, t.runs.apiUnavailable]);

  useEffect(() => {
    if (!monitorRunId) {
      setMonitorRun(null);
      setRunState(null);
      return;
    }

    let cancelled = false;
    const loadMonitor = async () => {
      try {
        const [runValue, stateValue] = await Promise.all([
          fetchRun(monitorRunId),
          fetchRunState(monitorRunId).catch(() => null)
        ]);
        if (cancelled) {
          return;
        }
        setMonitorRun(runValue);
        setRunState(stateValue);
        if (isTerminalRun(runValue.status)) {
          const summaryConfig = configs.find((config) => config.id === runValue.configId);
          const note = terminalRunNote(runValue.status, t.runs);
          setLastSummary(makeSummary(runValue, note, { config: summaryConfig, runState: stateValue }));
          setNotice("");
          setSelectedStageKey("summary");
          setStageSelectionPinned(false);
          setCancelConfirmOpen(false);
          refreshBase().catch(() => undefined);
        }
      } catch {
        if (!cancelled) {
          setNotice(t.runs.apiUnavailable);
        }
      }
    };
    loadMonitor();
    const timer = window.setInterval(loadMonitor, autoRefreshSeconds * 1000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [
    configs,
    monitorRunId,
    autoRefreshSeconds,
    t.runs.apiUnavailable,
    t.runs.runCancelledNotice,
    t.runs.runFailedNotice,
    t.runs.runFinishedNotice,
    t.runs.stopSavedNotice
  ]);

  useEffect(() => {
    if (!tuningActive || !tuningJob?.id) {
      return;
    }
    const timer = window.setInterval(() => {
      fetchParallelTuning(tuningJob.id)
        .then(setTuningJob)
        .catch(() => undefined);
    }, autoRefreshSeconds * 1000);
    return () => window.clearInterval(timer);
  }, [autoRefreshSeconds, tuningJob?.id, tuningActive]);

  useEffect(() => {
    if (
      selectedTuningProcessKey !== "all" &&
      !tuningProcessGroups.some((group) => group.key === selectedTuningProcessKey)
    ) {
      setSelectedTuningProcessKey("all");
    }
  }, [selectedTuningProcessKey, tuningProcessGroups]);

  useEffect(() => {
    if (tuningActive) {
      setTuningWorkspaceOpen(true);
    }
  }, [tuningActive]);

  const openStartDialog = () => {
    setStartDialogOpen(true);
    setNotice("");
    setTaskNameInput((current) => current || `${t.runs.defaultTaskName} ${new Date().toLocaleString()}`);
    refreshBase()
      .then(({ manageableRuns }) => {
        if (!manageableRuns.length) {
          setStartMode("new");
        }
      })
      .catch(() => undefined);
  };

  const submitStartDialog = async () => {
    setBusy(true);
    setNotice("");
    try {
      if (startMode === "new") {
        if (!selectedConfig) {
          setNotice(t.runs.configRequired);
          return;
        }
        const trimmedName = taskNameInput.trim();
        if (!trimmedName) {
          setNotice(t.runs.taskNameRequired);
          return;
        }
        const nextRun = await createRun(selectedConfig.id, trimmedName);
	        setMonitorRunId(nextRun.id);
	        setMonitorRun(nextRun);
	        setRunState(null);
	        setLastSummary(null);
        setSelectedStageKey("canonical");
        setStageSelectionPinned(false);
        setStartDialogOpen(false);
        setNotice(t.runs.createdTaskNotice);
      } else {
        if (!selectedResumeRun) {
          setNotice(t.runs.resumeTaskRequired);
          return;
        }
        const updated = isActiveRun(selectedResumeRun.status)
          ? selectedResumeRun
          : await resumeRun(selectedResumeRun.id);
	        setMonitorRunId(updated.id);
	        setMonitorRun(updated);
	        setRunState(null);
	        setLastSummary(null);
        setSelectedStageKey("canonical");
        setStageSelectionPinned(false);
        setStartDialogOpen(false);
        setNotice(t.runs.resumedTaskNotice);
      }
      refreshBase().catch(() => undefined);
    } catch {
      setNotice(startMode === "new" ? t.runs.createTaskFailed : t.runs.resumeTaskFailed);
    } finally {
      setBusy(false);
    }
  };

  const pauseCurrentRun = async () => {
    if (!monitorRun) {
      return;
    }
    setBusy(true);
    try {
      const updated = await pauseRun(monitorRun.id);
      setMonitorRun(updated);
      if (isTerminalRun(updated.status)) {
        const summary = makeSummary(updated, terminalRunNote(updated.status, t.runs), {
          config: monitoredConfig,
          runState
        });
        setLastSummary(summary);
        setNotice("");
        setSelectedStageKey("summary");
        setStageSelectionPinned(false);
      } else {
        setNotice(t.runs.pauseRequestedNotice);
      }
      refreshBase().catch(() => undefined);
    } catch {
      setNotice(t.runs.stopFailed);
    } finally {
      setBusy(false);
    }
  };

  const requestCancelCurrentRun = () => {
    if (!monitorRun) {
      return;
    }
    setCancelConfirmOpen(true);
  };

  const confirmCancelCurrentRun = async () => {
    if (!monitorRun) {
      return;
    }
    setBusy(true);
    try {
      const updated = await cancelRun(monitorRun.id);
      setCancelConfirmOpen(false);
      setMonitorRun(updated);
      if (isTerminalRun(updated.status)) {
        const summary = makeSummary(updated, terminalRunNote(updated.status, t.runs), {
          config: monitoredConfig,
          runState
        });
        setLastSummary(summary);
        setNotice("");
        setSelectedStageKey("summary");
        setStageSelectionPinned(false);
      } else {
        setNotice(t.runs.cancelRequestedNotice);
      }
      refreshBase().catch(() => undefined);
    } catch {
      setNotice(t.runs.cancelFailed);
    } finally {
      setBusy(false);
    }
  };

  const resumeSummaryRun = async () => {
    if (!lastSummary) {
      return;
    }
    setBusy(true);
    setNotice("");
    try {
      const updated = await resumeRun(lastSummary.runId);
      setMonitorRunId(updated.id);
      setMonitorRun(updated);
      setLastSummary(null);
      setSelectedStageKey("canonical");
      setStageSelectionPinned(false);
      setStartDialogOpen(false);
      setNotice(t.runs.resumedTaskNotice);
      refreshBase().catch(() => undefined);
    } catch {
      setNotice(t.runs.resumeTaskFailed);
    } finally {
      setBusy(false);
    }
  };

  const openTuningDialog = () => {
    setTuningDialogOpen(true);
    setTuningNotice("");
    setTuningDraftNotice("");
  };

  const setTuningMode = (mode: TuningMode) => {
    setTuningForm((current) => {
      const defaults = mode === "full" ? fullTuningDefaults : quickTuningDefaults;
      return {
        ...defaults,
        selectedWatermarkMethods: current.selectedWatermarkMethods,
        selectedAttackMethods: normalizeTuningAttackMethods(current.selectedAttackMethods),
        selectedQualityMetrics: current.selectedQualityMetrics.length ? current.selectedQualityMetrics : allTuningQualityMetrics,
        tuneWatermarks: current.selectedWatermarkMethods.length > 0,
        tuneAttacks: normalizeTuningAttackMethods(current.selectedAttackMethods).length > 0,
        tuneQuality: current.selectedQualityMetrics.length > 0,
        includeViewpoint3dAttacks: normalizeTuningAttackMethods(current.selectedAttackMethods).some(isViewpointTuningMethod)
      };
    });
  };

  const updateTuningForm = (updates: Partial<TuningForm>) => {
    setTuningForm((current) => ({ ...current, ...updates }));
    setTuningDraftNotice("");
  };

  const updateTuningStage = (key: TuningStageDetail["key"], checked: boolean) => {
    if (key === "watermarks") {
      updateTuningForm({
        tuneWatermarks: checked,
        selectedWatermarkMethods: checked
          ? addIds(tuningForm.selectedWatermarkMethods, tuningWatermarkOptions.filter((option) => option.available).map((option) => option.method))
          : []
      });
      return;
    }
    if (key === "attacks") {
      const nextAttackMethods = checked
        ? addIds(tuningForm.selectedAttackMethods, tuningAttackOptions.filter((option) => option.available).map((option) => option.method))
        : [];
      updateTuningForm({
        tuneAttacks: checked,
        selectedAttackMethods: nextAttackMethods,
        includeViewpoint3dAttacks: nextAttackMethods.some(isViewpointTuningMethod)
      });
      return;
    }
    updateTuningForm({
      tuneQuality: checked,
      selectedQualityMetrics: checked
        ? (tuningForm.selectedQualityMetrics.length ? tuningForm.selectedQualityMetrics : allTuningQualityMetrics)
        : []
    });
  };

  const updateSelectedQualityMetrics = (metric: string, checked: boolean) => {
    const nextMetrics = checked
      ? addIds(tuningForm.selectedQualityMetrics, [metric])
      : tuningForm.selectedQualityMetrics.filter((item) => item !== metric);
    updateTuningForm({
      selectedQualityMetrics: allTuningQualityMetrics.filter((item) => nextMetrics.includes(item)),
      tuneQuality: nextMetrics.length > 0
    });
  };

  const updateSelectedWatermarkMethods = (methods: string[]) => {
    updateTuningForm({
      selectedWatermarkMethods: methods,
      tuneWatermarks: methods.length > 0
    });
  };

  const updateSelectedAttackMethods = (methods: string[]) => {
    const normalizedMethods = normalizeTuningAttackMethods(methods);
    const hasViewpointMethod = normalizedMethods.some(isViewpointTuningMethod);
    updateTuningForm({
      selectedAttackMethods: normalizedMethods,
      tuneAttacks: normalizedMethods.length > 0,
      includeViewpoint3dAttacks: hasViewpointMethod
    });
  };

  const saveTuningDraft = () => {
    try {
      window.localStorage.setItem(tuningDraftStorageKey, JSON.stringify(tuningForm));
      setTuningDraftNotice("配置已保存到本地草稿。");
    } catch {
      setTuningDraftNotice("保存失败，请检查浏览器存储权限。");
    }
  };

  const submitTuningDialog = async () => {
    setTuningBusy(true);
    setTuningNotice("");
    try {
      const payload = buildTuningPayload(tuningForm);
      const job = await startParallelTuning(payload);
      setTuningJob(job);
      setTuningWorkspaceOpen(true);
      setTuningDialogOpen(false);
      setTuningNotice(
        `调参任务已启动，single pass sampleCount=${payload.sampleCount}，最大 batch=${payload.maxBatchSize}。`
      );
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (error) {
      setTuningNotice(error instanceof Error ? error.message : "调参任务启动失败。");
    } finally {
      setTuningBusy(false);
    }
  };

  const closeTuningWorkspace = () => {
    if (tuningActive) {
      return;
    }
    setTuningWorkspaceOpen(false);
  };

  const persistTuning = async () => {
    if (!tuningJob?.id) {
      return;
    }
    setTuningBusy(true);
    setTuningNotice("");
    try {
      const saved = await saveParallelTuning(tuningJob.id);
      setTuningNotice(
        `已保存 ${saved.savedKeys.length} 个参数到 ${saved.envPath}，并已写入运行时配置，后续实验会直接使用。`
      );
      setTuningJob(await fetchParallelTuning(tuningJob.id));
    } catch (error) {
      setTuningNotice(error instanceof Error ? error.message : "保存参数失败。");
    } finally {
      setTuningBusy(false);
    }
  };

  const pauseTuning = async () => {
    if (!tuningJob?.id) {
      return;
    }
    setTuningBusy(true);
    setTuningNotice("");
    try {
      const updated = await pauseParallelTuning(tuningJob.id);
      setTuningJob(updated);
      setTuningNotice(
        updated.status === "paused"
          ? "调参任务已暂停，可稍后从已保存检查点继续。"
          : "暂停请求已发送，当前候选完成清理后会保存检查点。"
      );
    } catch (error) {
      setTuningNotice(error instanceof Error ? error.message : "暂停调参失败。");
    } finally {
      setTuningBusy(false);
    }
  };

  const resumeTuning = async () => {
    if (!tuningJob?.id) {
      return;
    }
    setTuningBusy(true);
    setTuningNotice("");
    try {
      const updated = await resumeParallelTuning(tuningJob.id);
      setTuningJob(updated);
      setTuningWorkspaceOpen(true);
      setTuningNotice("调参任务已继续，将跳过已完成的算法记录。");
    } catch (error) {
      setTuningNotice(error instanceof Error ? error.message : "继续调参失败。");
    } finally {
      setTuningBusy(false);
    }
  };

  const stopTuning = async () => {
    if (!tuningJob?.id) {
      return;
    }
    setTuningBusy(true);
    setTuningNotice("");
    try {
      const updated = await cancelParallelTuning(tuningJob.id);
      setTuningJob(updated);
      setTuningNotice(
        updated.status === "cancelled"
          ? "调参任务已停止，已完成的候选记录保留在该任务目录中。"
          : "停止请求已发送，当前候选完成清理后会结束调参。"
      );
    } catch (error) {
      setTuningNotice(error instanceof Error ? error.message : "停止调参失败。");
    } finally {
      setTuningBusy(false);
    }
  };

  const closeSummaryDialog = () => {
    const summaryRunId = lastSummary?.runId;
    setLastSummary(null);
    setNotice("");
    setMonitorRunId("");
    setMonitorRun(null);
    setStartDialogOpen(false);
    if (summaryRunId) {
      setActiveRuns((current) => current.filter((run) => run.id !== summaryRunId || (run.status === "running" && !run.cancelRequested)));
    }
    refreshBase().catch(() => undefined);
  };

  const formatOptionalDate = (value?: string | null) => (value ? localizedDate(language, value) : "n/a");
  const statusLabels = t.common.status as Record<string, string>;
  const currentStopNotice = monitorRun ? stopIntentNotice(monitorRun, t.runs) : null;
  const inlineSummary = lastSummary && monitorRun && lastSummary.runId === monitorRun.id ? lastSummary : null;
  const startActionDisabled =
    busy ||
    (startMode === "new"
      ? !selectedConfig || !taskNameInput.trim()
      : !selectedResumeRun || !isActiveRun(selectedResumeRun.status));

  return (
    <AppShell active="runs">
      <div className="topbar">
        <div className="title-block">
          <h1>{t.runs.title}</h1>
          <p>{t.runs.subtitle}</p>
        </div>
        <div className="toolbar">
          <label className="select-button refresh-slider-control">
            <span>{t.console.autoRefresh}</span>
            <input
              className="refresh-slider"
              max={30}
              min={1}
              onChange={(event) => setAutoRefreshSeconds(Number(event.target.value))}
              step={1}
              type="range"
              value={autoRefreshSeconds}
            />
            <strong>{autoRefreshSeconds}s</strong>
          </label>
          <button className="button" onClick={() => refreshBase()} title={t.common.updated} type="button">
            <RefreshCw size={16} />
          </button>
        </div>
      </div>

      {showEntryCards ? (
        <div className="run-entry-grid">
          <section className="panel run-start-panel run-tuning-start-panel">
            <div className="panel-body run-start-body">
              <div className="run-start-copy">
                <h2>并行参数自动调优</h2>
                <p>先选择调参规则与算法范围，启动后再进入搜索进度、吞吐量趋势和推荐参数面板。</p>
              </div>
              <div className="run-start-actions">
                <button className="button primary run-start-button" disabled={tuningBusy} onClick={openTuningDialog} type="button">
                  <SlidersHorizontal size={18} />
                  配置并开始调参
                </button>
                {tuningJob?.id ? (
                  <button
                    className="button run-start-button"
                    disabled={tuningBusy}
                    onClick={() => {
                      setTuningWorkspaceOpen(true);
                      setTuningNotice("");
                    }}
                    type="button"
                  >
                    <LineChart size={18} />
                    查看最近结果
                  </button>
                ) : null}
              </div>
              {tuningNotice ? <div className="risk ok">{tuningNotice}</div> : null}
            </div>
          </section>

          <section className="panel run-start-panel">
            <div className="panel-body run-start-body">
              <div className="run-start-copy">
                <h2>{t.runs.startExperiment}</h2>
                <p>{t.runs.startExperimentHint}</p>
              </div>
              <div className="run-start-actions">
                <button className="button primary run-start-button" onClick={openStartDialog} type="button">
                  <PlayCircle size={18} />
                  {t.runs.startExperiment}
                </button>
              </div>

              {notice ? <div className="risk ok">{notice}</div> : null}
            </div>
          </section>
        </div>
      ) : null}

      {showTuningSection ? (
        <section className="panel run-tuning-panel">
          <div className="panel-header">
            <div>
              <h2>并行参数自动调优</h2>
              <p>
                {tuningJob
                  ? `${tuningJob.id} · ${tuningJob.message ?? tuningJob.status}`
                  : "在开始实验之外单独搜索 batch size 与 CPU worker 参数。"}
              </p>
            </div>
          </div>
          <div className="panel-body run-tuning-body">
          <div className="run-tuning-toolbar">
            <button className="button primary" disabled={tuningBusy || tuningActive} onClick={openTuningDialog} type="button">
              <SlidersHorizontal size={16} />
              开始调参
            </button>
            <button
              className="button"
              disabled={tuningBusy || tuningJob?.status !== "succeeded" || tuningEnvEntries.length === 0}
              onClick={persistTuning}
              type="button"
            >
              <Save size={16} />
              保存参数
            </button>
            <button className="button" disabled={tuningBusy || !tuningRunning} onClick={pauseTuning} type="button">
              <PauseCircle size={16} />
              {tuningPausing ? "正在暂停" : "暂停调参"}
            </button>
            <button className="button" disabled={tuningBusy || !tuningPaused} onClick={resumeTuning} type="button">
              <PlayCircle size={16} />
              继续调参
            </button>
            <button
              className="button danger"
              disabled={tuningBusy || (!tuningRunning && !tuningPausing && !tuningPaused)}
              onClick={stopTuning}
              type="button"
            >
              <Square size={16} />
              {tuningStopping ? "正在停止" : "停止调参"}
            </button>
            <button
              className="button"
              disabled={!tuningJob?.id}
              onClick={() => tuningJob?.id && fetchParallelTuning(tuningJob.id).then(setTuningJob)}
              type="button"
            >
              <RefreshCw size={16} />
              刷新调参
            </button>
            <button className="button" disabled={tuningActive} onClick={closeTuningWorkspace} type="button">
              返回入口
            </button>
          </div>

          <div className="run-tuning-progress-card">
            <div className="run-overview-title">
              <span>搜索进度</span>
              <strong>{tuningJob?.progress ?? 0}%</strong>
            </div>
            <div className="progress-track run-progress-large">
              <div className="progress-bar" style={{ width: progressWidth(tuningJob?.progress ?? 0) }} />
            </div>
            <p>{tuningJob?.message ?? "等待启动调参任务。"}</p>
          </div>

          {tuningNotice ? <div className="risk ok">{tuningNotice}</div> : null}
          {tuningJob?.error ? <div className="risk error">{tuningJob.error}</div> : null}

          <div className="run-tuning-grid">
            <section className="run-tuning-card chart">
              <div className="run-tuning-card-head">
                <div>
                  <strong>吞吐量趋势</strong>
                  <span>{tuningChartScope}</span>
                </div>
                <LineChart size={17} />
              </div>
              <ThroughputChart points={visibleTuningChartPoints} />
            </section>

            <section className="run-tuning-card">
              <div className="run-tuning-card-head">
                <div>
                  <strong>运行过程</strong>
                  <span>点击过程后，左侧只显示对应吞吐量趋势。</span>
                </div>
                <Zap size={17} />
              </div>
              {tuningProcessGroups.length ? (
                <div className="run-tuning-process-list">
                  <button
                    className={selectedTuningProcessKey === "all" ? "run-tuning-process active" : "run-tuning-process"}
                    onClick={() => setSelectedTuningProcessKey("all")}
                    type="button"
                  >
                    <span>
                      <strong>全部候选点</strong>
                      <small>{tuningChartPoints.length} 次测量 · 查看整体趋势</small>
                    </span>
                    <em>{Math.max(0, ...tuningChartPoints.map((point) => point.throughput)).toFixed(1)} img/s</em>
                  </button>
                  {tuningProcessGroups.map((group) => (
                    <button
                      className={selectedTuningProcessKey === group.key ? "run-tuning-process active" : "run-tuning-process"}
                      key={group.key}
                      onClick={() => setSelectedTuningProcessKey(group.key)}
                      type="button"
                    >
                      <span>
                        <strong>{group.label}</strong>
                        <small>
                          {group.description}
                          {group.failedCount > 0 ? ` · ${group.failedCount} 个失败` : ""}
                        </small>
                      </span>
                      <em>{group.bestThroughput.toFixed(1)} img/s</em>
                    </button>
                  ))}
                </div>
              ) : (
                <RunEmptyState
                  description="每个候选配置完成后会追加耗时、吞吐量和推荐结果。"
                  title="暂无调参事件"
                  variant="events"
                />
              )}
            </section>
          </div>

          <section className="run-tuning-card env">
            <div className="run-tuning-card-head">
              <div>
                <strong>推荐参数</strong>
                <span>{tuningJob?.summary?.reportPath ? `报告：${tuningJob.summary.reportPath}` : "调参完成后生成可保存的 .env 参数。"}</span>
              </div>
            </div>
            {tuningEnvEntries.length ? (
              <div className="env-suggestion-list run-env-suggestion-list">
                {tuningEnvEntries.map(([key, value]) => (
                  <div key={key} className="env-suggestion-row">
                    <code>{key}</code>
                    <span>{value}</span>
                  </div>
                ))}
              </div>
            ) : (
              <RunEmptyState
                description="调参完成后会生成可保存到 .env 的参数摘要。"
                title="等待推荐参数"
                variant="env"
              />
            )}
          </section>
          </div>
        </section>
      ) : null}

      {showExperimentSection && monitorRun ? (
        <section className="panel run-execution-panel">
          <div className="panel-body run-execution-body">
            <div className="run-monitor-toolbar">
              <div>
                <span>{t.runs.selectedTask}</span>
                <strong>{taskName(monitorRun)}</strong>
                <code>{monitorRun.id}</code>
                {currentStopNotice ? <small className="run-pause-state">{currentStopNotice}</small> : null}
              </div>
              <div className="run-monitor-actions">
                <span className={badgeClass(monitorRun.status)}>{runStatusLabel(monitorRun.status, statusLabels)}</span>
                {isPausableRun(monitorRun) ? (
                  <button className="button" disabled={busy} onClick={pauseCurrentRun} type="button">
                    <PauseCircle size={15} />
                    {t.runs.pauseAndSave}
                  </button>
                ) : null}
                {isCancellableRun(monitorRun) ? (
                  <button className="button danger" disabled={busy || monitorRun.stopIntent === "cancel"} onClick={requestCancelCurrentRun} type="button">
                    <XCircle size={15} />
                    {t.runs.cancelExperiment}
                  </button>
                ) : null}
              </div>
            </div>

            <StageTimeline
              onSelect={(key) => {
                setSelectedStageKey(key);
                setStageSelectionPinned(true);
              }}
	              overallProgress={runState?.overallProgress ?? monitorRun.progress}
              selectedKey={selectedStage?.key ?? "canonical"}
              stages={experimentStages}
            />

            {notice ? <div className="risk ok">{notice}</div> : null}
            {monitorRun.error ? <div className="risk error">{monitorRun.error}</div> : null}

            <div className="run-stage-content-grid">
              <section className="run-stage-detail-panel">
                <div className="run-stage-detail-head">
                  <div>
                    <span>{selectedStage?.phase?.key ?? selectedStage?.key ?? "n/a"}</span>
                    <strong>{selectedStage?.label ?? t.runs.waitingForStage}</strong>
                  </div>
                </div>
                <StageCellProgressMeter progress={selectedStageProgress} />

                {selectedStage?.key === "summary" && inlineSummary ? (
                  <InlineRunSummary
                    busy={busy}
                    formatOptionalDate={formatOptionalDate}
                    language={language}
                    onExit={closeSummaryDialog}
                    onResume={resumeSummaryRun}
                    statusLabels={statusLabels}
                    summary={inlineSummary}
                    t={t}
                  />
	                ) : (
	                  <PhaseDetailPanel
                      expandedShardIds={expandedShardIds}
                      gpuTelemetry={gpuTelemetry}
                      language={language}
                      onToggleShard={toggleExpandedShard}
	                    resourceNames={resourceNames}
                      shards={selectedStageShards}
	                    stage={selectedStage}
	                  />
	                )}
              </section>
            </div>
          </div>
        </section>
      ) : null}

      {tuningDialogOpen ? (
        <div className="modal-backdrop" role="presentation">
          <div aria-modal="true" className="config-modal tuning-config-modal" role="dialog">
            <div className="modal-header tuning-modal-header">
              <div>
                <h2>并行参数搜索</h2>
                <p>配置搜索范围与搜索模式。sampleCount 会自动不小于最大 batch。</p>
              </div>
              <button className="icon-button tuning-close-button" onClick={() => setTuningDialogOpen(false)} title="关闭" type="button">
                ×
              </button>
            </div>
            <div className="modal-body tuning-config-body">
              <div className="run-mode-grid tuning-mode-grid">
                <button
                  className={tuningForm.mode === "quick" ? "run-mode-card tuning-mode-card selected" : "run-mode-card tuning-mode-card"}
                  onClick={() => setTuningMode("quick")}
                  type="button"
                >
                  <span className="tuning-mode-icon">
                    <SlidersHorizontal size={18} />
                  </span>
                  <strong>快速模式</strong>
                  <span>每个候选只测一轮，适合快速得到一版可用参数。</span>
                </button>
                <button
                  className={tuningForm.mode === "full" ? "run-mode-card tuning-mode-card selected" : "run-mode-card tuning-mode-card"}
                  onClick={() => setTuningMode("full")}
                  type="button"
                >
                  <span className="tuning-mode-icon">
                    <BarChart3 size={18} />
                  </span>
                  <strong>完整模式</strong>
                  <span>扩大候选范围并检测边界，但每个候选仍只测一轮。</span>
                </button>
              </div>

              <div className="tuning-config-grid">
                <section className="run-dialog-section">
                  <h3>样本与测量</h3>
                  <div className="field-grid">
                    <div className="field">
                      <label htmlFor="tuning-samples">sampleCount</label>
                      <div className="tuning-input-wrap">
                        <input
                          id="tuning-samples"
                          min={2}
                          onChange={(event) => updateTuningForm({ sampleCount: Number(event.target.value) })}
                          type="number"
                          value={tuningForm.sampleCount}
                        />
                        <span>count</span>
                      </div>
                    </div>
                    <div className="field">
                      <label htmlFor="tuning-warmup">warmupCount</label>
                      <div className="tuning-input-wrap">
                        <input
                          id="tuning-warmup"
                          min={1}
                          onChange={(event) => updateTuningForm({ warmupCount: Number(event.target.value) })}
                          type="number"
                          value={tuningForm.warmupCount}
                        />
                        <span>count</span>
                      </div>
                    </div>
                    <div className="field">
                      <label htmlFor="tuning-candidate-batches">每候选 batch 数</label>
                      <div className="tuning-input-wrap">
                        <input
                          id="tuning-candidate-batches"
                          min={1}
                          onChange={(event) => updateTuningForm({ candidateBatchCount: Number(event.target.value) })}
                          type="number"
                          value={tuningForm.candidateBatchCount}
                        />
                        <span>batches</span>
                      </div>
                    </div>
                    <div className="field">
                      <label htmlFor="tuning-improve">最小提升比例</label>
                      <div className="tuning-input-wrap">
                        <input
                          id="tuning-improve"
                          min={0}
                          onChange={(event) => updateTuningForm({ minImprovementRatio: Number(event.target.value) })}
                          step={0.01}
                          type="number"
                          value={tuningForm.minImprovementRatio}
                        />
                        <span>ratio</span>
                      </div>
                    </div>
                  </div>
                </section>

                <section className="run-dialog-section">
                  <h3>搜索范围</h3>
                  <div className="field-grid">
                    <div className="field">
                      <label htmlFor="tuning-batch-min">最小 batch</label>
                      <div className="tuning-input-wrap">
                        <input
                          id="tuning-batch-min"
                          min={1}
                          onChange={(event) => updateTuningForm({ minBatchSize: Number(event.target.value) })}
                          type="number"
                          value={tuningForm.minBatchSize}
                        />
                        <span>batch</span>
                      </div>
                    </div>
                    <div className="field">
                      <label htmlFor="tuning-batch-max">最大 batch</label>
                      <div className="tuning-input-wrap">
                        <input
                          id="tuning-batch-max"
                          min={1}
                          onChange={(event) => updateTuningForm({ maxBatchSize: Number(event.target.value) })}
                          type="number"
                          value={tuningForm.maxBatchSize}
                        />
                        <span>batch</span>
                      </div>
                    </div>
                    <div className="field">
                      <label htmlFor="tuning-worker-min">最小 workers</label>
                      <div className="tuning-input-wrap">
                        <input
                          id="tuning-worker-min"
                          min={1}
                          onChange={(event) => updateTuningForm({ minWorkerCount: Number(event.target.value) })}
                          type="number"
                          value={tuningForm.minWorkerCount}
                        />
                        <span>workers</span>
                      </div>
                    </div>
                    <div className="field">
                      <label htmlFor="tuning-worker-max">最大 workers</label>
                      <div className="tuning-input-wrap">
                        <input
                          id="tuning-worker-max"
                          min={1}
                          onChange={(event) => updateTuningForm({ maxWorkerCount: Number(event.target.value) })}
                          type="number"
                          value={tuningForm.maxWorkerCount}
                        />
                        <span>workers</span>
                      </div>
                    </div>
                  </div>
                </section>

                <section className="run-dialog-section tuning-stage-section">
                  <div className="tuning-section-heading">
                    <div>
                      <h3>调参阶段</h3>
                      <p>按调参完成后会写入的参数分项组织，便于确认每个阶段影响正式实验的哪一部分。</p>
                    </div>
                    <span>{effectiveTuningStages} 阶段 · {effectiveTuningSegments} 分项</span>
                  </div>
                  <div className="tuning-stage-detail-grid">
                    {tuningStageDetails.map((detail) => {
                      const hasMethodSelector = detail.key === "watermarks" || detail.key === "attacks";
                      return (
                        <section
                          className={[
                            "tuning-stage-detail",
                            detail.checked ? "enabled" : "muted",
                            detail.disabled ? "disabled" : "",
                            hasMethodSelector ? "with-selector" : "",
                            detail.key === "quality" ? "full-row" : ""
                          ]
                            .filter(Boolean)
                            .join(" ")}
                          key={detail.key}
                        >
                          <div className="tuning-stage-detail-switch">
                            <input
                              checked={detail.checked}
                              disabled={detail.disabled}
                              onChange={(event) => updateTuningStage(detail.key, event.target.checked)}
                              type="checkbox"
                            />
                          </div>
                          <div className="tuning-stage-detail-copy">
                            <div className="tuning-stage-title-line">
                              <strong>{detail.title}</strong>
                              <em>{detail.badge}</em>
                            </div>
                            <small>{detail.description}</small>
                            {detail.key === "watermarks" ? (
                              <TuningMethodSelector
                                description="名称与资源页水印算法列表保持一致。"
                                embedded
                                filter={tuningWatermarkFilter}
                                onFilterChange={setTuningWatermarkFilter}
                                onSelectMethods={updateSelectedWatermarkMethods}
                                options={filteredTuningWatermarkOptions}
                                searchPlaceholder="搜索水印算法、方法或类别"
                                selectedMethods={tuningForm.selectedWatermarkMethods}
                                title="水印算法"
                                totalCount={tuningWatermarkOptions.length}
                                visibleMethods={visibleTuningWatermarkMethods}
                              />
                            ) : null}
                            {detail.key === "attacks" ? (
                              <TuningMethodSelector
                                description="名称与资源页攻击算法列表保持一致，3D 视角重渲染也在这里选择。"
                                embedded
                                filter={tuningAttackFilter}
                                onFilterChange={setTuningAttackFilter}
                                onSelectMethods={updateSelectedAttackMethods}
                                options={filteredTuningAttackOptions}
                                searchPlaceholder="搜索攻击算法、方法或类别"
                                selectedMethods={tuningForm.selectedAttackMethods}
                                title="攻击算法"
                                totalCount={tuningAttackOptions.length}
                                visibleMethods={visibleTuningAttackMethods}
                              />
                            ) : null}
                            {detail.key === "quality" ? (
                              <div className="tuning-quality-metrics">
                                {tuningQualityMetricOptions.map((metric) => (
                                  <label className="tuning-quality-metric" key={metric.id}>
                                    <input
                                      checked={tuningForm.selectedQualityMetrics.includes(metric.id)}
                                      onChange={(event) => updateSelectedQualityMetrics(metric.id, event.target.checked)}
                                      type="checkbox"
                                    />
                                    <span>
                                      <strong>{metric.label}</strong>
                                      <small>{metric.kind === "batch" ? "batch size" : "CPU workers"}</small>
                                    </span>
                                  </label>
                                ))}
                              </div>
                            ) : null}
                          </div>
                        </section>
                      );
                    })}
                  </div>
                </section>

                <section className="run-dialog-section tuning-estimate-section">
                  <h3>搜索预估</h3>
                  <div className="tuning-estimate-list">
                    <div>
                      <span>当前模式</span>
                      <strong>{effectiveTuningPayload.mode === "quick" ? "快速模式" : "完整模式"}</strong>
                    </div>
                    <div>
                      <span>候选 batch</span>
                      <strong>{effectiveTuningPayload.batchCandidates.length} 个</strong>
                    </div>
                    <div>
                      <span>候选 workers</span>
                      <strong>{effectiveTuningPayload.workerCandidates.length} 个</strong>
                    </div>
                    <div>
                      <span>粗略测量</span>
                      <strong>约 {effectiveTuningCombinations} 组</strong>
                    </div>
                    <div>
                      <span>水印 method</span>
                      <strong>{selectedTuningWatermarkCount} 个</strong>
                    </div>
                    <div>
                      <span>攻击 method</span>
                      <strong>{selectedTuningAttackCount} 个</strong>
                    </div>
                    <div>
                      <span>调参阶段</span>
                      <strong>{effectiveTuningStages} 个</strong>
                    </div>
                    <div>
                      <span>输出分项</span>
                      <strong>{effectiveTuningSegments} 类</strong>
                    </div>
                    <div>
                      <span>粗略耗时</span>
                      <strong>{tuningEstimateText(effectiveTuningPayload)}</strong>
                    </div>
                    <div>
                      <span>sampleCount 已校正</span>
                      <strong>{effectiveTuningPayload.sampleCount}</strong>
                    </div>
                    <div>
                      <span>每候选 batch 数</span>
                      <strong>{effectiveTuningPayload.candidateBatchCount}</strong>
                    </div>
                  </div>
                  <p className="tuning-estimate-note">
                    实际执行会以后端检测到的算法能力为准；batch 搜索按每个候选固定 batch 次数取样，workers 搜索按 sampleCount 取样。
                  </p>
                </section>

                <section className="run-dialog-section tuning-effective-section">
                  <h3>实际提交参数 Preview</h3>
                  <div className="tuning-preview-grid">
                    <div className="tuning-preview-stat">
                      <span>sampleCount</span>
                      <strong>{effectiveTuningPayload.sampleCount}</strong>
                    </div>
                    <div className="tuning-preview-stat">
                      <span>最大 batch</span>
                      <strong>{effectiveTuningPayload.maxBatchSize}</strong>
                    </div>
                    <div className="tuning-preview-stat">
                      <span>最大 workers</span>
                      <strong>{effectiveTuningPayload.maxWorkerCount}</strong>
                    </div>
                    <div className="tuning-preview-stat">
                      <span>水印 method</span>
                      <strong>{selectedTuningWatermarkCount}</strong>
                    </div>
                    <div className="tuning-preview-stat">
                      <span>攻击 method</span>
                      <strong>{selectedTuningAttackCount}</strong>
                    </div>
                    <div className="tuning-preview-stat">
                      <span>每候选测量</span>
                      <strong>{effectiveTuningPayload.candidateBatchCount} batches</strong>
                    </div>
                    <div className="tuning-preview-stat">
                      <span>搜索策略</span>
                      <strong>single pass</strong>
                    </div>
                    <div className="tuning-preview-stat">
                      <span>3D 攻击</span>
                      <strong>{effectiveTuningPayload.includeViewpoint3dAttacks ? "包含" : "排除"}</strong>
                    </div>
                    <div className="tuning-preview-stat">
                      <span>输出分项</span>
                      <strong>{effectiveTuningSegments}</strong>
                    </div>
                  </div>
                </section>
              </div>
            </div>
            <div className="modal-footer tuning-modal-footer">
              <span className={tuningDraftNotice ? "tuning-footer-note saved" : "tuning-footer-note"}>
                {tuningDraftNotice || `粗略预计约 ${effectiveTuningCombinations} 组候选参数，覆盖 ${effectiveTuningSegments} 类输出分项`}
              </span>
              <div className="toolbar">
                <button className="button" onClick={() => setTuningDialogOpen(false)} type="button">
                  取消
                </button>
                <button className="button" disabled={tuningBusy} onClick={saveTuningDraft} type="button">
                  <Save size={16} />
                  保存配置
                </button>
                <button className="button primary" disabled={tuningStartDisabled} onClick={submitTuningDialog} type="button">
                  <Zap size={16} />
                  开始搜索
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {startDialogOpen ? (
        <div className="modal-backdrop" role="presentation">
          <div aria-modal="true" className="config-modal run-start-modal" role="dialog">
            <div className="modal-header">
              <div>
                <h2>{t.runs.startDialogTitle}</h2>
                <p>{t.runs.startDialogHint}</p>
              </div>
              <button className="icon-button" onClick={() => setStartDialogOpen(false)} title={t.runs.cancel} type="button">
                ×
              </button>
            </div>
            <div className="modal-body run-start-modal-body">
              <div className="run-mode-grid">
                <button
                  className={startMode === "new" ? "run-mode-card selected" : "run-mode-card"}
                  onClick={() => setStartMode("new")}
                  type="button"
                >
                  <PlayCircle size={18} />
                  <strong>{t.runs.newTask}</strong>
                  <span>{t.runs.newTaskHint}</span>
                </button>
                <button
                  className={startMode === "resume" ? "run-mode-card selected" : "run-mode-card"}
                  disabled={!activeRuns.length}
                  onClick={() => setStartMode("resume")}
                  type="button"
                >
                  <RotateCcw size={18} />
                  <strong>{t.runs.continueTask}</strong>
                  <span>{t.runs.continueTaskHint}</span>
                </button>
              </div>

              {startMode === "new" ? (
                <div className="run-dialog-section">
                  <div className="field">
                    <label htmlFor="run-task-name">{t.runs.taskName}</label>
                    <input
                      id="run-task-name"
                      onChange={(event) => setTaskNameInput(event.target.value)}
                      placeholder={t.runs.taskNamePlaceholder}
                      value={taskNameInput}
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="run-config">{t.runs.selectConfig}</label>
                    <select
                      id="run-config"
                      onChange={(event) => setSelectedConfigId(event.target.value)}
                      value={selectedConfigId}
                    >
                      {configs.map((config) => (
                        <option key={config.id} value={config.id}>
                          {config.name}
                        </option>
                      ))}
                    </select>
                  </div>
                  {selectedConfig ? <ConfigPreview config={selectedConfig} t={t} /> : <div className="empty">{t.runs.noConfigs}</div>}
                </div>
              ) : (
                <div className="run-dialog-section">
                  <div className="run-resume-list">
                    {activeRuns.length ? (
                      activeRuns.map((run) => (
                        <button
                          className={run.id === selectedResumeRunId ? "run-resume-card selected" : "run-resume-card"}
                          key={run.id}
                          onClick={() => setSelectedResumeRunId(run.id)}
                          type="button"
                        >
                          <div>
                            <strong>{taskName(run)}</strong>
                            <span>{run.configName}</span>
                            <code>{run.id}</code>
                          </div>
                          <div>
                            <span className={badgeClass(run.status)}>{runStatusLabel(run.status, statusLabels)}</span>
                            <small>{run.progress}%</small>
                          </div>
                        </button>
                      ))
                    ) : (
                      <div className="empty">{t.runs.noUnfinishedTasks}</div>
                    )}
                  </div>
                </div>
              )}
            </div>
            <div className="modal-footer">
              <button className="button" onClick={() => setStartDialogOpen(false)} type="button">
                {t.runs.cancel}
              </button>
              <button className="button primary" disabled={startActionDisabled} onClick={submitStartDialog} type="button">
                {startMode === "resume" ? <RotateCcw size={16} /> : <PlayCircle size={16} />}
                {startMode === "resume" ? t.runs.resume : t.runs.beginExecution}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {cancelConfirmOpen && monitorRun ? (
        <div className="modal-backdrop" role="presentation">
          <div aria-modal="true" className="config-modal run-confirm-modal" role="dialog">
            <div className="modal-header">
              <div>
                <h2>{t.runs.cancelConfirmTitle}</h2>
                <p>{taskName(monitorRun)}</p>
              </div>
              <button className="icon-button" onClick={() => setCancelConfirmOpen(false)} title={t.runs.cancel} type="button">
                ×
              </button>
            </div>
            <div className="modal-body run-confirm-body">
              <div className="risk warn">{t.runs.cancelConfirmBody}</div>
              <code>{monitorRun.id}</code>
            </div>
            <div className="modal-footer">
              <button className="button" disabled={busy} onClick={() => setCancelConfirmOpen(false)} type="button">
                {t.runs.keepRunning}
              </button>
              <button className="button danger" disabled={busy} onClick={confirmCancelCurrentRun} type="button">
                <XCircle size={16} />
                {t.runs.confirmCancelExperiment}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {lastSummary && !inlineSummary ? (
        <div className="modal-backdrop" role="presentation">
          <div aria-modal="true" className="config-modal run-summary-modal" role="dialog">
            <div className="modal-header">
              <div>
                <h2>{t.runs.executionSummary}</h2>
                <p>{lastSummary.note}</p>
              </div>
              <button className="icon-button" onClick={closeSummaryDialog} title={t.runs.closeSummary} type="button">
                ×
              </button>
            </div>
            <div className="modal-body run-summary-modal-body">
              <div className="run-summary-head">
                <div>
                  <strong>{lastSummary.taskName}</strong>
                  <code>{lastSummary.runId}</code>
                </div>
                <span className={badgeClass(lastSummary.status)}>{runStatusLabel(lastSummary.status, statusLabels)}</span>
              </div>
              <div className="run-meta-grid">
                <Metric label={t.common.progress} value={`${lastSummary.progress}%`} />
                <Metric label={t.runs.duration} value={formatDurationMs(lastSummary.durationMs, language)} />
                <Metric label={language === "zh" ? "完成结果单元" : "Completed result units"} value={lastSummary.resultUnits.toString()} />
                <Metric label={language === "zh" ? "成功结果单元" : "Succeeded result units"} value={lastSummary.succeededResultUnits.toString()} />
                <Metric label={language === "zh" ? "失败结果单元" : "Failed result units"} value={lastSummary.failedResultUnits.toString()} />
                <Metric label={language === "zh" ? "剩余结果单元" : "Remaining result units"} value={lastSummary.remainingResultUnits.toString()} />
                <Metric label={t.common.config} value={lastSummary.configName} />
                <Metric label={t.runs.worker} value={lastSummary.workerId ?? "n/a"} />
                <Metric label={t.runs.created} value={formatOptionalDate(lastSummary.createdAt)} />
                <Metric label={t.runs.started} value={formatOptionalDate(lastSummary.startedAt)} />
                <Metric label={t.runs.finished} value={formatOptionalDate(lastSummary.finishedAt)} />
                <Metric label={t.runs.updated} value={formatOptionalDate(lastSummary.updatedAt)} />
                <Metric label={language === "zh" ? "结果单元" : "Result units"} value={lastSummary.resultUnits.toString()} />
              </div>
              {lastSummary.selection ? (
                <div className="run-summary-section">
                  <div className="run-artifacts-head">
                    <CheckCircle2 size={15} />
                    <span>{t.runs.selectionScope}</span>
                  </div>
                  <div className="run-meta-grid run-summary-compact-grid">
                    <Metric label={t.runs.datasets} value={lastSummary.selection.datasets.toString()} />
                    <Metric label={t.runs.watermarks} value={lastSummary.selection.watermarks.toString()} />
                    <Metric label={t.runs.attacks} value={lastSummary.selection.attacks.toString()} />
                    <Metric label={t.runs.seeds} value={lastSummary.selection.seeds.toString()} />
                    <Metric label={t.common.samples} value={lastSummary.selection.sampleCount.toString()} />
                    <Metric label={t.console.ops} value={lastSummary.selection.imageOperationCount.toString()} />
                  </div>
                </div>
              ) : null}
              {lastSummary.materialized ? (
                <div className="run-summary-section">
                  <div className="run-artifacts-head">
                    <FolderOpen size={15} />
                    <span>{t.runs.materializedCache}</span>
                  </div>
                  <div className="run-meta-grid run-summary-compact-grid">
                    <Metric label={t.runs.materializedRoot} value={lastSummary.materialized.root} />
                    <Metric label={t.runs.latestMaterializedDir} value={lastSummary.materialized.latestDir} />
                    <Metric label={t.runs.cacheHits} value={lastSummary.materialized.cacheHits.toString()} />
                  </div>
                </div>
              ) : null}
              <div className="run-artifacts">
                <div className="run-artifacts-head">
                  <FolderOpen size={15} />
                  <span>{t.runs.rawArtifacts}</span>
                </div>
                <code>{lastSummary.artifactRoot ?? "n/a"}</code>
                <div className="artifact-chip-grid">
                  {rawArtifactFiles.map((file) => (
                    <span key={file}>{file}</span>
                  ))}
                </div>
              </div>
            </div>
            <div className="modal-footer">
              {isRestartableTerminalRun(lastSummary.status) ? (
                <button className="button" disabled={busy} onClick={resumeSummaryRun} type="button">
                  <RotateCcw size={16} />
                  {t.runs.resumeFromCheckpoint}
                </button>
              ) : null}
              <button className="button primary" onClick={closeSummaryDialog} type="button">
                {t.runs.closeSummary}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </AppShell>
  );
}

function phaseValue(value: unknown) {
  if (value === null || value === undefined || value === "") {
    return "n/a";
  }
  if (typeof value === "number") {
    return formatNumber(value);
  }
  if (typeof value === "string") {
    return value;
  }
  return JSON.stringify(value);
}

function PhaseDetailPanel({
  stage,
  language,
  resourceNames,
  shards,
  gpuTelemetry,
  expandedShardIds,
  onToggleShard
}: {
  stage: ExperimentStageTab | undefined;
  language: Language;
  resourceNames: Record<string, string>;
  shards: RunShardProgress[];
  gpuTelemetry: GpuTelemetry | null;
  expandedShardIds: string[];
  onToggleShard: (shardId: string) => void;
}) {
  const phase = stage?.phase;
  return (
    <>
      {phase?.error ? <div className="risk error">{phase.error}</div> : null}
      <ShardProgressPanel
        expandedShardIds={expandedShardIds}
        gpuTelemetry={gpuTelemetry}
        language={language}
        onToggleShard={onToggleShard}
        resourceNames={resourceNames}
        shards={shards}
        stageKey={stage?.key ?? "canonical"}
      />
    </>
  );
}

function StageCellProgressMeter({ progress }: { progress: StageCellProgress }) {
  return (
    <div className="run-progress-meter run-stage-cell-meter">
      <div className="run-progress-head">
        <span>阶段总体进度</span>
        <strong>{progress.current}/{progress.total} {progress.unit}</strong>
      </div>
      <div className="progress-track">
        <div className="progress-bar" style={{ width: progressWidth(progress.percent) }} />
      </div>
    </div>
  );
}

function ShardProgressPanel({
  shards,
  stageKey,
  resourceNames,
  language,
  gpuTelemetry,
  expandedShardIds,
  onToggleShard
}: {
  shards: RunShardProgress[];
  stageKey: ExperimentStageKey;
  resourceNames: Record<string, string>;
  language: Language;
  gpuTelemetry: GpuTelemetry | null;
  expandedShardIds: string[];
  onToggleShard: (shardId: string) => void;
}) {
  if (!shards.length) {
    return null;
  }
  const sorted = [...shards].sort((left, right) => left.index - right.index || left.id.localeCompare(right.id));
  return (
    <div className="run-shard-grid">
      {sorted.map((shard) => (
        <ShardProgressCard
          key={shard.id}
          expanded={expandedShardIds.includes(shard.id)}
          gpuTelemetry={gpuTelemetry}
          language={language}
          onToggle={() => onToggleShard(shard.id)}
          resourceNames={resourceNames}
          shard={shard}
          stageKey={stageKey}
        />
      ))}
    </div>
  );
}

function ShardProgressCard({
  shard,
  stageKey,
  resourceNames,
  language,
  gpuTelemetry,
  expanded,
  onToggle
}: {
  shard: RunShardProgress;
  stageKey: ExperimentStageKey;
  resourceNames: Record<string, string>;
  language: Language;
  gpuTelemetry: GpuTelemetry | null;
  expanded: boolean;
  onToggle: () => void;
}) {
  const item = shard.currentItem ?? {};
  const imageProgress = shardImageProgress(shard, stageKey);
  const cellProgress = normalizedShardProgress(
    shardPhaseCellProgress(shard, phaseKeyForStage(stageKey), stageKey),
    shard.expectedCells
  );
  const contextRows = shardContextRows(stageKey, item, resourceNames, language);
  const gpu = gpuTelemetryForShard(shard, gpuTelemetry);
  const gpuTitle = gpu?.name || shard.device || shard.id;
  const gpuSubtitle = gpu ? `${shard.device} · GPU ${gpu.index}` : shard.device;
  const gpuStats = formatGpuCardStats(gpu, language);
  const toggleLabel = expanded ? "收起 GPU 曲线" : "展开 GPU 曲线";
  return (
    <div
      aria-expanded={expanded}
      className={expanded ? "run-shard-card expanded" : "run-shard-card"}
      onClick={onToggle}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onToggle();
        }
      }}
      role="button"
      tabIndex={0}
      title={toggleLabel}
    >
      <div className="run-shard-identity">
        <div className="run-shard-card-head">
          <div>
            <span>{gpuSubtitle}</span>
            <strong>{gpuTitle}</strong>
          </div>
          <small>{shard.status}</small>
        </div>
        <div className="run-shard-foot">
          <span>{gpuStats.temperature}</span>
          <code>{gpuStats.power}</code>
        </div>
      </div>
      {contextRows.length ? (
        <div className="run-shard-meta">
          {contextRows.map(([label, value]) => (
            <span key={label} title={value}>
              <b>{label}</b>
              <em>{value}</em>
            </span>
          ))}
        </div>
      ) : null}
      <div className="run-shard-progress-stack">
        <ShardMiniProgress label="图片" progress={imageProgress} />
        <ShardMiniProgress label="cell" progress={cellProgress} />
      </div>
      {expanded ? <GpuShardTelemetryPanel gpu={gpu} language={language} telemetry={gpuTelemetry} /> : null}
    </div>
  );
}

function ShardMiniProgress({
  label,
  progress
}: {
  label: string;
  progress: { current: number; total: number; percent: number };
}) {
  return (
    <div className="run-shard-progress">
      <div>
        <span>{label}</span>
        <strong>{progress.current}/{progress.total}</strong>
      </div>
      <div className="progress-track">
        <div className="progress-bar" style={{ width: progressWidth(progress.percent) }} />
      </div>
    </div>
  );
}

function gpuIndexForShard(shard: RunShardProgress) {
  const match = /^cuda:(\d+)$/.exec(shard.device);
  if (match) {
    return Number(match[1]);
  }
  const trailing = /cuda[_:-]?(\d+)/.exec(shard.id);
  if (trailing) {
    return Number(trailing[1]);
  }
  return shard.index;
}

function gpuTelemetryForShard(shard: RunShardProgress, telemetry: GpuTelemetry | null) {
  const gpuIndex = gpuIndexForShard(shard);
  return telemetry?.devices.find((device) => device.index === gpuIndex) ?? null;
}

function formatGpuCardStats(gpu: GpuTelemetryDevice | null, language: Language) {
  const temperature =
    gpu?.temperatureC == null
      ? language === "zh"
        ? "温度 n/a"
        : "Temp n/a"
      : language === "zh"
      ? `温度 ${formatNumber(gpu.temperatureC)}°C`
      : `Temp ${formatNumber(gpu.temperatureC)}°C`;
  const powerDraw = gpu?.powerDrawW == null ? "n/a" : `${formatNumber(gpu.powerDrawW)}W`;
  const powerLimit = gpu?.powerLimitW == null ? "" : ` / ${formatNumber(gpu.powerLimitW)}W`;
  const power = language === "zh" ? `功率 ${powerDraw}${powerLimit}` : `Power ${powerDraw}${powerLimit}`;
  return { power, temperature };
}

function GpuShardTelemetryPanel({
  gpu,
  telemetry,
  language
}: {
  gpu: GpuTelemetryDevice | null;
  telemetry: GpuTelemetry | null;
  language: Language;
}) {
  if (!gpu || !telemetry?.available) {
    return (
      <div className="run-gpu-telemetry-panel">
        <RunEmptyState
          description="GPU telemetry 暂不可用，等待下一次 nvidia-smi 采样。"
          title="等待 GPU 监控数据"
          variant="chart"
        />
      </div>
    );
  }
  const utilizationSeries = gpuMetricSeries(telemetry, gpu.index, "utilizationPercent");
  const memorySeries = gpuMetricSeries(telemetry, gpu.index, "memoryUsedMiB");
  const memoryLimit = gpu.memoryTotalMiB ?? Math.max(1, ...memorySeries.map((point) => point.value));
  return (
    <div className="run-gpu-telemetry-panel">
      <GpuTelemetryChart
        maxY={100}
        points={utilizationSeries}
        title={language === "zh" ? "GPU 使用率" : "GPU utilization"}
        unit="%"
        valueFormatter={(value) => `${formatNumber(value)}%`}
      />
      <GpuTelemetryChart
        limitLabel={`${language === "zh" ? "显存上限" : "Memory limit"} ${formatMiB(memoryLimit)}`}
        maxY={memoryLimit}
        points={memorySeries}
        title={language === "zh" ? "显存使用大小" : "Memory usage"}
        unit="MiB"
        valueFormatter={formatMiB}
      />
    </div>
  );
}

function gpuMetricSeries(
  telemetry: GpuTelemetry,
  gpuIndex: number,
  key: "utilizationPercent" | "memoryUsedMiB"
) {
  return telemetry.history
    .map((sample) => {
      const device = sample.devices.find((item) => item.index === gpuIndex);
      const value = device?.[key];
      return typeof value === "number" && Number.isFinite(value)
        ? { epochMs: sample.epochMs, value }
        : null;
    })
    .filter((point): point is { epochMs: number; value: number } => point !== null);
}

function GpuTelemetryChart({
  title,
  points,
  maxY,
  unit,
  valueFormatter,
  limitLabel
}: {
  title: string;
  points: Array<{ epochMs: number; value: number }>;
  maxY: number;
  unit: string;
  valueFormatter: (value: number) => string;
  limitLabel?: string;
}) {
  const width = 520;
  const height = 190;
  const padding = { top: 18, right: 14, bottom: 28, left: 46 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const safeMaxY = Math.max(1, maxY);
  const minTime = points[0]?.epochMs ?? 0;
  const maxTime = points[points.length - 1]?.epochMs ?? minTime + 1;
  const xFor = (epochMs: number) =>
    padding.left + ((epochMs - minTime) / Math.max(1, maxTime - minTime)) * plotWidth;
  const yFor = (value: number) =>
    padding.top + plotHeight * (1 - Math.max(0, Math.min(value, safeMaxY)) / safeMaxY);
  const linePath = points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${xFor(point.epochMs).toFixed(2)} ${yFor(point.value).toFixed(2)}`)
    .join(" ");
  const currentValue = points[points.length - 1]?.value;
  const yTicks = [0, safeMaxY / 2, safeMaxY];
  return (
    <div className="run-gpu-chart">
      <div className="run-gpu-chart-head">
        <div>
          <strong>{title}</strong>
          {limitLabel ? <span>{limitLabel}</span> : <span>{unit}</span>}
        </div>
        <em>{currentValue == null ? "n/a" : valueFormatter(currentValue)}</em>
      </div>
      <svg aria-label={title} preserveAspectRatio="none" viewBox={`0 0 ${width} ${height}`}>
        {yTicks.map((tick) => {
          const y = yFor(tick);
          return (
            <g key={tick}>
              <line className="run-gpu-grid-line" x1={padding.left} x2={width - padding.right} y1={y} y2={y} />
              <text className="run-gpu-axis-label" x={8} y={y + 4}>
                {valueFormatter(tick)}
              </text>
            </g>
          );
        })}
        {points.length > 1 ? <path className="run-gpu-line" d={linePath} /> : null}
        {points.length === 1 ? (
          <circle className="run-gpu-point" cx={xFor(points[0].epochMs)} cy={yFor(points[0].value)} r={3} />
        ) : null}
        <text className="run-gpu-time-label" x={padding.left} y={height - 7}>
          {formatTelemetryTime(minTime)}
        </text>
        <text className="run-gpu-time-label end" x={width - padding.right} y={height - 7}>
          {formatTelemetryTime(maxTime)}
        </text>
      </svg>
    </div>
  );
}

function formatMiB(value: number) {
  return `${formatNumber(value)}MiB`;
}

function formatTelemetryTime(epochMs: number) {
  if (!epochMs) {
    return "n/a";
  }
  return new Date(epochMs).toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });
}

function normalizedShardProgress(
  raw: { current: number; total: number } | undefined,
  fallbackTotal: number
) {
  const current = Math.max(0, Number(raw?.current ?? 0));
  const total = Math.max(current, Number(raw?.total ?? fallbackTotal ?? 0));
  return {
    current,
    total,
    percent: percent(current, total)
  };
}

function numericProgressValue(value: unknown) {
  const parsed = Number(value ?? 0);
  return Number.isFinite(parsed) ? Math.max(0, parsed) : 0;
}

function shardImageProgress(shard: RunShardProgress, stageKey: ExperimentStageKey) {
  const item = shard.currentItem ?? {};
  const sampleTotal = numericProgressValue(shard.sampleCount);
  let current: number | null = null;
  let total = sampleTotal;

  if (item.processedImages !== undefined) {
    current = numericProgressValue(item.processedImages);
  } else if (item.sampleCount !== undefined) {
    current = numericProgressValue(item.sampleCount);
  } else {
    const pairedImages = Math.max(numericProgressValue(item.positiveImages), numericProgressValue(item.negativeImages));
    if (pairedImages > 0) {
      current = pairedImages;
    } else if (item.pairCount !== undefined) {
      current = numericProgressValue(item.pairCount);
      if (stageKey === "quality") {
        total = Math.max(total * 2, current);
      }
    }
  }

  if (current === null) {
    const rawCurrent = numericProgressValue(shard.imageProgress?.current);
    const rawTotal = numericProgressValue(shard.imageProgress?.total);
    total = sampleTotal || rawTotal;
    current = total > 0 && rawCurrent <= total ? rawCurrent : 0;
  }

  total = Math.max(total, current);
  return {
    current: total > 0 ? Math.min(current, total) : current,
    total,
    percent: percent(current, total)
  };
}

const SHARD_ATTACK_PARAM_ORDER = [
  "strength",
  "step",
  "scale",
  "quality",
  "vae_model_name",
  "xy",
  "correct_perspective"
];

function shardAttackLabel(item: Record<string, unknown>, resourceNames: Record<string, string>) {
  const presetId = typeof item.attackPresetId === "string" ? item.attackPresetId : "";
  const method = typeof item.attackMethod === "string" ? item.attackMethod : "";
  if (presetId && resourceNames[presetId]) {
    return resourceNames[presetId];
  }
  if (method && resourceNames[method]) {
    return resourceNames[method];
  }
  return phaseValue(presetId || method);
}

function shardAttackVariantLabel(item: Record<string, unknown>, language: Language) {
  const params = parseShardParamRecord(item.attackParams);
  const fragments = formatShardParamFragments(params, language);
  if (fragments.length > 0) {
    return fragments.join(" · ");
  }
  if (item.attackStrength !== undefined && item.attackStrength !== null && item.attackStrength !== "") {
    const label = language === "zh" ? "强度" : "Strength";
    return `${label} ${formatShardParamValue(item.attackStrength, language)}`;
  }
  return language === "zh" ? "默认" : "Default";
}

function parseShardParamRecord(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  if (typeof value !== "string") {
    return {};
  }
  const trimmed = value.trim();
  if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
    return {};
  }
  try {
    const parsed = JSON.parse(trimmed);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function formatShardParamFragments(params: Record<string, unknown>, language: Language) {
  const keys = Object.keys(params).filter((key) => params[key] !== undefined && params[key] !== null && params[key] !== "");
  const orderedKeys = [
    ...SHARD_ATTACK_PARAM_ORDER.filter((key) => keys.includes(key)),
    ...keys.filter((key) => !SHARD_ATTACK_PARAM_ORDER.includes(key)).sort()
  ];
  return orderedKeys.map((key) => `${shardParamLabel(key, language)} ${formatShardParamValue(params[key], language)}`);
}

function shardParamLabel(key: string, language: Language) {
  const zh: Record<string, string> = {
    correct_perspective: "透视校正",
    quality: "质量",
    scale: "倍率",
    step: "步长",
    strength: "强度",
    vae_model_name: "VAE",
    xy: "XY"
  };
  const en: Record<string, string> = {
    correct_perspective: "Perspective",
    quality: "Quality",
    scale: "Scale",
    step: "Step",
    strength: "Strength",
    vae_model_name: "VAE",
    xy: "XY"
  };
  const labels = language === "zh" ? zh : en;
  return labels[key] ?? key.replace(/_/g, " ");
}

function formatShardParamValue(value: unknown, language: Language): string {
  if (typeof value === "number") {
    return formatNumber(value);
  }
  if (typeof value === "boolean") {
    return language === "zh" ? (value ? "开" : "关") : value ? "on" : "off";
  }
  if (Array.isArray(value)) {
    return value.map((item) => formatShardParamValue(item, language)).join("/");
  }
  if (typeof value === "string") {
    return displayMethodToken(value);
  }
  if (value && typeof value === "object") {
    return Object.entries(value as Record<string, unknown>)
      .map(([key, nestedValue]) => `${shardParamLabel(key, language)} ${formatShardParamValue(nestedValue, language)}`)
      .join(" · ");
  }
  return phaseValue(value);
}

function shardContextRows(
  stageKey: ExperimentStageKey,
  item: Record<string, unknown>,
  resourceNames: Record<string, string>,
  language: Language
) {
  const rows: Array<[string, unknown]> = [];
  if (stageKey === "canonical") {
    rows.push(["数据集", item.datasetId], ["样本", item.sampleCount]);
  } else if (stageKey === "watermark") {
    rows.push(["数据集", item.datasetId], ["水印", item.algorithmId], ["method", item.algorithmMethod], ["seed", item.seed]);
  } else if (stageKey === "attack") {
    rows.push(
      ["数据集", item.datasetId],
      ["水印", item.algorithmId],
      ["攻击", shardAttackLabel(item, resourceNames)],
      ["变体", shardAttackVariantLabel(item, language)]
    );
  } else if (stageKey === "extract") {
    rows.push(
      ["数据集", item.datasetId],
      ["水印", item.algorithmId],
      ["攻击", shardAttackLabel(item, resourceNames)],
      ["变体", shardAttackVariantLabel(item, language)]
    );
  } else if (stageKey === "quality") {
    rows.push(
      ["数据集", item.datasetId],
      ["水印", item.algorithmId],
      ["攻击", shardAttackLabel(item, resourceNames)],
      ["变体", shardAttackVariantLabel(item, language)]
    );
  } else {
    rows.push(["状态", item.status], ["result", item.resultUnitCount]);
  }
  return rows
    .map(([label, raw]) => {
      const rawValue = phaseValue(raw);
      const value = typeof raw === "string" && resourceNames[raw] ? resourceNames[raw] : rawValue;
      return [label, value] as [string, string];
    })
    .filter(([, value]) => value !== "n/a");
}

function StageTimeline({
  stages,
  selectedKey,
  overallProgress,
  onSelect
}: {
  stages: ExperimentStageTab[];
  selectedKey: ExperimentStageKey;
  overallProgress: number;
  onSelect: (key: ExperimentStageKey) => void;
}) {
  return (
    <div className="run-stage-timeline">
      <div className="run-stage-track" aria-hidden="true">
        <div className="run-stage-track-fill" style={{ width: progressWidth(overallProgress) }} />
      </div>
      <div className="run-stage-option-grid">
        {stages.map((stage, index) => (
          <button
            className={[
              "run-stage-option",
              stage.reached ? "reached" : "locked",
              stage.active ? "active" : "",
              stage.completed ? "completed" : "",
              selectedKey === stage.key ? "selected" : ""
            ]
              .filter(Boolean)
              .join(" ")}
            disabled={!stage.reached}
            key={stage.key}
            onClick={() => onSelect(stage.key)}
            type="button"
          >
            <span>{index + 1}</span>
            <strong>{stage.label}</strong>
            <small>{stage.phase?.status ?? "pending"}</small>
          </button>
        ))}
      </div>
    </div>
  );
}

function InlineRunSummary({
  summary,
  t,
  language,
  statusLabels,
  formatOptionalDate,
  busy,
  onResume,
  onExit
}: {
  summary: ExecutionSummary;
  t: Translation;
  language: Language;
  statusLabels: Record<string, string>;
  formatOptionalDate: (value?: string | null) => string;
  busy: boolean;
  onResume: () => void;
  onExit: () => void;
}) {
  return (
    <div className="run-inline-summary">
      <div className="run-summary-head">
        <div>
          <strong>{summary.taskName}</strong>
          <code>{summary.runId}</code>
        </div>
        <span className={badgeClass(summary.status)}>{runStatusLabel(summary.status, statusLabels)}</span>
      </div>
      <p>{summary.note}</p>
      <div className="run-meta-grid">
        <Metric label={t.common.progress} value={`${summary.progress}%`} />
        <Metric label={t.runs.duration} value={formatDurationMs(summary.durationMs, language)} />
        <Metric label={language === "zh" ? "完成结果单元" : "Completed result units"} value={summary.resultUnits.toString()} />
        <Metric label={language === "zh" ? "成功结果单元" : "Succeeded result units"} value={summary.succeededResultUnits.toString()} />
        <Metric label={language === "zh" ? "失败结果单元" : "Failed result units"} value={summary.failedResultUnits.toString()} />
        <Metric label={language === "zh" ? "剩余结果单元" : "Remaining result units"} value={summary.remainingResultUnits.toString()} />
        <Metric label={t.common.config} value={summary.configName} />
        <Metric label={t.runs.worker} value={summary.workerId ?? "n/a"} />
        <Metric label={t.runs.created} value={formatOptionalDate(summary.createdAt)} />
        <Metric label={t.runs.started} value={formatOptionalDate(summary.startedAt)} />
        <Metric label={t.runs.finished} value={formatOptionalDate(summary.finishedAt)} />
        <Metric label={t.runs.updated} value={formatOptionalDate(summary.updatedAt)} />
      </div>
      {summary.selection ? (
        <div className="run-summary-section">
          <div className="run-artifacts-head">
            <CheckCircle2 size={15} />
            <span>{t.runs.selectionScope}</span>
          </div>
          <div className="run-meta-grid run-summary-compact-grid">
            <Metric label={t.runs.datasets} value={summary.selection.datasets.toString()} />
            <Metric label={t.runs.watermarks} value={summary.selection.watermarks.toString()} />
            <Metric label={t.runs.attacks} value={summary.selection.attacks.toString()} />
            <Metric label={t.runs.seeds} value={summary.selection.seeds.toString()} />
            <Metric label={t.common.samples} value={summary.selection.sampleCount.toString()} />
            <Metric label={t.console.ops} value={summary.selection.imageOperationCount.toString()} />
          </div>
        </div>
      ) : null}
      {summary.materialized ? (
        <div className="run-summary-section">
          <div className="run-artifacts-head">
            <FolderOpen size={15} />
            <span>{t.runs.materializedCache}</span>
          </div>
          <div className="run-meta-grid run-summary-compact-grid">
            <Metric label={t.runs.materializedRoot} value={summary.materialized.root} />
            <Metric label={t.runs.latestMaterializedDir} value={summary.materialized.latestDir} />
            <Metric label={t.runs.cacheHits} value={summary.materialized.cacheHits.toString()} />
          </div>
        </div>
      ) : null}
      <div className="run-artifacts">
        <div className="run-artifacts-head">
          <FolderOpen size={15} />
          <span>{t.runs.rawArtifacts}</span>
        </div>
        <code>{summary.artifactRoot ?? "n/a"}</code>
        <div className="artifact-chip-grid">
          {rawArtifactFiles.map((file) => (
            <span key={file}>{file}</span>
          ))}
        </div>
      </div>
      <div className="run-inline-summary-actions">
        {isRestartableTerminalRun(summary.status) ? (
          <button className="button" disabled={busy} onClick={onResume} type="button">
            <RotateCcw size={16} />
            {t.runs.resumeFromCheckpoint}
          </button>
        ) : null}
        <button className="button primary" onClick={onExit} type="button">
          <Save size={16} />
          {t.runs.saveResultsAndExit}
        </button>
      </div>
    </div>
  );
}

function ProgressMeter({ step }: { step: ProgressStep }) {
  return (
    <div className="run-progress-meter">
      <div className="run-progress-head">
        <span>{step.label}</span>
        <strong>{step.meta}</strong>
      </div>
      <div className="progress-track">
        <div className="progress-bar" style={{ width: progressWidth(step.percent) }} />
      </div>
    </div>
  );
}

function formatTuningThroughput(value: number) {
  return new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits: value >= 100 ? 0 : 1,
    notation: value >= 10000 ? "compact" : "standard"
  }).format(value);
}

function ThroughputChart({ points }: { points: TuningPoint[] }) {
  const width = 720;
  const height = 240;
  const padding = { top: 16, right: 18, bottom: 42, left: 46 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const maxVisiblePoints = 72;
  const visiblePoints = points.length > maxVisiblePoints ? points.slice(-maxVisiblePoints) : points;
  const maxThroughput = Math.max(1, ...visiblePoints.map((point) => point.throughput));
  const xStep = visiblePoints.length > 1 ? plotWidth / (visiblePoints.length - 1) : 0;
  const pointCoordinates = visiblePoints.map((point, index) => {
    const x = padding.left + (visiblePoints.length === 1 ? plotWidth / 2 : index * xStep);
    const y = padding.top + plotHeight * (1 - point.throughput / maxThroughput);
    return { point, x, y };
  });
  const linePath = pointCoordinates
    .map(({ x, y }, index) => `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`)
    .join(" ");
  const labelStep = Math.max(1, Math.ceil(visiblePoints.length / 6));

  if (!visiblePoints.length) {
    return (
      <RunEmptyState
        description="每个候选配置会在这里记录耗时、吞吐量与推荐判断。"
        title="搜索开始后显示 images/sec 曲线"
        variant="chart"
      />
    );
  }

  return (
    <div className="throughput-chart">
      <svg aria-label="throughput chart" preserveAspectRatio="none" viewBox={`0 0 ${width} ${height}`}>
        <line className="chart-axis" x1={padding.left} x2={width - padding.right} y1={height - padding.bottom} y2={height - padding.bottom} />
        <line className="chart-axis" x1={padding.left} x2={padding.left} y1={padding.top} y2={height - padding.bottom} />
        {[0.25, 0.5, 0.75, 1].map((ratio) => {
          const y = padding.top + plotHeight * (1 - ratio);
          return (
            <g key={ratio}>
              <line className="chart-grid-line" x1={padding.left} x2={width - padding.right} y1={y} y2={y} />
              <text className="chart-label" x={8} y={y + 4}>
                {formatTuningThroughput(maxThroughput * ratio)}
              </text>
            </g>
          );
        })}
        {linePath ? <path className="chart-trend-line" d={linePath} /> : null}
        {pointCoordinates.map(({ point, x, y }, index) => (
          <g key={point.key}>
            <circle
              className={`${point.kind === "batch" ? "chart-point batch" : "chart-point workers"}${point.ok ? "" : " failed"}`}
              cx={x}
              cy={y}
              r={point.ok ? 4.2 : 5}
            />
            <title>{`${point.groupLabel} · ${point.kind === "batch" ? "batch" : "workers"}=${point.candidate} · ${formatTuningThroughput(
              point.throughput
            )} img/s${point.ok ? "" : " · failed"}`}</title>
            {(index === 0 || index === visiblePoints.length - 1 || index % labelStep === 0) && (
              <text className="chart-x-label" textAnchor="middle" x={x} y={height - 20}>
                {point.kind === "batch" ? `b${point.candidate}` : `w${point.candidate}`}
              </text>
            )}
          </g>
        ))}
      </svg>
      <div className="throughput-chart-meta">
        <span>
          <i className="batch" /> batch 候选
        </span>
        <span>
          <i className="workers" /> workers 候选
        </span>
        <strong>
          显示 {visiblePoints.length}/{points.length} 次 · 最高 {formatTuningThroughput(maxThroughput)} img/s
        </strong>
      </div>
    </div>
  );
}

function RunEmptyState({
  description,
  title,
  variant = "default"
}: {
  description: string;
  title: string;
  variant?: "chart" | "events" | "env" | "default";
}) {
  return (
    <div className={`run-empty-state ${variant}`}>
      <div className="run-empty-visual" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
      <div>
        <strong>{title}</strong>
        <span>{description}</span>
      </div>
    </div>
  );
}

function TuningMethodSelector({
  description,
  embedded = false,
  filter,
  onFilterChange,
  onSelectMethods,
  options,
  searchPlaceholder,
  selectedMethods,
  title,
  totalCount,
  visibleMethods
}: {
  description: string;
  embedded?: boolean;
  filter: string;
  onFilterChange: (value: string) => void;
  onSelectMethods: (methods: string[]) => void;
  options: TuningMethodOption[];
  searchPlaceholder: string;
  selectedMethods: string[];
  title: string;
  totalCount: number;
  visibleMethods: string[];
}) {
  const selectedSet = new Set(selectedMethods);
  const selectedVisibleCount = visibleMethods.filter((method) => selectedSet.has(method)).length;

  return (
    <section className={["tuning-method-panel", embedded ? "embedded" : ""].filter(Boolean).join(" ")}>
      <div className="tuning-method-head">
        <div>
          <strong>{title}</strong>
          <span>{description}</span>
        </div>
        <em>
          {selectedMethods.length}/{totalCount}
        </em>
      </div>
      <div className="selector-tools tuning-method-tools">
        <div className="field-icon-input">
          <Search size={15} />
          <input
            aria-label={searchPlaceholder}
            onChange={(event) => onFilterChange(event.target.value)}
            placeholder={searchPlaceholder}
            value={filter}
          />
        </div>
        <div className="bulk-actions">
          <button
            className="button compact"
            disabled={visibleMethods.length === 0 || selectedVisibleCount === visibleMethods.length}
            onClick={() => onSelectMethods(addIds(selectedMethods, visibleMethods))}
            type="button"
          >
            <Check size={14} />
            选择当前
          </button>
          <button
            className="button compact"
            disabled={visibleMethods.length === 0 || selectedVisibleCount === 0}
            onClick={() => onSelectMethods(removeIds(selectedMethods, visibleMethods))}
            type="button"
          >
            <X size={14} />
            清空当前
          </button>
        </div>
      </div>
      {options.length ? (
        <div className="tuning-method-grid">
          {options.map((option) => (
            <label className="check-tile resource-check-tile method-check-tile tuning-method-tile" key={option.method}>
              <input
                checked={selectedSet.has(option.method)}
                disabled={!option.available}
                onChange={() => onSelectMethods(toggle(selectedMethods, option.method))}
                type="checkbox"
              />
              <span className="tile-copy">
                <strong>{option.label}</strong>
                <small>{option.subtitle}</small>
              </span>
              {option.viewpoint ? <span className="badge warn">3D</span> : null}
              {option.requiresGpu ? <span className="badge warn">GPU</span> : null}
              {option.weighted ? <span className="badge">权重</span> : null}
              {!option.available ? <span className="badge error">Missing</span> : null}
            </label>
          ))}
        </div>
      ) : (
        <RunEmptyState
          description="没有匹配当前搜索条件的 method。"
          title="暂无可选 method"
          variant="events"
        />
      )}
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div title={value}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ConfigPreview({ config, t }: { config: SavedExperimentConfig; t: ReturnType<typeof useLanguage>["t"] }) {
  return (
    <div className="run-config-summary">
      <strong>{t.runs.configSummary}</strong>
      <span>{config.name}</span>
      <small>{config.id}</small>
      <div className="stats">
        <div className="stat">
          <span>{t.console.cells}</span>
          <strong>{config.cellCount}</strong>
        </div>
        <div className="stat">
          <span>{t.common.samples}</span>
          <strong>{config.sampleCount}</strong>
        </div>
        <div className="stat">
          <span>{t.console.ops}</span>
          <strong>{config.imageOperationCount}</strong>
        </div>
      </div>
    </div>
  );
}
