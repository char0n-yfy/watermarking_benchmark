"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Archive, Braces, Check, Database, Edit3, Gauge, Loader2, Plus, Save, Search, Shield, Trash2, X } from "lucide-react";
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
import { localizedName, type Language } from "@/lib/i18n";
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
  resource: { id: string; name: string; method?: string; category?: string; description?: string },
  extraTerms: Array<string | undefined> = []
) {
  const normalizedQuery = query.trim().toLowerCase();
  if (!normalizedQuery) {
    return true;
  }
  return [resource.id, resource.name, resource.method, resource.category, resource.description, ...extraTerms]
    .filter(Boolean)
    .join(" ")
    .toLowerCase()
    .includes(normalizedQuery);
}

function categoryLabel(language: Language, category: string) {
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
const CONSUMER_CATEGORY = "consumer_enhancement_workflow_attacks";
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
const CONSUMER_STRENGTH_METHODS = ["cew_e1", "cew_e2", "cew_e3", "cew_e4"] as const;
const CONSUMER_SUPER_RESOLUTION_METHODS = ["cew_s1", "cew_s2", "cew_s3"] as const;
const CONSUMER_SUPER_RESOLUTION_SCALES = [2, 4];
const VIEWPOINT_METHOD_PATTERN = /^3d_viewpoint_rerendering_(swipe|shake|rotate|rotate_forward)_phase(\d+)_(point|ahead)$/;
const VIEWPOINT_MOTION_ORDER = ["swipe", "shake", "rotate", "rotate_forward"] as const;
const VIEWPOINT_LOOKAT_MODES = ["point", "ahead"] as const;
const VIEWPOINT_PHASES = [0, 1, 2, 3, 4, 5, 6, 7];

type DisplayMeta = {
  en: string;
  rank: number;
  short?: string;
  zh: string;
};

const ALGORITHM_DISPLAY: Record<string, DisplayMeta> = {
  "invisible-watermark-dwtdct": { en: "Invisible Watermark DWT-DCT", rank: 10, short: "DWT-DCT", zh: "Invisible Watermark DWT-DCT" },
  "invisible-watermark-dwtdctsvd": {
    en: "Invisible Watermark DWT-DCT-SVD",
    rank: 11,
    short: "DWT-DCT-SVD",
    zh: "Invisible Watermark DWT-DCT-SVD"
  },
  "traditional-spread-dct": {
    en: "Traditional Spread-DCT",
    rank: 12,
    short: "Spread-DCT",
    zh: "传统扩频 DCT 水印"
  },
  hidden: { en: "HiDDeN", rank: 30, zh: "HiDDeN" },
  stegastamp: { en: "StegaStamp", rank: 31, zh: "StegaStamp" },
  "ssl-watermarking": { en: "SSL Watermarking", rank: 32, zh: "SSL Watermarking" },
  mbrs: { en: "MBRS", rank: 33, zh: "MBRS" },
  cin: { en: "CIN", rank: 34, zh: "CIN" },
  pimog: { en: "PIMoG", rank: 35, zh: "PIMoG" },
  invismark: { en: "InvisMark", rank: 36, zh: "InvisMark" },
  "invisible-watermark-rivagan": {
    en: "Invisible Watermark RivaGAN",
    rank: 37,
    short: "RivaGAN",
    zh: "Invisible Watermark RivaGAN"
  },
  "trustmark-c": { en: "TrustMark-C", rank: 38, zh: "TrustMark-C" },
  "trustmark-q": { en: "TrustMark-Q", rank: 39, zh: "TrustMark-Q" },
  rawatermark: { en: "RAWatermark", rank: 40, zh: "RAWatermark" },
  "maskwm-d32": { en: "MaskWM-D32", rank: 41, zh: "MaskWM-D32" },
  wam: { en: "WAM", rank: 42, zh: "WAM" },
  videoseal: { en: "VideoSeal", rank: 60, zh: "VideoSeal" },
  pixelseal: { en: "PixelSeal", rank: 61, zh: "PixelSeal" },
  chunkyseal: { en: "ChunkySeal", rank: 62, zh: "ChunkySeal" },
  vine: { en: "VINE", rank: 70, zh: "VINE" }
};

const ALGORITHM_CATEGORY_ORDER: Record<string, number> = {
  traditional_watermark: 10,
  deep_watermark: 20
};

const ATTACK_CATEGORY_ORDER: Record<string, number> = {
  distortion_attacks: 10,
  physical_channel_attacks: 20,
  "3d_viewpoint_rerendering": 30,
  regeneration_attacks: 40,
  consumer_enhancement_workflow_attacks: 50
};

const ATTACK_DISPLAY: Record<string, DisplayMeta> = {
  brightness: { en: "Brightness", rank: 10, zh: "亮度调整" },
  contrast: { en: "Contrast", rank: 11, zh: "对比度调整" },
  gaussian_blur: { en: "Gaussian Blur", rank: 12, zh: "高斯模糊" },
  gaussian_noise: { en: "Gaussian Noise", rank: 13, zh: "高斯噪声" },
  jpeg: { en: "JPEG Compression", rank: 14, zh: "JPEG 压缩" },
  resize: { en: "Resize", rank: 15, zh: "缩放" },
  resized_crop: { en: "Resized Crop", rank: 16, zh: "缩放裁剪" },
  rotation: { en: "Rotation", rank: 17, zh: "旋转" },
  erasing: { en: "Random Erasing", rank: 18, zh: "区域擦除" },
  screen_shoot: { en: "PIMoG-style Screen-Camera", rank: 20, short: "Screen-camera", zh: "屏幕-拍摄信道" },
  print_camera: { en: "CamMark-style Print-Camera", rank: 21, short: "Print-camera", zh: "打印-拍摄信道" },
  combined_physical: { en: "Combined Physical Channel", rank: 22, short: "Combined", zh: "组合物理信道" },
  "2x_regen": { en: "2-pass Diffusion Regeneration", rank: 40, short: "2-pass Regen", zh: "2轮扩散再生成" },
  "4x_regen": { en: "4-pass Diffusion Regeneration", rank: 41, short: "4-pass Regen", zh: "4轮扩散再生成" },
  regen_diffusion: { en: "WAVES Diffusion Regeneration", rank: 42, short: "Diffusion", zh: "扩散再生成" },
  noise_to_image: { en: "CtrlRegen Noise-to-Image", rank: 43, short: "Noise-to-image", zh: "噪声到图像再生成" },
  regen_vae: { en: "CompressAI VAE Reconstruction", rank: 44, short: "VAE", zh: "VAE 再生成" },
  image_to_vedio: { en: "NFPA Image-to-Video", rank: 45, short: "Image-to-video", zh: "图像到视频再生成" },
  cew_e1: { en: "Auto-Tone", rank: 50, short: "Auto tone", zh: "自动色调" },
  cew_e2: { en: "Warm-Vivid", rank: 51, short: "Warm vivid", zh: "暖色鲜艳" },
  cew_e3: { en: "Film-Faded", rank: 52, short: "Film faded", zh: "胶片褪色" },
  cew_e4: { en: "Local-Clarity HDR", rank: 53, short: "Local HDR", zh: "局部清晰 HDR" },
  cew_c1: { en: "Basic Auto-Fix SR", rank: 54, short: "Auto fix", zh: "自动修复+超分" },
  cew_c2: { en: "Color Retouch SR", rank: 55, short: "Retouch", zh: "色彩修饰+超分" },
  cew_c3: { en: "Detail Enhance SR", rank: 56, short: "Detail", zh: "细节增强+超分" },
  cew_c4: { en: "Full Enhancement Chain", rank: 57, short: "Full chain", zh: "完整增强链" },
  cew_d1: { en: "Zero-DCE++ Auto-Light", rank: 58, short: "Auto light", zh: "自动补光" },
  cew_d2: { en: "DeepWB Auto-WhiteBalance", rank: 59, short: "White balance", zh: "自动白平衡" },
  cew_d3: { en: "Image-Adaptive 3D LUT", rank: 60, short: "AI color", zh: "自适应 AI 色彩" },
  cew_d4: { en: "Retinexformer Detail Low-Light Enhance", rank: 61, short: "Low light", zh: "低光细节增强" },
  cew_d5: { en: "NAFNet/Restormer AI-Denoise", rank: 62, short: "Denoise", zh: "AI 去噪" },
  cew_s1: { en: "Real-ESRGAN", rank: 63, short: "Real-ESRGAN", zh: "Real-ESRGAN" },
  cew_s2: { en: "SwinIR", rank: 64, short: "SwinIR", zh: "SwinIR" },
  cew_s3: { en: "BSRGAN", rank: 65, short: "BSRGAN", zh: "BSRGAN" }
};

const ATTACK_ENGLISH_NAME: Record<string, string> = {
  brightness: "Brightness",
  contrast: "Contrast",
  gaussian_blur: "Gaussian Blur",
  gaussian_noise: "Gaussian Noise",
  jpeg: "JPEG Compression",
  resize: "Resize",
  resized_crop: "Resized Crop",
  rotation: "Rotation",
  erasing: "Random Erasing",
  screen_shoot: "PIMoG-style Screen-Camera",
  print_camera: "CamMark-style Print-Camera",
  combined_physical: "Combined Physical Channel",
  "2x_regen": "2-pass Diffusion Regeneration",
  "4x_regen": "4-pass Diffusion Regeneration",
  regen_diffusion: "WAVES Diffusion Regeneration",
  noise_to_image: "CtrlRegen Noise-to-Image",
  regen_vae: "CompressAI VAE Reconstruction",
  image_to_vedio: "NFPA Image-to-Video",
  cew_e1: "Auto-Tone",
  cew_e2: "Warm-Vivid",
  cew_e3: "Film-Faded",
  cew_e4: "Local-Clarity HDR",
  cew_c1: "Basic Auto-Fix SR",
  cew_c2: "Color Retouch SR",
  cew_c3: "Detail Enhance SR",
  cew_c4: "Full Enhancement Chain",
  cew_d1: "Zero-DCE++ Auto-Light",
  cew_d2: "DeepWB Auto-WhiteBalance",
  cew_d3: "Image-Adaptive 3D LUT",
  cew_d4: "Retinexformer Detail Low-Light Enhance",
  cew_d5: "NAFNet/Restormer AI-Denoise",
  cew_s1: "Real-ESRGAN",
  cew_s2: "SwinIR",
  cew_s3: "BSRGAN"
};

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
type ConsumerEnhancementSettings = StrengthRangeSettings & {
  superResolutionScales: number[];
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
const defaultConsumerEnhancementSettings: ConsumerEnhancementSettings = {
  strengthMin: 0,
  strengthMax: 1,
  strengthLevelCount: 5,
  superResolutionScales: [2, 4]
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
  return {
    phaseIndex: Number(match[2]),
    motion: match[1] as ViewpointMotion,
    lookatMode: match[3] as ViewpointLookatMode
  };
}

function viewpointMotionLabel(language: Language, motion: ViewpointMotion) {
  const zh: Record<ViewpointMotion, string> = {
    swipe: "横向扫动",
    shake: "抖动",
    rotate: "环绕旋转",
    rotate_forward: "前向环绕"
  };
  const en: Record<ViewpointMotion, string> = {
    swipe: "Swipe",
    shake: "Shake",
    rotate: "Rotate",
    rotate_forward: "Rotate forward"
  };
  return (language === "zh" ? zh : en)[motion];
}

function viewpointMotionEnglishLabel(motion: ViewpointMotion) {
  const labels: Record<ViewpointMotion, string> = {
    swipe: "Swipe",
    shake: "Shake",
    rotate: "Rotate",
    rotate_forward: "Rotate Forward"
  };
  return labels[motion];
}

function viewpointLookatLabel(language: Language, lookatMode: ViewpointLookatMode) {
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

function displayText(language: Language, meta: DisplayMeta | undefined, fallback: string) {
  if (meta === undefined) {
    return fallback;
  }
  return language === "zh" ? meta.zh : meta.en;
}

function isAsciiText(value: string) {
  return /^[\x00-\x7F]+$/.test(value.trim());
}

function englishSubtitleForTitle(language: Language, title: string, englishName: string | undefined) {
  const normalizedTitle = title.trim();
  const normalizedEnglish = englishName?.trim();
  if (
    language !== "zh" ||
    !normalizedEnglish ||
    !normalizedTitle ||
    isAsciiText(normalizedTitle) ||
    normalizedTitle.toLowerCase() === normalizedEnglish.toLowerCase()
  ) {
    return null;
  }
  return normalizedEnglish;
}

function algorithmMethod(algorithm: AlgorithmVersion) {
  return algorithm.method ?? algorithm.id.replace(/^alg-/, "");
}

function displayMethodToken(method: string) {
  if (method === "image_to_vedio") {
    return "image_to_video";
  }
  return method;
}

function algorithmDisplayName(language: Language, algorithm: AlgorithmVersion) {
  const method = algorithmMethod(algorithm);
  return displayText(language, ALGORITHM_DISPLAY[method], algorithm.name);
}

function algorithmCategoryDisplay(language: Language, category: string | undefined) {
  const normalized = normalizeAlgorithmCategory(category);
  const zh: Record<string, string> = {
    traditional_watermark: "传统水印",
    deep_watermark: "深度水印"
  };
  const en: Record<string, string> = {
    traditional_watermark: "Traditional watermark",
    deep_watermark: "Deep watermark"
  };
  return (language === "zh" ? zh : en)[normalized] ?? normalized;
}

function normalizeAlgorithmCategory(category: string | undefined) {
  if (category === "classical" || category === "traditional_watermark") {
    return "traditional_watermark";
  }
  return "deep_watermark";
}

function algorithmRank(algorithm: AlgorithmVersion) {
  const method = algorithmMethod(algorithm);
  const categoryRank = ALGORITHM_CATEGORY_ORDER[normalizeAlgorithmCategory(algorithm.category)] ?? 90;
  return categoryRank * 100 + (ALGORITHM_DISPLAY[method]?.rank ?? 99);
}

function attackDisplayName(language: Language, attack: AttackPreset) {
  return displayText(language, ATTACK_DISPLAY[attack.method], localizedName(language, attack.id, attack.name));
}

function attackEnglishName(attack: AttackPreset) {
  return ATTACK_ENGLISH_NAME[attack.method] ?? ATTACK_DISPLAY[attack.method]?.en ?? attack.name;
}

function attackRank(attack: AttackPreset) {
  const categoryRank = ATTACK_CATEGORY_ORDER[attackCategory(attack)] ?? 90;
  const parsed = parseViewpointMethod(attack.method);
  const methodRank = parsed
    ? VIEWPOINT_MOTION_ORDER.indexOf(parsed.motion) * 16 +
      parsed.phaseIndex * 2 +
      (parsed.lookatMode === "point" ? 0 : 1)
    : ATTACK_DISPLAY[attack.method]?.rank ?? 99;
  return categoryRank * 100 + methodRank;
}

function compareByRankAndName<T>(
  left: T,
  right: T,
  rank: (value: T) => number,
  name: (value: T) => string
) {
  const rankDelta = rank(left) - rank(right);
  if (rankDelta !== 0) {
    return rankDelta;
  }
  return name(left).localeCompare(name(right), undefined, { numeric: true });
}

function attackVariantSummary(language: Language, attack: AttackPreset) {
  if (attack.strengthParam === "strength" || attack.strengthParam === "step") {
    return language === "zh" ? "0-1 强度" : "0-1 strength";
  }
  if (attack.strengthParam === "xy") {
    return "XY";
  }
  if (attack.strengthParam === "scale") {
    return language === "zh" ? "倍率" : "scale";
  }
  return language === "zh" ? "固定参数" : "fixed";
}

function consumerAttackDisplayName(language: Language, attack: AttackPreset) {
  if (language === "en") {
    return attackEnglishName(attack);
  }
  return attackDisplayName(language, attack);
}

type StrengthRangeAxisProps = {
  id: string;
  language: Language;
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

function isConsumerStrengthAttack(attack: AttackPreset) {
  return (CONSUMER_STRENGTH_METHODS as readonly string[]).includes(attack.method);
}

function isConsumerSuperResolutionAttack(attack: AttackPreset) {
  return (CONSUMER_SUPER_RESOLUTION_METHODS as readonly string[]).includes(attack.method);
}

function consumerSuperResolutionScalesForSettings(settings: ConsumerEnhancementSettings) {
  return sortedSelectedNumbers(settings.superResolutionScales, CONSUMER_SUPER_RESOLUTION_SCALES);
}

function applyConsumerEnhancementSettings(
  current: ExperimentSelection,
  attacks: AttackPreset[],
  settings: ConsumerEnhancementSettings
): ExperimentSelection {
  const attackIds = attacks.map((attack) => attack.id);
  const selectedIds = new Set(current.attackPresetIds);
  const strengthLevels = strengthLevelsForSettings(settings);
  const superResolutionScales = consumerSuperResolutionScalesForSettings(settings);
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
    if (isConsumerStrengthAttack(attack) && attackSupportsStrengthOverride(attack)) {
      attackStrengthOverrides[attack.id] = strengthLevels;
    } else if (isConsumerSuperResolutionAttack(attack) && attack.strengthParam === "scale") {
      attackStrengthOverrides[attack.id] = superResolutionScales;
    }
  }

  return {
    ...current,
    attackStrengthOverrides: cleanAttackStrengthOverrides(attackStrengthOverrides, selectedIds),
    attackParamOverrides: cleanAttackParamOverrides(attackParamOverrides, selectedIds)
  };
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
  const [consumerEnhancementSettings, setConsumerEnhancementSettings] = useState<ConsumerEnhancementSettings>(
    defaultConsumerEnhancementSettings
  );
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [renameTarget, setRenameTarget] = useState<SavedExperimentConfig | null>(null);
  const [renameName, setRenameName] = useState("");
  const [message, setMessage] = useState("");
  const [isSavingConfig, setIsSavingConfig] = useState(false);
  const isSavingConfigRef = useRef(false);

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
  const sortedDatasets = useMemo(
    () =>
      [...datasets].sort((left, right) =>
        localizedName(language, left.id, left.name).localeCompare(localizedName(language, right.id, right.name), undefined, {
          numeric: true
        })
      ),
    [datasets, language]
  );
  const filteredAlgorithms = useMemo(
    () =>
      algorithms
        .filter((algorithm) =>
          matchesResource(algorithmFilter, algorithm, [
            algorithmDisplayName(language, algorithm),
            algorithmCategoryDisplay(language, algorithm.category),
            algorithmMethod(algorithm)
          ])
        )
        .sort((left, right) =>
          compareByRankAndName(
            left,
            right,
            algorithmRank,
            (algorithm) => `${algorithmDisplayName(language, algorithm)} ${algorithmMethod(algorithm)}`
          )
        ),
    [algorithmFilter, algorithms, language]
  );
  const filteredAttacks = useMemo(
    () =>
      configurableAttacks
        .filter((attack) =>
          matchesResource(attackFilter, attack, [
            attackDisplayName(language, attack),
            categoryLabel(language, attackCategory(attack)),
            attackVariantSummary(language, attack)
          ])
        )
        .sort((left, right) =>
          compareByRankAndName(
            left,
            right,
            attackRank,
            (attack) => `${attackDisplayName(language, attack)} ${attack.method}`
          )
        ),
    [attackFilter, configurableAttacks, language]
  );
  const visibleDatasetIds = useMemo(() => sortedDatasets.map((dataset) => dataset.id), [sortedDatasets]);
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
    return [...groups.entries()].sort(
      ([left], [right]) => (ATTACK_CATEGORY_ORDER[left] ?? 90) - (ATTACK_CATEGORY_ORDER[right] ?? 90)
    );
  }, [filteredAttacks]);

  const renderAttackTile = (
    attack: AttackPreset,
    onToggle?: (attack: AttackPreset) => void,
    options: { englishName?: string; title?: string } = {}
  ) => {
    const title = options.title ?? attackDisplayName(language, attack);
    const subtitle = englishSubtitleForTitle(language, title, options.englishName ?? attackEnglishName(attack));

    return (
      <label className="check-tile resource-check-tile method-check-tile" key={attack.id}>
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
          <strong>{title}</strong>
          {subtitle ? <small translate="no">{subtitle}</small> : null}
        </span>
        {attack.requiresGpu ? <span className="badge warn">{t.common.gpu}</span> : null}
        {attack.available === false ? <span className="badge error">Missing</span> : null}
      </label>
    );
  };

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
          saving: "保存中",
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
          saving: "Saving",
          modalHint: "After saving, launch this config from the Runs page."
        };

  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchDatasets(), fetchAlgorithms(), fetchAttacks(), fetchSavedConfigs()])
      .then(([apiDatasets, apiAlgorithms, apiAttacks, apiConfigs]) => {
        if (cancelled) {
          return;
        }
        setDatasets(apiDatasets);
        setAlgorithms(apiAlgorithms);
        setAttacks(apiAttacks);
        setSavedConfigs(apiConfigs);

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
    setConsumerEnhancementSettings(defaultConsumerEnhancementSettings);
    setMessage("");
    setIsCreateOpen(true);
  };

  const saveConfig = async () => {
    if (isSavingConfigRef.current) {
      return;
    }
    if (!canSave) {
      setMessage("请至少选择一个数据集、一个水印算法，并填写配置名称。");
      return;
    }
    isSavingConfigRef.current = true;
    setIsSavingConfig(true);
    try {
      const config = await createSavedConfig(configName.trim(), effectiveSelection);
      setSavedConfigs((current) => [config, ...current]);
      setIsCreateOpen(false);
      setMessage(t.configs.savedToast);
    } catch {
      setMessage("API 保存失败，请先启动 FastAPI 服务后再保存。");
    } finally {
      isSavingConfigRef.current = false;
      setIsSavingConfig(false);
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
                  {sortedDatasets.map((dataset) => (
                    <label className="check-tile resource-check-tile method-check-tile" key={dataset.id}>
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
                    <label className="check-tile resource-check-tile method-check-tile" key={algorithm.id}>
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
                        <strong>{algorithmDisplayName(language, algorithm)}</strong>
                        <small>
                          {algorithmCategoryDisplay(language, algorithm.category)} ·{" "}
                          {displayMethodToken(algorithmMethod(algorithm))}
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
                      onClick={() => {
                        const visibleViewpointAttacks = filteredAttacks.filter(
                          (attack) => attackCategory(attack) === VIEWPOINT_CATEGORY
                        );
                        const visibleViewpointMotions = VIEWPOINT_MOTION_ORDER.filter((motion) =>
                          visibleViewpointAttacks.some((attack) => parseViewpointMethod(attack.method)?.motion === motion)
                        );
                        const nextViewpointSettings =
                          visibleViewpointAttacks.length > 0
                            ? {
                                ...viewpointSettings,
                                enabled: true,
                                motions:
                                  viewpointSettings.motions.length > 0
                                    ? viewpointSettings.motions
                                    : visibleViewpointMotions,
                                lookatModes:
                                  viewpointSettings.lookatModes.length > 0
                                    ? viewpointSettings.lookatModes
                                    : [...VIEWPOINT_LOOKAT_MODES],
                                phases:
                                  viewpointSettings.phases.length > 0
                                    ? viewpointSettings.phases
                                    : [...VIEWPOINT_PHASES]
                              }
                            : viewpointSettings;
                        if (visibleViewpointAttacks.length > 0) {
                          setViewpointSettings(nextViewpointSettings);
                        }
                        setSelection((current) => {
                          let nextSelection = addAttackIdsToSelection(current, visibleAttackIds);
                          if (visibleViewpointAttacks.length > 0) {
                            nextSelection = applyViewpointSettings(
                              nextSelection,
                              visibleViewpointAttacks,
                              nextViewpointSettings
                            );
                          }
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
                          const visibleConsumerAttacks = filteredAttacks.filter(
                            (attack) => attackCategory(attack) === CONSUMER_CATEGORY
                          );
                          if (visibleConsumerAttacks.length > 0) {
                            nextSelection = applyConsumerEnhancementSettings(
                              nextSelection,
                              visibleConsumerAttacks,
                              consumerEnhancementSettings
                            );
                          }
                          return nextSelection;
                        });
                      }}
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
                    const isConsumerCategory = category === CONSUMER_CATEGORY;
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
                    const consumerStrengthAttacks = CONSUMER_STRENGTH_METHODS.map((method) =>
                      categoryAttacks.find((attack) => attack.method === method)
                    ).filter((attack): attack is AttackPreset => attack !== undefined);
                    const consumerSuperResolutionAttacks = CONSUMER_SUPER_RESOLUTION_METHODS.map((method) =>
                      categoryAttacks.find((attack) => attack.method === method)
                    ).filter((attack): attack is AttackPreset => attack !== undefined);
                    const remainingConsumerAttacks = categoryAttacks.filter(
                      (attack) => !isConsumerStrengthAttack(attack) && !isConsumerSuperResolutionAttack(attack)
                    );
                    const selectedViewpointIds = categoryAttacks
                      .map((attack) => attack.id)
                      .filter((attackId) => selection.attackPresetIds.includes(attackId));
                    const selectedCategoryCount = categoryAttacks.filter((attack) =>
                      selection.attackPresetIds.includes(attack.id)
                    ).length;
                    const selectedViewpointAttacks = categoryAttacks.filter((attack) =>
                      selection.attackPresetIds.includes(attack.id)
                    );
                    const inferredViewpointMotions = VIEWPOINT_MOTION_ORDER.filter((motion) =>
                      selectedViewpointAttacks.some((attack) => parseViewpointMethod(attack.method)?.motion === motion)
                    );
                    const inferredViewpointLookatModes = VIEWPOINT_LOOKAT_MODES.filter((lookatMode) =>
                      selectedViewpointAttacks.some((attack) => parseViewpointMethod(attack.method)?.lookatMode === lookatMode)
                    );
                    const inferredViewpointPhases = VIEWPOINT_PHASES.filter((phase) =>
                      selectedViewpointAttacks.some((attack) => parseViewpointMethod(attack.method)?.phaseIndex === phase)
                    );
                    const renderedViewpointSettings =
                      isViewpointCategory && selectedViewpointIds.length > 0 && !viewpointSettings.enabled
                        ? {
                            ...viewpointSettings,
                            enabled: true,
                            motions: inferredViewpointMotions.length > 0 ? inferredViewpointMotions : [...VIEWPOINT_MOTION_ORDER],
                            lookatModes:
                              inferredViewpointLookatModes.length > 0
                                ? inferredViewpointLookatModes
                                : [...VIEWPOINT_LOOKAT_MODES],
                            phases: inferredViewpointPhases.length > 0 ? inferredViewpointPhases : [...VIEWPOINT_PHASES]
                          }
                        : viewpointSettings;
                    const displayedCategoryTotal = isViewpointCategory ? VIEWPOINT_MOTION_ORDER.length : categoryAttacks.length;
                    const displayedSelectedCategoryCount = isViewpointCategory
                      ? VIEWPOINT_MOTION_ORDER.filter((motion) =>
                          renderedViewpointSettings.enabled && renderedViewpointSettings.motions.includes(motion)
                        ).length
                      : selectedCategoryCount;
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
                    const updateConsumerEnhancementSettings = (nextSettings: ConsumerEnhancementSettings) => {
                      setConsumerEnhancementSettings(nextSettings);
                      setSelection((current) =>
                        applyConsumerEnhancementSettings(current, categoryAttacks, nextSettings)
                      );
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
                    const toggleConsumerEnhancementAttack = (attack: AttackPreset) => {
                      setSelection((current) =>
                        applyConsumerEnhancementSettings(
                          toggleAttackIdInSelection(current, attack.id),
                          categoryAttacks,
                          consumerEnhancementSettings
                        )
                      );
                    };
                    const selectedConsumerSuperResolutionScales =
                      consumerSuperResolutionScalesForSettings(consumerEnhancementSettings);
                    const toggleConsumerSuperResolutionScale = (scale: number) => {
                      const selected = selectedConsumerSuperResolutionScales.includes(scale)
                        ? selectedConsumerSuperResolutionScales.filter((value) => value !== scale)
                        : sortStrengths([...selectedConsumerSuperResolutionScales, scale]);
                      if (selected.length === 0) {
                        return;
                      }
                      updateConsumerEnhancementSettings({
                        ...consumerEnhancementSettings,
                        superResolutionScales: selected
                      });
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
                          <span>
                            {displayedSelectedCategoryCount}/{displayedCategoryTotal}
                          </span>
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
                                if (isConsumerCategory) {
                                  setSelection((current) =>
                                    applyConsumerEnhancementSettings(
                                      addAttackIdsToSelection(
                                        current,
                                        categoryAttacks
                                          .filter((attack) => attack.available !== false)
                                          .map((attack) => attack.id)
                                      ),
                                      categoryAttacks,
                                      consumerEnhancementSettings
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
                                if (isConsumerCategory) {
                                  setSelection((current) =>
                                    applyConsumerEnhancementSettings(
                                      removeAttackIdsFromSelection(
                                        current,
                                        categoryAttacks.map((attack) => attack.id)
                                      ),
                                      categoryAttacks,
                                      consumerEnhancementSettings
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
                                const motionTitle = viewpointMotionLabel(language, motion);
                                const motionSubtitle = englishSubtitleForTitle(
                                  language,
                                  motionTitle,
                                  viewpointMotionEnglishLabel(motion)
                                );

                                return (
                                  <label className="check-tile resource-check-tile" key={motion}>
                                    <input
                                      checked={checked}
                                      disabled={motionIds.length === 0}
                                      onChange={() => toggleViewpointMotion(motion)}
                                      type="checkbox"
                                    />
                                    <span className="tile-copy">
                                      <strong>{motionTitle}</strong>
                                      {motionSubtitle ? <small translate="no">{motionSubtitle}</small> : null}
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
                                <span className="viewpoint-settings-label">
                                  {language === "zh" ? "视线模式" : "Lookat mode"}
                                </span>
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
                                <span className="viewpoint-settings-label">
                                  {language === "zh" ? "相位" : "Phase"}
                                </span>
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
                                <strong>{language === "zh" ? "强度型再生成" : "Strength-based regeneration"}</strong>
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
                                  <strong>
                                    {attackDisplayName(language, regenerationVaeAttack)}
                                  </strong>
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
                                    <span className="viewpoint-settings-label">
                                      {language === "zh" ? "质量" : "Quality"}
                                    </span>
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
                                  <strong>
                                    {attackDisplayName(language, regenerationImageToVideoAttack)}
                                  </strong>
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
                        ) : isConsumerCategory ? (
                          <>
                            {consumerStrengthAttacks.length > 0 ? (
                              <div className="attack-subgroup">
                                <div className="attack-subgroup-title">
                                  <strong>{language === "zh" ? "强度型增强" : "Strength-based enhancement"}</strong>
                                  <span>
                                    {
                                      consumerStrengthAttacks.filter((attack) =>
                                        selection.attackPresetIds.includes(attack.id)
                                      ).length
                                    }
                                    /{consumerStrengthAttacks.length}
                                  </span>
                                </div>
                                <div className="option-grid dense-options viewpoint-variant-grid">
                                  {consumerStrengthAttacks.map((attack) =>
                                    renderAttackTile(attack, toggleConsumerEnhancementAttack, {
                                      title: consumerAttackDisplayName(language, attack)
                                    })
                                  )}
                                </div>

                                <div className="viewpoint-settings-panel">
                                  <StrengthRangeAxis
                                    id="consumer-enhancement"
                                    language={language}
                                    onChange={(nextSettings) =>
                                      updateConsumerEnhancementSettings({
                                        ...consumerEnhancementSettings,
                                        ...nextSettings
                                      })
                                    }
                                    settings={consumerEnhancementSettings}
                                  />
                                </div>
                              </div>
                            ) : null}

                            {consumerSuperResolutionAttacks.length > 0 ? (
                              <div className="attack-subgroup">
                                <div className="attack-subgroup-title">
                                  <strong>{language === "zh" ? "超分辨率" : "Super-resolution"}</strong>
                                  <span>
                                    {
                                      consumerSuperResolutionAttacks.filter((attack) =>
                                        selection.attackPresetIds.includes(attack.id)
                                      ).length
                                    }
                                    /{consumerSuperResolutionAttacks.length}
                                  </span>
                                </div>
                                <div className="option-grid dense-options">
                                  {consumerSuperResolutionAttacks.map((attack) =>
                                    renderAttackTile(attack, toggleConsumerEnhancementAttack, {
                                      title: consumerAttackDisplayName(language, attack)
                                    })
                                  )}
                                </div>

                                <div className="viewpoint-settings-panel">
                                  <div className="viewpoint-settings-row">
                                    <span className="viewpoint-settings-label">
                                      {language === "zh" ? "倍率" : "Scale"}
                                    </span>
                                    <div className="viewpoint-toggle-grid consumer-scale-options">
                                      {CONSUMER_SUPER_RESOLUTION_SCALES.map((scale) => (
                                        <label className="viewpoint-toggle" key={scale}>
                                          <input
                                            checked={selectedConsumerSuperResolutionScales.includes(scale)}
                                            onChange={() => toggleConsumerSuperResolutionScale(scale)}
                                            type="checkbox"
                                          />
                                          <span>{scale}x</span>
                                        </label>
                                      ))}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            ) : null}

                            {remainingConsumerAttacks.length > 0 ? (
                              <div className="attack-subgroup">
                                <div className="attack-subgroup-title">
                                  <strong>{language === "zh" ? "消费级增强流程" : "Consumer enhancement workflows"}</strong>
                                  <span>
                                    {
                                      remainingConsumerAttacks.filter((attack) =>
                                        selection.attackPresetIds.includes(attack.id)
                                      ).length
                                    }
                                    /{remainingConsumerAttacks.length}
                                  </span>
                                </div>
                                <div className="option-grid dense-options">
                                  {remainingConsumerAttacks.map((attack) =>
                                    renderAttackTile(attack, toggleConsumerEnhancementAttack, {
                                      title: consumerAttackDisplayName(language, attack)
                                    })
                                  )}
                                </div>
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
                <button className="button primary" disabled={!canSave || isSavingConfig} onClick={saveConfig} type="button">
                  {isSavingConfig ? <Loader2 className="loading-spinner" size={16} /> : <Save size={16} />}
                  {isSavingConfig ? copy.saving : copy.save}
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
