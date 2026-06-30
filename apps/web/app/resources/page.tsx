"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Boxes,
  Database,
  Gauge,
  PackageCheck,
  Search,
  Shield,
  SlidersHorizontal
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { ResourcePagination } from "@/components/ResourcePagination";
import { useLanguage } from "@/components/LanguageProvider";
import { fetchAlgorithms, fetchAttacks, fetchAttackWeightDownloadJob, fetchDatasetCatalog, fetchDatasetDetail, fetchDatasetDownloadJob, fetchWeightDownloadJob, startAttackWeightDownload, startDatasetDownload, startWeightDownload, uninstallAttackInstallation, uninstallDatasetInstallation, uninstallWatermarkInstallation } from "@/lib/api";
import { localizedName } from "@/lib/i18n";
import {
  algorithms as fallbackAlgorithms,
  attacks as fallbackAttacks,
  datasets as fallbackDatasets
} from "@/lib/mock-data";
import type { AlgorithmVersion, AttackPreset, DatasetCatalogItem, DatasetDownloadJob, DatasetDownloadMode, DatasetVersion, ResourceStatus, WeightDownloadJob } from "@/lib/types";

type ResourceType = "datasets" | "watermarks" | "attacks";
type DeviceFilter = "all" | "cpu" | "gpu";

interface BrowserResource {
  id: string;
  type: ResourceType;
  name: string;
  subtitle: string;
  category: string;
  categoryLabel?: string;
  status: ResourceStatus | "indexed" | "missing";
  statusTone: "ok" | "warn" | "error" | "neutral";
  description?: string;
  method?: string;
  path?: string;
  version?: string;
  sampleCount?: number;
  strengths?: number[];
  params?: Record<string, unknown>;
  requiresGpu?: boolean;
  recommended?: boolean;
  available?: boolean;
  catalog?: DatasetCatalogItem;
  algorithm?: AlgorithmVersion;
  attack?: AttackPreset;
  attacks?: AttackPreset[];
  attackDetail?: AttackResourceDetail;
}

interface AttackResourceVariant {
  id: string;
  label: string;
  sublabel?: string;
}

interface AttackResourceMapping {
  id: string;
  label: string;
  zero: string;
  one: string;
  note?: string;
}

interface AttackResourceWeight {
  id: string;
  label: string;
  value: string;
  tone?: BrowserResource["statusTone"];
}

interface AttackResourceDetail {
  presetCount: number;
  presetIds: string[];
  variants: AttackResourceVariant[];
  mappings: AttackResourceMapping[];
  weights: AttackResourceWeight[];
  notes: string[];
}

const DEFAULT_RESOURCE_PAGE_SIZE = 8;
const HIDDEN_RESOURCE_ATTACK_METHODS = new Set(["identity"]);
const VIEWPOINT_MOTION_ORDER = ["swipe", "shake", "rotate", "rotate_forward"] as const;
const VIEWPOINT_MAX_DISPARITY_LEVELS = [0.01, 0.02, 0.04] as const;
const REGENERATION_UNIT_METHODS = ["2x_regen", "4x_regen", "regen_diffusion", "noise_to_image"] as const;
const REGENERATION_VAE_METHOD = "regen_vae";
const REGENERATION_IMAGE_TO_VIDEO_METHOD = "image_to_vedio";
const REGENERATION_IMAGE_TO_VIDEO_XY = [0, 10, 20, 30, 40, 60] as const;
const REGENERATION_VAE_MODEL_NAMES = [
  "bmshj2018-factorized",
  "cheng2020-anchor",
  "bmshj2018-hyperprior",
  "mbt2018-mean"
] as const;
const REGENERATION_VAE_QUALITIES = [1, 2, 3, 4, 5, 6] as const;
const CONSUMER_STRENGTH_METHODS = ["cew_e1", "cew_e2", "cew_e3", "cew_e4"] as const;
const CONSUMER_SUPER_RESOLUTION_METHODS = ["cew_s1", "cew_s2", "cew_s3"] as const;
const CONSUMER_SUPER_RESOLUTION_SCALES = [2, 4] as const;
const DATASET_CATEGORY_ORDER: Record<string, number> = {
  "natural-benchmark": 10,
  aigc: 20,
  "hd-copyright": 30,
  "open-world": 40,
  document: 50,
  ecommerce: 60,
  "mobile-ui": 70,
  local: 90
};
const DATASET_METHOD_ORDER: Record<string, number> = {
  "ms-coco": 10,
  imagenet: 11,
  diffusiondb: 20,
  "w-bench": 21,
  "4k-benchmark": 30,
  flickr2k: 31,
  "openimages-v7": 40,
  "mapillary-vistas": 41,
  doclaynet: 50,
  publaynet: 51,
  "shopee-product-matching": 60,
  "products-10k": 61,
  rico: 70,
  mobileviews: 71
};
const WATERMARK_CATEGORY_ORDER: Record<string, number> = {
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
const ATTACK_METHOD_ORDER: Record<string, number> = {
  brightness: 10,
  contrast: 11,
  gaussian_blur: 12,
  gaussian_noise: 13,
  jpeg: 14,
  resize: 15,
  resized_crop: 16,
  rotation: 17,
  erasing: 18,
  screen_shoot: 20,
  print_camera: 21,
  combined_physical: 22,
  "2x_regen": 40,
  "4x_regen": 41,
  regen_diffusion: 42,
  noise_to_image: 43,
  regen_vae: 44,
  image_to_vedio: 45,
  cew_e1: 50,
  cew_e2: 51,
  cew_e3: 52,
  cew_e4: 53,
  cew_c1: 54,
  cew_c2: 55,
  cew_c3: 56,
  cew_c4: 57,
  cew_d1: 58,
  cew_d2: 59,
  cew_d3: 60,
  cew_d4: 61,
  cew_d5: 62,
  cew_s1: 63,
  cew_s2: 64,
  cew_s3: 65
};
const WATERMARK_METHOD_ORDER: Record<string, number> = {
  "invisible-watermark-dwtdct": 10,
  "invisible-watermark-dwtdctsvd": 11,
  "traditional-spread-dct": 12,
  hidden: 30,
  stegastamp: 31,
  "ssl-watermarking": 32,
  mbrs: 33,
  cin: 34,
  pimog: 35,
  invismark: 36,
  "invisible-watermark-rivagan": 37,
  "trustmark-c": 38,
  "trustmark-q": 39,
  rawatermark: 40,
  "maskwm-d32": 41,
  wam: 42,
  videoseal: 60,
  pixelseal: 61,
  chunkyseal: 62,
  vine: 70
};
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
const VIEWPOINT_METHOD_PATTERN = /^3d_viewpoint_rerendering_(swipe|shake|rotate|rotate_forward)_phase(\d+)_(point|ahead)$/;
const VIEWPOINT_MOTION_LABELS: Record<(typeof VIEWPOINT_MOTION_ORDER)[number], { en: string; zh: string }> = {
  swipe: { en: "Swipe", zh: "横向扫动" },
  shake: { en: "Shake", zh: "抖动" },
  rotate: { en: "Rotate", zh: "环绕旋转" },
  rotate_forward: { en: "Rotate Forward", zh: "前向环绕" }
};
const DISTORTION_STRENGTH_MAPPINGS: Record<string, { param: string; zero: string; one: string }> = {
  brightness: { param: "factor", zero: "1.0", one: "2.0" },
  contrast: { param: "factor", zero: "1.0", one: "2.0" },
  gaussian_blur: { param: "radius", zero: "0", one: "20" },
  gaussian_noise: { param: "sigma", zero: "0", one: "0.1" },
  jpeg: { param: "quality", zero: "90", one: "10" },
  resize: { param: "scale", zero: "0.5", one: "0.5" },
  resized_crop: { param: "crop scale", zero: "1.0", one: "0.5" },
  rotation: { param: "angle", zero: "0°", one: "45°" },
  erasing: { param: "area ratio", zero: "0", one: "0.25" }
};

function getResponsiveResourcePageSize(width: number, height: number): number {
  if (width >= 1680 && height >= 950) {
    return 8;
  }
  if (width >= 1440 && height >= 820) {
    return 6;
  }
  if (width >= 1200 && height >= 720) {
    return 5;
  }
  if (width >= 900 && height >= 680) {
    return 4;
  }
  return 3;
}

function useResponsiveResourcePageSize(): number {
  const [pageSize, setPageSize] = useState(DEFAULT_RESOURCE_PAGE_SIZE);

  useEffect(() => {
    function updatePageSize() {
      setPageSize(getResponsiveResourcePageSize(window.innerWidth, window.innerHeight));
    }

    updatePageSize();
    window.addEventListener("resize", updatePageSize);
    return () => window.removeEventListener("resize", updatePageSize);
  }, []);

  return pageSize;
}

function compareText(left: string, right: string) {
  return left.localeCompare(right, undefined, { numeric: true });
}

function compareResources(left: BrowserResource, right: BrowserResource) {
  const categoryDelta = categoryRank(left.type, left.category) - categoryRank(right.type, right.category);
  if (categoryDelta !== 0) {
    return categoryDelta;
  }
  const itemDelta = resourceRank(left) - resourceRank(right);
  if (itemDelta !== 0) {
    return itemDelta;
  }
  return compareText(left.name, right.name);
}

function categoryRank(type: ResourceType, category: string) {
  if (type === "datasets") {
    return DATASET_CATEGORY_ORDER[category] ?? 90;
  }
  if (type === "watermarks") {
    return WATERMARK_CATEGORY_ORDER[category] ?? 90;
  }
  if (type === "attacks") {
    return ATTACK_CATEGORY_ORDER[category] ?? 90;
  }
  return 0;
}

function resourceRank(resource: BrowserResource) {
  if (resource.type === "datasets") {
    return DATASET_METHOD_ORDER[resource.id] ?? 90;
  }
  if (resource.type === "watermarks") {
    return WATERMARK_METHOD_ORDER[resource.method ?? ""] ?? 90;
  }
  if (resource.type === "attacks") {
    return attackMethodRank(resource.method ?? "");
  }
  return 0;
}

function parseViewpointMethod(method: string) {
  const match = VIEWPOINT_METHOD_PATTERN.exec(method);
  if (!match) {
    return null;
  }
  return {
    motion: match[1],
    phaseIndex: Number(match[2]),
    lookatMode: match[3]
  };
}

export default function ResourcesPage() {
  const { language, t } = useLanguage();
  const [datasets, setDatasets] = useState<DatasetVersion[]>(fallbackDatasets);
  const [catalogItems, setCatalogItems] = useState<DatasetCatalogItem[]>([]);
  const [algorithms, setAlgorithms] = useState<AlgorithmVersion[]>(fallbackAlgorithms);
  const [attacks, setAttacks] = useState<AttackPreset[]>(fallbackAttacks);
  const [activeType, setActiveType] = useState<ResourceType>("datasets");
  const [query, setQuery] = useState("");
  const [categoryFilter, setCategoryFilter] = useState("all");
  const [deviceFilter, setDeviceFilter] = useState<DeviceFilter>("all");
  const [recommendedOnly, setRecommendedOnly] = useState(false);
  const [availableOnly, setAvailableOnly] = useState(false);
  const [selectedResourceId, setSelectedResourceId] = useState("");
  const [page, setPage] = useState(1);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const pageSize = useResponsiveResourcePageSize();

  useEffect(() => {
    let cancelled = false;

    const applyCatalog = (catalog: Awaited<ReturnType<typeof fetchDatasetCatalog>>) => {
      setCatalogItems(catalog.items);
      setDatasets(
        catalog.items.map((item) => ({
          id: item.id,
          name: item.name,
          sampleCount: item.compactAvailable ? item.compactSampleCount : item.fullSampleCount,
          version: item.installed ? "local" : "catalog",
          path: item.rootPath
        }))
      );
    };

    setCatalogLoading(true);

    fetchDatasetCatalog()
      .then((catalog) => {
        if (cancelled) {
          return;
        }
        applyCatalog(catalog);
      })
      .catch(() => undefined)
      .finally(() => {
        if (!cancelled) {
          setCatalogLoading(false);
        }
      });

    Promise.all([fetchAlgorithms(), fetchAttacks()])
      .then(([apiAlgorithms, apiAttacks]) => {
        if (cancelled) {
          return;
        }
        setAlgorithms(apiAlgorithms.length > 0 ? apiAlgorithms : fallbackAlgorithms);
        setAttacks(apiAttacks.length > 0 ? apiAttacks : fallbackAttacks);
        return Promise.all([fetchAlgorithms({ remote: true }), fetchAttacks({ remote: true })]);
      })
      .then((remote) => {
        if (cancelled || !remote) {
          return;
        }
        const [remoteAlgorithms, remoteAttacks] = remote;
        if (remoteAlgorithms.length > 0) {
          setAlgorithms(remoteAlgorithms);
        }
        if (remoteAttacks.length > 0) {
          setAttacks(remoteAttacks);
        }
      })
      .catch(() => undefined);

    return () => {
      cancelled = true;
    };
  }, []);

  const resourceGroups = useMemo(
    () => ({
      datasets: catalogItems.map((item) => catalogToResource(item, language)).sort(compareResources),
      watermarks: algorithms.map((algorithm) => algorithmToResource(algorithm, language)).sort(compareResources),
      attacks: attackMethodResources(
        attacks.filter((attack) => !HIDDEN_RESOURCE_ATTACK_METHODS.has(attack.method)),
        language
      ).sort(compareResources)
    }),
    [algorithms, attacks, catalogItems, language]
  );

  const activeResources = resourceGroups[activeType];
  const categories = useMemo(() => {
    const labels = new Map<string, string>();
    for (const resource of activeResources) {
      labels.set(resource.category, resource.categoryLabel ?? resource.category);
    }
    return [
      { value: "all", label: t.resources.allResources },
      ...Array.from(labels.entries())
        .sort(([left], [right]) => {
          const rankDelta = categoryRank(activeType, left) - categoryRank(activeType, right);
          return rankDelta !== 0 ? rankDelta : compareText(left, right);
        })
        .map(([value, label]) => ({ value, label }))
    ];
  }, [activeResources, activeType, t.resources.allResources]);

  const filteredResources = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return activeResources.filter((resource) => {
      const queryMatch = !normalizedQuery || searchableText(resource).includes(normalizedQuery);
      const categoryMatch = categoryFilter === "all" || resource.category === categoryFilter;
      const deviceMatch =
        deviceFilter === "all" ||
        (deviceFilter === "gpu" && resource.requiresGpu) ||
        (deviceFilter === "cpu" && !resource.requiresGpu);
      const recommendedMatch = activeType === "datasets" || !recommendedOnly || resource.recommended;
      const availableMatch = !availableOnly || resource.available !== false;
      return queryMatch && categoryMatch && deviceMatch && recommendedMatch && availableMatch;
    });
  }, [activeResources, activeType, availableOnly, categoryFilter, deviceFilter, query, recommendedOnly]);

  const pageCount = Math.max(1, Math.ceil(filteredResources.length / pageSize));
  const usesPagination = pageCount > 1;
  const visibleResources = filteredResources.slice((page - 1) * pageSize, page * pageSize);

  const groupedResources = useMemo(() => {
    const groups = new Map<string, { label: string; resources: BrowserResource[] }>();
    for (const resource of visibleResources) {
      const current = groups.get(resource.category) ?? {
        label: resource.categoryLabel ?? resource.category,
        resources: []
      };
      current.resources.push(resource);
      groups.set(resource.category, current);
    }
    return Array.from(groups.entries())
      .map(([category, group]) => ({ category, ...group }))
      .sort((left, right) => {
        const rankDelta = categoryRank(activeType, left.category) - categoryRank(activeType, right.category);
        return rankDelta !== 0 ? rankDelta : compareText(left.label, right.label);
      });
  }, [activeType, visibleResources]);

  const selectedResource =
    visibleResources.find((resource) => resource.id === selectedResourceId) ?? visibleResources[0] ?? null;
  const totalSamples = datasets.reduce((total, dataset) => total + dataset.sampleCount, 0);
  const renderResourceRow = (resource: BrowserResource) => (
    <button
      className={selectedResource?.id === resource.id ? "resource-row active" : "resource-row"}
      key={resource.id}
      onClick={() => setSelectedResourceId(resource.id)}
      type="button"
    >
      <span className="resource-row-main">
        <strong>{resource.name}</strong>
        <small translate={resource.type === "attacks" ? "no" : undefined}>{resource.subtitle}</small>
      </span>
      <span className="resource-row-meta">
        {resource.type === "datasets" ? (
          resource.catalog?.installed ? (
            <span className="badge ok">{t.resources.installed}</span>
          ) : (
            <span className="badge">{t.resources.notInstalled}</span>
          )
        ) : (
          <>
            {resource.requiresGpu ? <span className="badge warn">{t.common.gpu}</span> : null}
            {resource.recommended ? <span className="badge ok">{t.resources.recommended}</span> : null}
            <span className={badgeClass(resource.statusTone)}>{statusLabel(resource, t)}</span>
          </>
        )}
      </span>
    </button>
  );

  useEffect(() => {
    setCategoryFilter("all");
    setDeviceFilter("all");
    setRecommendedOnly(false);
    setAvailableOnly(false);
    setPage(1);
    setSelectedResourceId("");
  }, [activeType]);

  useEffect(() => {
    setPage(1);
  }, [availableOnly, categoryFilter, deviceFilter, query, recommendedOnly]);

  useEffect(() => {
    if (page > pageCount) {
      setPage(pageCount);
    }
  }, [page, pageCount]);

  useEffect(() => {
    if (visibleResources.length === 0) {
      return;
    }
    if (!visibleResources.some((resource) => resource.id === selectedResourceId)) {
      setSelectedResourceId(visibleResources[0].id);
    }
  }, [selectedResourceId, visibleResources]);

  async function refreshDatasetCatalog() {
    const [localCatalog, remoteCatalog] = await Promise.all([
      fetchDatasetCatalog(),
      fetchDatasetCatalog({ remote: true }).catch(() => null)
    ]);
    const catalog = remoteCatalog ?? localCatalog;
    setCatalogItems(catalog.items);
    setDatasets(
      catalog.items.map((item) => ({
        id: item.id,
        name: item.name,
        sampleCount: item.compactAvailable ? item.compactSampleCount : item.fullSampleCount,
        version: item.installed ? "local" : "catalog",
        path: item.rootPath
      }))
    );
  }

  const handleCatalogItemUpdate = useCallback((item: DatasetCatalogItem) => {
    setCatalogItems((items) =>
      items.map((entry) => {
        if (entry.id !== item.id) {
          return entry;
        }
        if (
          entry.compactAvailable === item.compactAvailable &&
          entry.remoteCompactAvailable === item.remoteCompactAvailable &&
          entry.compactSampleCount === item.compactSampleCount &&
          entry.installed === item.installed &&
          entry.customDownloadReady === item.customDownloadReady
        ) {
          return entry;
        }
        return item;
      })
    );
  }, []);

  async function refreshAlgorithms() {
    const remoteAlgorithms = await fetchAlgorithms({ remote: true });
    if (remoteAlgorithms.length > 0) {
      setAlgorithms(remoteAlgorithms);
    }
  }

  async function refreshAttacks() {
    const remoteAttacks = await fetchAttacks({ remote: true });
    if (remoteAttacks.length > 0) {
      setAttacks(remoteAttacks);
    }
  }

  return (
    <AppShell active="resources">
      <div className="topbar">
        <div className="title-block">
          <h1>{t.resources.title}</h1>
          <p>{t.resources.subtitle}</p>
        </div>
      </div>

      <section className="resource-summary-grid">
        <SummaryCard icon={Database} label={t.console.datasets} value={catalogLoading && catalogItems.length === 0 ? "…" : catalogItems.length.toString()} meta={`${totalSamples.toLocaleString()} ${t.common.samples}`} />
        <SummaryCard icon={Shield} label={t.console.algorithms} value={resourceGroups.watermarks.length.toString()} meta={countByGpu(resourceGroups.watermarks)} />
        <SummaryCard icon={Gauge} label={t.console.attacks} value={resourceGroups.attacks.length.toString()} meta={countByGpu(resourceGroups.attacks)} />
      </section>

      <section className="resources-browser-grid">
        <aside className="panel resource-filter-panel">
          <div className="panel-header">
            <h2>{t.resources.resourceBrowser}</h2>
            <Boxes size={16} />
          </div>
          <div className="panel-body resource-filter-stack">
            <div className="resource-type-list">
              {(["datasets", "watermarks", "attacks"] as ResourceType[]).map((type) => (
                <button
                  className={activeType === type ? "resource-type-button active" : "resource-type-button"}
                  key={type}
                  onClick={() => setActiveType(type)}
                  type="button"
                >
                  {resourceTypeIcon(type)}
                  <span>{resourceTypeLabel(type, t)}</span>
                  <strong>{resourceGroups[type].length}</strong>
                </button>
              ))}
            </div>

            <div className="field">
              <label htmlFor="resource-search">{t.resources.search}</label>
              <div className="input-with-icon">
                <Search size={15} />
                <input
                  id="resource-search"
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder={t.resources.searchPlaceholder}
                  value={query}
                />
              </div>
            </div>

            <div className="field">
              <label htmlFor="resource-category">{t.resources.category}</label>
              <select
                id="resource-category"
                onChange={(event) => setCategoryFilter(event.target.value)}
                value={categoryFilter}
              >
                {categories.map((category) => (
                  <option key={category.value} value={category.value}>
                    {category.label}
                  </option>
                ))}
              </select>
            </div>

            {activeType !== "datasets" ? (
              <div className="segmented-control" aria-label={t.resources.device}>
                {(["all", "cpu", "gpu"] as DeviceFilter[]).map((value) => (
                  <button
                    className={deviceFilter === value ? "active" : ""}
                    key={value}
                    onClick={() => setDeviceFilter(value)}
                    type="button"
                  >
                    {value === "all" ? t.resources.allResources : value === "gpu" ? t.common.gpu : t.common.cpu}
                  </button>
                ))}
              </div>
            ) : null}

            {activeType === "watermarks" || activeType === "attacks" ? (
              <label className="toggle-row">
                <input
                  checked={recommendedOnly}
                  onChange={(event) => setRecommendedOnly(event.target.checked)}
                  type="checkbox"
                />
                <span>{t.resources.recommendedOnly}</span>
              </label>
            ) : null}
            <label className="toggle-row">
              <input
                checked={availableOnly}
                onChange={(event) => setAvailableOnly(event.target.checked)}
                type="checkbox"
              />
              <span>{t.resources.availableOnly}</span>
            </label>
          </div>
        </aside>

        <div className="panel resource-list-panel">
          <div className="panel-header">
            <h2>{resourceTypeLabel(activeType, t)}</h2>
            <span className="count-pill">
              {filteredResources.length}/{activeResources.length}
            </span>
          </div>
          <div
            className="panel-body resource-list-panel-body"
            style={{ ["--resource-page-size" as string]: pageSize }}
          >
            <div className="resource-result-note">
              <SlidersHorizontal size={14} />
              <span>
                {t.resources.showingResults}: {visibleResources.length} / {filteredResources.length}
              </span>
            </div>
            {groupedResources.length > 0 ? (
              <div className="resource-page-list">
                {groupedResources.map((group) => (
                  <div className="dataset-category-group" key={group.category}>
                    <div className="dataset-category-heading">
                      <span>{group.label}</span>
                      <strong>{group.resources.length}</strong>
                    </div>
                    {group.resources.map(renderResourceRow)}
                  </div>
                ))}
              </div>
            ) : null}
            {filteredResources.length === 0 ? (
              <div className="empty compact-empty">
                {catalogLoading && activeType === "datasets" ? t.resources.catalogLoading : t.common.noData}
              </div>
            ) : null}
            {usesPagination ? (
              <ResourcePagination
                onPageChange={setPage}
                page={page}
                pageCount={pageCount}
                previousLabel={t.resources.previousPage}
                nextLabel={t.resources.nextPage}
              />
            ) : null}
          </div>
        </div>

        <aside className="panel resource-detail-browser-panel">
          <div className="panel-header">
            <h2>{t.resources.resourceDetails}</h2>
            <PackageCheck size={16} />
          </div>
          <div className="panel-body">
            {selectedResource ? (
              <ResourceDetail
                language={language}
                onAlgorithmInstalled={refreshAlgorithms}
                onAttackInstalled={refreshAttacks}
                onCatalogItemUpdate={handleCatalogItemUpdate}
                onDatasetInstalled={refreshDatasetCatalog}
                resource={selectedResource}
                t={t}
              />
            ) : (
              <div className="empty compact-empty">{t.common.noData}</div>
            )}
          </div>
        </aside>
      </section>
    </AppShell>
  );
}

function ResourceDetail({
  resource,
  language,
  t,
  onCatalogItemUpdate,
  onDatasetInstalled,
  onAlgorithmInstalled,
  onAttackInstalled
}: {
  resource: BrowserResource;
  language: "zh" | "en";
  t: ReturnType<typeof useLanguage>["t"];
  onCatalogItemUpdate: (item: DatasetCatalogItem) => void;
  onDatasetInstalled: () => void;
  onAlgorithmInstalled: () => void;
  onAttackInstalled: () => void;
}) {
  const attackDetail = resource.type === "attacks" ? resource.attackDetail : undefined;
  const attackWeightTarget =
    resource.type === "attacks"
      ? resource.attack ?? resource.attacks?.find((item) => item.weightsPackRequired === true)
      : undefined;
  const attackGroupWeightsInstalled =
    resource.type === "attacks" && resource.attacks
      ? resource.attacks.every((item) => item.weightsPackRequired !== true || item.weightsInstalled === true)
      : attackWeightTarget?.weightsInstalled === true;
  return (
    <div className="resource-detail-stack">
      <div>
        <div className="detail-title-row">
          <h3>{resource.name}</h3>
          <span className={badgeClass(resource.statusTone)}>{statusLabel(resource, t)}</span>
        </div>
        <p>{resource.description || resource.subtitle}</p>
      </div>

      <div className={resource.type === "datasets" ? "detail-metrics-grid dataset-only-category" : "detail-metrics-grid"}>
        {resource.type === "datasets" ? (
          <DetailMetric label={t.resources.category} value={resource.category} />
        ) : attackDetail ? (
          <>
            <DetailMetric label="ID" value={resource.id} />
            <DetailMetric label={t.resources.category} value={resource.categoryLabel ?? resource.category} />
            <DetailMetric label="Method" value={resource.method ?? "n/a"} />
            <DetailMetric label={t.resources.device} value={resource.requiresGpu ? t.common.gpu : t.common.cpu} />
            <DetailMetric label={language === "zh" ? "底层 preset" : "Presets"} value={attackDetail.presetCount.toString()} />
            <DetailMetric
              label={language === "zh" ? "权重" : "Weights"}
              value={attackDetail.weights.length > 0 ? attackDetail.weights.length.toString() : language === "zh" ? "无需额外权重" : "none"}
            />
          </>
        ) : (
          <>
            <DetailMetric label="ID" value={resource.id} />
            <DetailMetric label={t.resources.category} value={resource.categoryLabel ?? resource.category} />
            <DetailMetric label="Method" value={resource.method ?? "n/a"} />
            <DetailMetric label={t.resources.device} value={resource.requiresGpu ? t.common.gpu : t.common.cpu} />
            {resource.sampleCount != null ? (
              <DetailMetric label={t.common.samples} value={resource.sampleCount.toLocaleString()} />
            ) : null}
            {resource.version ? <DetailMetric label="Version" value={resource.version} /> : null}
          </>
        )}
      </div>

      {attackDetail ? <AttackResourceDetailPanel detail={attackDetail} language={language} /> : null}

      {resource.strengths && resource.type !== "attacks" ? (
        <div className="detail-section">
          <strong>{t.resources.strengthGrid}</strong>
          <div className="strength-chip-row">
            {resource.strengths.map((strength) => (
              <span className="badge" key={strength}>
                {strength}
              </span>
            ))}
          </div>
        </div>
      ) : null}

      {resource.params && Object.keys(resource.params).length > 0 && resource.type !== "attacks" ? (
        <div className="detail-section">
          <strong>Params</strong>
          <pre className="detail-json">{JSON.stringify(resource.params, null, 2)}</pre>
        </div>
      ) : null}

      {resource.path && resource.type !== "datasets" ? (
        <div className="detail-section">
          <strong>Path</strong>
          <code>{resource.path}</code>
        </div>
      ) : null}

      {resource.type === "datasets" && resource.catalog ? (
        <DatasetDownloadPanel
          catalog={resource.catalog}
          language={language}
          onCatalogUpdate={onCatalogItemUpdate}
          onInstalled={onDatasetInstalled}
          t={t}
        />
      ) : null}

      {resource.type === "watermarks" && resource.algorithm?.weightsPackRequired ? (
        <WeightDownloadPanel
          algorithm={resource.algorithm}
          key={`watermark-${resource.id}`}
          onInstalled={onAlgorithmInstalled}
          t={t}
          variant="watermark"
        />
      ) : null}

      {resource.type === "attacks" && attackWeightTarget?.weightsPackRequired ? (
        <WeightDownloadPanel
          attack={attackWeightTarget}
          groupWeightsInstalled={attackGroupWeightsInstalled}
          key={`attack-${resource.id}-${attackWeightTarget.id}`}
          onInstalled={onAttackInstalled}
          t={t}
          variant="attack"
        />
      ) : null}

    </div>
  );
}

function AttackResourceDetailPanel({
  detail,
  language
}: {
  detail: AttackResourceDetail;
  language: "zh" | "en";
}) {
  return (
    <>
      <div className="detail-section">
        <strong>{detail.variants.length > 1 ? (language === "zh" ? "底层执行 preset" : "Execution presets") : language === "zh" ? "执行接口" : "Execution interface"}</strong>
        <div className="attack-method-chip-grid">
          {detail.variants.map((variant) => (
            <span className="attack-method-chip" key={variant.id}>
              <strong>{variant.label}</strong>
              {variant.sublabel ? <small translate="no">{variant.sublabel}</small> : null}
            </span>
          ))}
        </div>
      </div>

      {detail.mappings.length > 0 ? (
        <div className="detail-section">
          <strong>{language === "zh" ? "强度 0/1 映射" : "Strength 0/1 mapping"}</strong>
          <div className="attack-mapping-list">
            {detail.mappings.map((mapping) => (
              <div className="attack-mapping-row" key={mapping.id}>
                <span>{mapping.label}</span>
                <code>0 → {mapping.zero}</code>
                <code>1 → {mapping.one}</code>
                {mapping.note ? <small>{mapping.note}</small> : null}
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <div className="detail-section">
        <strong>{language === "zh" ? "权重 / 模型" : "Weights / models"}</strong>
        {detail.weights.length > 0 ? (
          <div className="attack-weight-list">
            {detail.weights.map((weight) => (
              <div className="attack-weight-row" key={weight.id}>
                <span>{weight.label}</span>
                {weight.tone ? <span className={badgeClass(weight.tone)}>{weightStatusLabel(weight.tone, language)}</span> : null}
              </div>
            ))}
          </div>
        ) : (
          <p className="dataset-download-hint">
            {language === "zh" ? "该攻击族不需要额外攻击权重。" : "This attack family does not require extra attack weights."}
          </p>
        )}
      </div>

      {detail.notes.length > 0 ? (
        <div className="detail-section">
          <strong>{language === "zh" ? "配置说明" : "Configuration notes"}</strong>
          <div className="attack-family-notes">
            {detail.notes.map((note) => (
              <span key={note}>{note}</span>
            ))}
          </div>
        </div>
      ) : null}
    </>
  );
}

function SummaryCard({
  icon: Icon,
  label,
  value,
  meta
}: {
  icon: typeof Database;
  label: string;
  value: string;
  meta: string;
}) {
  return (
    <div className="summary-card">
      <Icon size={18} />
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{meta}</small>
    </div>
  );
}

function DetailMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="detail-metric">
      <span>{label}</span>
      <strong title={value}>{value}</strong>
    </div>
  );
}

function formatDatasetSubtitle(item: DatasetCatalogItem, language: "zh" | "en"): string {
  if (item.officialTotalImages && item.officialTotalImages > 0) {
    return language === "zh"
      ? `官方约 ${item.officialTotalImages.toLocaleString()} 张`
      : `official ~${item.officialTotalImages.toLocaleString()}`;
  }
  return language === "zh" ? "官方总量待公布" : "official total TBD";
}

function catalogToResource(item: DatasetCatalogItem, language: "zh" | "en"): BrowserResource {
  const displayName = language === "zh" ? item.nameZh : item.name;
  const description = language === "zh" ? item.descriptionZh : item.description;
  const categoryLabel = language === "zh" ? item.categoryZh : item.category;
  const sampleCount = item.compactAvailable ? item.compactSampleCount : item.fullSampleCount;
  return {
    id: item.id,
    type: "datasets",
    name: displayName,
    subtitle: formatDatasetSubtitle(item, language),
    category: item.category,
    categoryLabel,
    status: item.installed ? "enabled" : "reviewed",
    statusTone: item.installed ? "ok" : "warn",
    available: true,
    sampleCount,
    version: item.installed ? "local" : "catalog",
    path: item.rootPath,
    description,
    catalog: item
  };
}

function WeightDownloadPanel({
  algorithm,
  attack,
  groupWeightsInstalled,
  onInstalled,
  t,
  variant
}: {
  algorithm?: AlgorithmVersion;
  attack?: AttackPreset;
  groupWeightsInstalled?: boolean;
  onInstalled: () => void;
  t: ReturnType<typeof useLanguage>["t"];
  variant: "watermark" | "attack";
}) {
  const resource = variant === "watermark" ? algorithm : attack;
  const identifier = resource?.id ?? "";
  const [job, setJob] = useState<WeightDownloadJob | null>(null);
  const [busy, setBusy] = useState(false);
  const [uninstallBusy, setUninstallBusy] = useState(false);
  const [uninstallError, setUninstallError] = useState<string | null>(null);
  const [uninstallNotice, setUninstallNotice] = useState<string | null>(null);
  const installedJobRef = useRef<string | null>(null);

  useEffect(() => {
    if (!job || job.status === "succeeded" || job.status === "failed" || job.status === "cancelled") {
      return;
    }
    const fetchJob = variant === "watermark" ? fetchWeightDownloadJob : fetchAttackWeightDownloadJob;
    let cancelled = false;
    const poll = () => {
      fetchJob(job.id)
        .then((next) => {
          if (!cancelled) {
            setJob(next);
          }
        })
        .catch(() => undefined);
    };
    poll();
    const timer = window.setInterval(poll, 500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [job, variant]);

  useEffect(() => {
    if (job?.status !== "succeeded" || installedJobRef.current === job.id) {
      return;
    }
    installedJobRef.current = job.id;
    void onInstalled();
  }, [job, onInstalled]);

  useEffect(() => {
    if (resource?.weightsInstalled === true) {
      setJob(null);
    }
  }, [resource?.weightsInstalled, resource?.id]);

  useEffect(() => {
    setUninstallError(null);
    setUninstallNotice(null);
  }, [resource?.id, variant]);

  if (!resource) {
    return null;
  }

  const installed =
    (variant === "attack" ? groupWeightsInstalled : resource.weightsInstalled === true) || job?.status === "succeeded";
  const jobInFlight = job?.status === "queued" || job?.status === "running";
  const canStart = resource.weightsDownloadReady === true && !installed && !jobInFlight;
  const progressPercent =
    job && job.totalItems > 0 ? Math.round((job.completedItems / job.totalItems) * 100) : job?.progress ?? 0;
  const panelTitle = variant === "watermark" ? t.resources.weightDownloadPanel : t.resources.attackWeightDownloadPanel;
  const alreadyInstalled =
    variant === "watermark" ? t.resources.weightsAlreadyInstalled : t.resources.attackWeightsAlreadyInstalled;
  const unavailable = variant === "watermark" ? t.resources.weightsUnavailable : t.resources.attackWeightsUnavailable;
  const readyLabel = variant === "watermark" ? t.resources.weightDownloadReady : t.resources.attackWeightDownloadReady;
  const canUninstall = installed && !jobInFlight && !uninstallBusy;
  const showUninstallOnly = Boolean(uninstallNotice);

  async function handleStart() {
    setUninstallNotice(null);
    setUninstallError(null);
    setBusy(true);
    try {
      const startDownload = variant === "watermark" ? startWeightDownload : startAttackWeightDownload;
      const fetchJob = variant === "watermark" ? fetchWeightDownloadJob : fetchAttackWeightDownloadJob;
      const created = await startDownload(identifier);
      setJob(created);
      if (created.status === "queued" || created.status === "running") {
        const latest = await fetchJob(created.id);
        setJob(latest);
      }
    } finally {
      setBusy(false);
    }
  }

  async function handleUninstall() {
    setUninstallBusy(true);
    setUninstallError(null);
    setUninstallNotice(null);
    try {
      const uninstall =
        variant === "watermark" ? uninstallWatermarkInstallation : uninstallAttackInstallation;
      const result = await uninstall(identifier);
      setJob(null);
      installedJobRef.current = null;
      setUninstallNotice(t.resources.uninstallSuccess);
      void onInstalled();
    } catch (error) {
      setUninstallError(error instanceof Error ? error.message : t.resources.uninstallFailed);
    } finally {
      setUninstallBusy(false);
    }
  }

  return (
    <div className="detail-section dataset-download-panel">
      <strong>{panelTitle}</strong>
      {!showUninstallOnly && installed ? <div className="risk ok">{alreadyInstalled}</div> : null}
      {!showUninstallOnly && !canStart && !installed && !jobInFlight ? <div className="risk warn">{unavailable}</div> : null}
      <div className="download-action-row">
        <button className="button primary" disabled={!canStart || busy || jobInFlight || installed} onClick={handleStart} type="button">
          {t.resources.startDownload}
        </button>
        <button className="button danger" disabled={!canUninstall || uninstallBusy} onClick={handleUninstall} type="button">
          {t.resources.uninstallLocal}
        </button>
      </div>
      {uninstallNotice ? <div className="risk ok">{uninstallNotice}</div> : null}
      {uninstallError ? <div className="risk error">{t.resources.uninstallFailed}: {uninstallError}</div> : null}
      {job ? (
        <div className="dataset-download-progress">
          <div className="dataset-download-progress-head">
            <span>{t.resources.downloadProgress}</span>
            <strong>{progressPercent}%</strong>
          </div>
          <div className="progress-track">
            <div className="progress-bar" style={{ width: `${Math.max(0, Math.min(100, progressPercent))}%` }} />
          </div>
          <small>
            {job.completedItems}/{job.totalItems} · {job.status}
          </small>
          {job.status === "failed" && job.error ? <div className="risk error">{t.resources.downloadFailed}: {job.error}</div> : null}
          {job.status === "succeeded" ? (
            <div className="dataset-download-success">
              <div className="badge ok">{readyLabel}</div>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function DatasetDownloadPanel({
  catalog,
  language,
  onCatalogUpdate,
  onInstalled,
  t
}: {
  catalog: DatasetCatalogItem;
  language: "zh" | "en";
  onCatalogUpdate: (item: DatasetCatalogItem) => void;
  onInstalled: () => void;
  t: ReturnType<typeof useLanguage>["t"];
}) {
  const [mode, setMode] = useState<DatasetDownloadMode>("compact");
  const [seed, setSeed] = useState(42);
  const [sampleCount, setSampleCount] = useState(100);
  const [job, setJob] = useState<DatasetDownloadJob | null>(null);
  const [busy, setBusy] = useState(false);
  const [uninstallBusy, setUninstallBusy] = useState(false);
  const [uninstallError, setUninstallError] = useState<string | null>(null);
  const [uninstallNotice, setUninstallNotice] = useState<string | null>(null);
  const [detail, setDetail] = useState<DatasetCatalogItem>(catalog);
  const [detailLoading, setDetailLoading] = useState(false);
  const installedJobRef = useRef<string | null>(null);
  const probeTokenRef = useRef(0);

  useEffect(() => {
    setDetail(catalog);
    setDetailLoading(false);
    setJob(null);
    setUninstallError(null);
    setUninstallNotice(null);
  }, [catalog.id]);

  useEffect(() => {
    setUninstallError(null);
    setUninstallNotice(null);
  }, [mode, seed, sampleCount]);

  useEffect(() => {
    if (catalog.installed || catalog.compactAvailable || catalog.remoteCompactAvailable) {
      return;
    }
    const token = probeTokenRef.current + 1;
    probeTokenRef.current = token;
    let cancelled = false;
    setDetailLoading(true);
    fetchDatasetDetail(catalog.id)
      .then((item) => {
        if (cancelled || token !== probeTokenRef.current) {
          return;
        }
        setDetail(item);
        onCatalogUpdate(item);
      })
      .catch(() => undefined)
      .finally(() => {
        if (!cancelled && token === probeTokenRef.current) {
          setDetailLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [catalog.id, catalog.installed, catalog.compactAvailable, catalog.remoteCompactAvailable, onCatalogUpdate]);

  useEffect(() => {
    if (!job || job.status === "succeeded" || job.status === "failed" || job.status === "cancelled") {
      return;
    }
    const timer = window.setInterval(() => {
      fetchDatasetDownloadJob(job.id)
        .then((next) => setJob(next))
        .catch(() => undefined);
    }, 1000);
    return () => window.clearInterval(timer);
  }, [job]);

  useEffect(() => {
    if (job?.status !== "succeeded" || installedJobRef.current === job.id) {
      return;
    }
    installedJobRef.current = job.id;
    onInstalled();
    fetchDatasetDetail(catalog.id)
      .then((item) => {
        setDetail(item);
        onCatalogUpdate(item);
      })
      .catch(() => undefined);
  }, [job, onInstalled, catalog.id, onCatalogUpdate]);

  const compactReady = detail.compactAvailable;
  const customReady = detail.customDownloadReady;
  const compactInstalled = detail.installed === true;
  const jobSucceededForMode =
    job?.status === "succeeded" &&
    job.mode === mode &&
    (mode === "compact" || (job.seed === seed && job.sampleCount === sampleCount));
  const installedForMode = mode === "compact" ? compactInstalled || jobSucceededForMode : jobSucceededForMode;
  const canStart = mode === "compact" ? compactReady && !installedForMode : customReady && !installedForMode;
  const totalImages = detail.officialTotalImages ?? 0;
  const ossProbing = detailLoading && !compactReady && !installedForMode;
  const progressPercent =
    job && job.totalItems > 0 ? Math.round((job.completedItems / job.totalItems) * 100) : job?.progress ?? 0;
  const jobInFlight = job?.status === "queued" || job?.status === "running";
  const canUninstall = installedForMode && !busy && !uninstallBusy && !jobInFlight;
  const showUninstallOnly = Boolean(uninstallNotice);

  async function handleStart() {
    setUninstallNotice(null);
    setUninstallError(null);
    setBusy(true);
    try {
      const created = await startDatasetDownload(detail.id, {
        mode,
        seed,
        sampleCount: mode === "compact" ? 1000 : sampleCount
      });
      setJob(created);
    } finally {
      setBusy(false);
    }
  }

  async function handleUninstall() {
    setUninstallBusy(true);
    setUninstallError(null);
    setUninstallNotice(null);
    try {
      const result = await uninstallDatasetInstallation(detail.id, {
        mode,
        seed,
        sampleCount: mode === "compact" ? 1000 : sampleCount
      });
      setJob(null);
      installedJobRef.current = null;
      setUninstallNotice(t.resources.uninstallSuccess);
      const refreshed = await fetchDatasetDetail(detail.id);
      setDetail(refreshed);
      onCatalogUpdate(refreshed);
      onInstalled();
    } catch (error) {
      setUninstallError(error instanceof Error ? error.message : t.resources.uninstallFailed);
    } finally {
      setUninstallBusy(false);
    }
  }

  return (
    <div className="detail-section dataset-download-panel">
      <strong>{t.resources.downloadPanel}</strong>
      <div className="segmented-control dataset-download-mode">
        <button className={mode === "compact" ? "active" : ""} onClick={() => setMode("compact")} type="button">
          {t.resources.compactDownload}
        </button>
        <button className={mode === "custom" ? "active" : ""} onClick={() => setMode("custom")} type="button">
          {t.resources.customDownload}
        </button>
      </div>
      {!showUninstallOnly && mode === "compact" ? (
        <p className="dataset-download-hint">
          {`${t.resources.compactPack}：${detail.compactSampleCount.toLocaleString()} ${t.resources.imagesUnit}`}
        </p>
      ) : null}
      {!showUninstallOnly && mode === "custom" ? (
        <div className="dataset-pool-summary">
          <span>
            {t.resources.datasetTotalImages}：
            {totalImages > 0 ? `${totalImages.toLocaleString()} ${t.resources.imagesUnit}` : "—"}
          </span>
        </div>
      ) : null}
      {!showUninstallOnly && ossProbing && mode === "compact" ? <div className="risk warn">{t.resources.ossProbing}</div> : null}
      {!showUninstallOnly && !compactReady && !ossProbing && mode === "compact" ? (
        <div className="risk warn">{t.resources.compactUnavailable}</div>
      ) : null}
      {!showUninstallOnly && compactReady && !installedForMode && detail.remoteCompactAvailable && mode === "compact" ? (
        <div className="risk ok">{t.resources.compactRemoteReady}</div>
      ) : null}
      {!showUninstallOnly && installedForMode && mode === "compact" ? (
        <div className="risk ok">{t.resources.compactAlreadyInstalled}</div>
      ) : null}
      {!showUninstallOnly && !customReady && mode === "custom" ? <div className="risk warn">{t.resources.customUnavailable}</div> : null}
      {!showUninstallOnly && mode === "custom" ? (
        <div className="dataset-download-fields">
          <label>
            {t.resources.randomSeed}
            <input
              min={0}
              onChange={(event) => setSeed(Number(event.target.value) || 0)}
              type="number"
              value={seed}
            />
          </label>
          <label>
            {t.resources.downloadCount}
            <input
              min={1}
              max={10000}
              onChange={(event) => setSampleCount(Number(event.target.value) || 1)}
              type="number"
              value={sampleCount}
            />
          </label>
        </div>
      ) : null}
      <div className="download-action-row">
        <button className="button primary" disabled={!canStart || busy || jobInFlight} onClick={handleStart} type="button">
          {t.resources.startDownload}
        </button>
        <button className="button danger" disabled={!canUninstall || uninstallBusy} onClick={handleUninstall} type="button">
          {t.resources.uninstallLocal}
        </button>
      </div>
      {uninstallNotice ? <div className="risk ok">{uninstallNotice}</div> : null}
      {uninstallError ? <div className="risk error">{t.resources.uninstallFailed}: {uninstallError}</div> : null}
      {job ? (
        <div className="dataset-download-progress">
          <div className="dataset-download-progress-head">
            <span>{t.resources.downloadProgress}</span>
            <strong>{progressPercent}%</strong>
          </div>
          <div className="progress-track">
            <div className="progress-bar" style={{ width: `${Math.max(0, Math.min(100, progressPercent))}%` }} />
          </div>
          <small>
            {job.completedItems}/{job.totalItems} · {job.status}
            {job.message ? ` · ${job.message}` : ""}
          </small>
          {job.status === "failed" && job.error ? <div className="risk error">{t.resources.downloadFailed}: {job.error}</div> : null}
          {job.status === "succeeded" ? (
            <div className="dataset-download-success">
              <div className="badge ok">{t.resources.downloadReady}</div>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function datasetToResource(dataset: DatasetVersion, language: "zh" | "en"): BrowserResource {
  return {
    id: dataset.id,
    type: "datasets",
    name: localizedName(language, dataset.id, dataset.name),
    subtitle: `${dataset.sampleCount.toLocaleString()} images · ${dataset.version}`,
    category: "dataset",
    status: "enabled",
    statusTone: "ok",
    available: true,
    sampleCount: dataset.sampleCount,
    version: dataset.version,
    path: dataset.path,
    description: dataset.path
  };
}

function normalizeWatermarkCategory(category: string | undefined) {
  if (category === "classical" || category === "traditional_watermark") {
    return "traditional_watermark";
  }
  return "deep_watermark";
}

function watermarkCategoryLabel(category: string, language: "zh" | "en") {
  const labels = {
    traditional_watermark: language === "zh" ? "传统水印" : "Traditional watermark",
    deep_watermark: language === "zh" ? "深度水印" : "Deep watermark"
  };
  return labels[category as keyof typeof labels] ?? category;
}

function algorithmDisplayName(algorithm: AlgorithmVersion) {
  return algorithm.name;
}

function algorithmSubtitle(algorithm: AlgorithmVersion, categoryLabel: string) {
  return `${algorithm.method ?? algorithm.id} · ${categoryLabel}`;
}

function isAsciiText(value: string) {
  return /^[\x00-\x7F]+$/.test(value.trim());
}

function englishSubtitleForTitle(language: "zh" | "en", title: string, englishName: string | undefined) {
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

function algorithmToResource(algorithm: AlgorithmVersion, language: "zh" | "en"): BrowserResource {
  const available = algorithm.available !== false && algorithm.status === "enabled";
  const needsWeights = algorithm.weightsPackRequired === true;
  const weightsReady = !needsWeights || algorithm.weightsInstalled === true;
  const category = normalizeWatermarkCategory(algorithm.category);
  const categoryLabel = watermarkCategoryLabel(category, language);
  return {
    id: algorithm.id,
    type: "watermarks",
    name: algorithmDisplayName(algorithm),
    subtitle: algorithmSubtitle(algorithm, categoryLabel),
    category,
    categoryLabel,
    status: algorithm.status,
    statusTone: available && weightsReady ? "ok" : needsWeights && !weightsReady ? "warn" : available ? "ok" : "warn",
    available,
    method: algorithm.method,
    version: algorithm.version,
    description: algorithm.description,
    params: algorithm.params,
    requiresGpu: algorithm.requiresGpu,
    recommended: algorithm.recommended,
    algorithm
  };
}

function normalizeAttackCategory(attack: AttackPreset) {
  return attack.category ?? "attack";
}

function attackCategoryLabel(category: string, language: "zh" | "en") {
  const zh: Record<string, string> = {
    "3d_viewpoint_rerendering": "3D 视角重渲染",
    consumer_enhancement_workflow_attacks: "消费级增强",
    distortion_attacks: "经典失真",
    physical_channel_attacks: "物理信道",
    regeneration_attacks: "再生成"
  };
  const en: Record<string, string> = {
    "3d_viewpoint_rerendering": "3D viewpoint re-rendering",
    consumer_enhancement_workflow_attacks: "Consumer enhancement",
    distortion_attacks: "Distortion",
    physical_channel_attacks: "Physical channel",
    regeneration_attacks: "Regeneration"
  };
  return (language === "zh" ? zh : en)[category] ?? category;
}

function attackDisplayName(attack: AttackPreset, language: "zh" | "en") {
  const viewpointName = viewpointDisplayName(attack.method, language);
  if (viewpointName) {
    return viewpointName;
  }
  const display = ATTACK_DISPLAY_NAMES[attack.method];
  if (display) {
    return language === "zh" ? display.zh : display.en;
  }
  return localizedName(language, attack.id, attack.name);
}

function attackEnglishName(attack: AttackPreset) {
  const viewpointName = viewpointDisplayName(attack.method, "en");
  return viewpointName ?? ATTACK_DISPLAY_NAMES[attack.method]?.en ?? attack.name;
}

function viewpointDisplayName(method: string, language: "zh" | "en") {
  const parsed = parseViewpointMethod(method);
  if (!parsed) {
    return null;
  }
  const mode = parsed.lookatMode === "point" ? "point" : "ahead";
  return language === "zh"
    ? `3D 视角 ${viewpointMotionLabel(parsed.motion, language)} Phase ${parsed.phaseIndex} (${mode})`
    : `3D Viewpoint ${viewpointMotionLabel(parsed.motion, language)} Phase ${parsed.phaseIndex} (${mode})`;
}

function attackLabelByMethod(method: string, language: "zh" | "en") {
  const display = ATTACK_DISPLAY_NAMES[method];
  return display ? (language === "zh" ? display.zh : display.en) : method;
}

function viewpointMotionLabel(motion: string, language: "zh" | "en") {
  const labels = VIEWPOINT_MOTION_LABELS[motion as (typeof VIEWPOINT_MOTION_ORDER)[number]];
  return labels ? (language === "zh" ? labels.zh : labels.en) : motion;
}

function attackResourceMethod(attack: AttackPreset) {
  if (attack.displayMethod) {
    return attack.displayMethod;
  }
  const parsed = parseViewpointMethod(attack.method);
  return parsed ? parsed.motion : attack.method;
}

function attackResourceGroup(attack: AttackPreset) {
  return attack.displayGroup ?? normalizeAttackCategory(attack);
}

function attackMethodRank(method: string) {
  const viewpointRank = VIEWPOINT_MOTION_ORDER.indexOf(method as (typeof VIEWPOINT_MOTION_ORDER)[number]);
  if (viewpointRank >= 0) {
    return 30 + viewpointRank;
  }
  return ATTACK_METHOD_ORDER[method] ?? 90;
}

function viewpointExecutionRank(method: string) {
  const parsed = parseViewpointMethod(method);
  if (!parsed) {
    return attackMethodRank(method);
  }
  const motionRank = VIEWPOINT_MOTION_ORDER.indexOf(parsed.motion as (typeof VIEWPOINT_MOTION_ORDER)[number]);
  return Math.max(0, motionRank) * 16 + parsed.phaseIndex * 2 + (parsed.lookatMode === "point" ? 0 : 1);
}

function resourceSlug(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "attack";
}

function attackMethodResources(attacks: AttackPreset[], language: "zh" | "en"): BrowserResource[] {
  const groups = new Map<string, { category: string; method: string; attacks: AttackPreset[] }>();
  for (const attack of attacks) {
    const category = attackResourceGroup(attack);
    const method = attackResourceMethod(attack);
    const key = `${category}:${method}`;
    const current = groups.get(key) ?? { category, method, attacks: [] };
    current.attacks.push(attack);
    groups.set(key, current);
  }
  return Array.from(groups.values()).map((group) =>
    attackMethodToResource(group.category, group.method, group.attacks, language)
  );
}

function attackMethodToResource(
  category: string,
  method: string,
  methodAttacks: AttackPreset[],
  language: "zh" | "en"
): BrowserResource {
  const categoryLabel = attackCategoryLabel(category, language);
  const detail = buildAttackResourceDetail(category, method, methodAttacks, language);
  const available = methodAttacks.some((attack) => attack.available !== false);
  const requiresGpu = methodAttacks.some((attack) => attack.requiresGpu);
  const needsWeights = methodAttacks.some((attack) => attack.weightsPackRequired === true);
  const weightsReady =
    !needsWeights || methodAttacks.every((attack) => attack.weightsPackRequired !== true || attack.weightsInstalled === true);
  const name = attackResourceDisplayName(category, method, methodAttacks, language);
  const englishName = attackResourceEnglishName(category, method, methodAttacks);
  const singleAttack = methodAttacks.length === 1 ? methodAttacks[0] : undefined;
  return {
    id: singleAttack?.id ?? `atk-${resourceSlug(category)}-${resourceSlug(method)}`,
    type: "attacks",
    name,
    subtitle: attackMethodSubtitle(name, englishName, categoryLabel, category, method, detail, language),
    category,
    categoryLabel,
    status: available ? "enabled" : "missing",
    statusTone: available && weightsReady ? "ok" : needsWeights && !weightsReady ? "warn" : available ? "ok" : "error",
    available,
    method,
    path: methodAttacks[0]?.categoryPath ?? `evaluator/attacks/${category}`,
    description: attackMethodDescription(category, method, methodAttacks, language),
    requiresGpu,
    recommended: methodAttacks.some((attack) => attack.recommended),
    attack: singleAttack,
    attacks: methodAttacks,
    attackDetail: detail
  };
}

function attackResourceDisplayName(
  category: string,
  method: string,
  methodAttacks: AttackPreset[],
  language: "zh" | "en"
) {
  if (category === "3d_viewpoint_rerendering") {
    return viewpointMotionLabel(method, language);
  }
  const display = ATTACK_DISPLAY_NAMES[method];
  if (display) {
    return language === "zh" ? display.zh : display.en;
  }
  return methodAttacks[0] ? attackDisplayName(methodAttacks[0], language) : method;
}

function attackResourceEnglishName(category: string, method: string, methodAttacks: AttackPreset[]) {
  if (category === "3d_viewpoint_rerendering") {
    return viewpointMotionLabel(method, "en");
  }
  return ATTACK_DISPLAY_NAMES[method]?.en ?? methodAttacks[0]?.name ?? method;
}

function attackMethodSubtitle(
  name: string,
  englishName: string,
  categoryLabel: string,
  category: string,
  method: string,
  detail: AttackResourceDetail,
  language: "zh" | "en"
) {
  const english = englishSubtitleForTitle(language, name, englishName);
  const summary = attackMethodControlSummary(category, method, detail, language);
  return [english, categoryLabel, summary].filter(Boolean).join(" · ");
}

function attackMethodControlSummary(
  category: string,
  method: string,
  detail: AttackResourceDetail,
  language: "zh" | "en"
) {
  if (category === "3d_viewpoint_rerendering") {
    return language === "zh" ? `${detail.presetCount} 个底层 preset` : `${detail.presetCount} execution presets`;
  }
  if (method === REGENERATION_VAE_METHOD) {
    return language === "zh" ? "VAE 权重类型 + quality" : "VAE model + quality";
  }
  if (method === REGENERATION_IMAGE_TO_VIDEO_METHOD) {
    return language === "zh" ? "XY 离散参数" : "XY options";
  }
  if ((CONSUMER_SUPER_RESOLUTION_METHODS as readonly string[]).includes(method)) {
    return language === "zh" ? "超分倍率" : "super-resolution scale";
  }
  if (detail.mappings.length > 0 && detail.mappings.some((mapping) => mapping.zero !== mapping.one)) {
    return language === "zh" ? "0-1 强度映射" : "0-1 strength mapping";
  }
  return language === "zh" ? "固定/离散参数" : "fixed/discrete parameters";
}

function attackMethodDescription(
  category: string,
  method: string,
  methodAttacks: AttackPreset[],
  language: "zh" | "en"
) {
  if (category === "3d_viewpoint_rerendering") {
    const motion = viewpointMotionLabel(method, language);
    return language === "zh"
      ? `${motion} 运动变体；底层由 phase 与 look-at mode 组合展开，攻击强度映射到 max_disparity。`
      : `${motion} motion variant; execution expands by phase and look-at mode, with strength mapped to max_disparity.`;
  }
  return methodAttacks[0]?.description ?? attackResourceEnglishName(category, method, methodAttacks);
}

function buildAttackResourceDetail(
  category: string,
  method: string,
  attacks: AttackPreset[],
  language: "zh" | "en"
): AttackResourceDetail {
  return {
    presetCount: attacks.length,
    presetIds: attacks.map((attack) => attack.id),
    variants: attackResourceVariants(category, method, attacks),
    mappings: attackResourceMappings(category, method, language),
    weights: attackResourceWeights(category, method, attacks, language),
    notes: attackResourceNotes(category, method, language)
  };
}

function attackResourceVariants(category: string, method: string, attacks: AttackPreset[]): AttackResourceVariant[] {
  if (category === "3d_viewpoint_rerendering") {
    return [...attacks]
      .sort((left, right) => viewpointExecutionRank(left.method) - viewpointExecutionRank(right.method))
      .map((attack) => {
        const parsed = parseViewpointMethod(attack.method);
        const phase = attack.viewpointPhase ?? parsed?.phaseIndex;
        const lookat = attack.viewpointLookatMode ?? parsed?.lookatMode;
        return {
          id: attack.id,
          label: `phase ${phase ?? "?"}`,
          sublabel: lookat ? `look-at ${lookat}` : attack.method
        };
      });
  }
  const first = attacks[0];
  return [
    {
      id: first?.id ?? method,
      label: first?.method ?? method,
      sublabel: first?.id
    }
  ];
}

function attackResourceMappings(category: string, method: string, language: "zh" | "en"): AttackResourceMapping[] {
  if (category === "distortion_attacks") {
    const mapping = DISTORTION_STRENGTH_MAPPINGS[method];
    if (!mapping) {
      return [];
    }
    return [
      {
        id: method,
        label: mapping.param,
        zero: mapping.zero,
        one: mapping.one,
        note:
          method === "resize"
            ? language === "zh"
              ? "固定参数，不参与强度连续调节。"
              : "Fixed parameter; not part of continuous strength control."
            : undefined
      }
    ];
  }
  if (category === "physical_channel_attacks") {
    if (method === "combined_physical") {
      return [
        {
          id: method,
          label: attackLabelByMethod(method, language),
          zero: "print 0 + screen 0",
          one: "print 0.5 + screen 0.5",
          note:
            language === "zh"
              ? "0-0.5 先增强打印信道，0.5-1 再叠加屏幕-拍摄信道。"
              : "0-0.5 raises the print channel first; 0.5-1 adds the screen-camera channel."
        }
      ];
    }
    return [
      {
        id: method,
        label: attackLabelByMethod(method, language),
        zero: "mild",
        one: "strong",
        note:
          language === "zh"
            ? "0.5 对应 medium；数值参数逐项线性插值。"
            : "0.5 maps to medium; numeric parameters are interpolated."
      }
    ];
  }
  if (category === "3d_viewpoint_rerendering") {
    return [
      {
        id: "max_disparity",
        label: "max_disparity",
        zero: `${VIEWPOINT_MAX_DISPARITY_LEVELS[0]}`,
        one: `${VIEWPOINT_MAX_DISPARITY_LEVELS[2]}`,
        note:
          language === "zh"
            ? "0.5 对应 0.02；phase 和 look-at mode 在实验配置页继续细分。"
            : "0.5 maps to 0.02; phase and look-at mode remain configurable in the experiment page."
      }
    ];
  }
  if (category === "regeneration_attacks") {
    if ((REGENERATION_UNIT_METHODS as readonly string[]).includes(method)) {
      if (method === "noise_to_image") {
        return [
          {
            id: method,
            label: "step",
            zero: "0",
            one: "1",
            note: language === "zh" ? "CtrlRegen 管线中的 img2img 强度。" : "Img2img strength inside the CtrlRegen pipeline."
          }
        ];
      }
      return [
        {
          id: method,
          label: "noise_step",
          zero: "20",
          one: "100",
          note:
            language === "zh"
              ? "通过 strength 在 20-100 的噪声步范围内线性映射。"
              : "The strength value maps linearly to the 20-100 noise-step range."
        }
      ];
    }
    if (method === REGENERATION_VAE_METHOD) {
      return [
        {
          id: method,
          label: "quality",
          zero: "1",
          one: "6",
          note:
            language === "zh"
              ? "VAE 使用权重类型和 quality 勾选，不使用连续强度轴。"
              : "VAE uses selected model and quality rather than the continuous strength axis."
        }
      ];
    }
    if (method === REGENERATION_IMAGE_TO_VIDEO_METHOD) {
      return [
        {
          id: method,
          label: "xy",
          zero: "0",
          one: "60",
          note: `xy ∈ ${REGENERATION_IMAGE_TO_VIDEO_XY.join("/")}`
        }
      ];
    }
  }
  if (category === "consumer_enhancement_workflow_attacks") {
    if ((CONSUMER_STRENGTH_METHODS as readonly string[]).includes(method)) {
      return [
        {
          id: method,
          label: "strength",
          zero: "light",
          one: "strong",
          note:
            language === "zh"
              ? "实验配置页按 0-1 轴生成档位，执行时映射到 CEW 编辑强度。"
              : "Experiment configuration generates 0-1 levels and maps them to CEW edit strength."
        }
      ];
    }
    if ((CONSUMER_SUPER_RESOLUTION_METHODS as readonly string[]).includes(method)) {
      return [
        {
          id: method,
          label: "scale",
          zero: "2",
          one: "4",
          note: `scale ∈ ${CONSUMER_SUPER_RESOLUTION_SCALES.join("/")}`
        }
      ];
    }
  }
  return [];
}

function attackResourceWeights(
  category: string,
  method: string,
  attacks: AttackPreset[],
  language: "zh" | "en"
): AttackResourceWeight[] {
  const rows = weightRowsFromAttacks(attacks, language);
  if (method === REGENERATION_VAE_METHOD) {
    rows.push(
      {
        id: "vae-models",
        label: language === "zh" ? "VAE 权重类型" : "VAE model types",
        value: REGENERATION_VAE_MODEL_NAMES.join(", "),
        tone: "neutral"
      },
      {
        id: "vae-quality",
        label: "VAE Quality",
        value: REGENERATION_VAE_QUALITIES.join(", "),
        tone: "neutral"
      }
    );
  }
  if (category === "consumer_enhancement_workflow_attacks" && (CONSUMER_SUPER_RESOLUTION_METHODS as readonly string[]).includes(method)) {
    rows.push({
      id: "sr-scales",
      label: language === "zh" ? "可选倍率" : "Selectable scales",
      value: CONSUMER_SUPER_RESOLUTION_SCALES.join(", "),
      tone: "neutral"
    });
  }
  return rows;
}

function attackResourceNotes(category: string, method: string, language: "zh" | "en"): string[] {
  if (category === "physical_channel_attacks") {
    return [
      language === "zh"
        ? "透视矫正可在实验配置页选择：开启、不开启，或两者都跑。"
        : "Perspective correction can be configured as enabled, disabled, or both."
    ];
  }
  if (category === "3d_viewpoint_rerendering") {
    return [
      language === "zh"
        ? "资源页按 4 个运动方法计数；底层执行仍由所选 phase × look-at mode 展开。"
        : "The resource page counts four motion methods; execution still expands by selected phase and look-at mode."
    ];
  }
  if (method === REGENERATION_VAE_METHOD) {
    return [
      language === "zh"
        ? "当前前端只开放 quality 1-6，与实验配置页面保持一致。"
        : "The UI exposes quality 1-6 only, matching the experiment configuration page."
    ];
  }
  if (method === REGENERATION_IMAGE_TO_VIDEO_METHOD) {
    return [`xy ∈ ${REGENERATION_IMAGE_TO_VIDEO_XY.join("/")}`];
  }
  if (category === "consumer_enhancement_workflow_attacks" && method.startsWith("cew_c")) {
    return [
      language === "zh"
        ? "组合增强链是固定流程；具体链路由后端攻击实现定义。"
        : "Composite enhancement chains are fixed workflows defined by the backend attack implementation."
    ];
  }
  if (category === "consumer_enhancement_workflow_attacks" && method.startsWith("cew_d")) {
    return [
      language === "zh"
        ? "深度增强方法可能依赖权重；实现允许 fallback 的方法会在缺失权重时使用本地近似。"
        : "Deep enhancement methods may require weights; fallback-enabled methods use local approximations when weights are missing."
    ];
  }
  return [];
}

function weightRowsFromAttacks(attacks: AttackPreset[], language: "zh" | "en"): AttackResourceWeight[] {
  const rows = new Map<string, AttackResourceWeight>();
  for (const attack of attacks) {
    if (attack.weightsPackRequired !== true || !attack.weightsDir) {
      continue;
    }
    const id = attack.weightsDir;
    const current = rows.get(id);
    const tone: BrowserResource["statusTone"] = attack.weightsInstalled === true ? "ok" : attack.weightsDownloadReady ? "warn" : "error";
    if (current) {
      current.tone = current.tone === "ok" && tone !== "ok" ? tone : current.tone;
      continue;
    }
    rows.set(id, {
      id,
      label: language === "zh" ? "攻击权重包" : "Attack weight pack",
      value: "",
      tone
    });
  }
  return Array.from(rows.values());
}

function weightStatusLabel(tone: BrowserResource["statusTone"], language: "zh" | "en") {
  if (tone === "ok") {
    return language === "zh" ? "已安装" : "installed";
  }
  if (tone === "warn") {
    return language === "zh" ? "可下载" : "downloadable";
  }
  if (tone === "error") {
    return language === "zh" ? "缺失" : "missing";
  }
  return language === "zh" ? "可选" : "optional";
}

function searchableText(resource: BrowserResource): string {
  return [
    resource.id,
    resource.name,
    resource.subtitle,
    resource.category,
    resource.categoryLabel,
    resource.description,
    resource.method,
    resource.path,
    ...(resource.attackDetail?.variants.flatMap((variant) => [variant.label, variant.sublabel]) ?? []),
    ...(resource.attackDetail?.mappings.flatMap((mapping) => [mapping.label, mapping.zero, mapping.one, mapping.note]) ?? []),
    ...(resource.attackDetail?.weights.flatMap((weight) => [weight.label, weight.value]) ?? [])
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function resourceTypeLabel(type: ResourceType, t: ReturnType<typeof useLanguage>["t"]): string {
  if (type === "datasets") {
    return t.console.datasets;
  }
  if (type === "watermarks") {
    return t.console.algorithms;
  }
  if (type === "attacks") {
    return t.console.attacks;
  }
  return t.console.attacks;
}

function resourceTypeIcon(type: ResourceType) {
  if (type === "datasets") {
    return <Database size={16} />;
  }
  if (type === "watermarks") {
    return <Shield size={16} />;
  }
  if (type === "attacks") {
    return <Gauge size={16} />;
  }
  return <Gauge size={16} />;
}

function badgeClass(tone: BrowserResource["statusTone"]) {
  if (tone === "ok") {
    return "badge ok";
  }
  if (tone === "warn") {
    return "badge warn";
  }
  if (tone === "error") {
    return "badge error";
  }
  return "badge";
}

function statusLabel(resource: BrowserResource, t: ReturnType<typeof useLanguage>["t"]) {
  if (resource.status === "indexed") {
    return t.common.indexed;
  }
  if (resource.status === "missing") {
    return "Missing";
  }
  return t.common.status[resource.status];
}

function countByGpu(resources: Array<{ requiresGpu?: boolean }>) {
  const gpu = resources.filter((resource) => resource.requiresGpu).length;
  const cpu = resources.length - gpu;
  return `${cpu} CPU · ${gpu} GPU`;
}
