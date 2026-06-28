"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Archive,
  Boxes,
  ChevronLeft,
  ChevronRight,
  Cpu,
  Database,
  Gauge,
  HardDrive,
  PackageCheck,
  Search,
  Shield,
  SlidersHorizontal
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { useLanguage } from "@/components/LanguageProvider";
import { fetchAlgorithms, fetchAttacks, fetchDatasets } from "@/lib/api";
import { localizedName } from "@/lib/i18n";
import {
  algorithms as fallbackAlgorithms,
  artifacts,
  attacks as fallbackAttacks,
  datasets as fallbackDatasets
} from "@/lib/mock-data";
import type { AlgorithmVersion, AttackPreset, DatasetVersion, ModelArtifact, ResourceStatus } from "@/lib/types";

type ResourceType = "datasets" | "watermarks" | "attacks" | "weights";
type DeviceFilter = "all" | "cpu" | "gpu";

interface BrowserResource {
  id: string;
  type: ResourceType;
  name: string;
  subtitle: string;
  category: string;
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
  size?: string;
  checksum?: string;
}

const PAGE_SIZE = 15;

export default function ResourcesPage() {
  const { language, t } = useLanguage();
  const [datasets, setDatasets] = useState<DatasetVersion[]>(fallbackDatasets);
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

  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchDatasets(), fetchAlgorithms(), fetchAttacks()])
      .then(([apiDatasets, apiAlgorithms, apiAttacks]) => {
        if (cancelled) {
          return;
        }
        setDatasets(apiDatasets);
        setAlgorithms(apiAlgorithms.length > 0 ? apiAlgorithms : fallbackAlgorithms);
        setAttacks(apiAttacks.length > 0 ? apiAttacks : fallbackAttacks);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  const resourceGroups = useMemo(
    () => ({
      datasets: datasets.map((dataset) => datasetToResource(dataset, language)),
      watermarks: algorithms.map(algorithmToResource),
      attacks: attacks.map((attack) => attackToResource(attack, language)),
      weights: artifacts.map(weightToResource)
    }),
    [algorithms, attacks, datasets, language]
  );

  const activeResources = resourceGroups[activeType];
  const categories = useMemo(() => {
    const values = Array.from(new Set(activeResources.map((resource) => resource.category))).sort();
    return ["all", ...values];
  }, [activeResources]);

  const filteredResources = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return activeResources.filter((resource) => {
      const queryMatch = !normalizedQuery || searchableText(resource).includes(normalizedQuery);
      const categoryMatch = categoryFilter === "all" || resource.category === categoryFilter;
      const deviceMatch =
        deviceFilter === "all" ||
        (deviceFilter === "gpu" && resource.requiresGpu) ||
        (deviceFilter === "cpu" && !resource.requiresGpu);
      const recommendedMatch =
        activeType === "datasets" || activeType === "weights" || !recommendedOnly || resource.recommended;
      const availableMatch = !availableOnly || resource.available !== false;
      return queryMatch && categoryMatch && deviceMatch && recommendedMatch && availableMatch;
    });
  }, [activeResources, activeType, availableOnly, categoryFilter, deviceFilter, query, recommendedOnly]);

  const pageCount = Math.max(1, Math.ceil(filteredResources.length / PAGE_SIZE));
  const visibleResources = filteredResources.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
  const selectedResource =
    filteredResources.find((resource) => resource.id === selectedResourceId) ?? visibleResources[0] ?? null;
  const totalSamples = datasets.reduce((total, dataset) => total + dataset.sampleCount, 0);

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
    if (!selectedResource && filteredResources.length > 0) {
      setSelectedResourceId(filteredResources[0].id);
    }
  }, [filteredResources, selectedResource]);

  return (
    <AppShell active="resources">
      <div className="topbar">
        <div className="title-block">
          <h1>{t.resources.title}</h1>
          <p>{t.resources.subtitle}</p>
        </div>
      </div>

      <section className="resource-summary-grid">
        <SummaryCard icon={Database} label={t.console.datasets} value={datasets.length.toString()} meta={`${totalSamples.toLocaleString()} ${t.common.samples}`} />
        <SummaryCard icon={Shield} label={t.console.algorithms} value={algorithms.length.toString()} meta={countByGpu(algorithms)} />
        <SummaryCard icon={Gauge} label={t.console.attacks} value={attacks.length.toString()} meta={countByGpu(attacks)} />
        <SummaryCard icon={HardDrive} label={t.resources.weightFolders} value={artifacts.length.toString()} meta="resources/weights" />
      </section>

      <section className="resources-browser-grid">
        <aside className="panel resource-filter-panel">
          <div className="panel-header">
            <h2>{t.resources.resourceBrowser}</h2>
            <Boxes size={16} />
          </div>
          <div className="panel-body resource-filter-stack">
            <div className="resource-type-list">
              {(["datasets", "watermarks", "attacks", "weights"] as ResourceType[]).map((type) => (
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
                  <option key={category} value={category}>
                    {category === "all" ? t.resources.allResources : category}
                  </option>
                ))}
              </select>
            </div>

            {activeType !== "datasets" && activeType !== "weights" ? (
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
          <div className="panel-body resource-browser-list">
            <div className="resource-result-note">
              <SlidersHorizontal size={14} />
              <span>
                {t.resources.showingResults}: {visibleResources.length} / {filteredResources.length}
              </span>
            </div>
            {visibleResources.map((resource) => (
              <button
                className={selectedResource?.id === resource.id ? "resource-row active" : "resource-row"}
                key={resource.id}
                onClick={() => setSelectedResourceId(resource.id)}
                type="button"
              >
                <span className="resource-row-main">
                  <strong>{resource.name}</strong>
                  <small>{resource.subtitle}</small>
                </span>
                <span className="resource-row-meta">
                  {resource.requiresGpu ? <span className="badge warn">{t.common.gpu}</span> : null}
                  {resource.recommended ? <span className="badge ok">{t.resources.recommended}</span> : null}
                  <span className={badgeClass(resource.statusTone)}>{statusLabel(resource, t)}</span>
                </span>
              </button>
            ))}
            {visibleResources.length === 0 ? <div className="empty compact-empty">{t.common.noData}</div> : null}
            <div className="pagination-row">
              <button
                className="icon-button"
                disabled={page <= 1}
                onClick={() => setPage((current) => Math.max(1, current - 1))}
                title={t.resources.previousPage}
                type="button"
              >
                <ChevronLeft size={16} />
              </button>
              <span>
                {page} / {pageCount}
              </span>
              <button
                className="icon-button"
                disabled={page >= pageCount}
                onClick={() => setPage((current) => Math.min(pageCount, current + 1))}
                title={t.resources.nextPage}
                type="button"
              >
                <ChevronRight size={16} />
              </button>
            </div>
          </div>
        </div>

        <aside className="panel resource-detail-browser-panel">
          <div className="panel-header">
            <h2>{t.resources.resourceDetails}</h2>
            <PackageCheck size={16} />
          </div>
          <div className="panel-body">
            {selectedResource ? (
              <ResourceDetail resource={selectedResource} />
            ) : (
              <div className="empty compact-empty">{t.common.noData}</div>
            )}
          </div>
        </aside>
      </section>
    </AppShell>
  );

  function ResourceDetail({ resource }: { resource: BrowserResource }) {
    const configHref = buildConfigHref(resource);
    return (
      <div className="resource-detail-stack">
        <div>
          <div className="detail-title-row">
            <h3>{resource.name}</h3>
            <span className={badgeClass(resource.statusTone)}>{statusLabel(resource, t)}</span>
          </div>
          <p>{resource.description || resource.subtitle}</p>
        </div>

        <div className="detail-metrics-grid">
          <DetailMetric label="ID" value={resource.id} />
          <DetailMetric label={t.resources.category} value={resource.category} />
          <DetailMetric label="Method" value={resource.method ?? "n/a"} />
          <DetailMetric label={t.resources.device} value={resource.requiresGpu ? t.common.gpu : t.common.cpu} />
          {resource.sampleCount != null ? (
            <DetailMetric label={t.common.samples} value={resource.sampleCount.toLocaleString()} />
          ) : null}
          {resource.version ? <DetailMetric label="Version" value={resource.version} /> : null}
          {resource.size ? <DetailMetric label="Size" value={resource.size} /> : null}
          {resource.checksum ? <DetailMetric label="Checksum" value={resource.checksum} /> : null}
        </div>

        {resource.strengths ? (
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

        {resource.params && Object.keys(resource.params).length > 0 ? (
          <div className="detail-section">
            <strong>Params</strong>
            <pre className="detail-json">{JSON.stringify(resource.params, null, 2)}</pre>
          </div>
        ) : null}

        {resource.path ? (
          <div className="detail-section">
            <strong>Path</strong>
            <code>{resource.path}</code>
          </div>
        ) : null}

        {configHref ? (
          <a className="button primary resource-config-link" href={configHref}>
            <Archive size={16} />
            {t.resources.useInConfig}
          </a>
        ) : (
          <div className="risk warn">{t.resources.weightsConfigHint}</div>
        )}
      </div>
    );
  }
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

function algorithmToResource(algorithm: AlgorithmVersion): BrowserResource {
  const available = algorithm.available !== false && algorithm.status === "enabled";
  return {
    id: algorithm.id,
    type: "watermarks",
    name: algorithm.name,
    subtitle: `${algorithm.method ?? algorithm.id} · ${algorithm.category ?? "watermark"}`,
    category: algorithm.category ?? "watermark",
    status: algorithm.status,
    statusTone: available ? "ok" : "warn",
    available,
    method: algorithm.method,
    version: algorithm.version,
    description: algorithm.description,
    params: algorithm.params,
    requiresGpu: algorithm.requiresGpu,
    recommended: algorithm.recommended
  };
}

function attackToResource(attack: AttackPreset, language: "zh" | "en"): BrowserResource {
  const available = attack.available !== false;
  return {
    id: attack.id,
    type: "attacks",
    name: localizedName(language, attack.id, attack.name),
    subtitle: `${attack.method} · ${attack.category ?? "attack"} · ${attack.strengths.length} strength${
      attack.strengths.length === 1 ? "" : "s"
    }`,
    category: attack.category ?? "attack",
    status: available ? "enabled" : "missing",
    statusTone: available ? "ok" : "error",
    available,
    method: attack.method,
    description: attack.description,
    params: attack.params,
    strengths: attack.strengths,
    requiresGpu: attack.requiresGpu,
    recommended: attack.recommended
  };
}

function weightToResource(artifact: ModelArtifact): BrowserResource {
  return {
    id: artifact.id,
    type: "weights",
    name: artifact.name,
    subtitle: artifact.size,
    category: "artifact",
    status: "indexed",
    statusTone: "neutral",
    available: true,
    size: artifact.size,
    checksum: artifact.checksum,
    description: "Indexed model artifact. Weight folders are stored under resources/weights."
  };
}

function searchableText(resource: BrowserResource): string {
  return [
    resource.id,
    resource.name,
    resource.subtitle,
    resource.category,
    resource.description,
    resource.method,
    resource.path
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function buildConfigHref(resource: BrowserResource): string | null {
  const params = new URLSearchParams();
  if (resource.type === "datasets") {
    params.set("datasetIds", resource.id);
  } else if (resource.type === "watermarks") {
    params.set("algorithmIds", resource.id);
  } else if (resource.type === "attacks") {
    params.set("attackPresetIds", resource.id);
  } else {
    return null;
  }
  return `/configs?${params.toString()}`;
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
  return t.resources.weightFolders;
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
  return <HardDrive size={16} />;
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
