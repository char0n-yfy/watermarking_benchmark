"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Archive,
  Boxes,
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
import { ResourcePagination } from "@/components/ResourcePagination";
import { useLanguage } from "@/components/LanguageProvider";
import { fetchAlgorithms, fetchAttacks, fetchAttackWeightDownloadJob, fetchDatasetCatalog, fetchDatasetDownloadJob, fetchWeightDownloadJob, startAttackWeightDownload, startDatasetDownload, startWeightDownload } from "@/lib/api";
import { localizedName } from "@/lib/i18n";
import {
  algorithms as fallbackAlgorithms,
  artifacts,
  attacks as fallbackAttacks,
  datasets as fallbackDatasets
} from "@/lib/mock-data";
import type { AlgorithmVersion, AttackPreset, DatasetCatalogItem, DatasetDownloadJob, DatasetDownloadMode, DatasetVersion, ModelArtifact, ResourceStatus, WeightDownloadJob } from "@/lib/types";

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
  catalog?: DatasetCatalogItem;
  algorithm?: AlgorithmVersion;
  attack?: AttackPreset;
}

const PAGE_SIZE = 8;

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

    fetchDatasetCatalog()
      .then((catalog) => {
        if (cancelled) {
          return;
        }
        applyCatalog(catalog);
        setCatalogLoading(false);
        return fetchDatasetCatalog({ remote: true });
      })
      .then((catalog) => {
        if (cancelled || !catalog) {
          return;
        }
        applyCatalog(catalog);
      })
      .catch(() => {
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
      datasets: catalogItems.map((item) => catalogToResource(item, language)),
      watermarks: algorithms.map(algorithmToResource),
      attacks: attacks.map((attack) => attackToResource(attack, language)),
      weights: artifacts.map(weightToResource)
    }),
    [algorithms, attacks, catalogItems, language]
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

  const usesPagination = activeType === "watermarks" || activeType === "datasets";
  const pageCount = usesPagination ? Math.max(1, Math.ceil(filteredResources.length / PAGE_SIZE)) : 1;
  const visibleResources = usesPagination
    ? filteredResources.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)
    : filteredResources;

  const visibleGroupedDatasetResources = useMemo(() => {
    if (activeType !== "datasets") {
      return null;
    }
    const groups = new Map<string, BrowserResource[]>();
    for (const resource of visibleResources) {
      const bucket = groups.get(resource.category) ?? [];
      bucket.push(resource);
      groups.set(resource.category, bucket);
    }
    return Array.from(groups.entries());
  }, [activeType, visibleResources]);
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
    if (!usesPagination) {
      return;
    }
    const firstVisible = visibleResources[0];
    if (!firstVisible) {
      return;
    }
    const selectionVisible = visibleResources.some((resource) => resource.id === selectedResourceId);
    if (!selectionVisible) {
      setSelectedResourceId(firstVisible.id);
    }
  }, [page, selectedResourceId, usesPagination, visibleResources]);

  useEffect(() => {
    if (!selectedResource && filteredResources.length > 0) {
      setSelectedResourceId(filteredResources[0].id);
    }
  }, [filteredResources, selectedResource]);

  async function refreshDatasetCatalog() {
    const catalog = await fetchDatasetCatalog();
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
        <SummaryCard icon={Shield} label={t.console.algorithms} value={algorithms.length.toString()} meta={countByGpu(algorithms)} />
        <SummaryCard icon={Gauge} label={t.console.attacks} value={attacks.length.toString()} meta={countByGpu(attacks)} />
        <SummaryCard icon={HardDrive} label={t.resources.weightFolders} value={artifacts.length.toString()} meta="resources/weights/{attacks,watermarking}" />
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
          <div
            className={
              usesPagination ? "panel-body resource-list-panel-body" : "panel-body resource-browser-list"
            }
          >
            <div className="resource-result-note">
              <SlidersHorizontal size={14} />
              <span>
                {t.resources.showingResults}: {visibleResources.length} / {filteredResources.length}
              </span>
            </div>
            {activeType === "datasets" && visibleGroupedDatasetResources ? (
              <div className="resource-page-list" style={{ ["--resource-page-size" as string]: PAGE_SIZE }}>
                {visibleGroupedDatasetResources.map(([category, resources]) => (
                  <div className="dataset-category-group" key={category}>
                    <div className="dataset-category-heading">{category}</div>
                    {resources.map((resource) => (
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
                          {resource.catalog?.installed ? (
                            <span className="badge ok">{t.resources.installed}</span>
                          ) : (
                            <span className="badge">{t.resources.notInstalled}</span>
                          )}
                        </span>
                      </button>
                    ))}
                  </div>
                ))}
              </div>
            ) : activeType === "watermarks" ? (
              <div className="resource-page-list" style={{ ["--resource-page-size" as string]: PAGE_SIZE }}>
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
              </div>
            ) : (
              filteredResources.map((resource) => (
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
              ))
            )}
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
  onDatasetInstalled,
  onAlgorithmInstalled,
  onAttackInstalled
}: {
  resource: BrowserResource;
  language: "zh" | "en";
  t: ReturnType<typeof useLanguage>["t"];
  onDatasetInstalled: () => void;
  onAlgorithmInstalled: () => void;
  onAttackInstalled: () => void;
}) {
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

      <div className={resource.type === "datasets" ? "detail-metrics-grid dataset-only-category" : "detail-metrics-grid"}>
        {resource.type === "datasets" ? (
          <DetailMetric label={t.resources.category} value={resource.category} />
        ) : (
          <>
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
          </>
        )}
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

      {resource.path && resource.type !== "datasets" ? (
        <div className="detail-section">
          <strong>Path</strong>
          <code>{resource.path}</code>
        </div>
      ) : null}

      {resource.type === "datasets" && resource.catalog ? (
        <DatasetDownloadPanel catalog={resource.catalog} language={language} onInstalled={onDatasetInstalled} t={t} />
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

      {resource.type === "attacks" && resource.attack?.weightsPackRequired ? (
        <WeightDownloadPanel
          attack={resource.attack}
          key={`attack-${resource.id}`}
          onInstalled={onAttackInstalled}
          t={t}
          variant="attack"
        />
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
  const category = language === "zh" ? item.categoryZh : item.category;
  const sampleCount = item.compactAvailable ? item.compactSampleCount : item.fullSampleCount;
  return {
    id: item.id,
    type: "datasets",
    name: displayName,
    subtitle: formatDatasetSubtitle(item, language),
    category,
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
  onInstalled,
  t,
  variant
}: {
  algorithm?: AlgorithmVersion;
  attack?: AttackPreset;
  onInstalled: () => void;
  t: ReturnType<typeof useLanguage>["t"];
  variant: "watermark" | "attack";
}) {
  const resource = variant === "watermark" ? algorithm : attack;
  const identifier = resource?.id ?? "";
  const weightsDir = resource?.weightsDir;
  const [job, setJob] = useState<WeightDownloadJob | null>(null);
  const [busy, setBusy] = useState(false);
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

  if (!resource) {
    return null;
  }

  const installed = resource.weightsInstalled === true || job?.status === "succeeded";
  const jobInFlight = job?.status === "queued" || job?.status === "running";
  const canStart = resource.weightsDownloadReady === true && !installed && !jobInFlight;
  const progressPercent =
    job && job.totalItems > 0 ? Math.round((job.completedItems / job.totalItems) * 100) : job?.progress ?? 0;
  const panelTitle = variant === "watermark" ? t.resources.weightDownloadPanel : t.resources.attackWeightDownloadPanel;
  const panelHint = variant === "watermark" ? t.resources.weightDownloadHint : t.resources.attackWeightDownloadHint;
  const alreadyInstalled =
    variant === "watermark" ? t.resources.weightsAlreadyInstalled : t.resources.attackWeightsAlreadyInstalled;
  const unavailable = variant === "watermark" ? t.resources.weightsUnavailable : t.resources.attackWeightsUnavailable;
  const readyLabel = variant === "watermark" ? t.resources.weightDownloadReady : t.resources.attackWeightDownloadReady;
  const weightsRoot = variant === "watermark" ? "weights/watermarking" : "weights/attacks";

  async function handleStart() {
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

  return (
    <div className="detail-section dataset-download-panel">
      <strong>{panelTitle}</strong>
      <p className="dataset-download-hint">{panelHint}</p>
      {weightsDir ? (
        <div className="dataset-pool-summary">
          <span>{weightsRoot}/{weightsDir}</span>
        </div>
      ) : null}
      {installed ? <div className="risk ok">{alreadyInstalled}</div> : null}
      {!canStart && !installed && !jobInFlight ? <div className="risk warn">{unavailable}</div> : null}
      {!installed ? (
        <button className="button primary" disabled={!canStart || busy || jobInFlight} onClick={handleStart} type="button">
          {t.resources.startDownload}
        </button>
      ) : null}
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
              <div className="badge ok">{readyLabel}</div>
              {job.outputDir ? (
                <div className="dataset-download-path">
                  <span>{t.resources.downloadInstalledTo}</span>
                  <code>{job.outputDir}</code>
                </div>
              ) : null}
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
  onInstalled,
  t
}: {
  catalog: DatasetCatalogItem;
  language: "zh" | "en";
  onInstalled: () => void;
  t: ReturnType<typeof useLanguage>["t"];
}) {
  const [mode, setMode] = useState<DatasetDownloadMode>("compact");
  const [seed, setSeed] = useState(42);
  const [sampleCount, setSampleCount] = useState(100);
  const [job, setJob] = useState<DatasetDownloadJob | null>(null);
  const [busy, setBusy] = useState(false);
  const installedJobRef = useRef<string | null>(null);

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
  }, [job, onInstalled]);

  const compactReady = catalog.compactAvailable;
  const customReady = catalog.customDownloadReady;
  const compactInstalled = (catalog.compactSampleCount ?? 0) > 0;
  const canStart = mode === "compact" ? compactReady && !compactInstalled : customReady;
  const totalImages = catalog.officialTotalImages ?? 0;
  const progressPercent =
    job && job.totalItems > 0 ? Math.round((job.completedItems / job.totalItems) * 100) : job?.progress ?? 0;

  async function handleStart() {
    setBusy(true);
    try {
      const created = await startDatasetDownload(catalog.id, {
        mode,
        seed,
        sampleCount: mode === "compact" ? 1000 : sampleCount
      });
      setJob(created);
    } finally {
      setBusy(false);
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
      {mode === "compact" ? (
        <p className="dataset-download-hint">
          {`${t.resources.compactDownloadHint}（${t.resources.compactPack}：${catalog.compactSampleCount.toLocaleString()} ${t.resources.imagesUnit}）`}
        </p>
      ) : null}
      {mode === "custom" ? (
        <>
          <p className="dataset-download-hint">{t.resources.customDownloadHint}</p>
          <p className="dataset-download-hint muted">{t.resources.customFolderHint}</p>
          <div className="dataset-pool-summary">
            <span>
              {t.resources.datasetTotalImages}：
              {totalImages > 0 ? `${totalImages.toLocaleString()} ${t.resources.imagesUnit}` : "—"}
            </span>
          </div>
        </>
      ) : null}
      {!compactReady && mode === "compact" ? <div className="risk warn">{t.resources.compactUnavailable}</div> : null}
      {compactInstalled && mode === "compact" ? (
        <div className="risk ok">{t.resources.compactAlreadyInstalled}</div>
      ) : null}
      {!customReady && mode === "custom" ? <div className="risk warn">{t.resources.customUnavailable}</div> : null}
      {mode === "custom" ? (
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
      <button className="button primary" disabled={!canStart || busy || job?.status === "running"} onClick={handleStart} type="button">
        {t.resources.startDownload}
      </button>
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
              {job.outputDir ? (
                <div className="dataset-download-path">
                  <span>{t.resources.downloadInstalledTo}</span>
                  <code>{job.outputDir}</code>
                </div>
              ) : null}
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

function algorithmToResource(algorithm: AlgorithmVersion): BrowserResource {
  const available = algorithm.available !== false && algorithm.status === "enabled";
  const needsWeights = algorithm.weightsPackRequired === true;
  const weightsReady = !needsWeights || algorithm.weightsInstalled === true;
  return {
    id: algorithm.id,
    type: "watermarks",
    name: algorithm.name,
    subtitle: `${algorithm.method ?? algorithm.id} · ${algorithm.category ?? "watermark"}`,
    category: algorithm.category ?? "watermark",
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

function attackToResource(attack: AttackPreset, language: "zh" | "en"): BrowserResource {
  const available = attack.available !== false;
  const needsWeights = attack.weightsPackRequired === true;
  const weightsReady = !needsWeights || attack.weightsInstalled === true;
  return {
    id: attack.id,
    type: "attacks",
    name: localizedName(language, attack.id, attack.name),
    subtitle: `${attack.method} · ${attack.category ?? "attack"} · ${attack.strengths.length} strength${
      attack.strengths.length === 1 ? "" : "s"
    }`,
    category: attack.category ?? "attack",
    status: available ? "enabled" : "missing",
    statusTone: available && weightsReady ? "ok" : needsWeights && !weightsReady ? "warn" : available ? "ok" : "error",
    available,
    method: attack.method,
    description: attack.description,
    params: attack.params,
    strengths: attack.strengths,
    requiresGpu: attack.requiresGpu,
    recommended: attack.recommended,
    attack
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
    description: "Indexed model artifact. Watermark weights live under resources/weights/watermarking/; attack weights under resources/weights/attacks/."
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
