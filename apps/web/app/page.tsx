"use client";

import { type CSSProperties, type ReactNode, useEffect, useMemo, useState } from "react";
import {
  Activity,
  CheckCircle2,
  Cpu,
  HardDrive,
  MemoryStick,
  PlayCircle,
  Zap
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { useLanguage } from "@/components/LanguageProvider";
import {
  fetchAlgorithms,
  fetchAttacks,
  fetchDatasets,
  fetchRuns,
  fetchRuntime,
  fetchSavedConfigs,
  fetchSystemMetrics
} from "@/lib/api";
import {
  buildActiveRunRows,
  statusBadgeClass,
  summarizeRuns
} from "@/lib/insights";
import type {
  AlgorithmVersion,
  AttackPreset,
  DatasetVersion,
  DemoRunRecord,
  RuntimeInfo,
  SavedExperimentConfig,
  SystemMetrics
} from "@/lib/types";

function formatPercent(value: number | null | undefined) {
  return value == null ? "n/a" : `${value.toFixed(1)}%`;
}

function formatBytes(value: number | null | undefined) {
  if (value == null) {
    return "n/a";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let nextValue = value;
  let unitIndex = 0;
  while (nextValue >= 1024 && unitIndex < units.length - 1) {
    nextValue /= 1024;
    unitIndex += 1;
  }
  return `${nextValue.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatBytesPerSecond(value: number | null | undefined) {
  return value == null ? "n/a" : `${formatBytes(value)}/s`;
}

function metricLevel(percent: number | null | undefined) {
  if (percent == null) {
    return "unknown";
  }
  if (percent >= 90) {
    return "danger";
  }
  if (percent >= 75) {
    return "warn";
  }
  return "ok";
}

function metricColor(level: string) {
  if (level === "danger") {
    return "var(--red)";
  }
  if (level === "warn") {
    return "var(--amber)";
  }
  if (level === "unknown") {
    return "#66736c";
  }
  return "var(--teal)";
}

function GaugeCard({
  detail,
  icon,
  label,
  meta = [],
  value
}: {
  detail: string;
  icon: ReactNode;
  label: string;
  meta?: Array<{ label: string; value: string }>;
  value: number | null | undefined;
}) {
  const level = metricLevel(value);
  const normalized = Math.max(0, Math.min(100, value ?? 0));
  const gaugeStyle = {
    "--gauge-color": metricColor(level),
    "--gauge-value": `${normalized}%`
  } as CSSProperties;

  return (
    <div className={`gauge-card ${level}`} style={gaugeStyle}>
      <div className="gauge-card-top">
        <span className="gauge-title">{label}</span>
        <span className="gauge-icon">{icon}</span>
      </div>
      <div className="gauge-ring">
        <div className="gauge-core">
          <strong>{formatPercent(value)}</strong>
          <span>{value == null ? "未采集" : "实时占用"}</span>
        </div>
      </div>
      <p className="gauge-detail">{detail}</p>
      {meta.length ? (
        <div className="gauge-meta">
          {meta.map((item) => (
            <span key={item.label}>
              <small>{item.label}</small>
              <strong>{item.value}</strong>
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export default function ExperimentConsole() {
  const { t } = useLanguage();
  const [configs, setConfigs] = useState<SavedExperimentConfig[]>([]);
  const [runs, setRuns] = useState<DemoRunRecord[]>([]);
  const [runtime, setRuntime] = useState<RuntimeInfo | null>(null);
  const [datasets, setDatasets] = useState<DatasetVersion[]>([]);
  const [algorithms, setAlgorithms] = useState<AlgorithmVersion[]>([]);
  const [attacks, setAttacks] = useState<AttackPreset[]>([]);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [autoRefreshSeconds, setAutoRefreshSeconds] = useState(10);
  const [notice, setNotice] = useState("");

  const loadDashboard = async () => {
    try {
      const [loadedConfigs, loadedRuns, loadedRuntime, loadedDatasets, loadedAlgorithms, loadedAttacks] =
        await Promise.all([
          fetchSavedConfigs(),
          fetchRuns(),
          fetchRuntime(),
          fetchDatasets(),
          fetchAlgorithms(),
          fetchAttacks()
        ]);
      setConfigs(loadedConfigs);
      setRuns(loadedRuns);
      setRuntime(loadedRuntime);
      setDatasets(loadedDatasets);
      setAlgorithms(loadedAlgorithms);
      setAttacks(loadedAttacks);
      setNotice("");
    } catch {
      setNotice("API 未启动或不可访问，请先启动 FastAPI 服务。");
      return;
    }

    try {
      setSystemMetrics(await fetchSystemMetrics());
    } catch {
      setSystemMetrics(null);
      setNotice("系统性能指标暂时不可用，队列和资源数据已加载。");
    }

  };

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      loadDashboard().catch(() => {
        if (!cancelled) {
          setNotice("API 未启动或不可访问，请先启动 FastAPI 服务。");
        }
      });
    };
    load();
    const timer = window.setInterval(load, autoRefreshSeconds * 1000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [autoRefreshSeconds]);

  const stats = useMemo(() => summarizeRuns(runs), [runs]);
  const activeRows = useMemo(
    () => buildActiveRunRows(runs, configs, datasets, algorithms, attacks),
    [runs, configs, datasets, algorithms, attacks]
  );
  const systemHealthy = Boolean(runtime) && !(runtime?.workers ?? []).some((worker) => worker.status === "error");
  const primaryGpu = systemMetrics?.gpu.devices[0] ?? null;
  const cpuTemperatureLabel =
    systemMetrics?.cpu.temperatureC == null ? "n/a" : `${systemMetrics.cpu.temperatureC}°C`;
  const cpuPowerLabel =
    systemMetrics?.cpu.powerDrawW == null ? "n/a" : `${systemMetrics.cpu.powerDrawW.toFixed(1)}W`;
  const gpuName = primaryGpu?.name ?? "未检测到 NVIDIA GPU 指标";
  const vramUsedBytes = primaryGpu?.memoryUsedMiB == null ? null : primaryGpu.memoryUsedMiB * 1024 * 1024;
  const vramTotalBytes = primaryGpu?.memoryTotalMiB == null ? null : primaryGpu.memoryTotalMiB * 1024 * 1024;
  const memoryFreeBytes =
    systemMetrics?.memory.availableBytes ?? (
      systemMetrics?.memory.totalBytes != null && systemMetrics?.memory.usedBytes != null
        ? systemMetrics.memory.totalBytes - systemMetrics.memory.usedBytes
        : null
    );
  return (
    <AppShell active="console">
      <div className="topbar console-topbar">
        <div className="title-block console-title">
          <h1>{t.console.title}</h1>
          <span className={systemHealthy ? "status-dot ok" : "status-dot warn"}>
            {systemHealthy ? t.console.systemHealthy : t.console.systemDegraded}
          </span>
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
        </div>
      </div>

      <section className="metric-card-grid metric-card-grid-console">
        <div className="metric-card">
          <div>
            <span>{t.console.runningRuns}</span>
            <strong>{stats.running}</strong>
          </div>
          <PlayCircle size={20} />
        </div>
        <div className="metric-card">
          <div>
            <span>{t.console.completedRuns}</span>
            <strong>{stats.completed}</strong>
          </div>
          <CheckCircle2 size={20} />
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>{t.console.activeRuns}</h2>
          <Activity size={16} />
        </div>
        <div className="panel-body table-scroll">
          <table className="table compact-table">
            <thead>
              <tr>
                <th>{t.runs.run}</th>
                <th>{t.common.config}</th>
                <th>{t.common.dataset}</th>
                <th>{t.common.algorithm}</th>
                <th>{t.common.attackPreset}</th>
                <th>{t.common.progress}</th>
                <th>{t.runs.status}</th>
                <th>{t.console.startedAt}</th>
              </tr>
            </thead>
            <tbody>
              {activeRows.map((run) => (
                <tr key={run.id}>
                  <td>{run.id}</td>
                  <td>{run.configName}</td>
                  <td>{run.datasetLabel}</td>
                  <td>{run.algorithmLabel}</td>
                  <td>{run.attackLabel}</td>
                  <td>
                    <div className="progress-cell">
                      <div className="progress-track">
                        <div className="progress-bar" style={{ width: `${run.progress}%` }} />
                      </div>
                      <span>{run.progress}%</span>
                    </div>
                  </td>
                  <td>
                    <span className={statusBadgeClass(run.status)}>{t.common.status[run.status]}</span>
                  </td>
                  <td>{run.startedAt ?? run.updatedAt}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {activeRows.length === 0 ? <div className="empty compact-empty">{t.console.noActiveRuns}</div> : null}
          {notice ? <div className="risk warn">{notice}</div> : null}
        </div>
      </section>

      <section className="console-bottom-grid">
        <div className="panel hardware-monitor-panel">
          <div className="panel-header">
            <h2>当前电脑性能</h2>
            <Activity size={16} />
          </div>
          <div className="panel-body gauge-panel-body">
            <div className="gauge-grid">
              <GaugeCard
                detail={gpuName}
                icon={<Zap size={17} />}
                label="GPU 占用率"
                meta={[
                  ...(primaryGpu?.temperatureC == null ? [] : [{ label: "温度", value: `${primaryGpu.temperatureC}°C` }]),
                  ...(primaryGpu?.powerDrawW == null ? [] : [{ label: "功耗", value: `${primaryGpu.powerDrawW.toFixed(1)}W` }])
                ]}
                value={primaryGpu?.utilizationPercent}
              />
              <GaugeCard
                detail={
                  primaryGpu
                    ? `${formatBytes(vramUsedBytes)} / ${formatBytes(vramTotalBytes)}`
                    : "显存数据不可用"
                }
                icon={<MemoryStick size={17} />}
                label="显存占用率"
                meta={
                  primaryGpu
                    ? [
                        { label: "已用", value: formatBytes(vramUsedBytes) },
                        { label: "总量", value: formatBytes(vramTotalBytes) }
                      ]
                    : []
                }
                value={primaryGpu?.memoryUsedPercent}
              />
              <GaugeCard
                detail={`${formatBytes(systemMetrics?.memory.usedBytes)} / ${formatBytes(systemMetrics?.memory.totalBytes)}`}
                icon={<MemoryStick size={17} />}
                label="内存占用率"
                meta={[
                  { label: "可用", value: formatBytes(memoryFreeBytes) },
                  { label: "API RSS", value: formatBytes(systemMetrics?.process.rssBytes) }
                ]}
                value={systemMetrics?.memory.usedPercent}
              />
              <GaugeCard
                detail={`${systemMetrics?.cpu.logicalCores ?? "n/a"} logical cores`}
                icon={<Cpu size={17} />}
                label="CPU 占用率"
                meta={[
                  { label: "温度", value: cpuTemperatureLabel },
                  { label: "功耗", value: cpuPowerLabel }
                ]}
                value={systemMetrics?.cpu.usagePercent}
              />
              <GaugeCard
                detail={`${formatBytes(systemMetrics?.disk.usedBytes)} / ${formatBytes(systemMetrics?.disk.totalBytes)}`}
                icon={<HardDrive size={17} />}
                label="硬盘占用率"
                meta={[
                  { label: "可用", value: formatBytes(systemMetrics?.disk.freeBytes) },
                  { label: "I/O", value: formatBytesPerSecond(systemMetrics?.disk.ioTotalBytesPerSecond) }
                ]}
                value={systemMetrics?.disk.usedPercent}
              />
            </div>
          </div>
        </div>
      </section>
    </AppShell>
  );
}
