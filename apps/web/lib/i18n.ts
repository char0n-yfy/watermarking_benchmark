export type Language = "zh" | "en";

export const languages: Array<{ code: Language; label: string }> = [
  { code: "zh", label: "中" },
  { code: "en", label: "EN" }
];

export const translations = {
  zh: {
    languageLabel: "界面语言",
    nav: {
      console: "控制台",
      resources: "资源",
      runs: "运行",
      results: "结果",
      schema: "数据结构"
    },
    common: {
      samples: "样本",
      dataset: "数据集",
      algorithm: "水印算法",
      attackPreset: "攻击预设",
      weight: "权重",
      gpu: "GPU",
      cpu: "CPU",
      enabled: "已启用",
      versioned: "已版本化",
      indexed: "已索引",
      table: "表",
      status: {
        draft: "草稿",
        queued: "排队中",
        running: "运行中",
        succeeded: "成功",
        failed: "失败",
        cancelled: "已取消",
        partially_failed: "部分失败",
        enabled: "已启用",
        reviewed: "已审核",
        built: "已构建",
        uploaded: "已上传"
      }
    },
    console: {
      title: "实验控制台",
      subtitle: "配置测评矩阵草稿",
      reset: "重置草稿",
      save: "保存草稿",
      materialize: "生成运行",
      resources: "资源",
      matrix: "测评矩阵",
      datasets: "数据集",
      algorithms: "水印算法",
      attacks: "攻击算法",
      parameters: "参数",
      seeds: "随机种子",
      maxSamples: "最大样本数",
      inspector: "检查器",
      cells: "单元",
      ops: "操作数",
      okRisk: "运行规模在默认队列保护范围内。",
      warnRisk: "运行规模超过默认队列保护范围。"
    },
    resources: {
      title: "资源",
      subtitle: "数据集、水印算法、权重和攻击预设",
      catalog: "资源目录",
      name: "名称",
      type: "类型",
      status: "状态",
      details: "详情"
    },
    runs: {
      title: "运行",
      subtitle: "队列、worker 和执行日志",
      recent: "最近运行",
      run: "运行",
      status: "状态",
      cells: "单元数",
      updated: "更新时间"
    },
    results: {
      title: "结果",
      subtitle: "指标汇总和 artifact 索引",
      matrixCells: "矩阵单元",
      cell: "单元",
      manifest: "Manifest"
    },
    schema: {
      title: "数据结构",
      subtitle: "元数据实体",
      coreTables: "核心表",
      tableDescription: "PostgreSQL 元数据表"
    },
    resourceNames: {
      "ds-coco-v1": "MS-COCO 验证集切片",
      "ds-diffusiondb-v1": "DiffusionDB 精选集",
      "ds-demo-v1": "Demo 冒烟集",
      "atk-identity": "不攻击",
      "atk-jpeg-sweep": "JPEG 强度扫描",
      "atk-blur-sweep": "模糊强度扫描",
      "atk-crop-sweep": "裁剪强度扫描",
      "run_20260626_001": "JPEG 基线扫描",
      "run_20260625_004": "Demo 冒烟集",
      "run_20260624_002": "上传包沙箱"
    },
    dates: {
      Yesterday: "昨天",
      "Jun 24": "6月24日"
    }
  },
  en: {
    languageLabel: "Interface language",
    nav: {
      console: "Console",
      resources: "Resources",
      runs: "Runs",
      results: "Results",
      schema: "Schema"
    },
    common: {
      samples: "samples",
      dataset: "dataset",
      algorithm: "Algorithm",
      attackPreset: "Attack preset",
      weight: "Weight",
      gpu: "gpu",
      cpu: "CPU",
      enabled: "enabled",
      versioned: "versioned",
      indexed: "indexed",
      table: "table",
      status: {
        draft: "draft",
        queued: "queued",
        running: "running",
        succeeded: "succeeded",
        failed: "failed",
        cancelled: "cancelled",
        partially_failed: "partially_failed",
        enabled: "enabled",
        reviewed: "reviewed",
        built: "built",
        uploaded: "uploaded"
      }
    },
    console: {
      title: "Experiment Console",
      subtitle: "Draft experiment matrix",
      reset: "Reset draft",
      save: "Save draft",
      materialize: "Materialize run",
      resources: "Resources",
      matrix: "Matrix",
      datasets: "Datasets",
      algorithms: "Algorithms",
      attacks: "Attacks",
      parameters: "Parameters",
      seeds: "Seeds",
      maxSamples: "Max samples",
      inspector: "Inspector",
      cells: "Cells",
      ops: "Ops",
      okRisk: "Run size is within the default queue guard.",
      warnRisk: "Run size exceeds the default queue guard."
    },
    resources: {
      title: "Resources",
      subtitle: "Datasets, algorithms, weights, and attack presets",
      catalog: "Catalog",
      name: "Name",
      type: "Type",
      status: "Status",
      details: "Details"
    },
    runs: {
      title: "Runs",
      subtitle: "Queue, workers, and execution logs",
      recent: "Recent runs",
      run: "Run",
      status: "Status",
      cells: "Cells",
      updated: "Updated"
    },
    results: {
      title: "Results",
      subtitle: "Metric summaries and artifact indexes",
      matrixCells: "Matrix cells",
      cell: "Cell",
      manifest: "Manifest"
    },
    schema: {
      title: "Schema",
      subtitle: "Metadata entities",
      coreTables: "Core tables",
      tableDescription: "PostgreSQL metadata table"
    },
    resourceNames: {},
    dates: {}
  }
} as const;

export type Translation = (typeof translations)[Language];

export function localizedName(
  language: Language,
  id: string,
  fallback: string,
): string {
  const names = translations[language].resourceNames as Record<string, string>;
  return names[id] ?? fallback;
}

export function localizedDate(language: Language, value: string): string {
  const dates = translations[language].dates as Record<string, string>;
  return dates[value] ?? value;
}
