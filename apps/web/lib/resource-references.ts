import attackReferencesJson from "@/data/attack-references.json";
import watermarkReferencesJson from "@/data/watermark-references.json";
import { normalizeExternalHref } from "@/lib/reference-inline";

export interface ResourceReferenceLink {
  label: string;
  url: string;
}

export interface ResourceReference {
  summary: {
    zh: string;
    en?: string;
  };
  papers: ResourceReferenceLink[];
  repos: ResourceReferenceLink[];
  paperNotes?: string[];
  repoNotes?: string[];
}

const watermarkReferences = watermarkReferencesJson as unknown as Record<string, ResourceReference>;
const attackReferences = attackReferencesJson as unknown as Record<string, ResourceReference>;

export function getWatermarkReference(method: string | undefined): ResourceReference | undefined {
  if (!method) {
    return undefined;
  }
  return watermarkReferences[method];
}

export function getAttackReference(
  method: string | undefined,
  executionMethods: string[] = []
): ResourceReference | undefined {
  if (method && attackReferences[method]) {
    return attackReferences[method];
  }
  for (const executionMethod of executionMethods) {
    if (attackReferences[executionMethod]) {
      return attackReferences[executionMethod];
    }
  }
  return undefined;
}

export function referenceOverviewText(
  reference: ResourceReference | undefined,
  language: "zh" | "en"
): string {
  if (!reference) {
    return "";
  }
  if (language === "zh") {
    return reference.summary.zh || reference.summary.en || "";
  }
  return reference.summary.en || reference.summary.zh || "";
}

export function buildDatasetReference(
  description: string,
  descriptionZh: string,
  sourceUrl: string
): ResourceReference | undefined {
  const summaryZh = descriptionZh.trim();
  const summaryEn = description.trim();
  const papers: ResourceReferenceLink[] = [];
  const repos: ResourceReferenceLink[] = [];

  const normalizedSourceUrl = normalizeExternalHref(sourceUrl);
  if (normalizedSourceUrl) {
    repos.push({
      label: normalizedSourceUrl.replace(/^https?:\/\//, ""),
      url: normalizedSourceUrl
    });
  }

  if (!summaryZh && !summaryEn && repos.length === 0) {
    return undefined;
  }

  return {
    summary: { zh: summaryZh || summaryEn, en: summaryEn || summaryZh },
    papers,
    repos
  };
}
