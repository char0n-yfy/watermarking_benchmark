import { ExternalLink } from "lucide-react";
import type { ResourceReferenceLink } from "@/lib/resource-references";
import { referenceOverviewText, type ResourceReference } from "@/lib/resource-references";
import {
  normalizeExternalHref,
  openExternalReferenceLink,
  projectSourceHref,
  ReferenceInlineText,
  splitProjectSourceNotes
} from "@/lib/reference-inline";

interface ResourceReferencePanelProps {
  reference: ResourceReference;
  language: "zh" | "en";
  hideOverview?: boolean;
  forceLayeredRepos?: boolean;
  labels: {
    overview: string;
    papers: string;
    repos: string;
    projectSource?: string;
    upstreamRepos?: string;
    noUpstream?: string;
  };
}

function referenceSummary(reference: ResourceReference, language: "zh" | "en") {
  return referenceOverviewText(reference, language);
}

function noteLines(notes: string[] | undefined) {
  if (!notes?.length) {
    return null;
  }
  return notes.map((note) => (
    <p className="resource-reference-note" key={note}>
      <ReferenceInlineText text={note} />
    </p>
  ));
}

function externalLinkList(links: ResourceReference["papers"], limit?: number) {
  const externalLinks = links.flatMap((link) => {
    const href = normalizeExternalHref(link.url);
    return href ? [{ ...link, href }] : [];
  });
  const visibleLinks = limit ? externalLinks.slice(0, limit) : externalLinks;
  if (visibleLinks.length === 0) {
    return null;
  }
  return (
    <ul className="resource-reference-links">
      {visibleLinks.map((link) => (
        <li key={`${link.href}-${link.label}`}>
          <a
            className="resource-reference-external-link"
            href={link.href}
            onClick={(event) => openExternalReferenceLink(event, link.href)}
            onMouseDown={(event) => event.stopPropagation()}
            rel="noreferrer noopener"
            target="_blank"
            title={link.href}
          >
            <span>{link.label}</span>
            <ExternalLink aria-hidden size={13} />
          </a>
        </li>
      ))}
    </ul>
  );
}

function projectSourceLinkList(links: ResourceReference["projectSources"]) {
  const projectLinks = (links ?? []).flatMap((link) => {
    const href = projectSourceHref(link.url);
    return href ? [{ ...link, href }] : [];
  });
  if (projectLinks.length === 0) {
    return null;
  }
  return (
    <ul className="resource-reference-links">
      {projectLinks.map((link) => (
        <li key={`${link.href}-${link.label}`}>
          <a
            className="resource-reference-external-link"
            href={link.href}
            onClick={(event) => openExternalReferenceLink(event, link.href)}
            onMouseDown={(event) => event.stopPropagation()}
            rel="noreferrer noopener"
            target="_blank"
            title={link.href}
          >
            <span>{link.label}</span>
            <ExternalLink aria-hidden size={13} />
          </a>
        </li>
      ))}
    </ul>
  );
}

function mergeUpstreamLinks(reference: ResourceReference): ResourceReferenceLink[] {
  const links = [...(reference.repos ?? [])];
  const seen = new Set(links.map((link) => link.url));
  for (const link of reference.upstreamRepos ?? []) {
    if (seen.has(link.url)) {
      continue;
    }
    seen.add(link.url);
    const label =
      link.label.includes("原始") || /original/i.test(link.label) ? link.label : `${link.label}（原始）`;
    links.push({ ...link, label });
  }
  return links;
}

export function ResourceReferencePanel({
  reference,
  language,
  hideOverview = false,
  forceLayeredRepos = false,
  labels
}: ResourceReferencePanelProps) {
  const summary = referenceSummary(reference, language);
  const showOverview = Boolean(summary) && !hideOverview;
  const legacyProjectSources = splitProjectSourceNotes(reference.repoNotes).sourceLinks;
  const projectSources =
    reference.projectSources && reference.projectSources.length > 0
      ? reference.projectSources
      : legacyProjectSources;
  const upstreamLinks = mergeUpstreamLinks(reference);
  const primaryUpstreamLink = upstreamLinks.find((link) => normalizeExternalHref(link.url));
  const visibleUpstreamLinks =
    forceLayeredRepos && primaryUpstreamLink ? [primaryUpstreamLink] : upstreamLinks;
  const repoTextNotes = splitProjectSourceNotes(reference.repoNotes).textNotes;
  const showLayeredRepos = forceLayeredRepos || projectSources.length > 0;
  const hasRepos = visibleUpstreamLinks.length > 0 || projectSources.length > 0 || repoTextNotes.length > 0;
  const primaryPaperLink = reference.papers.find((link) => normalizeExternalHref(link.url));
  const primaryPaperNote = reference.paperNotes?.find((note) => note.trim());
  const hasPapers = Boolean(primaryPaperLink || primaryPaperNote);
  const projectSourceLabel = labels.projectSource ?? (language === "zh" ? "项目实现" : "Project implementation");
  const upstreamReposLabel = labels.upstreamRepos ?? (language === "zh" ? "上游仓库" : "Upstream repository");

  if (!showOverview && !hasPapers && !hasRepos) {
    return null;
  }

  return (
    <div className="detail-section resource-reference-panel">
      {showOverview ? (
        <div className="resource-reference-block">
          <strong>{labels.overview}</strong>
          <p>{summary}</p>
        </div>
      ) : null}

      {hasPapers ? (
        <div className="resource-reference-block">
          <strong>{labels.papers}</strong>
          {primaryPaperLink ? externalLinkList([primaryPaperLink], 1) : null}
          {primaryPaperLink ? null : primaryPaperNote ? noteLines([primaryPaperNote]) : null}
        </div>
      ) : null}

      {hasRepos ? (
        <div className="resource-reference-block">
          <strong>{labels.repos}</strong>
          {showLayeredRepos ? (
            <>
              {projectSources.length > 0 ? (
                <div className="resource-reference-layer">
                  <span className="resource-reference-layer-label">{projectSourceLabel}</span>
                  {projectSourceLinkList(projectSources)}
                </div>
              ) : null}
              {visibleUpstreamLinks.length > 0 ? (
                <div className="resource-reference-layer">
                  <span className="resource-reference-layer-label">{upstreamReposLabel}</span>
                  {externalLinkList(visibleUpstreamLinks, forceLayeredRepos ? 1 : undefined)}
                </div>
              ) : null}
            </>
          ) : (
            externalLinkList(upstreamLinks)
          )}
          {noteLines(repoTextNotes)}
        </div>
      ) : null}
    </div>
  );
}
