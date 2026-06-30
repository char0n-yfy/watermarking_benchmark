import { ExternalLink } from "lucide-react";
import type { ResourceReference } from "@/lib/resource-references";
import { referenceOverviewText } from "@/lib/resource-references";
import { normalizeExternalHref, ReferenceInlineText, splitProjectSourceNotes } from "@/lib/reference-inline";

interface ResourceReferencePanelProps {
  reference: ResourceReference;
  language: "zh" | "en";
  hideOverview?: boolean;
  labels: {
    overview: string;
    papers: string;
    repos: string;
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

function linkList(links: ResourceReference["papers"]) {
  const externalLinks = links.flatMap((link) => {
    const href = normalizeExternalHref(link.url);
    return href ? [{ ...link, href }] : [];
  });
  if (externalLinks.length === 0) {
    return null;
  }
  return (
    <ul className="resource-reference-links">
      {externalLinks.map((link) => (
        <li key={`${link.href}-${link.label}`}>
          <a
            className="resource-reference-external-link"
            href={link.href}
            onClick={(event) => event.stopPropagation()}
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

export function ResourceReferencePanel({
  reference,
  language,
  hideOverview = false,
  labels
}: ResourceReferencePanelProps) {
  const summary = referenceSummary(reference, language);
  const showOverview = Boolean(summary) && !hideOverview;
  const { sourceLinks, textNotes: repoTextNotes } = splitProjectSourceNotes(reference.repoNotes);
  const repoLinks = [...reference.repos, ...sourceLinks];
  const hasPapers =
    reference.papers.some((link) => normalizeExternalHref(link.url)) || Boolean(reference.paperNotes?.length);
  const hasRepos =
    repoLinks.some((link) => normalizeExternalHref(link.url)) || repoTextNotes.length > 0;

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
          {linkList(reference.papers)}
          {noteLines(reference.paperNotes)}
        </div>
      ) : null}

      {hasRepos ? (
        <div className="resource-reference-block">
          <strong>{labels.repos}</strong>
          {linkList(repoLinks)}
          {noteLines(repoTextNotes)}
        </div>
      ) : null}
    </div>
  );
}
