import { ExternalLink } from "lucide-react";
import type { ReactNode } from "react";

const DEFAULT_PROJECT_SOURCE_BASE =
  "https://github.com/char0n-yfy/watermarking_benchmark/blob/main";

export function isExternalUrl(value: string) {
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

export function normalizeExternalHref(value: string): string | null {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  const candidate = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;
  return isExternalUrl(candidate) ? candidate : null;
}

export function projectSourceHref(path: string): string | null {
  const trimmed = path.trim().replace(/^\/+/, "").replace(/\\/g, "/");
  if (!trimmed || /^(https?:|mailto:|javascript:)/i.test(trimmed) || trimmed.includes("..")) {
    return null;
  }
  const base = (process.env.NEXT_PUBLIC_PROJECT_SOURCE_BASE_URL ?? DEFAULT_PROJECT_SOURCE_BASE).replace(
    /\/$/,
    ""
  );
  return `${base}/${trimmed}`;
}

export function splitProjectSourceNotes(notes: string[] | undefined): {
  sourceLinks: Array<{ label: string; url: string }>;
  textNotes: string[];
} {
  const sourceLinks: Array<{ label: string; url: string }> = [];
  const textNotes: string[] = [];

  for (const note of notes ?? []) {
    const pathMatch = /`([^`]+)`/.exec(note);
    if (!pathMatch) {
      textNotes.push(note);
      continue;
    }
    const href = projectSourceHref(pathMatch[1]);
    if (!href) {
      textNotes.push(note);
      continue;
    }
    sourceLinks.push({ label: pathMatch[1], url: href });
    const remainder = note
      .replace(/`[^`]+`/g, "")
      .replace(/^[；;:\s]+|[；;:\s]+$/g, "")
      .trim();
    if (remainder && !/^(项目源码|组合源码)[：:]?$/.test(remainder)) {
      textNotes.push(remainder);
    }
  }

  return { sourceLinks, textNotes };
}

function renderExternalAnchor(href: string, label: string, key: string) {
  return (
    <a
      className="resource-reference-external-link"
      href={href}
      key={key}
      onClick={(event) => event.stopPropagation()}
      rel="noreferrer noopener"
      target="_blank"
      title={href}
    >
      <span>{label}</span>
      <ExternalLink aria-hidden size={13} />
    </a>
  );
}

function renderProjectSourceLink(path: string, label: string, key: string) {
  const href = projectSourceHref(path);
  if (!href) {
    return <code key={key}>{label}</code>;
  }
  return renderExternalAnchor(href, label, key);
}

function renderToken(token: string, key: string) {
  if (token.startsWith("[")) {
    const match = /\[([^\]]+)\]\(([^)]+)\)/.exec(token);
    if (!match) {
      return token;
    }
    const [, label, url] = match;
    if (isExternalUrl(url)) {
      return renderExternalAnchor(url, label, key);
    }
    const sourceHref = projectSourceHref(url);
    if (sourceHref) {
      return renderExternalAnchor(sourceHref, label, key);
    }
    return `${label} (${url})`;
  }

  if (token.startsWith("`") && token.endsWith("`")) {
    return renderProjectSourceLink(token.slice(1, -1), token.slice(1, -1), key);
  }

  return token;
}

export function ReferenceInlineText({ text }: { text: string }) {
  const parts: ReactNode[] = [];
  let lastIndex = 0;
  let tokenIndex = 0;
  const tokenPattern = /(\[[^\]]+\]\([^)]+\)|`[^`]+`)/g;
  let match = tokenPattern.exec(text);
  while (match) {
    if (match.index > lastIndex) {
      parts.push(<span key={`text-${tokenIndex}`}>{text.slice(lastIndex, match.index)}</span>);
    }
    parts.push(renderToken(match[0], `token-${tokenIndex}`));
    lastIndex = match.index + match[0].length;
    tokenIndex += 1;
    match = tokenPattern.exec(text);
  }
  if (lastIndex < text.length) {
    parts.push(<span key={`text-${tokenIndex}`}>{text.slice(lastIndex)}</span>);
  }
  return <>{parts}</>;
}
