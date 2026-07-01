#!/usr/bin/env python3
"""Parse resource metadata markdown tables into frontend JSON."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WATERMARK_MD = ROOT / "resources" / "metadata" / "WATERMARK_ALGORITHM_DESCRIPTIONS.md"
ATTACK_MD = ROOT / "resources" / "metadata" / "ATTACK_ALGORITHM_DESCRIPTIONS.md"
OUT_DIR = ROOT / "apps" / "web" / "data"

LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
ID_RE = re.compile(r"`([^`]+)`")
VIEWPOINT_ID_RE = re.compile(
    r"^3d_viewpoint_rerendering_(?P<motion>swipe|shake|rotate|rotate_forward)_phase\d+_(?P<lookat>point|ahead)$"
)
PROJECT_SOURCE_PREFIXES = ("项目源码", "项目实现", "项目后端", "组合源码")
UPSTREAM_MARKERS = ("原始实现", "上游参考")
UPSTREAM_REPO_LABEL = "上游仓库"

WATERMARK_PROJECT_FILES = {
    "chunkyseal": "evaluator/watermarking/methods/chunkyseal.py",
    "cin": "evaluator/watermarking/methods/cin.py",
    "hidden": "evaluator/watermarking/methods/hidden.py",
    "invismark": "evaluator/watermarking/methods/invismark.py",
    "invisible-watermark-dwtdct": "evaluator/watermarking/methods/invisible_watermark.py",
    "invisible-watermark-dwtdctsvd": "evaluator/watermarking/methods/invisible_watermark.py",
    "invisible-watermark-rivagan": "evaluator/watermarking/methods/invisible_watermark.py",
    "maskwm-d32": "evaluator/watermarking/methods/maskwm.py",
    "mbrs": "evaluator/watermarking/methods/mbrs.py",
    "pimog": "evaluator/watermarking/methods/pimog.py",
    "pixelseal": "evaluator/watermarking/methods/pixelseal.py",
    "rawatermark": "evaluator/watermarking/methods/rawatermark.py",
    "ssl-watermarking": "evaluator/watermarking/methods/ssl_watermarking.py",
    "stegastamp": "evaluator/watermarking/methods/stegastamp.py",
    "traditional-spread-dct": "evaluator/watermarking/methods/traditional.py",
    "trustmark-c": "evaluator/watermarking/methods/trustmark.py",
    "trustmark-q": "evaluator/watermarking/methods/trustmark.py",
    "videoseal": "evaluator/watermarking/methods/videoseal.py",
    "vine": "evaluator/watermarking/methods/vine.py",
    "wam": "evaluator/watermarking/methods/wam.py",
}


def split_cell(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"<br\s*/?>", text.strip()) if part.strip()]


def project_source_link(path: str) -> dict[str, str]:
    normalized = path.strip().replace("\\", "/")
    return {"label": normalized, "url": normalized}


def is_upstream_part(part: str, match_start: int) -> bool:
    prefix = part[:match_start]
    return any(marker in prefix for marker in UPSTREAM_MARKERS)


def parse_links(cell: str) -> tuple[list[dict[str, str]], list[str]]:
    links: list[dict[str, str]] = []
    notes: list[str] = []
    for part in split_cell(cell):
        matches = list(LINK_RE.finditer(part))
        if matches:
            for match in matches:
                links.append({"label": match.group(1).strip(), "url": match.group(2).strip()})
            remainder = LINK_RE.sub("", part).strip(" ;，。")
            if remainder:
                notes.append(remainder)
        elif part:
            notes.append(part)
    return links, notes


def parse_repo_cell(cell: str) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[str]]:
    project_sources: list[dict[str, str]] = []
    repos: list[dict[str, str]] = []
    upstream_repos: list[dict[str, str]] = []
    notes: list[str] = []

    for part in split_cell(cell):
        path_match = ID_RE.search(part)
        if path_match and any(prefix in part for prefix in PROJECT_SOURCE_PREFIXES):
            project_sources.append(project_source_link(path_match.group(1)))
            remainder = ID_RE.sub("", part)
            for prefix in PROJECT_SOURCE_PREFIXES:
                remainder = remainder.replace(f"{prefix}：", "").replace(f"{prefix}:", "")
            remainder = remainder.strip(" ;，。")
            if remainder:
                notes.append(remainder)
            continue

        matches = list(LINK_RE.finditer(part))
        if matches:
            for match in matches:
                link = {"label": match.group(1).strip(), "url": match.group(2).strip()}
                if is_upstream_part(part, match.start()):
                    upstream_repos.append(link)
                else:
                    repos.append(link)
            remainder = LINK_RE.sub("", part).strip(" ;，。")
            if remainder and not any(marker in remainder for marker in (*UPSTREAM_MARKERS, UPSTREAM_REPO_LABEL)):
                notes.append(remainder)
            continue

        if part:
            notes.append(part)

    return project_sources, repos, upstream_repos, notes


def dedupe_links(links: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, str]] = []
    for link in links:
        key = (link["label"], link["url"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(link)
    return unique


def is_placeholder_note(note: str) -> bool:
    normalized = note.strip()
    if not normalized:
        return True
    collapsed = normalized.replace("、", "").replace("，", "").replace(",", "").replace(" ", "")
    if collapsed in {
        "子攻击来源：",
        "子攻击来源:",
        "子攻击仓库：",
        "子攻击仓库:",
        "组成来源：",
        "组成来源:",
        "基础参考：",
        "基础参考:",
        "上游仓库：",
        "上游仓库:",
    }:
        return True
    markers = (
        "无独立论文",
        "上游未绑定",
        "属于传统",
        "属于项目内",
        "项目中未记录",
        "无；benchmark baseline",
        "无；标准图像变换",
        "子攻击来源",
        "子攻击仓库",
        "组成来源",
        "基础参考",
    )
    return any(marker in normalized for marker in markers)


def clean_notes(entry: dict, field: str) -> None:
    notes = entry.get(field)
    if not notes:
        return
    cleaned = [note for note in notes if not is_placeholder_note(note)]
    if cleaned:
        entry[field] = cleaned
    else:
        entry.pop(field, None)


def merge_upstream_repos(entry: dict) -> dict:
    upstream = entry.pop("upstreamRepos", None)
    if not upstream:
        return entry
    repos = entry.setdefault("repos", [])
    existing_urls = {link["url"] for link in repos}
    for link in upstream:
        if link["url"] in existing_urls:
            continue
        label = link["label"]
        if "原始" not in label and "original" not in label.lower():
            label = f"{label}（原始）"
        repos.append({"label": label, "url": link["url"]})
    entry["repos"] = dedupe_links(repos)
    return entry


def normalize_entry(entry: dict) -> dict:
    for field, target in (("paperNotes", "papers"),):
        notes = entry.get(field) or []
        if not notes:
            continue
        cleaned_notes: list[str] = []
        for note in notes:
            extracted_links, remainder = parse_links(note)
            if extracted_links:
                entry.setdefault(target, []).extend(extracted_links)
            if remainder:
                cleaned_notes.extend(remainder)
        if cleaned_notes:
            entry[field] = cleaned_notes
        else:
            entry.pop(field, None)

    repo_notes = entry.pop("repoNotes", None)
    if repo_notes:
        cleaned_notes: list[str] = []
        for note in repo_notes:
            project_sources, repos, upstream_repos, remainder = parse_repo_cell(note)
            if project_sources:
                entry.setdefault("projectSources", []).extend(project_sources)
            if repos:
                entry.setdefault("repos", []).extend(repos)
            if upstream_repos:
                entry.setdefault("upstreamRepos", []).extend(upstream_repos)
            if remainder:
                cleaned_notes.extend(remainder)
        if cleaned_notes:
            entry["repoNotes"] = cleaned_notes

    for key in ("projectSources", "repos", "upstreamRepos", "papers"):
        if key in entry:
            entry[key] = dedupe_links(entry[key])

    clean_notes(entry, "paperNotes")
    clean_notes(entry, "repoNotes")

    return merge_upstream_repos(entry)


def attach_default_watermark_project_source(method_id: str, entry: dict) -> dict:
    default_path = WATERMARK_PROJECT_FILES.get(method_id)
    if not default_path:
        return entry
    default_link = project_source_link(default_path)
    existing_urls = {link["url"] for link in entry.get("projectSources", [])}
    if default_link["url"] not in existing_urls:
        entry.setdefault("projectSources", []).insert(0, default_link)
    return entry


def parse_table_row(line: str) -> list[str] | None:
    if not line.startswith("|"):
        return None
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if len(cells) < 5 or cells[0] in {"序号", "---"} or set(cells[0]) == {"-"}:
        return None
    if not cells[0].isdigit():
        return None
    return cells


def parse_markdown_tables(path: Path, *, attach_watermark_defaults: bool = False) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = parse_table_row(line)
        if cells is None:
            continue
        method_match = ID_RE.search(cells[1])
        if method_match is None:
            continue
        method_id = method_match.group(1)
        summary_zh = cells[2].strip()
        paper_links, paper_notes = parse_links(cells[3])
        project_sources, repo_links, upstream_repos, repo_notes = parse_repo_cell(cells[4])
        entry: dict = {
            "summary": {"zh": summary_zh},
            "papers": paper_links,
            "repos": repo_links,
        }
        if project_sources:
            entry["projectSources"] = project_sources
        if upstream_repos:
            entry["upstreamRepos"] = upstream_repos
        if paper_notes:
            entry["paperNotes"] = paper_notes
        if repo_notes:
            entry["repoNotes"] = repo_notes
        entry = normalize_entry(entry)
        if attach_watermark_defaults:
            entry = attach_default_watermark_project_source(method_id, entry)
        entries[method_id] = entry
    return entries


def add_viewpoint_motion_aliases(entries: dict[str, dict]) -> None:
    family_intro = (
        "本组方法属于 REG-3D-SHARP 视角重渲染攻击族：先用 SHARP 从单张水印图预测 3D Gaussian 表示，"
        "再渲染新视角图像。前端按 motion 聚合多个 phase/lookat 变体。"
    )
    motion_samples: dict[str, dict] = {}
    for method_id, entry in list(entries.items()):
        match = VIEWPOINT_ID_RE.match(method_id)
        if match is None:
            continue
        motion = match.group("motion")
        if motion not in motion_samples:
            motion_samples[motion] = entry

    for motion, sample in motion_samples.items():
        motion_label = motion.replace("_", " ")
        entries[motion] = {
            "summary": {
                "zh": f"{family_intro} 当前资源对应 {motion_label} 运动下的 16 个执行变体。"
            },
            "papers": sample.get("papers", []),
            "repos": sample.get("repos", []),
        }
        if sample.get("projectSources"):
            entries[motion]["projectSources"] = sample["projectSources"]
        if sample.get("paperNotes"):
            entries[motion]["paperNotes"] = sample["paperNotes"]
        if sample.get("repoNotes"):
            entries[motion]["repoNotes"] = sample["repoNotes"]


def main() -> int:
    if not WATERMARK_MD.is_file():
        print(f"Missing watermark metadata: {WATERMARK_MD}", file=sys.stderr)
        return 1
    if not ATTACK_MD.is_file():
        print(f"Missing attack metadata: {ATTACK_MD}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    watermark_entries = parse_markdown_tables(WATERMARK_MD, attach_watermark_defaults=True)
    attack_entries = parse_markdown_tables(ATTACK_MD)
    add_viewpoint_motion_aliases(attack_entries)

    watermark_out = OUT_DIR / "watermark-references.json"
    attack_out = OUT_DIR / "attack-references.json"
    watermark_out.write_text(json.dumps(watermark_entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    attack_out.write_text(json.dumps(attack_entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(watermark_entries)} watermark references -> {watermark_out.relative_to(ROOT)}")
    print(f"Wrote {len(attack_entries)} attack references -> {attack_out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
