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


def split_cell(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"<br\s*/?>", text.strip()) if part.strip()]


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


def parse_table_row(line: str) -> list[str] | None:
    if not line.startswith("|"):
        return None
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if len(cells) < 5 or cells[0] in {"序号", "---"} or set(cells[0]) == {"-"}:
        return None
    if not cells[0].isdigit():
        return None
    return cells


def normalize_entry(entry: dict) -> dict:
    for field, target in (("paperNotes", "papers"), ("repoNotes", "repos")):
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
    return entry


def parse_markdown_tables(path: Path) -> dict[str, dict]:
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
        repo_links, repo_notes = parse_links(cells[4])
        entry: dict = {
            "summary": {"zh": summary_zh},
            "papers": paper_links,
            "repos": repo_links,
        }
        if paper_notes:
            entry["paperNotes"] = paper_notes
        if repo_notes:
            entry["repoNotes"] = repo_notes
        entries[method_id] = normalize_entry(entry)
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

    watermark_entries = parse_markdown_tables(WATERMARK_MD)
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
