# Web Development Guidelines

This project is an operational benchmark console. The frontend should help users decide what can run, what must be downloaded, and what will happen next.

## Resource Page Standard

- Keep the top-level resource model to three groups: datasets, watermark algorithms, and attack algorithms.
- Resource rows should be scannable: name, short subtitle, GPU/recommended/status chips, and no long prose.
- Resource details should use this order:
  1. resource name and status;
  2. a one- or two-line purpose summary;
  3. key metrics such as category, method, device, samples, presets, and weight state;
  4. install/download action area;
  5. optional technical mappings, paths, params, and notes.
- Long descriptions from papers, manifests, or future curated files must be collapsed or linked, not shown as a full block by default.
- Dataset downloads must present the choice explicitly. Users on remote machines may not want full or unnecessary downloads, so always show sample count, disk-impact hints, and whether the option is already installed.
- Watermark and attack weight downloads should clearly distinguish "installed", "downloadable", "not available", and "shared by multiple methods".
- Attack algorithms should continue to be grouped by `evaluator/attacks/<folder>` categories, with friendly labels in the UI.

## Metadata Inputs

- Curated resource metadata can be supplied as Markdown or JSON.
- Prefer JSON when the UI must filter, sort, or render fields independently.
- Prefer Markdown when the content is mostly explanatory text for humans.
- Keep imported metadata outside React component bodies; load or map it through small typed helpers.

Suggested JSON shape:

```json
{
  "id": "resource-id",
  "summaryZh": "一句话说明",
  "summaryEn": "One-line summary",
  "recommendedUse": "适用场景",
  "downloadNotes": ["精简包适合快速验证", "全量包需要更大磁盘空间"],
  "references": [{ "label": "Paper", "url": "https://example.com" }]
}
```

Suggested Markdown sections:

```md
# Resource Name

## Summary

## When To Use

## Download Notes

## References
```

## Interaction And Layout

- Build the usable workflow first: browse, inspect, download/install, use in config.
- Avoid page-level horizontal scroll on desktop viewports from 1024px upward.
- Keep resource panels within the viewport where practical; use internal scrolling for long lists and detail content.
- Buttons should describe actions directly: start download, uninstall local, use in config.
- Disabled actions must have nearby visible reason text.
- Empty states should explain the next action, not just say there is no data.

## Verification

Before committing frontend changes, run:

```bash
pnpm --filter @wm-bench/web build
```

For API contract changes that feed the UI, also run:

```bash
python3 -m unittest apps.api.tests.test_api_routes apps.api.tests.test_resource_catalog
```
