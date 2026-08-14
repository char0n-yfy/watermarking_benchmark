# Web style architecture

`app/globals.css` is the single import entry. Keep its cascade order stable:

1. `tokens.css` — colors, typography, spacing, focus, and shell dimensions.
2. `foundation.css` — resets, AppShell, shared controls, panels, and page states.
3. Page modules — `analytics.css`, `resources.css`, `configs.css`, and `runs.css`.
4. `responsive.css` — breakpoint-only overrides.
5. `accessibility.css` — focus and reduced-motion guarantees; always last.

New page-specific selectors belong in the closest page module. Reusable values belong in `tokens.css`; reusable components belong in `foundation.css`. Avoid redefining the same selector in multiple page modules.
