# Worker

The worker package owns local file-backed experiment execution for the
local/AutoDL runtime profile.

Current scope:

- claim queued experiment runs from `WM_BENCH_RUNS_ROOT/_experiment_state`;
- execute runs through `app.services.experiment_service`;
- provide a dry-run attack helper that reuses `evaluator.attacks.runner`.
