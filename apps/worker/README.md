# Worker

The worker package owns local SQLite-queued experiment execution for the
local/AutoDL runtime profile.

Current scope:

- claim queued experiment runs from the SQLite metadata database;
- execute runs through `app.services.experiment_service`;
- provide a dry-run attack helper that reuses `evaluator.attacks.runner`.
