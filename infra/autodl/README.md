# AutoDL Profile

This is the primary no-Docker deployment profile for the MVP.

Recommended paths:

- Persistent state: `/root/autodl-fs/wm-bench`
- SQLite: `/root/autodl-fs/wm-bench/state/wmbench.sqlite`
- Datasets: `/root/autodl-fs/wm-bench/resources/datasets`
- Weights: `/root/autodl-fs/wm-bench/resources/weights`
- Run cache/results: `/root/autodl-tmp/wm-bench/runs`

## Setup

```bash
cp .env.autodl.example .env.autodl
bash infra/autodl/setup_env.sh
```

Edit `.env.autodl` when you need a different GPU or storage path.
The setup script creates a project-local Python environment at `.venv` by
default. On AutoDL it uses `--system-site-packages` unless
`WM_BENCH_VENV_SYSTEM_SITE_PACKAGES=0`, so the venv can still see CUDA packages
preinstalled in the base image.

## Start

Start both API and worker in `screen`:

```bash
bash infra/autodl/start_all_screen.sh
```

Or start them manually in two terminals:

```bash
bash infra/autodl/start_api.sh
bash infra/autodl/start_worker.sh
```

FastAPI listens on `6006` by default and serves the static web build from
`apps/web/out` when it exists.

## Runtime model

- `POST /runs` creates a queued run only.
- `apps/worker/local_worker.py` claims queued runs and executes them on `WM_BENCH_DEVICE`.
- Worker logs are written to `<run artifact root>/worker.log`.
- `GET /system/runtime` reports configured paths, device, and worker heartbeats.

Known MVP limits: no multi-user auth, no Docker sandbox, and no formal public leaderboard yet.
