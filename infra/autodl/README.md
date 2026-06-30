# AutoDL Profile

This is the primary no-Docker deployment profile for the MVP.

Recommended paths:

- Persistent state: `/root/autodl-fs/wm-bench`
- SQLite: `/root/autodl-fs/wm-bench/state/wmbench.sqlite`
- Datasets: `<repo>/resources/datasets`
- Weights: `<repo>/resources/weights`
- Run cache/results: `/root/autodl-tmp/wm-bench/runs`
- Logs: `/root/autodl-tmp/wm-bench/runs/logs`

## One Command

From the repository root:

```bash
bash infra/autodl/start.sh
```

The command creates `.env.autodl` when missing, creates/reuses `.venv`, installs Python dependencies, prepares screen and Node.js/pnpm when missing, installs web dependencies, builds `apps/web/out`, and starts API + worker in `screen`.

FastAPI listens on `6006` by default and serves the static web build from `apps/web/out` when it exists. Expose local port `6006` in the AutoDL console, or tunnel it with SSH.

## Stop

```bash
bash infra/autodl/stop.sh
```

The one-command startup output also prints this shutdown command, plus the manual `screen` commands for debugging.

## Configuration

Edit `.env.autodl` when you need a different GPU, port, storage path, or dependency mode. By default, AutoDL reads and downloads resources under the repository `resources/` directory, matching the macOS launcher.

Useful switches:

- `WM_BENCH_DEVICE=cuda:0`: worker device.
- `API_PORT=6006`: API and static web port.
- `WM_BENCH_INSTALL_SHARP_DEPS=0`: skip optional SHARP/3D heavy dependencies.
- `WM_BENCH_AUTO_INSTALL_NODE=0`: disable automatic Node.js/pnpm installation and fail fast if missing.
- `WM_BENCH_AUTO_INSTALL_SCREEN=0`: disable automatic `screen` installation and fail fast if missing.
- `WM_BENCH_LOG_DIR=/root/autodl-tmp/wm-bench/runs/logs`: screen log directory.

## Lower-Level Commands

All AutoDL deployment scripts live in this directory. Use these only when debugging the deployment pieces separately:

```bash
bash infra/autodl/setup_env.sh
bash infra/autodl/start_all_screen.sh
```

Or start API and worker manually in two terminals:

```bash
bash infra/autodl/start_api.sh
bash infra/autodl/start_worker.sh
```

## Runtime Model

- `POST /runs` creates a queued run only.
- `apps/worker/local_worker.py` claims queued runs and executes them on `WM_BENCH_DEVICE`.
- Worker logs are written to `<run artifact root>/worker.log`.
- Screen logs are written under `WM_BENCH_LOG_DIR`.
- `GET /system/runtime` reports configured paths, device, and worker heartbeats.

Known MVP limits: no multi-user auth, no Docker sandbox, and no formal public leaderboard yet.
