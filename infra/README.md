# Infrastructure

The active development profile is local/AutoDL:

- FastAPI stores demo metadata in SQLite.
- Runs and image artifacts are written under `runs/local` locally or
  `WM_BENCH_RUNS_ROOT` on AutoDL.
- The local worker is a plain Python process.
- Datasets live under `resources/datasets`.
- Model weights live under `resources/weights`.

## Local start

```bash
python -m uvicorn app.main:app --app-dir apps/api --host 127.0.0.1 --port 8000
pnpm --filter @wm-bench/web dev
```

`POST /runs` creates queued runs. Start the worker to execute them:

```bash
python apps/worker/local_worker.py --poll-seconds 2
```

For AutoDL, use the single entrypoint: `bash infra/autodl/start.sh`. The `infra/autodl/*` scripts are lower-level helpers for debugging.
