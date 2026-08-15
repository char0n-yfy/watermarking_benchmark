# Unified service manager

`scripts/wmbench.py` is the canonical cross-platform entrypoint for dependency preparation, service lifecycle, status, logs, and deployment checks.

```bash
python3 scripts/wmbench.py --profile local bootstrap
python3 scripts/wmbench.py --profile local up
python3 scripts/wmbench.py --profile local status
python3 scripts/wmbench.py --profile local logs --follow
python3 scripts/wmbench.py --profile local restart
python3 scripts/wmbench.py --profile local down
```

Use `py -3` instead of `python3` on Windows. Global options must appear before the command:

```bash
python3 scripts/wmbench.py --profile local --api-port 8001 --web-port 3001 --device cpu up
python3 scripts/wmbench.py --profile production --api-port 6006 up
python3 scripts/wmbench.py --profile autodl --skip-sharp check
```

Profiles:

- `local`: Next.js development server plus API and Worker; default ports 3000 and 8000.
- `production`: static Web build served by FastAPI plus Worker; default port 6006.
- `autodl`: production layout with `.env.autodl`, CUDA defaults, and optional conda Node installation.
- `auto`: chooses `autodl` on an AutoDL host and `local` elsewhere.

Configuration precedence is command line, current process environment, profile dotenv, then defaults. Profile dotenv files are `.env`, `.env.production`, and `.env.autodl`.

`up` prepares only dependency/build layers whose fingerprints changed. Use `up --no-bootstrap` after a completed bootstrap when no dependency or Web source changed. Service PID metadata is kept under `WM_BENCH_RUNS_ROOT/services/<profile>`; logs are kept under `WM_BENCH_LOG_DIR`.

`wmbench.py` is the only service lifecycle entrypoint. AutoDL is a profile rather than a separate deployment directory. `prepare_perceptual_weights.py` is an internal bootstrap helper invoked automatically for that profile; it is not a second entrypoint.
