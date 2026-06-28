#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ -f .env.autodl ]]; then
  set -a
  source .env.autodl
  set +a
fi

source infra/autodl/python_env.sh
autodl_require_python_env

export APP_ENV="${APP_ENV:-autodl}"
export WM_BENCH_DATA_ROOT="${WM_BENCH_DATA_ROOT:-/root/autodl-fs/wm-bench}"
export WM_BENCH_RESOURCES_ROOT="${WM_BENCH_RESOURCES_ROOT:-${WM_BENCH_DATA_ROOT}/resources}"
export WM_BENCH_RUNS_ROOT="${WM_BENCH_RUNS_ROOT:-/root/autodl-tmp/wm-bench/runs}"
export WM_BENCH_DB_PATH="${WM_BENCH_DB_PATH:-${WM_BENCH_DATA_ROOT}/state/wmbench.sqlite}"
export WM_BENCH_DEVICE="${WM_BENCH_DEVICE:-cuda:0}"
export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-6006}"

mkdir -p \
  "${WM_BENCH_RESOURCES_ROOT}/datasets" \
  "${WM_BENCH_RESOURCES_ROOT}/weights" \
  "${WM_BENCH_RUNS_ROOT}" \
  "$(dirname "${WM_BENCH_DB_PATH}")"

"${AUTODL_PYTHON}" -m uvicorn app.main:app --app-dir apps/api --host "${API_HOST}" --port "${API_PORT}"
