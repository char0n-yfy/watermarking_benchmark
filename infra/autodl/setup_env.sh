#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ ! -f .env.autodl ]]; then
  cp .env.autodl.example .env.autodl
fi

set -a
source .env.autodl
set +a

mkdir -p \
  "${WM_BENCH_RESOURCES_ROOT}/datasets" \
  "${WM_BENCH_RESOURCES_ROOT}/weights" \
  "${WM_BENCH_RUNS_ROOT}" \
  "$(dirname "${WM_BENCH_DB_PATH}")"

python -m pip install -r apps/api/requirements.txt
python -m pip install -r apps/worker/requirements.txt

corepack enable
pnpm install
NEXT_PUBLIC_API_BASE_URL="${NEXT_PUBLIC_API_BASE_URL:-}" pnpm --filter @wm-bench/web build

echo "AutoDL environment is ready."
echo "Datasets: ${WM_BENCH_RESOURCES_ROOT}/datasets"
echo "Weights: ${WM_BENCH_RESOURCES_ROOT}/weights"
echo "Runs: ${WM_BENCH_RUNS_ROOT}"
echo "SQLite: ${WM_BENCH_DB_PATH}"
