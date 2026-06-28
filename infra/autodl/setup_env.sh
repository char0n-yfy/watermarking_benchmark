#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ ! -f .env.autodl ]]; then
  cp .env.autodl.example .env.autodl
fi

set -a
source .env.autodl
set +a

source infra/autodl/python_env.sh
autodl_prepare_python_env

mkdir -p \
  "${WM_BENCH_RESOURCES_ROOT}/datasets" \
  "${WM_BENCH_RESOURCES_ROOT}/weights" \
  "${WM_BENCH_RUNS_ROOT}" \
  "$(dirname "${WM_BENCH_DB_PATH}")"

"${AUTODL_PYTHON}" -m pip install --upgrade pip
"${AUTODL_PYTHON}" -m pip install -r apps/api/requirements.txt
"${AUTODL_PYTHON}" -m pip install -r apps/worker/requirements.txt

corepack enable
pnpm install
NEXT_PUBLIC_API_BASE_URL="${NEXT_PUBLIC_API_BASE_URL:-}" pnpm --filter @wm-bench/web build

echo "AutoDL environment is ready."
echo "Python: ${AUTODL_PYTHON}"
echo "Datasets: ${WM_BENCH_RESOURCES_ROOT}/datasets"
echo "Weights: ${WM_BENCH_RESOURCES_ROOT}/weights"
echo "Runs: ${WM_BENCH_RUNS_ROOT}"
echo "SQLite: ${WM_BENCH_DB_PATH}"
