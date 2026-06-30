#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/node_env.sh"

autodl_load_env

bash infra/autodl/bootstrap_python.sh

source infra/autodl/python_env.sh
autodl_require_python_env

autodl_prepare_dirs

autodl_ensure_node
autodl_ensure_pnpm
autodl_pnpm install
NEXT_PUBLIC_API_BASE_URL="${NEXT_PUBLIC_API_BASE_URL:-}" autodl_pnpm --filter @wm-bench/web build

echo "AutoDL environment is ready."
echo "Python: ${AUTODL_PYTHON}"
echo "Node: $(node --version)"
echo "pnpm: $(autodl_pnpm --version)"
echo "Datasets: ${WM_BENCH_RESOURCES_ROOT}/datasets"
echo "Weights: ${WM_BENCH_RESOURCES_ROOT}/weights"
echo "Runs: ${WM_BENCH_RUNS_ROOT}"
echo "SQLite: ${WM_BENCH_DB_PATH}"
