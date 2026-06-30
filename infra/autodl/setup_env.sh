#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ ! -f .env.autodl ]]; then
  cp .env.autodl.example .env.autodl
fi

set -a
source .env.autodl
set +a

export WM_BENCH_DOTENV_PATH="${PWD}/.env.autodl"
export WM_BENCH_VENV="${WM_BENCH_VENV:-.venv}"
export WM_BENCH_VENV_SYSTEM_SITE_PACKAGES="${WM_BENCH_VENV_SYSTEM_SITE_PACKAGES:-1}"
export WM_BENCH_INSTALL_PYTHON_DEPS="${WM_BENCH_INSTALL_PYTHON_DEPS:-1}"
export WM_BENCH_INSTALL_SHARP_DEPS="${WM_BENCH_INSTALL_SHARP_DEPS:-1}"

bash scripts/bootstrap-python.sh

source infra/autodl/python_env.sh
autodl_require_python_env

mkdir -p \
  "${WM_BENCH_RESOURCES_ROOT}/datasets" \
  "${WM_BENCH_RESOURCES_ROOT}/weights" \
  "${WM_BENCH_RUNS_ROOT}" \
  "$(dirname "${WM_BENCH_DB_PATH}")"

corepack enable
pnpm install
NEXT_PUBLIC_API_BASE_URL="${NEXT_PUBLIC_API_BASE_URL:-}" pnpm --filter @wm-bench/web build

echo "AutoDL environment is ready."
echo "Python: ${AUTODL_PYTHON}"
echo "Datasets: ${WM_BENCH_RESOURCES_ROOT}/datasets"
echo "Weights: ${WM_BENCH_RESOURCES_ROOT}/weights"
echo "Runs: ${WM_BENCH_RUNS_ROOT}"
echo "SQLite: ${WM_BENCH_DB_PATH}"
