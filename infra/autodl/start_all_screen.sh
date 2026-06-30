#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
autodl_load_env
autodl_prepare_dirs

autodl_ensure_screen

screen -S wmbench-api -X quit >/dev/null 2>&1 || true
screen -S wmbench-worker -X quit >/dev/null 2>&1 || true

screen -L -Logfile "${WM_BENCH_LOG_DIR}/api.screen.log" -dmS wmbench-api bash infra/autodl/start_api.sh
screen -L -Logfile "${WM_BENCH_LOG_DIR}/worker.screen.log" -dmS wmbench-worker bash infra/autodl/start_worker.sh

echo "Started screen sessions:"
screen -ls | grep wmbench || true
