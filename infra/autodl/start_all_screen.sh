#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
autodl_load_env
autodl_prepare_dirs

autodl_ensure_screen

autodl_stop_wmbench_screen_sessions
autodl_wait_for_port_free "${API_PORT}" "${WM_BENCH_STOP_WAIT_SECONDS:-30}" || true
autodl_wait_for_no_orphan_workers "${WM_BENCH_STOP_WAIT_SECONDS:-30}" || true

autodl_assert_port_free "${API_PORT}"
autodl_assert_no_orphan_workers

screen -L -Logfile "${WM_BENCH_LOG_DIR}/api.screen.log" -dmS wmbench-api bash infra/autodl/start_api.sh
screen -L -Logfile "${WM_BENCH_LOG_DIR}/worker.screen.log" -dmS wmbench-worker bash infra/autodl/start_worker.sh

echo "Started screen sessions:"
screen -ls | grep wmbench || true
