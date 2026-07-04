#!/usr/bin/env bash
set -euo pipefail

AUTODL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${AUTODL_SCRIPT_DIR}/common.sh"
autodl_load_env

autodl_stop_wmbench_screen_sessions
autodl_wait_for_port_free "${API_PORT}" "${WM_BENCH_STOP_WAIT_SECONDS:-30}" || true
autodl_wait_for_no_orphan_workers "${WM_BENCH_STOP_WAIT_SECONDS:-30}" || true

echo "WM Bench AutoDL services stopped."
echo
echo "Remaining wmbench screen sessions:"
screen -ls | grep wmbench || echo "  none"

remaining_pids="$(autodl_port_pids "${API_PORT}" | tr '\n' ' ' | xargs || true)"
if [[ -n "${remaining_pids}" ]]; then
  echo
  echo "Port ${API_PORT} is still occupied by non-screen process(es):"
  autodl_describe_pids "${remaining_pids}"
  if [[ "${WM_BENCH_STOP_FOREIGN:-0}" == "1" ]]; then
    echo "Killing remaining port owner(s) because WM_BENCH_STOP_FOREIGN=1."
    kill ${remaining_pids} 2>/dev/null || true
  else
    echo "Set WM_BENCH_STOP_FOREIGN=1 to let this script kill these external owner(s)."
  fi
fi

remaining_worker_pids="$(autodl_worker_pids | tr '\n' ' ' | xargs || true)"
if [[ -n "${remaining_worker_pids}" ]]; then
  echo
  echo "Repo-local worker process(es) are still running outside the current screen sessions:"
  autodl_describe_pids "${remaining_worker_pids}"
  if [[ "${WM_BENCH_STOP_FOREIGN:-0}" == "1" ]]; then
    echo "Killing remaining worker process(es) because WM_BENCH_STOP_FOREIGN=1."
    kill ${remaining_worker_pids} 2>/dev/null || true
  else
    echo "Set WM_BENCH_STOP_FOREIGN=1 to let this script kill these external worker(s)."
  fi
fi
