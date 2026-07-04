#!/usr/bin/env bash
set -euo pipefail

AUTODL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${AUTODL_SCRIPT_DIR}/common.sh"
autodl_load_env
autodl_prepare_dirs

autodl_ensure_screen

bash "${AUTODL_SCRIPT_DIR}/setup_env.sh"
bash "${AUTODL_SCRIPT_DIR}/start_all_screen.sh"

CHECK_API_HOST="$(autodl_local_host "${API_HOST}")"
API_HEALTH_URL="http://${CHECK_API_HOST}:${API_PORT}/health"
API_RUNTIME_URL="http://${CHECK_API_HOST}:${API_PORT}/system/runtime"
PUBLIC_URL="$(autodl_public_url_for_port "${API_PORT}")"
PUBLIC_HEALTH_URL=""

if ! autodl_wait_for_url "API" "${API_HEALTH_URL}" 90 "${WM_BENCH_LOG_DIR}/api.screen.log"; then
  echo "Recent API screen log:" >&2
  tail -n 80 "${WM_BENCH_LOG_DIR}/api.screen.log" >&2 || true
  exit 1
fi
if ! autodl_validate_runtime "${API_RUNTIME_URL}" "autodl" "${WM_BENCH_DEVICE}"; then
  echo "Recent API screen log:" >&2
  tail -n 80 "${WM_BENCH_LOG_DIR}/api.screen.log" >&2 || true
  exit 1
fi

if [[ -n "${PUBLIC_URL}" ]]; then
  PUBLIC_HEALTH_URL="${PUBLIC_URL%/}/health"
  if ! autodl_wait_for_url "AutoDL public access" "${PUBLIC_HEALTH_URL}" "${WM_BENCH_PUBLIC_WAIT_SECONDS:-120}" "${WM_BENCH_LOG_DIR}/api.screen.log"; then
    echo "Local API is healthy, but the AutoDL public URL is not reachable yet." >&2
    echo "Public URL: ${PUBLIC_URL}" >&2
    echo "Expected local port: ${API_PORT}" >&2
    echo "Check the AutoDL custom service mapping for local port ${API_PORT}." >&2
    exit 1
  fi
fi

cat <<EOF
WM Bench AutoDL services started.

Server-local URL:
  http://${CHECK_API_HOST}:${API_PORT}

Health check:
  ${API_HEALTH_URL}

AutoDL/public access:
EOF

if [[ -n "${PUBLIC_URL}" ]]; then
  cat <<EOF
  ${PUBLIC_URL}

Public health check:
  ${PUBLIC_HEALTH_URL}
EOF
else
  cat <<EOF
  No AutoDL public URL was detected for local port ${API_PORT}.
  Available AutoDL service URLs:
$(autodl_list_public_service_urls || true)

  If you use SSH tunneling instead:
    ssh -L ${API_PORT}:127.0.0.1:${API_PORT} root@<server-ip>
    Then open http://127.0.0.1:${API_PORT} on your computer.
EOF
fi

cat <<EOF

Service sessions:
  screen -r wmbench-api
  screen -r wmbench-worker

Logs:
  ${WM_BENCH_LOG_DIR}/api.screen.log
  ${WM_BENCH_LOG_DIR}/worker.screen.log

Stop services:
  bash infra/autodl/stop.sh

Or stop screen sessions manually:
  screen -S wmbench-api -X quit
  screen -S wmbench-worker -X quit

Check sessions:
  screen -ls
EOF
