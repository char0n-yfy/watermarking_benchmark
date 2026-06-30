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

if ! autodl_wait_for_url "API" "${API_HEALTH_URL}" 90 "${WM_BENCH_LOG_DIR}/api.screen.log"; then
  echo "Recent API screen log:" >&2
  tail -n 80 "${WM_BENCH_LOG_DIR}/api.screen.log" >&2 || true
  exit 1
fi

cat <<EOF
WM Bench AutoDL services started.

Server-local URL:
  http://${CHECK_API_HOST}:${API_PORT}

Health check:
  ${API_HEALTH_URL}

AutoDL/public access:
  1. In the AutoDL console, expose or tunnel local port ${API_PORT}.
  2. Open the generated public URL in any browser.
  3. If you use SSH tunneling instead:
     ssh -L ${API_PORT}:127.0.0.1:${API_PORT} root@<server-ip>
     Then open http://127.0.0.1:${API_PORT} on your computer.

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
