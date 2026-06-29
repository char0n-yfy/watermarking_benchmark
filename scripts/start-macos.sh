#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

VENV_DIR="${WM_BENCH_VENV:-.venv}"
PYTHON_BIN="${PYTHON:-python3}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
WEB_HOST="${WEB_HOST:-127.0.0.1}"
WEB_PORT="${WEB_PORT:-3000}"

export APP_ENV="${APP_ENV:-development}"
export WM_BENCH_DATA_ROOT="${WM_BENCH_DATA_ROOT:-${ROOT_DIR}}"
export WM_BENCH_RESOURCES_ROOT="${WM_BENCH_RESOURCES_ROOT:-${ROOT_DIR}/resources}"
export WM_BENCH_RUNS_ROOT="${WM_BENCH_RUNS_ROOT:-${ROOT_DIR}/runs/local}"
export WM_BENCH_DB_PATH="${WM_BENCH_DB_PATH:-${WM_BENCH_RUNS_ROOT}/wmbench.sqlite}"
export WM_BENCH_DEVICE="${WM_BENCH_DEVICE:-cpu}"
export WM_BENCH_WORKER_POLL_SECONDS="${WM_BENCH_WORKER_POLL_SECONDS:-2}"
export WM_BENCH_RUN_TIMEOUT_SECONDS="${WM_BENCH_RUN_TIMEOUT_SECONDS:-3600}"
export API_HOST
export API_PORT

LOG_DIR="${WM_BENCH_LOG_DIR:-${WM_BENCH_RUNS_ROOT}/logs}"
PID_DIR="${WM_BENCH_PID_DIR:-${WM_BENCH_RUNS_ROOT}/pids}"

mkdir -p \
  "${WM_BENCH_RESOURCES_ROOT}/datasets" \
  "${WM_BENCH_RESOURCES_ROOT}/weights" \
  "${WM_BENCH_RUNS_ROOT}" \
  "$(dirname "${WM_BENCH_DB_PATH}")" \
  "${LOG_DIR}" \
  "${PID_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

PYTHON="${VENV_DIR}/bin/python"
"${PYTHON}" -m pip install --upgrade pip
"${PYTHON}" -m pip install -r apps/api/requirements.txt
"${PYTHON}" -m pip install -r apps/worker/requirements.txt

corepack enable
corepack pnpm install

stop_pid() {
  local name="$1"
  local pid_file="${PID_DIR}/${name}.pid"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
      for _ in {1..20}; do
        if ! kill -0 "${pid}" >/dev/null 2>&1; then
          break
        fi
        sleep 0.25
      done
    fi
    rm -f "${pid_file}"
  fi
}

wait_for_url() {
  local name="$1"
  local url="$2"
  local log_file="$3"
  if ! command -v curl >/dev/null 2>&1; then
    sleep 3
    return 0
  fi
  for _ in {1..60}; do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "${name} did not become reachable: ${url}" >&2
  echo "Check log: ${log_file}" >&2
  return 1
}

stop_pid api
stop_pid worker
stop_pid web

"${PYTHON}" -m uvicorn app.main:app \
  --app-dir apps/api \
  --host "${API_HOST}" \
  --port "${API_PORT}" \
  >"${LOG_DIR}/api.log" 2>&1 &
echo "$!" >"${PID_DIR}/api.pid"

"${PYTHON}" apps/worker/local_worker.py \
  --poll-seconds "${WM_BENCH_WORKER_POLL_SECONDS}" \
  >"${LOG_DIR}/worker.log" 2>&1 &
echo "$!" >"${PID_DIR}/worker.pid"

NEXT_PUBLIC_API_BASE_URL="http://localhost:${API_PORT}" \
  corepack pnpm --filter @wm-bench/web dev \
  --hostname "${WEB_HOST}" \
  --port "${WEB_PORT}" \
  >"${LOG_DIR}/web.log" 2>&1 &
echo "$!" >"${PID_DIR}/web.pid"

CHECK_API_HOST="${API_HOST}"
CHECK_WEB_HOST="${WEB_HOST}"
if [[ "${CHECK_API_HOST}" == "0.0.0.0" ]]; then
  CHECK_API_HOST="127.0.0.1"
fi
if [[ "${CHECK_WEB_HOST}" == "0.0.0.0" ]]; then
  CHECK_WEB_HOST="127.0.0.1"
fi

wait_for_url "API" "http://${CHECK_API_HOST}:${API_PORT}/health" "${LOG_DIR}/api.log"
wait_for_url "Web UI" "http://${CHECK_WEB_HOST}:${WEB_PORT}" "${LOG_DIR}/web.log"

cat <<EOF
WM Bench local services started.

Web UI:     http://${CHECK_WEB_HOST}:${WEB_PORT}
API health: http://${CHECK_API_HOST}:${API_PORT}/health

Logs:
  ${LOG_DIR}/api.log
  ${LOG_DIR}/worker.log
  ${LOG_DIR}/web.log

Stop:
  kill \$(cat "${PID_DIR}/api.pid") \$(cat "${PID_DIR}/worker.pid") \$(cat "${PID_DIR}/web.pid")
EOF
