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

API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-3000}"
RUNS_ROOT="${WM_BENCH_RUNS_ROOT:-${ROOT_DIR}/runs}"
PID_DIR="${WM_BENCH_PID_DIR:-${RUNS_ROOT}/pids}"

stop_pid_file() {
  local name="$1"
  local pid_file="${PID_DIR}/${name}.pid"
  local pid

  [[ -f "${pid_file}" ]] || return 0
  pid="$(tr -d '[:space:]' <"${pid_file}" || true)"
  if [[ -n "${pid}" ]] && [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
  fi
  rm -f "${pid_file}"
}

service_command_line() {
  local pid="$1"
  ps -p "${pid}" -o command= 2>/dev/null || true
}

command_matches_local_service() {
  local command_line="$1"
  [[ "${command_line}" == *"${ROOT_DIR}"* ]] && return 0
  [[ "${command_line}" == *"local_worker.py"* ]] && return 0
  [[ "${command_line}" == *"uvicorn app.main:app"* ]] && return 0
  [[ "${command_line}" == *"@wm-bench/web dev"* ]] && return 0
  [[ "${command_line}" == *"next dev"* ]] && return 0
  return 1
}

stop_repo_process() {
  local pid="$1"
  local command_line
  [[ -n "${pid}" ]] || return 0
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 0
  command_line="$(service_command_line "${pid}")"
  if command_matches_local_service "${command_line}"; then
    kill "${pid}" >/dev/null 2>&1 || true
  fi
}

for name in api worker web; do
  stop_pid_file "${name}"
done

for port in "${WEB_PORT}" "${API_PORT}"; do
  if command -v lsof >/dev/null 2>&1; then
    while IFS= read -r pid; do
      stop_repo_process "${pid}"
    done < <(lsof -ti "tcp:${port}" -sTCP:LISTEN 2>/dev/null || true)
  fi
done

if command -v pgrep >/dev/null 2>&1; then
  while IFS= read -r pid; do
    stop_repo_process "${pid}"
  done < <(pgrep -f "local_worker.py|uvicorn app.main:app|@wm-bench/web dev|next dev" 2>/dev/null || true)
fi

echo "WM Bench local services stopped."
