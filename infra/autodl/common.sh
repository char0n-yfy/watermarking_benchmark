#!/usr/bin/env bash

AUTODL_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

autodl_cd_repo() {
  cd "${AUTODL_REPO_ROOT}"
}

autodl_load_env_file() {
  local env_file="$1"
  local line key value

  [[ -f "${env_file}" ]] || return 0

  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" || "${line}" == \#* || "${line}" != *"="* ]] && continue

    key="${line%%=*}"
    value="${line#*=}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue

    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    if [[ "${value}" == \"*\" && "${value}" == *\" && "${#value}" -ge 2 ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "${value}" == \'*\' && "${value}" == *\' && "${#value}" -ge 2 ]]; then
      value="${value:1:${#value}-2}"
    fi
    export "${key}=${value}"
  done <"${env_file}"
}

autodl_load_platform_env() {
  local profile="/etc/profile.d/autodl.env.sh"
  if [[ -f "${profile}" ]]; then
    # shellcheck disable=SC1090
    source "${profile}"
  fi
}

autodl_load_env() {
  autodl_cd_repo

  if [[ ! -f .env.autodl ]]; then
    cp .env.autodl.example .env.autodl
  fi

  autodl_load_platform_env
  autodl_load_env_file "${AUTODL_REPO_ROOT}/.env.autodl"

  export WM_BENCH_DOTENV_PATH="${AUTODL_REPO_ROOT}/.env.autodl"
  export APP_ENV="${APP_ENV:-autodl}"
  export WM_BENCH_DATA_ROOT="${WM_BENCH_DATA_ROOT:-/root/autodl-fs/wm-bench}"
  export WM_BENCH_RESOURCES_ROOT="${WM_BENCH_RESOURCES_ROOT:-${AUTODL_REPO_ROOT}/resources}"
  export WM_BENCH_RUNS_ROOT="${WM_BENCH_RUNS_ROOT:-${AUTODL_REPO_ROOT}/runs}"
  export WM_BENCH_DEVICE="${WM_BENCH_DEVICE:-cuda:0}"
  export WM_BENCH_WORKER_POLL_SECONDS="${WM_BENCH_WORKER_POLL_SECONDS:-2}"
  export WM_BENCH_VENV="${WM_BENCH_VENV:-.venv}"
  export WM_BENCH_VENV_SYSTEM_SITE_PACKAGES="${WM_BENCH_VENV_SYSTEM_SITE_PACKAGES:-1}"
  export WM_BENCH_INSTALL_PYTHON_DEPS="${WM_BENCH_INSTALL_PYTHON_DEPS:-1}"
  export WM_BENCH_INSTALL_SHARP_DEPS="${WM_BENCH_INSTALL_SHARP_DEPS:-1}"
  export WM_BENCH_AUTO_INSTALL_NODE="${WM_BENCH_AUTO_INSTALL_NODE:-1}"
  export WM_BENCH_AUTO_INSTALL_SCREEN="${WM_BENCH_AUTO_INSTALL_SCREEN:-1}"
  export WM_BENCH_NODE_VERSION="${WM_BENCH_NODE_VERSION:-20}"
  export WM_BENCH_PNPM_VERSION="${WM_BENCH_PNPM_VERSION:-9.15.0}"
  export WM_BENCH_LOG_DIR="${WM_BENCH_LOG_DIR:-${WM_BENCH_RUNS_ROOT}/logs}"
  export API_HOST="${API_HOST:-0.0.0.0}"
  export API_PORT="${API_PORT:-6006}"
}

autodl_prepare_dirs() {
  mkdir -p \
    "${WM_BENCH_RESOURCES_ROOT}/datasets" \
    "${WM_BENCH_RESOURCES_ROOT}/weights" \
    "${WM_BENCH_RUNS_ROOT}" \
    "${WM_BENCH_LOG_DIR}"
}

autodl_ensure_screen() {
  if command -v screen >/dev/null 2>&1; then
    return 0
  fi

  if [[ "${WM_BENCH_AUTO_INSTALL_SCREEN:-1}" == "0" ]]; then
    echo "screen is required on AutoDL but was not found." >&2
    echo "Install screen or set WM_BENCH_AUTO_INSTALL_SCREEN=1." >&2
    return 1
  fi

  if command -v apt-get >/dev/null 2>&1; then
    echo "Installing screen with apt-get..."
    apt-get update
    apt-get install -y screen
    command -v screen >/dev/null 2>&1 && return 0
  fi

  echo "Unable to prepare screen automatically." >&2
  echo "Install screen and rerun: bash infra/autodl/start.sh" >&2
  return 1
}

autodl_wmbench_screen_sessions() {
  if ! command -v screen >/dev/null 2>&1; then
    return 0
  fi
  screen -ls 2>/dev/null | awk '/[.](wmbench-api|wmbench-worker)([^[:space:]]*)/ { print $1 }'
}

autodl_stop_wmbench_screen_sessions() {
  local session
  while IFS= read -r session; do
    [[ -n "${session}" ]] || continue
    screen -S "${session}" -X quit >/dev/null 2>&1 || true
  done < <(autodl_wmbench_screen_sessions)
}

autodl_local_host() {
  local host="$1"
  if [[ "${host}" == "0.0.0.0" ]]; then
    echo "127.0.0.1"
  else
    echo "${host}"
  fi
}

autodl_public_url_for_port() {
  local port="$1"
  local configured="${WM_BENCH_PUBLIC_URL:-}"
  local env_name value

  if [[ -n "${configured}" ]]; then
    echo "${configured%/}"
    return 0
  fi

  env_name="AutoDLService${port}URL"
  value="$(printenv "${env_name}" 2>/dev/null || true)"
  if [[ -z "${value}" && "${port}" == "6006" ]]; then
    value="$(printenv AutoDLServiceURL 2>/dev/null || true)"
  fi

  echo "${value%/}"
}

autodl_list_public_service_urls() {
  env | awk -F= '/^AutoDLService([0-9]+)?URL=/ { print "  " $0 }' | sort
}

autodl_wait_for_url() {
  local name="$1"
  local url="$2"
  local timeout_seconds="${3:-60}"
  local log_file="${4:-}"

  if ! command -v curl >/dev/null 2>&1; then
    sleep 3
    return 0
  fi

  for ((i = 0; i < timeout_seconds; i += 1)); do
    if curl -fsSk --connect-timeout 5 --max-time 10 "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  echo "${name} did not become reachable: ${url}" >&2
  if [[ -n "${log_file}" ]]; then
    echo "Check log: ${log_file}" >&2
  fi
  return 1
}

autodl_port_pids() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti TCP:"${port}" -sTCP:LISTEN 2>/dev/null | sort -u
    return 0
  fi
  if command -v fuser >/dev/null 2>&1; then
    fuser "${port}/tcp" 2>/dev/null | tr ' ' '\n' | sed '/^$/d' | sort -u
    return 0
  fi
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp "sport = :${port}" 2>/dev/null \
      | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
      | sort -u
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    python - "${port}" <<'PY'
from __future__ import annotations

import os
import sys

target_port = int(sys.argv[1])
listen_state = "0A"
socket_inodes: set[str] = set()

for proc_net in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        with open(proc_net, "r", encoding="utf-8", errors="replace") as handle:
            rows = handle.read().splitlines()[1:]
    except OSError:
        continue
    for row in rows:
        columns = row.split()
        if len(columns) < 10 or columns[3] != listen_state:
            continue
        local_address = columns[1]
        try:
            port = int(local_address.rsplit(":", 1)[1], 16)
        except (IndexError, ValueError):
            continue
        if port == target_port:
            socket_inodes.add(columns[9])

if not socket_inodes:
    raise SystemExit(0)

pids: set[int] = set()
for name in os.listdir("/proc"):
    if not name.isdigit():
        continue
    fd_dir = f"/proc/{name}/fd"
    try:
        fd_names = os.listdir(fd_dir)
    except OSError:
        continue
    for fd_name in fd_names:
        try:
            target = os.readlink(f"{fd_dir}/{fd_name}")
        except OSError:
            continue
        if target.startswith("socket:[") and target[8:-1] in socket_inodes:
            pids.add(int(name))
            break

for pid in sorted(pids):
    print(pid)
PY
    return 0
  fi
}

autodl_wait_for_port_free() {
  local port="$1"
  local timeout_seconds="${2:-30}"
  local pids

  for ((i = 0; i < timeout_seconds; i += 1)); do
    pids="$(autodl_port_pids "${port}" | tr '\n' ' ' | xargs || true)"
    if [[ -z "${pids}" ]]; then
      return 0
    fi
    sleep 1
  done

  return 1
}

autodl_describe_pids() {
  local pids="$1"
  [[ -n "${pids}" ]] || return 0
  ps -o pid,ppid,stat,command -p ${pids} 2>/dev/null || true
}

autodl_worker_pids() {
  if ! command -v python >/dev/null 2>&1; then
    return 0
  fi
  python - "${AUTODL_REPO_ROOT}" <<'PY'
from __future__ import annotations

import os
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
needle = b"apps/worker/local_worker.py"
current_pid = os.getpid()

for name in os.listdir("/proc"):
    if not name.isdigit():
        continue
    pid = int(name)
    if pid == current_pid:
        continue
    try:
        cmdline = Path(f"/proc/{name}/cmdline").read_bytes()
    except OSError:
        continue
    if needle not in cmdline:
        continue
    try:
        cwd = Path(os.readlink(f"/proc/{name}/cwd")).resolve()
    except OSError:
        continue
    if cwd == repo:
        print(pid)
PY
}

autodl_wait_for_no_orphan_workers() {
  local timeout_seconds="${1:-30}"
  local pids

  for ((i = 0; i < timeout_seconds; i += 1)); do
    pids="$(autodl_worker_pids | tr '\n' ' ' | xargs || true)"
    if [[ -z "${pids}" ]]; then
      return 0
    fi
    sleep 1
  done

  return 1
}

autodl_assert_port_free() {
  local port="$1"
  local pids
  pids="$(autodl_port_pids "${port}" | tr '\n' ' ' | xargs || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  echo "Port ${port} is already occupied before AutoDL startup." >&2
  echo "Refusing to start because an existing API could fool health checks." >&2
  autodl_describe_pids "${pids}" >&2
  echo "Stop the owner process first, or run: bash infra/autodl/stop.sh" >&2
  return 1
}

autodl_assert_no_orphan_workers() {
  local pids
  pids="$(autodl_worker_pids | tr '\n' ' ' | xargs || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  echo "Repo-local worker process(es) are already running before AutoDL startup." >&2
  echo "Refusing to start because an orphan worker can claim runs with the wrong environment." >&2
  autodl_describe_pids "${pids}" >&2
  echo "Stop them first, or run: WM_BENCH_STOP_FOREIGN=1 bash infra/autodl/stop.sh" >&2
  return 1
}

autodl_validate_runtime() {
  local runtime_url="$1"
  local expected_environment="$2"
  local expected_device="$3"

  python - "${runtime_url}" "${expected_environment}" "${expected_device}" <<'PY'
from __future__ import annotations

import json
import sys
import urllib.request

url, expected_environment, expected_device = sys.argv[1:4]
try:
    with urllib.request.urlopen(url, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
except Exception as exc:
    print(f"Unable to read runtime payload from {url}: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)

errors: list[str] = []
if payload.get("environment") != expected_environment:
    errors.append(f"environment={payload.get('environment')!r}, expected {expected_environment!r}")
if payload.get("device") != expected_device:
    errors.append(f"device={payload.get('device')!r}, expected {expected_device!r}")

if errors:
    print("AutoDL runtime validation failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
    raise SystemExit(1)
PY
}
