#!/usr/bin/env bash

AUTODL_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

autodl_cd_repo() {
  cd "${AUTODL_REPO_ROOT}"
}

autodl_load_env_file_defaults() {
  local env_file="$1"
  local line key value

  [[ -f "${env_file}" ]] || return 0

  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" || "${line}" == \#* || "${line}" != *= ]] && continue

    key="${line%%=*}"
    value="${line#*=}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [[ -n "${!key+x}" ]] && continue

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

autodl_load_env() {
  autodl_cd_repo

  if [[ ! -f .env.autodl ]]; then
    cp .env.autodl.example .env.autodl
  fi

  autodl_load_env_file_defaults "${AUTODL_REPO_ROOT}/.env.autodl"

  export WM_BENCH_DOTENV_PATH="${AUTODL_REPO_ROOT}/.env.autodl"
  export APP_ENV="${APP_ENV:-autodl}"
  export WM_BENCH_DATA_ROOT="${WM_BENCH_DATA_ROOT:-/root/autodl-fs/wm-bench}"
  export WM_BENCH_RESOURCES_ROOT="${WM_BENCH_RESOURCES_ROOT:-${AUTODL_REPO_ROOT}/resources}"
  export WM_BENCH_RUNS_ROOT="${WM_BENCH_RUNS_ROOT:-${AUTODL_REPO_ROOT}/runs}"
  export WM_BENCH_DB_PATH="${WM_BENCH_DB_PATH:-${WM_BENCH_DATA_ROOT}/state/wmbench.sqlite}"
  export WM_BENCH_DEVICE="${WM_BENCH_DEVICE:-cuda:0}"
  export WM_BENCH_WORKER_POLL_SECONDS="${WM_BENCH_WORKER_POLL_SECONDS:-2}"
  export WM_BENCH_RUN_TIMEOUT_SECONDS="${WM_BENCH_RUN_TIMEOUT_SECONDS:-3600}"
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
    "$(dirname "${WM_BENCH_DB_PATH}")" \
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

autodl_local_host() {
  local host="$1"
  if [[ "${host}" == "0.0.0.0" ]]; then
    echo "127.0.0.1"
  else
    echo "${host}"
  fi
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
    if curl -fsS "${url}" >/dev/null 2>&1; then
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
