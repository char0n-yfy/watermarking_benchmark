#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

load_dotenv_defaults() {
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

DOTENV_PATH="${WM_BENCH_DOTENV_PATH:-${ROOT_DIR}/.env}"
if [[ -n "${DOTENV_PATH}" ]]; then
  load_dotenv_defaults "${DOTENV_PATH}"
fi

VENV_DIR="${WM_BENCH_VENV:-.venv}"
PYTHON_BIN="${PYTHON:-python3}"
PYTHON_EXE="${VENV_DIR}/bin/python"

if [[ "${WM_BENCH_INSTALL_PYTHON_DEPS:-1}" == "0" ]]; then
  if [[ ! -x "${PYTHON_EXE}" ]]; then
    echo "Missing Python virtual environment: ${PYTHON_EXE}" >&2
    echo "Unset WM_BENCH_INSTALL_PYTHON_DEPS or create the venv manually." >&2
    exit 1
  fi
  echo "Python dependency install skipped: WM_BENCH_INSTALL_PYTHON_DEPS=0"
  exit 0
fi

if [[ ! -x "${PYTHON_EXE}" ]]; then
  venv_args=()
  if [[ "${WM_BENCH_VENV_SYSTEM_SITE_PACKAGES:-0}" != "0" ]]; then
    venv_args+=(--system-site-packages)
  fi
  "${PYTHON_BIN}" -m venv "${venv_args[@]}" "${VENV_DIR}"
fi

"${PYTHON_EXE}" -m pip install --upgrade pip setuptools wheel

if [[ "${WM_BENCH_INSTALL_SHARP_DEPS:-1}" != "0" ]]; then
  "${PYTHON_EXE}" -m pip install -r requirements.txt -r requirements/sharp.txt
else
  "${PYTHON_EXE}" -m pip install -r requirements.txt
fi

echo "Python environment is ready: ${PYTHON_EXE}"
