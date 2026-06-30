#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

autodl_load_env

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
  if [[ "${WM_BENCH_VENV_SYSTEM_SITE_PACKAGES:-1}" != "0" ]]; then
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
