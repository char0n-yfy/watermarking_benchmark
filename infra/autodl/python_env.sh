#!/usr/bin/env bash

autodl_prepare_python_env() {
  AUTODL_VENV_DIR="${WM_BENCH_VENV:-.venv}"
  AUTODL_PYTHON="${AUTODL_VENV_DIR}/bin/python"

  if [[ ! -x "${AUTODL_PYTHON}" ]]; then
    local bootstrap_python="${PYTHON:-python}"
    local venv_args=()

    if [[ "${WM_BENCH_VENV_SYSTEM_SITE_PACKAGES:-1}" != "0" ]]; then
      venv_args+=(--system-site-packages)
    fi

    "${bootstrap_python}" -m venv "${venv_args[@]}" "${AUTODL_VENV_DIR}"
  fi
}

autodl_require_python_env() {
  AUTODL_VENV_DIR="${WM_BENCH_VENV:-.venv}"
  AUTODL_PYTHON="${AUTODL_VENV_DIR}/bin/python"

  if [[ ! -x "${AUTODL_PYTHON}" ]]; then
    echo "Missing AutoDL Python environment: ${AUTODL_PYTHON}" >&2
    echo "Run: bash infra/autodl/setup_env.sh" >&2
    exit 1
  fi
}
