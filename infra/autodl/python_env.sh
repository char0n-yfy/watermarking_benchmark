#!/usr/bin/env bash

autodl_configure_python_env() {
  AUTODL_VENV_DIR="${WM_BENCH_VENV:-.venv}"

  if [[ "${AUTODL_VENV_DIR}" != /* ]]; then
    AUTODL_VENV_DIR="${AUTODL_REPO_ROOT:-$(pwd)}/${AUTODL_VENV_DIR}"
  fi

  AUTODL_PYTHON="${AUTODL_VENV_DIR}/bin/python"
  export AUTODL_VENV_DIR AUTODL_PYTHON

  case ":${PATH}:" in
    *":${AUTODL_VENV_DIR}/bin:"*) ;;
    *) export PATH="${AUTODL_VENV_DIR}/bin:${PATH}" ;;
  esac
}

autodl_prepare_python_env() {
  autodl_configure_python_env

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
  autodl_configure_python_env

  if [[ ! -x "${AUTODL_PYTHON}" ]]; then
    echo "Missing AutoDL Python environment: ${AUTODL_PYTHON}" >&2
    echo "Run: bash infra/autodl/start.sh" >&2
    exit 1
  fi
}
