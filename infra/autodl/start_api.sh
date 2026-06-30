#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
autodl_load_env

source infra/autodl/python_env.sh
autodl_require_python_env

autodl_prepare_dirs

"${AUTODL_PYTHON}" -m uvicorn app.main:app --app-dir apps/api --host "${API_HOST}" --port "${API_PORT}"
