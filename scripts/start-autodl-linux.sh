#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f .env.autodl ]]; then
  cp .env.autodl.example .env.autodl
fi

if ! command -v screen >/dev/null 2>&1; then
  echo "screen is required on AutoDL. Install it first or use infra/autodl/start_api.sh and start_worker.sh in two terminals." >&2
  exit 1
fi

bash infra/autodl/setup_env.sh

set -a
source .env.autodl
set +a

API_PORT="${API_PORT:-6006}"

screen -S wmbench-api -X quit >/dev/null 2>&1 || true
screen -S wmbench-worker -X quit >/dev/null 2>&1 || true

bash infra/autodl/start_all_screen.sh

if command -v curl >/dev/null 2>&1; then
  for _ in {1..60}; do
    if curl -fsS "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  if ! curl -fsS "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
    echo "API did not become reachable on http://127.0.0.1:${API_PORT}/health" >&2
    echo "Inspect logs with: screen -r wmbench-api" >&2
    exit 1
  fi
fi

cat <<EOF
WM Bench AutoDL services started.

Server-local URL:
  http://127.0.0.1:${API_PORT}

AutoDL/public access:
  1. In the AutoDL console, expose or tunnel local port ${API_PORT}.
  2. Open the generated public URL in any browser.
  3. If you use SSH tunneling instead:
     ssh -L ${API_PORT}:127.0.0.1:${API_PORT} root@<server-ip>
     Then open http://127.0.0.1:${API_PORT} on your computer.

Service sessions:
  screen -r wmbench-api
  screen -r wmbench-worker
EOF
