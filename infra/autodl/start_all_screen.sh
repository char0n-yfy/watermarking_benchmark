#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

screen -dmS wmbench-api bash infra/autodl/start_api.sh
screen -dmS wmbench-worker bash infra/autodl/start_worker.sh

echo "Started screen sessions:"
screen -ls | grep wmbench || true
