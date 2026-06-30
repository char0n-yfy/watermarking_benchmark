#!/usr/bin/env bash
set -euo pipefail

screen -S wmbench-api -X quit >/dev/null 2>&1 || true
screen -S wmbench-worker -X quit >/dev/null 2>&1 || true

echo "WM Bench AutoDL services stopped."
echo
echo "Remaining wmbench screen sessions:"
screen -ls | grep wmbench || echo "  none"
