#!/usr/bin/env bash
# Follow one hotkey's rounds live:   deploy/live.sh <hotkey-ss58> [--backfill N]
# Runs in .venv-monitor so wandb never touches the miner venv's pins.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$(dirname "$HERE")/.venv-monitor/bin/python" -u "$HERE/live.py" "$@"
