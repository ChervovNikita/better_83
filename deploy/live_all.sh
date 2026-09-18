#!/usr/bin/env bash
# Every miner of every round, live: one block per validator push, one row per
# miner it scored.   deploy/live_all.sh [--backfill N] [--validator PREFIX]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$(dirname "$HERE")/.venv-monitor/bin/python" -u "$HERE/live.py" --all "$@"
