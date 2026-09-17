#!/usr/bin/env bash
# Wrapper so the monitor always runs in its own venv.
#
# wandb is NOT installed in the miner venv on purpose: bittensor==10.3.2 pins
# bittensor-wallet==4.0.1 and a pile of other exact versions, and resolving
# wandb into that set risks moving a pin under a running miner.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# -u (unbuffered): with --watch the output is a long-lived stream, and Python
# block-buffers stdout whenever it is piped or redirected -- so `monitor.sh
# --watch 300 | tee log` showed nothing for minutes and looked hung.
exec "$(dirname "$HERE")/.venv-monitor/bin/python" -u "$HERE/monitor.py" "$@"
