#!/usr/bin/env bash
# One miner process per hotkey. Every extra argument is passed through to
# CliqueAI.miner, so anything in sn83.env can be overridden on the command line.
#
#   deploy/start_miner.sh                          # uses sn83.env
#   WALLET_HOTKEY=miner2 AXON_PORT=8092 deploy/start_miner.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/sn83.env"

cd "$REPO"

# A second miner on the same hotkey re-serves the axon on chain and silently
# steals traffic from the live one -- both boxes then fight over the record every
# healthcheck. Refuse to start where someone has marked the box retired.
if [ -f "$HERE/MINER_DISABLED" ]; then
    echo "REFUSING TO START: $HERE/MINER_DISABLED exists." >&2
    echo >&2
    sed 's/^/  /' "$HERE/MINER_DISABLED" >&2
    exit 1
fi

if [ -z "${AXON_IP:-}" ]; then
    echo "AXON_IP is empty. Set it in deploy/sn83.env to this box's PUBLIC ip:" >&2
    echo "  the validator dials whatever is published on chain, and a private" >&2
    echo "  address there means every round times out and scores zero." >&2
    exit 1
fi

# AXON_IP is PUBLISHED, never bound. This box is behind NAT (public
# 174.136.205.7, interface 10.0.2.15), so binding the public address fails
# outright -- the axon listens on [::] and advertises AXON_IP on chain.
if ip -4 -o addr show 2>/dev/null | grep -qw "$AXON_IP"; then
    echo "note: $AXON_IP is a local address on this box (not NAT'd)."
fi

if [ ! -d "$HOME/.bittensor/wallets/$WALLET_NAME/hotkeys" ]; then
    echo "no wallet at ~/.bittensor/wallets/$WALLET_NAME -- restore it first" >&2
    exit 1
fi

PROCESS_NAME="sn83-miner-$WALLET_HOTKEY"

# The dispatcher is not required to START, but a miner without one answers from
# its local CPU fallback on every round, so say so loudly rather than let it
# look healthy.
if ! "$PY" -c "
import sys; sys.path.insert(0, 'research_manual/eda')
import dispatch_client
sys.exit(0 if dispatch_client.health() else 1)"; then
    echo "WARNING: dispatcher not answering at $SN83_DISPATCH_URL." >&2
    echo "         Start it with deploy/start_dispatcher.sh." >&2
fi

pm2 delete "$PROCESS_NAME" >/dev/null 2>&1 || true
pm2 start "$PY" --name "$PROCESS_NAME" --cwd "$REPO" --interpreter none --time -- \
    -m CliqueAI.miner \
    --netuid "$NETUID" \
    --subtensor.network "$SUBTENSOR_NETWORK" \
    --wallet.name "$WALLET_NAME" \
    --wallet.hotkey "$WALLET_HOTKEY" \
    --axon.port "$AXON_PORT" \
    --axon.external_ip "$AXON_IP" \
    --axon.external_port "${AXON_EXTERNAL_PORT:-$AXON_PORT}" \
    --logging.info \
    --neuron.autoupdate 0 \
    "$@"

pm2 logs "$PROCESS_NAME" --lines 30 --nostream || true
