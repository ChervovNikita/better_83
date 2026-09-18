#!/usr/bin/env bash
# One miner process per hotkey, each on its own forwarded port.
#
#   deploy/start_fleet.sh                 # every hotkey in the wallet
#   deploy/start_fleet.sh hk001 hk002     # only these
#   DRY=1 deploy/start_fleet.sh           # print the port plan, start nothing
#
# Hotkeys CANNOT share a port: bittensor verifies the validator's signature
# against the hotkey that axon serves (axon.py default_verify), so a request
# for hotkey B arriving on A's port is rejected. Each hotkey therefore gets its
# own internal port, published as the external number this host maps it to.
#
# Start the dispatcher FIRST (deploy/start_dispatcher.sh): a miner whose
# dispatcher is down answers from its local CPU fallback on every round.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/sn83.env"
cd "$REPO"

# Internal ports handed to miners. Must be forwarded by the host, and must
# exclude anything already in use -- 22 (ssh) and 80 (taken) on this box.
PORT_LO="${FLEET_PORT_LO:-100}"
PORT_HI="${FLEET_PORT_HI:-249}"
RESERVED="${FLEET_RESERVED:-22 80}"

HOTKEY_DIR="$HOME/.bittensor/wallets/$WALLET_NAME/hotkeys"
[ -d "$HOTKEY_DIR" ] || { echo "no wallet at $HOTKEY_DIR" >&2; exit 1; }

if [ $# -gt 0 ]; then
    HOTKEYS=("$@")
else
    mapfile -t HOTKEYS < <(ls -1 "$HOTKEY_DIR" | grep -v 'pub\.txt$' | sort)
fi
[ "${#HOTKEYS[@]}" -gt 0 ] || { echo "no hotkeys found in $HOTKEY_DIR" >&2; exit 1; }

# Ports that are forwarded AND not reserved. A port with no VAST_TCP_PORT_<n>
# in the environment is not reachable from outside, and publishing it on chain
# is silently fatal -- is_serving reads true and every round scores zero.
PORTS=()
for p in $(seq "$PORT_LO" "$PORT_HI"); do
    skip=0
    for r in $RESERVED; do [ "$p" = "$r" ] && skip=1; done
    [ "$skip" = 1 ] && continue
    ext="$(printenv "VAST_TCP_PORT_$p" || true)"
    [ -n "$ext" ] && PORTS+=("$p:$ext")
done

if [ "${#PORTS[@]}" -lt "${#HOTKEYS[@]}" ]; then
    echo "only ${#PORTS[@]} usable ports in $PORT_LO-$PORT_HI for ${#HOTKEYS[@]} hotkeys." >&2
    echo "widen FLEET_PORT_LO/FLEET_PORT_HI, or host the rest elsewhere." >&2
    exit 1
fi

echo "${#HOTKEYS[@]} hotkeys, ${#PORTS[@]} usable ports (reserved: $RESERVED), ip $AXON_IP"
if [ "${DRY:-0}" = "1" ]; then
    for i in "${!HOTKEYS[@]}"; do
        echo "  ${HOTKEYS[$i]}  internal ${PORTS[$i]%%:*} -> external ${PORTS[$i]##*:}"
    done
    exit 0
fi

# The dispatcher is not required to start, but say so loudly: without it every
# round falls back to a local CPU solve.
"$PY" -c "
import sys; sys.path.insert(0, 'research_manual/eda')
import dispatch_client
sys.exit(0 if dispatch_client.health() else 1)" \
    || echo "WARNING: dispatcher not answering at $SN83_DISPATCH_URL" >&2

for i in "${!HOTKEYS[@]}"; do
    hk="${HOTKEYS[$i]}"
    WALLET_HOTKEY="$hk" \
    AXON_PORT="${PORTS[$i]%%:*}" \
    AXON_EXTERNAL_PORT="${PORTS[$i]##*:}" \
        "$HERE/start_miner.sh" >/dev/null 2>&1 \
        && echo "started $hk on ${PORTS[$i]}" \
        || echo "FAILED $hk on ${PORTS[$i]}" >&2
done

echo
echo "pm2 status | grep sn83-miner    # what is running"
echo "deploy/monitor.sh               # what the validators scored"
