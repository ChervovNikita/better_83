#!/usr/bin/env bash
# Every-20-minutes watchdog. Checks the things that have actually broken here,
# repairs the ones that are safe to repair unattended, and shouts about the rest.
#
#   */20 * * * * /home/dev/better_83/deploy/healthcheck.sh >> ~/sn83-health.log 2>&1
#
# Deliberately does NOT touch the wallet, the chain, or any solver constant.
# Auto-repair is limited to restarting a dead process and re-serving an axon
# whose on-chain record has drifted from sn83.env.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/sn83.env"

STATE="$HOME/.sn83-healthcheck.state"
# Counters below are CUMULATIVE since the dispatcher/miner started. Flagging them
# raw means one bad round at 01:27 is still reported as a problem hours later,
# which trains the reader to ignore the output. Compare against the last run and
# flag only what is NEW.
prev() { [ -f "$STATE" ] && grep -E "^$1=" "$STATE" 2>/dev/null | tail -1 | cut -d= -f2 || true; }
save() { printf '%s=%s\n' "$1" "$2" >> "$STATE.new"; }

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] $*"; }
PROBLEMS=0
flag() { say "PROBLEM: $*"; PROBLEMS=$((PROBLEMS + 1)); }

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" >/dev/null 2>&1

say "--- healthcheck ---"

# 1. dispatcher alive, workers warm
HEALTH=$(curl -s --max-time 5 "$SN83_DISPATCH_URL/health" || true)
if [ -z "$HEALTH" ]; then
    flag "dispatcher not answering; restarting"
    "$HERE/start_dispatcher.sh" >/dev/null 2>&1 && say "  dispatcher restarted" \
        || flag "  dispatcher restart FAILED"
else
    echo "$HEALTH" | "$PY" -c '
import json, sys
h = json.load(sys.stdin)
c = h["counters"]
free = h["workers_free"]; want = h["workers"] + h["cpu_workers"]
print("  dispatcher: %d/%d workers free, served=%d late=%d error=%d rejected=%d"
      % (free, want, c["served"], c["late"], c["error"], c["rejected"]))'
    D_LATE=$(echo "$HEALTH" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["counters"]["late"])')
    D_ERR=$(echo "$HEALTH" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["counters"]["error"])')
    P_LATE=$(prev dispatcher_late); P_ERR=$(prev dispatcher_error)
    if [ -n "$P_LATE" ] && [ "$D_LATE" -gt "$P_LATE" ]; then
        flag "$((D_LATE - P_LATE)) NEW late round(s) since the last check"
    elif [ "$D_LATE" -gt 0 ]; then
        say "  dispatcher late rounds to date: $D_LATE (none new)"
    fi
    if [ -n "$P_ERR" ] && [ "$D_ERR" -gt "$P_ERR" ]; then
        flag "$((D_ERR - P_ERR)) NEW dispatcher error(s) since the last check"
    fi
    save dispatcher_late "$D_LATE"
    save dispatcher_error "$D_ERR"
fi

# 2. miner process alive
if ! pm2 describe "sn83-miner-$WALLET_HOTKEY" 2>/dev/null | grep -q "online"; then
    flag "miner not online; restarting"
    "$HERE/start_miner.sh" >/dev/null 2>&1 && say "  miner restarted" \
        || flag "  miner restart FAILED"
else
    say "  miner: online"
fi

# 3. the axon the chain advertises must match what we intend to serve
CHAIN=$("$PY" - <<PYEOF 2>/dev/null
import bittensor as bt, json
try:
    st = bt.Subtensor(network="$SUBTENSOR_NETWORK")
    v = st.query_module("SubtensorModule", "Axons", params=[$NETUID, "$( "$PY" -c "
import json;print(json.load(open('$HOME/.bittensor/wallets/$WALLET_NAME/hotkeys/$WALLET_HOTKEY'))['ss58Address'])" )"])
    v = v.value if hasattr(v, "value") else v
    print("%s:%s" % (bt.utils.networking.int_to_ip(v["ip"]), v["port"]))
except Exception as e:
    print("ERROR")
PYEOF
)
WANT="$AXON_IP:$AXON_EXTERNAL_PORT"
if [ "$CHAIN" = "ERROR" ] || [ -z "$CHAIN" ]; then
    flag "could not read the axon record from chain"
elif [ "$CHAIN" != "$WANT" ]; then
    flag "chain says $CHAIN but sn83.env wants $WANT; restarting miner to re-serve"
    "$HERE/start_miner.sh" >/dev/null 2>&1 && say "  miner restarted to re-serve axon"
else
    say "  axon on chain: $CHAIN (matches sn83.env)"
fi

# 4. the Vast port map is regenerated when the container is recreated
if [ -n "${VAST_TCP_PORT_8000:-}" ] && [ "${VAST_TCP_PORT_8000}" != "$AXON_EXTERNAL_PORT" ]; then
    flag "Vast remapped internal 8000 to ${VAST_TCP_PORT_8000}, but sn83.env still says $AXON_EXTERNAL_PORT"
fi
if [ -n "${PUBLIC_IPADDR:-}" ] && [ "${PUBLIC_IPADDR}" != "$AXON_IP" ]; then
    flag "PUBLIC_IPADDR is now ${PUBLIC_IPADDR}, but sn83.env still says $AXON_IP"
fi

# 5. are requests actually arriving? this is the symptom that matters
LOG="$HOME/.pm2/logs/sn83-miner-${WALLET_HOTKEY//_/-}-out.log"
if [ -f "$LOG" ]; then
    RECENT=$(grep -a "source=" "$LOG" | tail -400 | \
             awk -v cutoff="$(date -u -d '25 minutes ago' +%Y-%m-%dT%H:%M:%S)" \
                 '{split($0,a,":"); if (substr($0,1,19) >= cutoff) n++} END {print n+0}')
    say "  requests handled in the last 25 min: $RECENT"
    [ "$RECENT" -eq 0 ] && flag "no validator requests arrived in 25 minutes"
    LOCAL=$(grep -ac "source=local" "$LOG" || true)
    P_LOCAL=$(prev local_fallbacks)
    if [ -n "$P_LOCAL" ] && [ "$LOCAL" -gt "$P_LOCAL" ]; then
        flag "$((LOCAL - P_LOCAL)) NEW local CPU fallback(s) since the last check"
    elif [ "$LOCAL" -gt 0 ]; then
        say "  local CPU fallbacks to date: $LOCAL (none new)"
    fi
    save local_fallbacks "$LOCAL"
fi

# 6. external reachability of the published endpoint
# TCP connect only -- deliberately NOT an HTTP request. An HTTP probe reaches
# the axon with no synapse name and makes it log
#     ERROR | UnknownSynapseError: Synapse name '' not found
# once per run, which pollutes the very log used to tell an arrived round from a
# missed one. A successful connect is all this check needs.
if timeout 5 bash -c ">/dev/tcp/$AXON_IP/$AXON_EXTERNAL_PORT" 2>/dev/null; then
    say "  endpoint $AXON_IP:$AXON_EXTERNAL_PORT accepts TCP"
else
    # Inconclusive from here: NAT hairpinning fails even when inbound works,
    # so this is reported and never acted on.
    say "  endpoint $AXON_IP:$AXON_EXTERNAL_PORT refused from THIS box (inconclusive: NAT hairpin)"
fi

# 7. metagraph snapshot freshness -- the picker reads it at request time
SNAP="$REPO/research_manual/artifacts/data/metagraph.json"
if [ -f "$SNAP" ]; then
    AGE=$(( ($(date +%s) - $(stat -c %Y "$SNAP")) / 60 ))
    say "  metagraph snapshot age: ${AGE} min"
    [ "$AGE" -gt 180 ] && { flag "metagraph snapshot is stale; refreshing"; \
        "$HERE/refresh_metagraph.sh" >/dev/null 2>&1 && say "  refreshed"; }
else
    flag "metagraph snapshot missing; refreshing"
    "$HERE/refresh_metagraph.sh" >/dev/null 2>&1
fi

[ -f "$STATE.new" ] && mv "$STATE.new" "$STATE"
say "--- $PROBLEMS problem(s) ---"
exit 0
