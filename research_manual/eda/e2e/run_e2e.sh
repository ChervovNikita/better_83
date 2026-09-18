#!/usr/bin/env bash
# End-to-end run: the production dispatcher + K production miner processes on
# real axons, queried through the validator's own transport.
#
#   K=100 research_manual/eda/e2e/run_e2e.sh          # fleet of 100
#   K=160 RUN=run2 SN83_WORKERS=1 .../run_e2e.sh      # on a one-GPU box
#
# Everything the miners run is production code: CliqueAI.miner.Miner behind a
# bt.Axon, talking to research_manual/eda/dispatcher.py. Only the chain is
# stubbed (see e2e_miner.py). Reduce a finished run with e2e_analyze.py.
#
# Env (all optional): K, RUN, PORT0, OUT, NREAL, NSTRESS, NSCEN, GAPREAL,
# GAPSTRESS, plus any SN83_* the dispatcher reads (SN83_WORKERS defaults to the
# GPU count, so this works on a 1-GPU pod as well as the 4-GPU one).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
cd "$REPO"

PY="${PY:-$REPO/.venv/bin/python}"
UVICORN="${UVICORN:-$REPO/.venv/bin/uvicorn}"
K="${K:-100}"
PORT0="${PORT0:-19000}"
RUN="${RUN:-run1}"
OUT="${OUT:-$REPO/../better_83_e2e}"          # outside the repo: runs are large
R="$OUT/$RUN"
WALLETS="${WALLETS:-$OUT/wallets}"
PORT="${SN83_DISPATCH_PORT:-8899}"

# One dispatcher worker per GPU unless told otherwise.
NGPU="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
[ "$NGPU" -lt 1 ] && NGPU=1
export SN83_BACKEND="${SN83_BACKEND:-gpu}"
export SN83_WORKERS="${SN83_WORKERS:-$NGPU}"
export SN83_CPU_WORKERS="${SN83_CPU_WORKERS:-1}"
export SN83_CPU_BUDGET="${SN83_CPU_BUDGET:-15}"
export SN83_OVERFLOW_THREADS="${SN83_OVERFLOW_THREADS:-1}"
export SN83_SOLVE_K="${SN83_SOLVE_K:-8192}"
export SN83_MINIMAX_N="${SN83_MINIMAX_N:-150}"
# the fleet under test is exactly the K hotkeys this script starts
export SN83_FLEET_N="$K"
export SN83_DISPATCH_URL="http://127.0.0.1:$PORT"

mkdir -p "$R/events" "$R/ready" "$WALLETS"
rm -f "$R"/events/* "$R"/ready/* "$R/validator.jsonl"
echo "=== E2E START $(date +%T) K=$K workers=$SN83_WORKERS out=$R"

"$PY" "$HERE/make_wallets.py" "$WALLETS" "$K" >/dev/null || {
    echo "wallet setup failed" >&2; exit 1; }

"$UVICORN" research_manual.eda.dispatcher:app --host 127.0.0.1 --port "$PORT" \
    > "$R/dispatcher.log" 2>&1 &
echo $! > "$R/dispatcher.pid"
for _ in $(seq 1 90); do
    "$PY" -c "
import json,sys,urllib.request
try: h=json.load(urllib.request.urlopen('http://127.0.0.1:$PORT/health',timeout=2))
except Exception: sys.exit(1)
sys.exit(0 if h['workers_free']>=h['workers']+h['cpu_workers'] else 1)" 2>/dev/null && break
    sleep 2
done
curl -s "127.0.0.1:$PORT/health" > "$R/health_start.json"
echo "=== DISPATCHER READY $(date +%T)"

VAL=$("$PY" -c "
import bittensor as bt
print(bt.Wallet(name='e2e_val', hotkey='default', path='$WALLETS').hotkey.ss58_address)")

: > "$R/miner.pids"
for i in $(seq 0 $((K-1))); do
    "$PY" "$HERE/e2e_miner.py" --index "$i" --port $((PORT0+i)) \
        --wallet-path "$WALLETS" --val-hotkey "$VAL" \
        --events "$R/events/m$i.jsonl" --ready "$R/ready/m$i" \
        > "$R/miner_$i.log" 2>&1 &
    echo $! >> "$R/miner.pids"
    [ $(( (i+1) % 10 )) -eq 0 ] && sleep 4
done
for _ in $(seq 1 300); do
    [ "$(ls "$R/ready" | wc -l)" -ge "$K" ] && break; sleep 2
done
echo "=== MINERS READY $(ls "$R/ready" | wc -l)/$K $(date +%T)"
for i in $(seq 0 $((K-1))); do cat "$R/ready/m$i"; echo; done > "$R/hotkeys.txt"

V=("$PY" "$HERE/e2e_validator.py" --wallet-path "$WALLETS" --ports "$PORT0:$K"
   --hotkeys "$R/hotkeys.txt"
   --rounds "$REPO/research_manual/artifacts/data/rounds.json"
   --out "$R/validator.jsonl")
# realistic: rounds arrive as the field's validators send them.
"${V[@]}" --tag realistic --n-random "${NREAL:-100}" --mean-gap "${GAPREAL:-8}" \
    --scenarios 0 --seed 11 > "$R/val_real.log" 2>&1
echo "=== PHASE realistic rc=$? $(date +%T)"
# stress: rounds overlap several deep, well past what has been observed.
"${V[@]}" --tag stress --n-random "${NSTRESS:-60}" --mean-gap "${GAPSTRESS:-1.5}" \
    --scenarios 0 --seed 12 > "$R/val_stress.log" 2>&1
echo "=== PHASE stress rc=$? $(date +%T)"
# scenarios: the rare-query alert contract (see e2e_validator.scenario_phase).
"${V[@]}" --tag scen --n-random 0 --scenarios "${NSCEN:-3}" --seed 13 \
    > "$R/val_scen.log" 2>&1
echo "=== PHASE scenarios rc=$? $(date +%T)"

sleep 10
curl -s "127.0.0.1:$PORT/health" > "$R/health_end.json"
while read -r p; do kill "$p" 2>/dev/null; done < "$R/miner.pids"
kill "$(cat "$R/dispatcher.pid")" 2>/dev/null
sleep 3
echo "=== E2E COMPLETE $(date +%T)"
echo "reduce with: $PY $HERE/e2e_analyze.py $R"
