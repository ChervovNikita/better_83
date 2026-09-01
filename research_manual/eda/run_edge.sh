#!/usr/bin/env bash
# Measure our_median - field_median for a given fleet size, with a fresh GPU solve.
#
#   research_manual/eda/run_edge.sh <N> [rounds] [--cache] [--only <file>]
#
# The picker chooses its own algorithm from N (see pick_derived.regime):
#   non-dominant operators > 10% of the field  -> "multi"   -> delegate to pick_value
#   one dominant rival                          -> "duel"    -> exact J best response
#   N >= SN83_MINIMAX_N (150)                   -> "minimax" -> mirror, J >= 0 guaranteed
#
# --cache reuses/creates research_manual/cache_n<N>_<tag>.jsonl so repeat runs skip
# the GPU. WITHOUT it every round is solved fresh, which is the honest end-to-end
# number and costs sum(time_limit - 2s) ~= 20 min per 100 rounds.
set -euo pipefail
cd "$(dirname "$0")/../.."

N="${1:?usage: run_edge.sh <N> [rounds] [--cache] [--only <file>]}"
ROUNDS="${2:-100}"
shift $(( $# > 1 ? 2 : 1 ))
CACHE=""; ONLY=""; TAG="first${ROUNDS}"
while [ $# -gt 0 ]; do
  case "$1" in
    --cache) CACHE="research_manual/cache_n${N}_${TAG}.jsonl"; shift ;;
    --only)  ONLY="--only $2"; TAG="$(basename "$2" .txt)"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
OUT_DIR="${SN83_OUT_DIR:-$HOME/autoresearch-runs/sn83-picker}"
mkdir -p "$OUT_DIR"

export SN83_GPU_MAX_OUT=8192      # take the device's whole result table
export SN83_GPU_CLOSURE=1         # one-swap closure to fixpoint
export SN83_FLEET_N="$N"          # lets the picker pick its own regime
[ -n "$CACHE" ] && export SN83_POOL_CACHE="$CACHE"

echo "== N=$N  rounds=$ROUNDS  regime: $(
  .venv/bin/python -c "
import sys; sys.path.insert(0,'research_manual/eda')
import pick_derived as P
pr=P.fleet_profile($N); f=sum(pr.values()); o=f-max(pr.values())
print('%s   (field %d, non-dominant %.1f%%)'%(P.regime($N),f,100*o/f))")"

for spec in "unified:pick_derived:picker_unified" "baseline:value"; do
  name="${spec%%:*}"; picker="${spec#*:}"
  SN83_PICKER="$picker" .venv/bin/python research_manual/simulate.py \
      -N "$N" --rounds "$ROUNDS" $ONLY \
      --out "$OUT_DIR/edge-N${N}-${name}.json" >/dev/null 2>&1
  printf '   %-9s ' "$name"
  .venv/bin/python research_manual/eda/metric_edge.py \
      "$OUT_DIR/edge-N${N}-${name}.json" --json
done
