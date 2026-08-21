#!/usr/bin/env bash
# Promote a winning variant to the shipped solver.
#
#   bash research/promote.sh v7_fastscan
#
# The deliverable is native/clique.cpp, loaded by fastsolver.py and therefore by
# score_submission.py. Variants live in native/variants/ and are never shipped from
# there, so promotion is an explicit, reviewable copy rather than a symlink.
set -euo pipefail

VARIANT=${1:?usage: promote.sh <variant-name>}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/native/variants/$VARIANT.cpp"
DST="$HERE/native/clique.cpp"

[ -f "$SRC" ] || { echo "no such variant: $SRC" >&2; exit 1; }

echo "==> backing up the current solver"
cp "$DST" "$HERE/native/clique.cpp.prev"

echo "==> promoting $VARIANT -> native/clique.cpp"
{
  echo "// PROMOTED FROM native/variants/$VARIANT.cpp"
  echo "// Do not edit here — edit the variant and re-run research/promote.sh."
  cat "$SRC"
} > "$DST"

echo "==> rebuilding"
bash "$HERE/native/build.sh"

echo "==> self-test (random graphs, validity + maximality)"
python3 "$HERE/fastsolver.py"

echo "==> regression: 40 held-out tasks at the honest 8-thread config"
python3 "$HERE/bench.py" --variant champion --set recent_val --limit 40 \
  --threads 8 --workers 1 --seed 0 2>&1 | grep -E "parity|REWARD|invalid"

cat <<EOF

Promoted $VARIANT. Previous solver saved at native/clique.cpp.prev.

Before shipping, run the full honest pass (about 106 minutes):
  python3 research/bench.py --set recent_val --threads 8 --workers 1
and the never-tuned check:
  python3 research/bench.py --set gauntlet --limit 300 --threads 8 --workers 1
EOF
