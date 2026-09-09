#!/usr/bin/env bash
# The solve service: owns both GPUs, batches siblings per round, admits or
# rejects. Start this BEFORE any miner -- a miner whose dispatcher is down
# answers from its local CPU fallback, which is strictly worse.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/sn83.env"

cd "$REPO"

# The picker reads this at request time; a missing file is a crash inside the
# deadline, not a degraded answer.
if [ ! -s research_manual/artifacts/data/metagraph.json ]; then
    echo "metagraph.json missing -- run deploy/refresh_metagraph.sh first" >&2
    exit 1
fi

# Build both native libraries HERE, not inside a worker's first request: nvcc
# takes ~16s and g++ ~2s, and either one inside a round spends its deadline.
"$PY" - <<'PYEOF'
import sys
sys.path.insert(0, "research_manual")
import gpu_lib, fleet_solver
gpu_lib.load(32, False)
print("libcliquegpu ready; libclique ready (THREADS=%d)" % fleet_solver.THREADS)
PYEOF

pm2 delete sn83-dispatcher >/dev/null 2>&1 || true
pm2 start "$VENV/bin/uvicorn" --name sn83-dispatcher --cwd "$REPO" \
    --interpreter none --time -- \
    research_manual.eda.dispatcher:app \
    --host "$SN83_DISPATCH_HOST" --port "$SN83_DISPATCH_PORT"

echo "waiting for both workers to warm..."
for i in $(seq 1 60); do
    if "$PY" -c "
import json,sys,urllib.request
try:
    h = json.load(urllib.request.urlopen('$SN83_DISPATCH_URL/health', timeout=2))
except Exception:
    sys.exit(1)
sys.exit(0 if h['workers_free'] >= h['workers'] + h['cpu_workers'] else 1)
" 2>/dev/null; then
        "$PY" -c "
import json,urllib.request
h = json.load(urllib.request.urlopen('$SN83_DISPATCH_URL/health', timeout=2))
print(json.dumps(h, indent=2))"
        exit 0
    fi
    sleep 2
done
echo "dispatcher did not become ready; pm2 logs sn83-dispatcher" >&2
exit 1
