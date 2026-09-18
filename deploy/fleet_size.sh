#!/usr/bin/env bash
# Our fleet size as the dispatcher and miners compute it: hotkeys registered on
# netuid 83 under the coldkeys in $SN83_COLDKEYS_FILE, per the current snapshot.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/sn83.env"
cd "$REPO"
exec "$PY" -c '
import sys; sys.path.insert(0, "research_manual")
import pick_derived as pd
keys = pd.our_coldkeys()
meta = pd._load_metagraph()
by = {}
for m in meta["miners"]:
    if m["coldkey"] in keys:
        by[m["coldkey"]] = by.get(m["coldkey"], 0) + 1
for k in sorted(keys):
    print("  %s  %d hotkeys" % (k, by.get(k, 0)))
print("fleet_n=%d  (%d coldkeys, metagraph block %d, SN83_FLEET_N=%s)" % (
    pd.resolve_fleet_n(), len(keys), meta["block"], __import__("os").environ.get("SN83_FLEET_N")))
'
