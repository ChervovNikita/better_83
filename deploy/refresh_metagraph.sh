#!/usr/bin/env bash
# Re-snapshot the netuid-83 miner set the picker models the field from.
#
# Run it before the first start, and on a timer afterwards: the snapshot decides
# which rivals our registrations displaced and how many hotkeys each operator
# holds, and both drift as miners register and deregister. Hourly is ample --
# immunity alone is 6000 blocks (~20 h).
#
#   crontab -e
#   17 * * * * /home/dev/better_83/deploy/refresh_metagraph.sh >> /home/dev/sn83-metagraph.log 2>&1
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/sn83.env"

cd "$REPO"
# Writes through a .tmp and os.replace, so a miner reading the file mid-refresh
# sees either the old snapshot or the new one, never half of one.
exec "$PY" research_manual/eda/dump_metagraph.py --live --netuid "$NETUID"
