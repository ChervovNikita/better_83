#!/usr/bin/env bash
# Provision the SN83 data pipeline on a fresh box and put it on a 5-minute cron.
#
#   WANDB_API_KEY=xxxx bash research/setup_server.sh
#
# Idempotent: safe to re-run after a git pull. The W&B key is read from the
# environment and written to ~/.netrc — it is never stored in the repo.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESEARCH="$REPO/research"
DATA="$RESEARCH/data"
PY="$(command -v python3)"

echo "==> repo:   $REPO"
echo "==> python: $PY ($($PY --version))"

echo "==> python dependencies"
"$PY" -m pip install --quiet --upgrade "wandb>=0.16" "numpy>=1.24" 2>&1 | tail -2 || true
"$PY" -c "import wandb, numpy; print('    wandb', wandb.__version__, '| numpy', numpy.__version__)"

echo "==> W&B credentials"
if [ -n "${WANDB_API_KEY:-}" ]; then
  umask 077
  # replace any existing api.wandb.ai stanza rather than appending a duplicate
  if [ -f "$HOME/.netrc" ]; then
    "$PY" - "$HOME/.netrc" <<'EOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
s = re.sub(r"machine api\.wandb\.ai\s+login \S+\s+password \S+\s*", "", s)
open(p, "w").write(s)
EOF
  fi
  printf 'machine api.wandb.ai\n  login user\n  password %s\n' "$WANDB_API_KEY" >> "$HOME/.netrc"
  chmod 600 "$HOME/.netrc"
  echo "    wrote ~/.netrc (0600)"
elif grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
  echo "    ~/.netrc already has api.wandb.ai, leaving it alone"
else
  echo "    !! no WANDB_API_KEY and no ~/.netrc entry — fetches will fail" >&2
fi

mkdir -p "$DATA"

echo "==> cron daemon"
if ! command -v crontab >/dev/null 2>&1; then
  echo "    installing cron"
  (apt-get update -qq && apt-get install -y -qq cron) >/dev/null 2>&1
fi
# containers have no init, so start the daemon directly if it isn't running
if ! pgrep -x cron >/dev/null 2>&1 && ! pgrep -x crond >/dev/null 2>&1; then
  (service cron start >/dev/null 2>&1) || (/usr/sbin/cron) || true
fi
pgrep -x cron >/dev/null 2>&1 && echo "    cron is running (pid $(pgrep -x cron | head -1))" \
                              || echo "    !! cron did not start" >&2

echo "==> crontab entry (every 5 minutes)"
MARK="# sn83-fetch"
CMD="*/5 * * * * cd $RESEARCH && $PY fetch_new.py --versions 0.0.17 >> $DATA/fetch.log 2>&1 $MARK"
( crontab -l 2>/dev/null | grep -v "$MARK" || true; echo "$CMD" ) | crontab -
crontab -l | grep "$MARK"

echo "==> first fetch (backfill 500 steps per run)"
cd "$RESEARCH"
"$PY" fetch_new.py --versions 0.0.17 --backfill 500 2>&1 | tail -8

echo "==> status"
"$PY" status.py || true

cat <<EOF

Done. The job appends to $DATA/v0.0.17/YYYY-MM-DD.jsonl every 5 minutes.
  health:  cd $RESEARCH && python3 status.py
  log:     tail -f $DATA/fetch.log
  backfill more: python3 build_dataset.py --versions 0.0.17 --limit 0 --out bench.jsonl
EOF
