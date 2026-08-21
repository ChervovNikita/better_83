#!/usr/bin/env python3
"""Snapshot the netuid-83 metagraph for fleet_sim: who is displaceable, and when.

    python3 snapshot_metagraph.py [--out data/metagraph.json]

Only miners (validator_trust == 0) are recorded. block_at_registration is what
decides immunity, so it has to come from the chain rather than be inferred.
"""
import argparse, json, os, sys, time
from _common import DATA_DIR

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--netuid", type=int, default=83)
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "metagraph.json"))
    a = ap.parse_args()
    import bittensor as bt
    mg = None
    for attempt in range(6):                    # the public endpoint is flaky
        try:
            mg = bt.Subtensor(network="finney").metagraph(netuid=a.netuid, lite=False)
            break
        except Exception as e:
            print(f"  retry {attempt}: {type(e).__name__}", file=sys.stderr)
            time.sleep(5)
    if mg is None:
        raise SystemExit("chain unreachable")
    miners = [
        {"uid": int(u), "hotkey": mg.hotkeys[u], "coldkey": mg.coldkeys[u],
         "incentive": float(mg.incentive[u]),
         "block_at_registration": int(mg.block_at_registration[u])}
        for u in range(int(mg.n)) if float(mg.validator_trust[u]) == 0.0
    ]
    snap = {"netuid": a.netuid, "block": int(mg.block), "n": int(mg.n), "miners": miners}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(snap, open(a.out, "w"), indent=1)
    print(f"block {snap['block']}: {len(miners)} miners -> {a.out}")

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
