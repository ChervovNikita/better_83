#!/usr/bin/env python3
"""Snapshot the netuid-83 miner set for pick_derived's field model.

Two modes, one output format:

    --live                      the CURRENT block on finney. What a running
                                miner needs: the picker models the field from
                                who is registered NOW.
    (default, replay)           the block at the first round in rounds.json,
                                read off the archive node. What simulate.py
                                needs, so a replay scores against the field
                                that actually answered those rounds.

`miners` is sorted weakest-first by (incentive, uid) in BOTH modes. That order
is not cosmetic: pick_derived._churn_order slices the head of this list to
decide which hotkeys our registrations displace, and simulate.py asserts the
ordering outright.
"""

import argparse
import json
import os
import sys
import time

import bittensor as bt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths

NETUID = 83
ROUNDS_PATH = paths.ROUNDS_JSON
DEST = paths.METAGRAPH_JSON
BLOCK_MS = 12000
RETRIES = 6
RETRY_S = 5.0


def timestamp_ms(subtensor, block):
    value = subtensor.query_module("Timestamp", "Now", block=block)
    ms = int(value.value)
    assert ms > 0, block
    return ms


def block_at_or_before(subtensor, target_ms):
    hi = int(subtensor.block)
    hi_ms = timestamp_ms(subtensor, hi)
    assert hi_ms >= target_ms, (hi_ms, target_ms)
    lo = hi - (hi_ms - target_ms) // BLOCK_MS - 8
    assert lo > 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if timestamp_ms(subtensor, mid) <= target_ms:
            lo = mid
        else:
            hi = mid - 1
    assert timestamp_ms(subtensor, lo) <= target_ms
    return lo


def miner_rows(mg):
    """The non-validator entries, weakest first."""
    miners = []
    for uid in range(int(mg.n)):
        if float(mg.validator_trust[uid]) != 0.0:
            continue
        miners.append({
            "uid": int(uid),
            "hotkey": mg.hotkeys[uid],
            "coldkey": mg.coldkeys[uid],
            "incentive": float(mg.incentive[uid]),
            "block_at_registration": int(mg.block_at_registration[uid]),
        })
    assert miners
    miners.sort(key=lambda miner: (miner["incentive"], miner["uid"]))
    return miners


def fetch(network, netuid, block=None):
    """The metagraph, retried: the public endpoints drop connections."""
    last = None
    for attempt in range(RETRIES):
        try:
            subtensor = bt.Subtensor(network=network)
            if block is None:
                return subtensor, subtensor.metagraph(netuid=netuid, lite=True)
            mg = subtensor.metagraph(netuid=netuid, lite=True, block=block)
            assert int(mg.block) == block
            return subtensor, mg
        except Exception as exc:                       # noqa: BLE001 - retried
            last = exc
            print("  retry %d: %r" % (attempt, exc), file=sys.stderr)
            time.sleep(RETRY_S)
    raise SystemExit("chain unreachable: %r" % (last,))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="snapshot the current block instead of replaying")
    parser.add_argument("--netuid", type=int, default=NETUID)
    parser.add_argument("--network", default=None,
                        help="default: finney with --live, archive without")
    parser.add_argument("--out", default=DEST)
    args = parser.parse_args()

    network = args.network or ("finney" if args.live else "archive")

    if args.live:
        subtensor, mg = fetch(network, args.netuid)
        block = int(mg.block)
        start_ts = time.time()
    else:
        with open(ROUNDS_PATH) as handle:
            rounds = json.load(handle)
        assert rounds
        start_ts = min(rec["timestamp"] for rec in rounds.values())
        subtensor, _ = fetch(network, args.netuid, block=None)
        block = block_at_or_before(subtensor, int(start_ts * 1000))
        _, mg = fetch(network, args.netuid, block=block)

    miners = miner_rows(mg)
    payload = {
        "netuid": args.netuid,
        "block": block,
        "n": int(mg.n),
        "timestamp": start_ts,
        "miners": miners,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=1)
    # Atomic: a miner reads this file at picker time, and a half-written
    # snapshot is a crash inside the request path.
    os.replace(tmp, args.out)
    print(args.out, block, start_ts, len(miners), file=sys.stderr)


if __name__ == "__main__":
    main()
