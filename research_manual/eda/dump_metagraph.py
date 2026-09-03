#!/usr/bin/env python3

import json
import os
import sys

import bittensor as bt

NETUID = 83
HERE = os.path.dirname(os.path.abspath(__file__))
ROUNDS_PATH = os.path.join(HERE, os.pardir, "artifacts", "data", "rounds.json")
DEST = os.path.join(HERE, os.pardir, "artifacts", "data", "metagraph.json")
BLOCK_MS = 12000


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


def main():
    with open(ROUNDS_PATH) as handle:
        rounds = json.load(handle)
    assert rounds
    start_ts = min(rec["timestamp"] for rec in rounds.values())
    target_ms = int(start_ts * 1000)
    subtensor = bt.Subtensor(network="archive")
    block = block_at_or_before(subtensor, target_ms)
    mg = subtensor.metagraph(netuid=NETUID, lite=True, block=block)
    assert int(mg.block) == block
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
    payload = {
        "netuid": NETUID,
        "block": block,
        "n": int(mg.n),
        "timestamp": start_ts,
        "miners": miners,
    }
    with open(DEST, "w") as handle:
        json.dump(payload, handle, indent=1)
    print(DEST, block, start_ts, len(miners), file=sys.stderr)


if __name__ == "__main__":
    main()
