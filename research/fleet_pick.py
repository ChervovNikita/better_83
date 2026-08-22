#!/usr/bin/env python3
"""Pick one clique per hotkey so a fleet never self-collides.

Measured gain: coordinating N hotkeys over distinct cliques lifts sole-holder rate
from 24.8% (every hotkey submits the natural endpoint) to 34.3%. It is the largest
single improvement found in this project and needs no solver change.

Two schemes, and the difference is not cosmetic:

  hash   AGENT.md's `hash(hotkey || uuid)`. Each miner picks independently, which is
         correct when you do NOT control the other miners. But N independent hashes
         into a pool of P collide by the birthday problem: at N=9, P=18.7 the expected
         number of DISTINCT picks is only ~7.1, so ~2 hotkeys duplicate and each pays
         1/2 diversity for nothing.

  rank   the fleet owner assigns hotkey i the index i. Distinct by construction, zero
         self-collision -- which is exactly what the 93-UID and 46-UID operators are
         measured doing (12.1 answers -> 12.1 distinct cliques, 0 repeats in 2778
         rounds).

Use `rank` when you own the hotkeys. `hash` is the fallback for an uncoordinated
miner, and it is strictly worse.
"""
import hashlib


def pick(pool, uuid, hotkey, rank=None, fleet_size=None):
    """Return this hotkey's clique from `pool` (a list of distinct cliques)."""
    if not pool:
        return []
    if rank is not None:
        # Coordinated: distinct by construction. Rotate by the round so the same
        # hotkey does not always take the same slot -- otherwise hotkey 0 eats every
        # crowded natural endpoint and its score alone carries the whole penalty.
        off = int(hashlib.sha1(str(uuid).encode()).hexdigest()[:8], 16)
        return pool[(rank + off) % len(pool)]
    h = hashlib.sha1(f"{hotkey}|{uuid}".encode()).hexdigest()
    return pool[int(h[:8], 16) % len(pool)]


def assign(pool, uuid, hotkeys):
    """Whole-fleet assignment: hotkey i gets a distinct clique while the pool lasts."""
    return {hk: pick(pool, uuid, hk, rank=i) for i, hk in enumerate(hotkeys)}
