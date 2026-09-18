#!/usr/bin/env python3
"""Validator side of the end-to-end run.

Sends rounds the way CliqueAI/validator.py does: a MaximumCliqueOfLambdaGraph
with the real encoded matrix and time limit, signed by a bt.Dendrite, delivered
through CliqueAI.transport.axon_requester.AxonRequester to the sampled axons.

Rounds are scheduled with random (exponential) gaps, so they overlap on the
dispatcher the way rounds from several validators do. The number of our hotkeys
queried per round is Binomial(fleet, selection_p(D)), as MinerSelector draws it.

Scripted scenarios for the rare-query alert run after the random phase.
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time
import uuid as uuid_mod

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))   # <repo>
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research_manual"))

import aiohttp  # noqa: E402
import bittensor as bt  # noqa: E402
import numpy as np  # noqa: E402
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph  # noqa: E402
from CliqueAI.transport.axon_requester import AxonRequester  # noqa: E402
import pick_derived  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--wallet-path", required=True)
ap.add_argument("--ports", required=True, help="first:count")
ap.add_argument("--hotkeys", required=True, help="file: one hotkey per line, port order")
ap.add_argument("--rounds", required=True)
ap.add_argument("--n-random", type=int, default=120)
ap.add_argument("--mean-gap", type=float, default=2.0)
ap.add_argument("--scenarios", type=int, default=1)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--tag", default="random")
ap.add_argument("--out", required=True)
args = ap.parse_args()

rng = random.Random(args.seed)
first, count = (int(x) for x in args.ports.split(":"))
hotkeys = [l.strip() for l in open(args.hotkeys) if l.strip()]
assert len(hotkeys) == count, (len(hotkeys), count)
FLEET = count

val = bt.Wallet(name="e2e_val", hotkey="default", path=args.wallet_path)
dendrite = bt.Dendrite(wallet=val)
axons = [bt.AxonInfo(version=__import__("bittensor.core.settings", fromlist=["x"]).version_as_int, ip="127.0.0.1", port=first + i,
                     ip_type=4, hotkey=hotkeys[i], coldkey=hotkeys[i])
         for i in range(count)]

rounds = json.load(open(args.rounds))
round_ids = sorted(rounds)

OUT = open(args.out, "a")
INFLIGHT = {"n": 0}


def log(**kw):
    OUT.write(json.dumps(kw) + "\n")
    OUT.flush()


class TimedRequester(AxonRequester):
    """Stamps send/receive per axon. Delivery and parsing are the parent's."""

    async def _send_request(self, url, headers, body, timeout):
        t0 = time.monotonic()
        try:
            return await super()._send_request(url, headers, body, timeout)
        finally:
            self.stamps[url] = (t0, time.monotonic())


async def send(tag, round_id, idx, delay_groups=None, fixed_uuid=None):
    """Query axons `idx` with the round's graph. delay_groups: [(delay_s, [idx...])]."""
    rec = rounds[round_id]
    uuid = fixed_uuid or str(uuid_mod.uuid4())
    tl = float(rec["time_limit"])
    synapse = MaximumCliqueOfLambdaGraph(
        uuid=uuid, label="general", number_of_nodes=rec["number_of_nodes"],
        encoded_matrix=rec["encoded_matrix"], timeout=tl)
    groups = delay_groups or [(0.0, idx)]
    depth = INFLIGHT["n"]
    INFLIGHT["n"] += 1
    t_round = time.monotonic()

    async def one_group(delay, members):
        if delay:
            await asyncio.sleep(delay)
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=256)) as session:
            req = TimedRequester(session=session, dendrite=dendrite)
            req.stamps = {}
            sel = [axons[i] for i in members]
            resp = await req.forward(synapse=synapse, axons=sel)
            t_done = time.monotonic()
            for i, ax, r in zip(members, sel, resp):
                url = dendrite._get_endpoint_url(ax, request_name="MaximumCliqueOfLambdaGraph")
                t0, t1 = req.stamps.get(url, (None, None))
                log(tag=tag, uuid=uuid, round_id=round_id, tl=tl, D=rec["difficulty"],
                    n=rec["number_of_nodes"], q=len(idx), depth_at_send=depth, index=i,
                    hotkey=ax.hotkey, delay=delay, t_round=t_round, t_send=t0, t_headers=t1,
                    t_parsed=t_done, status=r.dendrite.status_code,
                    msg=r.dendrite.status_message, clique=list(r.maximum_clique or []))

    try:
        await asyncio.gather(*[one_group(d, m) for d, m in groups])
    finally:
        INFLIGHT["n"] -= 1
    return uuid


async def random_phase():
    tasks = []
    t = 0.0
    start = time.monotonic()
    for k in range(args.n_random):
        rid = rng.choice(round_ids)
        D = rounds[rid]["difficulty"]
        p = pick_derived.selection_p(D)
        q = max(1, int(np.random.default_rng(args.seed * 1000 + k).binomial(FLEET, p)))
        idx = rng.sample(range(FLEET), q)
        t += rng.expovariate(1.0 / args.mean_gap)

        async def fire(at=t, rid=rid, idx=idx, k=k):
            await asyncio.sleep(max(0.0, start + at - time.monotonic()))
            await send(args.tag, rid, idx)
        tasks.append(asyncio.create_task(fire()))
    await asyncio.gather(*tasks)


def pick_round(pred):
    c = [r for r in round_ids if pred(rounds[r])]
    return rng.choice(c)


async def scenario_phase():
    """Deterministic alert cases, spaced so each runs on an idle dispatcher."""
    cases = []
    for rep in range(args.scenarios):
        # rare by construction: one sibling on a big-fleet low-D round, tl >= 10
        cases.append(("S1_rare_tl_ge10_q1", lambda r: r["time_limit"] >= 10 and r["difficulty"] <= 0.8, 1, None))
        # same rarity but tl < 10: the watch must not even start
        cases.append(("S2_rare_tl_lt10_q1", lambda r: r["time_limit"] < 10 and r["difficulty"] <= 0.8, 1, None))
        # typical sibling count on tl >= 10: must not fire
        cases.append(("S3_typical_tl_ge10", lambda r: r["time_limit"] >= 10 and r["difficulty"] == 0.7, "mean", None))
        # rare (3 siblings, D=0.7), all concurrent: fire once, after all three
        cases.append(("S4_rare_q3_concurrent", lambda r: r["time_limit"] >= 10 and r["difficulty"] == 0.7, 3, None))
        # rare, but one sibling's request lands 1.0s after the others (jitter)
        cases.append(("S5_rare_q3_one_late_1.0s", lambda r: r["time_limit"] >= 10 and r["difficulty"] == 0.7, 3, 1.0))
        # one sibling lands after the owner has already answered
        cases.append(("S6_rare_q3_one_late_2.6s", lambda r: r["time_limit"] >= 10 and r["difficulty"] == 0.7, 3, 2.6))
    for name, pred, q, late in cases:
        rid = pick_round(pred)
        D = rounds[rid]["difficulty"]
        if q == "mean":
            q = max(1, int(round(FLEET * pick_derived.selection_p(D))))
        idx = rng.sample(range(FLEET), q)
        groups = None
        if late is not None:
            groups = [(0.0, idx[:-1]), (late, idx[-1:])]
        await send(name, rid, idx, delay_groups=groups)
        await asyncio.sleep(8.0)   # longer than GATHER_FINISH_MAX_S + solve


async def main():
    log(tag="meta", fleet=FLEET, n_random=args.n_random, mean_gap=args.mean_gap,
        seed=args.seed, start_wall=time.time(), start_mono=time.monotonic())
    if args.n_random:
        await random_phase()
        await asyncio.sleep(8.0)
    if args.scenarios:
        await scenario_phase()
    log(tag="end", wall=time.time(), mono=time.monotonic())


asyncio.run(main())
