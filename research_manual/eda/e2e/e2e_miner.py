#!/usr/bin/env python3
"""One production miner process, off-chain.

Runs CliqueAI.miner.Miner UNCHANGED -- its __init__, forward_graph, gather watch,
local fallback, blacklist and priority -- behind a real bt.Axon on 127.0.0.1.
The only thing replaced is the chain: BaseNeuron.__init__ would connect to finney,
fetch the metagraph and exit if the hotkey is unregistered. Here that constructor
builds the same config/wallet/axon without a subtensor, and the metagraph is a
one-entry stand-in holding the test validator's hotkey with a validator permit,
so the production blacklist and priority code run on it as-is.

Instrumentation is by wrapping (never editing) and writes one JSONL per process.
"""
import argparse
import asyncio
import json
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))   # <repo>
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--index", type=int, required=True)
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--wallet-path", required=True)
ap.add_argument("--val-hotkey", required=True)
ap.add_argument("--events", required=True)
ap.add_argument("--ready", required=True)
args = ap.parse_args()

# production config parsing, with the same flags start_miner.sh passes
sys.argv = ["CliqueAI.miner", "--netuid", "83", "--wallet.name", "e2e_miner",
            "--wallet.hotkey", "hk%03d" % args.index, "--wallet.path", args.wallet_path,
            "--axon.port", str(args.port), "--axon.ip", "127.0.0.1",
            "--axon.external_ip", "127.0.0.1", "--axon.external_port", str(args.port),
            "--neuron.autoupdate", "0", "--logging.info",
            ]

import bittensor as bt  # noqa: E402
import common.base.miner as base_miner  # noqa: E402
import common.base.neuron as base_neuron  # noqa: E402
import CliqueAI.miner as prod  # noqa: E402
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph  # noqa: E402

_lock = threading.Lock()


def emit(kind, **kw):
    kw.update(kind=kind, mono=time.monotonic(), wall=time.time(), index=args.index)
    line = json.dumps(kw) + "\n"
    with _lock, open(args.events, "a") as h:
        h.write(line)


class _Meta(object):
    def __init__(self, val_hotkey):
        self.hotkeys = [val_hotkey]
        self.validator_permit = [True]
        self.S = [1.0]


def _offchain_neuron_init(self, config=None):
    """BaseNeuron.__init__ minus subtensor/metagraph/check_registered."""
    import copy
    self.config = copy.deepcopy(config) if config is not None else self.config()
    self.check_config(self.config)
    bt.logging.set_config(config=self.config.logging)
    self.device = self.config.neuron.device
    self.wallet = bt.Wallet(config=self.config)
    self.subtensor = None
    self.metagraph = _Meta(args.val_hotkey)
    self.uid = args.index
    self.last_set_weight = 0
    self.step = 0
    self.init_step = 0


base_neuron.BaseNeuron.__init__ = _offchain_neuron_init

# ---- instrumentation: wrap the module functions the production code calls ----
_solve_source = prod.dispatch_client.solve_source


def solve_source(uuid, hotkey, *a, **kw):
    t0 = time.monotonic()
    clique, role = _solve_source(uuid, hotkey, *a, **kw)
    emit("dispatch", uuid=uuid, hotkey=hotkey, t0=t0, t1=time.monotonic(),
         http_timeout=kw.get("timeout"), role=role, size=len(clique or []))
    return clique, role


prod.dispatch_client.solve_source = solve_source

_solve_source_async = getattr(prod.dispatch_client, "solve_source_async", None)


async def solve_source_async(uuid, hotkey, *a, **kw):
    t0 = time.monotonic()
    clique, role = await _solve_source_async(uuid, hotkey, *a, **kw)
    emit("dispatch", uuid=uuid, hotkey=hotkey, t0=t0, t1=time.monotonic(),
         http_timeout=kw.get("timeout"), role=role, size=len(clique or []))
    return clique, role


if _solve_source_async is not None:
    prod.dispatch_client.solve_source_async = solve_source_async


class E2EMiner(prod.Miner):
    async def forward_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> MaximumCliqueOfLambdaGraph:
        t0 = time.monotonic()
        emit("recv", uuid=synapse.uuid, tl=float(synapse.timeout or 0))
        out = await super().forward_graph(synapse)
        emit("forward_done", uuid=synapse.uuid, t0=t0, t1=time.monotonic(),
             size=len(out.maximum_clique))
        return out

    def _gather_watch(self, uuid, n_nodes, time_limit, encoded_matrix):
        emit("watch_start", uuid=uuid, tl=time_limit, n=n_nodes)
        try:
            return super()._gather_watch(uuid, n_nodes, time_limit, encoded_matrix)
        finally:
            emit("watch_end", uuid=uuid)

    def _started_count(self, uuid):
        n = super()._started_count(uuid)
        emit("claims_snapshot", uuid=uuid, claims=n)
        return n

    def _wait_until_finished(self, uuid, claims):
        ok = super()._wait_until_finished(uuid, claims)
        emit("wait_result", uuid=uuid, claims=claims, ok=ok,
             task=prod.dispatch_client.task_progress(uuid))
        return ok

    def _cpu_followup(self, encoded_matrix, uuid, n_nodes, time_limit, claims):
        # fresh dispatcher view at the instant the alert body runs
        emit("ALERT", uuid=uuid, claims=claims, tl=time_limit, n=n_nodes,
             task=prod.dispatch_client.task_progress(uuid))
        return super()._cpu_followup(encoded_matrix, uuid, n_nodes, time_limit, claims)

    def _solve_locally(self, encoded_matrix, seconds_left):
        t0 = time.monotonic()
        out = super()._solve_locally(encoded_matrix, seconds_left)
        emit("local_fallback", seconds_left=seconds_left, t0=t0, t1=time.monotonic(),
             size=len(out or []))
        return out


wallet_dir = os.path.join(args.wallet_path, "e2e_miner", "hotkeys")
assert os.path.isdir(wallet_dir), wallet_dir

miner = E2EMiner()          # production Miner.__init__: attach + dispatcher health
miner.axon.start()
emit("started", port=args.port, hotkey=miner.wallet.hotkey.ss58_address,
     fleet_n=prod.pick_derived.resolve_fleet_n(), solve_cap=prod.SOLVE_CAP_S)
with open(args.ready, "w") as h:
    h.write(miner.wallet.hotkey.ss58_address)
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
