import asyncio
import os
import sys
import time
import typing

import bittensor as bt
from CliqueAI.graph.codec import GraphCodec
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph
from common.base.miner import BaseMinerNeuron

# The solver lives in research_manual/, located relative to this file so a
# deployment that ships CliqueAI/ without research_manual/ fails at import --
# loudly, at startup -- rather than answering empty cliques for a whole epoch.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "research_manual"),
           os.path.join(_REPO, "research_manual", "eda")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dispatch_client  # noqa: E402
import fleet_solver  # noqa: E402  (builds libclique.so at import, never in a request)

# Round-trip to the validator, plus the axon's own serialization. The dispatcher
# subtracts the same figure again from what it is told; this one keeps the
# miner's local fallback from starting a solve it cannot finish in time.
LATENCY_S = float(os.environ.get("SN83_MINER_LATENCY_S", "2.0"))

# The dispatcher subtracts its own LATENCY_S from whatever time_limit it is
# given. Kept in step with SN83_LATENCY_S so the miner can work backwards from
# the answer time it actually wants.
DISPATCH_LATENCY_S = float(os.environ.get("SN83_LATENCY_S", "2.0"))

# Never hold the validator's connection longer than this, whatever the deadline
# allows. MEASURED: at tl=30 we answered at 27.0-27.5s and the validator
# recorded nothing on 3 of 4 such rounds -- including cliques of 44 and 76 that
# MATCHED the field's best. The same stack scores reliably at tl=10 and tl=15,
# where the connection is held 7.5s and 12.5s. Something between us and the
# validator does not keep a request alive for ~27s, so the extra harvest time is
# not just wasted, it is destructive.
#
# Time-to-omega is under 10% of budget on every round measured, so capping the
# hold costs pool breadth (fewer distinct cliques for the picker), not the
# clique itself. A narrower pool scores; a late answer scores zero.
MAX_HOLD_S = float(os.environ.get("SN83_MAX_HOLD_S", "18.0"))

# Reserve for the round trip, as a fraction of the deadline, floored at
# LATENCY_S. A flat 2s left only ~2.5s of margin at every deadline, which is
# proportionally far tighter at tl=30 than at tl=7.5.
#
# Raised 0.15 -> 0.25 on 2026-09-09 after two MEASURED losses where the answer
# was correct and simply did not complete the round trip:
#
#   tl=10  hold 8.0s   answered 7.63s (margin 2.37s)  size 82 = field best -> []
#   tl=15  hold 12.8s  answered 12.25s (margin 2.75s) size 25 = field best -> []
#
# The floor stays at 2.0s, so tl=6 and tl=7.5 are UNCHANGED -- neither has lost
# a round, and cutting their hold would trade solve time for margin they do not
# need. Only tl=10 (+0.5s) and tl=15 (+1.5s) move; tl=30 is already bounded by
# MAX_HOLD_S. The cost is harvest breadth, not clique size: time-to-omega is
# under 10% of budget, so a shorter hold narrows the pool the picker chooses
# from rather than lowering the clique we find.
LATENCY_FRAC = float(os.environ.get("SN83_LATENCY_FRAC", "0.25"))

# Fraction of the remaining budget the local fallback may spend. The rest is
# margin: a late answer scores zero on both reward terms, so a smaller clique
# delivered on time strictly dominates a better one delivered late.
FALLBACK_SHARE = 0.8

# Below this there is no time to run the native core at all, so the fallback
# degrades to a greedy maximal clique, which costs microseconds.
FALLBACK_MIN_S = 0.5

DEFAULT_TIMEOUT_S = 15.0


class Miner(BaseMinerNeuron):
    """Answers maximum-clique requests from the shared solve service."""

    def __init__(self, config=None):
        super().__init__(config=config)
        self.axon.attach(
            forward_fn=self.forward_graph,
            blacklist_fn=self.backlist_graph,
            priority_fn=self.priority_graph,
        )
        health = dispatch_client.health()
        if health is None:
            bt.logging.warning(
                f"Dispatcher unreachable at {dispatch_client.URL}; every round "
                f"will fall back to a local CPU solve until it comes up."
            )
        else:
            bt.logging.info(f"Dispatcher: {health}")

    async def forward_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> MaximumCliqueOfLambdaGraph:
        started = time.monotonic()
        timeout = float(getattr(synapse, "timeout", None) or DEFAULT_TIMEOUT_S)
        reserve = max(LATENCY_S, LATENCY_FRAC * timeout)
        # `budget` is when WE intend to answer; `hold` additionally caps how long
        # the connection is held open regardless of how generous the deadline is.
        budget = min(timeout - reserve, MAX_HOLD_S)
        hotkey = self.wallet.hotkey.ss58_address

        clique = None
        if budget > 0:
            # The HTTP deadline is `budget`, not `timeout`: a dispatcher that
            # hangs must not consume the margin the local fallback needs.
            clique = await asyncio.to_thread(
                dispatch_client.solve,
                synapse.uuid,
                hotkey,
                synapse.number_of_nodes,
                None,
                budget + DISPATCH_LATENCY_S,
                encoded_matrix=synapse.encoded_matrix,
                timeout=budget,
            )

        source = "dispatcher"
        if not clique:
            source = "local"
            bt.logging.warning("Dispatcher gave no answer; solving locally")
            left = budget - (time.monotonic() - started)
            clique = await asyncio.to_thread(
                self._solve_locally, synapse.encoded_matrix, left
            )

        synapse.maximum_clique = list(clique or [])
        bt.logging.info(
            f"uuid={synapse.uuid} n={synapse.number_of_nodes} "
            f"tl={timeout:.1f}s hold={budget:.1f}s source={source} "
            f"size={len(synapse.maximum_clique)} "
            f"elapsed={time.monotonic() - started:.2f}s"
        )
        return synapse

    def _solve_locally(self, encoded_matrix, seconds_left):
        """Error handling, not a mode -- nothing selects this path.

        It runs only when the dispatcher is down or rejected the round, and it
        stays on the CPU: the dispatcher's workers own the GPUs, and a second
        CUDA context on a busy device makes both solves miss the deadline.
        """
        try:
            import numpy as np

            matrix = GraphCodec().decode_matrix(encoded_matrix)
            adjacency = np.ascontiguousarray(matrix, dtype=np.uint8)
            if seconds_left >= FALLBACK_MIN_S:
                pool = fleet_solver.solve_many(
                    adjacency, seconds_left * FALLBACK_SHARE, 1)
                if pool and pool[0]:
                    return [int(v) for v in pool[0]]
            # Out of time for a search: extend the highest-degree vertex until
            # nothing can be added. Maximal by construction, so it scores the
            # diversity term even when it is far from omega.
            seed = int(np.argmax(adjacency.sum(axis=1)))
            return [int(v) for v in fleet_solver._extend(adjacency, [seed])]
        except Exception:
            bt.logging.error("Local fallback failed", exc_info=True)
            return []

    async def backlist_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_graph(self, synapse: MaximumCliqueOfLambdaGraph) -> float:
        return await self.priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Miner has started running.")
        while True:
            if miner.should_exit:
                bt.logging.info("Miner is exiting.")
                break
            time.sleep(1)
