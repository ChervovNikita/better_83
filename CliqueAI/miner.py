import asyncio
import os
import sys
import threading
import time
import typing

import bittensor as bt
from CliqueAI.graph.codec import GraphCodec
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph
from common.base.miner import BaseMinerNeuron

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "research_manual"),
           os.path.join(_REPO, "research_manual", "eda")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dispatch_client  # noqa: E402
import fleet_solver  # noqa: E402  (builds libclique.so at import, never in a request)
import pick_derived  # noqa: E402

LATENCY_S = float(os.environ.get("SN83_MINER_LATENCY_S", "2.0"))
DISPATCH_LATENCY_S = float(os.environ.get("SN83_LATENCY_S", "2.0"))
SOLVE_CAP_S = float(os.environ.get("SN83_SOLVE_CAP_S", "2.0"))
# Slack between the dispatcher's solve budget and how long we wait for it. Both
# used to be exactly `budget`, so an answer the dispatcher finished at 2.01 s
# was thrown away for a greedy local clique. Measured at fleet 160: 84 of 3,501
# answers lost that way on realistic traffic. The round's split is 2 s network
# + 2 s solve + < 1 s for everything else, and this spends part of the last.
DISPATCH_GRACE_S = float(os.environ.get("SN83_HTTP_GRACE_S", "0.5"))
MAX_HOLD_S = float(os.environ.get("SN83_MAX_HOLD_S", "18.0"))

RARE_QUERY_P = 0.05
RARE_QUERY_MIN_TL_S = 10.0
GATHER_FINISH_MAX_S = float(os.environ.get("SN83_GATHER_FINISH_MAX_S", "5.0"))
GATHER_POLL_S = 0.05


LATENCY_FRAC = float(os.environ.get("SN83_LATENCY_FRAC", "0.25"))
FALLBACK_SHARE = 0.8
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
        budget = min(timeout - reserve, MAX_HOLD_S, SOLVE_CAP_S)
        hotkey = self.wallet.hotkey.ss58_address

        clique = None
        role = None
        if budget > 0:
            clique, role = await dispatch_client.solve_source_async(
                synapse.uuid,
                hotkey,
                synapse.number_of_nodes,
                None,
                budget + DISPATCH_LATENCY_S,
                encoded_matrix=synapse.encoded_matrix,
                timeout=budget + DISPATCH_GRACE_S,
            )

        source = role or "dispatcher"
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
        if timeout >= RARE_QUERY_MIN_TL_S and role == "owner":
            threading.Thread(
                target=self._gather_watch,
                args=(synapse.uuid, synapse.number_of_nodes,
                      timeout, synapse.encoded_matrix),
                daemon=True,
            ).start()
        return synapse

    def _gather_watch(self, uuid, n_nodes, time_limit, encoded_matrix):
        try:
            claims = self._started_count(uuid)
            if not self._wait_until_finished(uuid, claims):
                return
            difficulty = pick_derived.difficulty_from_n(n_nodes)
            p_sel = pick_derived.selection_p(difficulty)
            fleet = pick_derived.resolve_fleet_n()
            p_tail = pick_derived.p_at_most_queried(claims, fleet, difficulty)
            if time_limit < RARE_QUERY_MIN_TL_S or p_tail >= RARE_QUERY_P:
                return
            bt.logging.warning(
                f"RARE QUERY uuid={uuid} claims={claims}/{fleet} "
                f"p={p_sel:.3f} P(X<=k)={p_tail:.4f} "
                f"D={difficulty} tl={time_limit:.1f}s"
            )
            self._cpu_followup(
                encoded_matrix, uuid, n_nodes, time_limit, claims)
        except Exception:
            bt.logging.error("Gather watch failed", exc_info=True)

    def _started_count(self, uuid):
        n = dispatch_client.started_waiting(uuid)
        return n if n else 1

    def _wait_until_finished(self, uuid, claims):
        """Block until every miner in the snapshot has left /solve."""
        deadline = time.monotonic() + GATHER_FINISH_MAX_S
        while time.monotonic() < deadline:
            info = dispatch_client.task_progress(uuid)
            if info is not None and info["finished"] >= claims:
                return True
            time.sleep(GATHER_POLL_S)
        bt.logging.warning(
            f"gather: {claims} started on {uuid} but not all finished"
        )
        return False

    def _cpu_followup(self, encoded_matrix, uuid, n_nodes, time_limit, claims):
        bt.logging.info(f"low probability event")
        return []

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
