import asyncio
import time
import typing

import bittensor as bt
from CliqueAI.clique_algorithms import native_algorithm, networkx_algorithm
from CliqueAI.clique_algorithms.native_algorithm_shim import (
    difficulty_from_n,
    solver_seed,
)
from CliqueAI.graph.codec import GraphCodec
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph
from common.base.miner import BaseMinerNeuron


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self, config=None):
        super().__init__(config=config)
        self.axon.attach(
            forward_fn=self.forward_graph,
            blacklist_fn=self.backlist_graph,
            priority_fn=self.priority_graph,
        )

    async def forward_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> MaximumCliqueOfLambdaGraph:
        codec = GraphCodec()
        adjacency_matrix = codec.decode_matrix(synapse.encoded_matrix)
        adjacency_list = codec.matrix_to_list(adjacency_matrix)
        # The researched champion, with upstream's approximation as the fallback.
        # nx.approximation.max_clique is a greedy heuristic and does not reach omega;
        # every result in research/ was measured against the native solver, which
        # nothing in this file used until now. native_algorithm validates its own
        # answer for validity AND maximality before returning it, and falls back on
        # any failure, so the miner always answers.
        # asyncio.to_thread, not a direct call: this handler is `async def` and the
        # solve blocks for up to the full deadline. Whether bittensor's axon dispatches
        # forward_fn onto a threadpool is version-dependent, and if it does not, a
        # blocking solve stalls the event loop for every other validator's request.
        # ctypes releases the GIL for the duration of the foreign call, so the thread
        # genuinely runs in parallel.
        maximum_clique: list[int] = await asyncio.to_thread(
            native_algorithm,
            synapse.number_of_nodes,
            adjacency_list,
            adjacency_matrix=adjacency_matrix,
            timeout=getattr(synapse, "timeout", None),
            # Without this every hotkey an operator runs submits the IDENTICAL clique:
            # the solver is reproducible in practice (same clique on 5 of 5 runs with
            # the default seed), and each hotkey is its own process with no shared
            # state. The scorer pays diversity = 1 / holders, so a fleet of N would
            # earn 1/N of the diversity term. Measured over 8 rounds against the real
            # scorer this is worth +0.2936 per answer, helping 7 rounds and hurting 0.
            # (The 0.8954 figure elsewhere is the CEILING -- all-same versus 8 distinct
            # omega cliques each assumed unique. Realized is a third of it, because
            # seeding only removes the holders that were OURS and the field already
            # holds 67.5% of our pool.)
            seed=solver_seed(self.wallet.hotkey.ss58_address, synapse.uuid),
            # Passed so the omega-1 spread rule can activate when SN83_SPREAD=1. Without
            # these three the rule silently does nothing, because it needs a per-hotkey
            # hash and an estimate of how many siblings are queried. difficulty is not in
            # the synapse, but the four problems in problem_selector.py have
            # non-overlapping vertex ranges, so number_of_nodes determines it exactly.
            hotkey=self.wallet.hotkey.ss58_address,
            uuid=synapse.uuid,
            difficulty=difficulty_from_n(synapse.number_of_nodes),
            fallback=lambda: networkx_algorithm(synapse.number_of_nodes, adjacency_list),
        )
        # or use GNN models
        # from CliqueAI.clique_algorithms import scattering_clique_algorithm
        # maximum_clique = scattering_clique_algorithm(synapse.number_of_nodes, adjacency_list)
        bt.logging.info(
            f"Maximum clique found: {maximum_clique} with size {len(maximum_clique)}"
        )
        synapse.maximum_clique = maximum_clique
        return synapse

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
