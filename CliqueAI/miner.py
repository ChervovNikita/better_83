import asyncio
import os
import sys
import time
import typing

import bittensor as bt
from CliqueAI.graph.codec import GraphCodec
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph
from common.base.miner import BaseMinerNeuron

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "research_manual", "eda"))
import dispatch_client


class Miner(BaseMinerNeuron):
    """Answers maximum-clique requests from the shared solve service."""

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
        timeout = getattr(synapse, "timeout", None)

        clique = await asyncio.to_thread(
            dispatch_client.solve,
            synapse.uuid,
            self.wallet.hotkey.ss58_address,
            synapse.number_of_nodes,
            adjacency_matrix,
            float(timeout or 0.0),
        )
        if not clique:
            bt.logging.warning("Dispatcher gave no answer")
            clique = []

        bt.logging.info("Maximum clique of size %d" % len(clique))
        synapse.maximum_clique = clique
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
