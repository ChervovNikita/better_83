"""Pin reward_reference.replay_reward to the validator's own CliqueScoreCalculator.

Runs on real logged rounds: for each one, insert a field answer a second time (so
the collision count is known) and check the replay matches what the validator
class computes on the full response list.

    python3 test_reward_reference.py /tmp/sn83ref_answers.jsonl
"""
import collections
import json
import sys

import numpy as np

from _common import REPO  # noqa: F401  (puts the repo on sys.path)
from reward_reference import count_hist_from_answers, replay_reward

from CliqueAI.graph.codec import GraphCodec
from CliqueAI.scoring.clique_scoring import CliqueScoreCalculator


class _Graph:
    """LambdaGraph stand-in; sets instead of lists is a pure speedup."""
    def __init__(self, n, adj):
        self.number_of_nodes, self.adjacency_list = n, adj


def main(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    recs = [r for r in recs if r.get("answers")]
    checked = worst = 0
    for rec in recs:
        M = np.array(GraphCodec().decode_matrix(rec["matrix_b92"]), dtype=np.uint8)
        graph = _Graph(M.shape[0],
                       [set(np.flatnonzero(M[i]).tolist()) for i in range(M.shape[0])])
        answers = [a["clique"] for a in rec["answers"]]
        valid = [a["clique"] for a in rec["answers"] if a.get("opt", 0) > 0]
        if not valid:
            continue
        dup = collections.Counter(tuple(sorted(c)) for c in valid)
        ch = count_hist_from_answers(rec["answers"])
        for key, cnt in list(dup.items())[:4]:          # a few per round is plenty
            ours = list(key)
            got, _, _ = replay_reward(len(ours), cnt, rec["size_hist"], ch,
                                      rec["difficulty"],
                                      n_invalid=rec["n_responders"] - rec["n_valid"])
            *_, rewards = CliqueScoreCalculator(
                graph=graph, difficulty=rec["difficulty"],
                responses=answers + [ours]).get_scores()
            worst = max(worst, abs(got - float(rewards[-1])))
            checked += 1
    print(f"checked {checked} (round, answer) pairs across {len(recs)} rounds")
    print(f"max |replay - CliqueScoreCalculator| = {worst:.3e}")
    if worst < 1e-9:
        print("PASS")
        return 0
    print("FAIL — replay_reward has drifted from the validator")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sn83ref_answers.jsonl"))
