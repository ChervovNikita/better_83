#!/usr/bin/env python3
"""Picker that scores the split instead of thresholding it.

    picker(pool, uuid, hotkeys) -> list[list[int]], one per hotkey

Same signature and rotation as fleet_pick.picker and pick_static.picker.  The
difference is how many hotkeys go to omega: pick_static uses the categorical
`t = q if P >= q else 0`, this evaluates strategy.plan's value function, which
needs the round's size.

WHY THE ROUND SIZE MATTERS

opt(omega-1) = exp(-((a+t)/N) * omega/(omega-1)) with N = n_others + q, so the
cost of declining omega scales with how many answers are in the round.  Writing
a = phi * n_others for phi the fraction of the field at omega,

    a/N = phi / (1 + q/n_others)   and   q/n_others = N_fleet/(n_miners-N_fleet)

which is constant in difficulty but NOT in fleet size: a bigger fleet dilutes pr
with its own omega-1 answers, so holding back gets cheaper as N_fleet grows
(0.92*phi at N=20, 0.84 at N=40, 0.76 at N=60).  The categorical rule cannot see
that; this can.

GETTING THE INPUTS

difficulty is NOT in the synapse -- production recovers it from the vertex count,
because problem_selector.py defines four problems whose ranges do not overlap
(290-300 -> 0.7, 490-500 -> 0.8, 690-700 -> 0.9, 890-900 -> 1.0).  This uses the
same function, native_algorithm_shim.difficulty_from_n, so the simulator and a
deployed miner see identical inputs.  Checked exact on all 1100 logged rounds.

Given difficulty, the round size needs no fleet count:

    n_others = p(difficulty) * n_miners - q

because the validator queries p*M miners and q of them are ours.  That matters:
a fleet size fluctuates as hotkeys register and deregister, and a stale one
biases the whole decision through q/n_others = N/(M-N) -- configuring 30 when
the true fleet is 60 measured as a 0.157 loss, wiping out the gain over
pick_static.  N_FLEET now survives only as a fallback for callers with neither
difficulty nor a vertex count.
"""

import hashlib
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import strategy

REF_R = 1.5

# Deployment constants: our fleet size and the metagraph population. Both are
# knowable at solve time (we own the hotkeys; the metagraph is on-chain).
N_FLEET = int(os.environ.get("SN83_FLEET", "40"))
N_MINERS = int(os.environ.get("SN83_MINERS", "249"))

# Fraction of the field's answers that reach omega, as a function of how many
# distinct omega-cliques WE found. Step model fitted on tuning_data only.
PHI_SPLIT = int(os.environ.get("SN83_PHI_SPLIT", "5"))
PHI_LOW = float(os.environ.get("SN83_PHI_LOW", "0.06"))
PHI_HIGH = float(os.environ.get("SN83_PHI_HIGH", "1.00"))


def selection_p(difficulty):
    """MinerSelector.miner_selection_probabilities, per hotkey per round."""
    return 1.0 - math.exp(-max(0.0, math.sqrt(1.0 + REF_R) - difficulty - 0.5))


def difficulty_from_p(p):
    """Inverse of selection_p. Exact; the error is in estimating p, not here."""
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.sqrt(1.0 + REF_R) - 0.5 + math.log(1.0 - p)


def round_shape(q, difficulty=None):
    """(n_others, difficulty) for this round.

    With `difficulty` the fleet size is not needed at all: the validator queries
    p*M miners in total and q of them are ours, so

        n_others = p(difficulty) * M - q

    and our own share comes from q exactly.  That matters because the fleet size
    fluctuates in production -- hotkeys deregister and register -- and a stale
    N_FLEET biases the whole decision: q/n_others is N/(M-N), which is 0.191 at
    N=40 and 0.137 at N=30, so a fleet 25% smaller than configured would look
    40% cheaper to hold back than it is.

    Without difficulty there is no choice but to estimate p from our own queried
    fraction, which does need N_FLEET. Prefer passing difficulty.
    """
    if difficulty is not None:
        p = selection_p(difficulty)
        return max(1, int(round(p * N_MINERS - q))), difficulty
    p = min(max(q / float(max(1, N_FLEET)), 1e-3), 0.999)
    return (max(1, int(round(p * (N_MINERS - N_FLEET)))),
            difficulty_from_p(p))


def phi(n_top):
    return PHI_LOW if n_top <= PHI_SPLIT else PHI_HIGH


def difficulty_from_n(number_of_nodes):
    """Recover difficulty from the vertex count, as the deployed miner does.

    Mirrors native_algorithm_shim.difficulty_from_n rather than importing it,
    because that module pulls in bittensor. Exact on all 1100 logged rounds.
    """
    n = int(number_of_nodes)
    if 290 <= n <= 300:
        return 0.7
    if 490 <= n <= 500:
        return 0.8
    if 690 <= n <= 700:
        return 0.9
    if 890 <= n <= 900:
        return 1.0
    return 0.8


def slots(pool, hotkeys, difficulty=None, n_nodes=None, n_top_true=0,
          n_spare_true=0):
    """The multiset of cliques the fleet submits, one entry per hotkey.

    n_top_true / n_spare_true are the distinct counts the solver actually found,
    BEFORE the pool was truncated to k.  They matter because crowding is
    a_hat / (number of omega-cliques that exist), and the pool's own length is a
    truncated proxy that inflates it -- which made the value function duplicate
    on rounds where duplication is worthless.  Falls back to the pool length when
    the caller cannot supply them.
    """
    q = len(hotkeys)
    assert pool and q > 0
    omega = max(len(c) for c in pool)
    n_top = max(n_top_true, sum(1 for c in pool if len(c) == omega))
    n_spare = max(n_spare_true, sum(1 for c in pool if len(c) < omega))
    if difficulty is None and n_nodes:
        difficulty = difficulty_from_n(n_nodes)
    n_others, difficulty = round_shape(q, difficulty)
    a_hat = phi(n_top) * n_others
    return strategy.slots(pool, q, n_others, a_hat, difficulty,
                          b_hat=n_others - a_hat,
                          n_top_supply=n_top, n_spare_supply=n_spare)


def _rotation(uuid):
    return int(hashlib.sha1(str(uuid).encode()).hexdigest()[:8], 16)


def picker(pool, uuid, hotkeys, difficulty=None, n_nodes=None, n_top_true=0,
           n_spare_true=0):
    """One answer per queried hotkey, in the order the hotkeys were given."""
    assert pool
    assert hotkeys
    use = slots(pool, hotkeys, difficulty, n_nodes, n_top_true, n_spare_true)
    assert len(use) == len(hotkeys)
    offset = _rotation(uuid)
    answers = [list(use[(index + offset) % len(use)])
               for index in range(len(hotkeys))]
    assert len(answers) == len(hotkeys)
    return answers
