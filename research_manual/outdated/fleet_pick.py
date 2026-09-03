#!/usr/bin/env python3
"""Hand one clique to each queried hotkey.

    picker(pool, uuid, hotkeys) -> list[list[int]], one per hotkey, in order

The reward is optimality*(1+difficulty) + diversity, where diversity for an answer
is 1/(number of miners submitting that identical vertex set). Two consequences
drive everything here.

**A group of c identical answers is worth c*(1/c) = 1 unit of diversity in total,
whatever c is.** So a repeat is not merely worth less than a distinct clique -- it
contributes nothing at all. Every distinct clique the fleet can put on the wire is
worth a full unit, which is why fleet_solver spends most of its budget widening the
pool rather than re-confirming omega.

**Never stay silent.** When the pool still runs short, the surplus must submit
something anyway. Returning [] is scored invalid and earns a hard 0; repeating a
sibling's clique earns the full optimality term and merely shares the diversity
term. Measured over 81 affected rounds (29.5% of all rounds, 6.6 hotkeys short on
average): duplicating is worth +12.15 total fleet reward per round and wins on
100% of them.

No environment variables; SPARE_CAP below is the one judgement call and is frozen
at the value the simulator settled on.
"""
import hashlib

# How many of the surplus hotkeys get a maximal clique one vertex SHORT of omega
# rather than repeating a sibling's omega clique.
#
# This is not obviously "all of them". A distinct omega-1 answer buys a full unit
# of diversity that a repeat does not, but it gives up part of the optimality
# term, and that term is multiplied by (1 + difficulty) -- up to 2x what diversity
# can pay back. The optimality of a short answer is exp(-pr/rel), where pr is the
# fraction of ALL answers that are strictly larger, so the cost grows as more of
# the fleet drops down: converting the whole shortfall is measurably worse than
# converting a little of it.
#
# The picker cannot compute that trade itself -- it never sees the difficulty or
# the field -- so the depth is capped instead of reasoned about per round.
SPARE_CAP = 2


def _rotation(uuid):
    """Where hotkey 0 starts in the pool.

    Rotating by the round stops hotkey 0 taking the crowded natural endpoint every
    time and carrying the whole collision penalty alone.
    """
    return int(hashlib.sha1(str(uuid).encode()).hexdigest()[:8], 16)


def slots(pool, hotkeys):
    """The exact multiset of cliques the fleet will submit, one entry per hotkey.

    Distinct max-size cliques first, then a capped number of one-short spares for
    the surplus, then repeats of max-size cliques -- never repeats of a spare, which
    would split the same diversity term AND give up a vertex of optimality on top.
    """
    q = len(hotkeys)
    mx = max(len(c) for c in pool)
    top = [list(c) for c in pool if len(c) == mx]
    if len(top) >= q:
        return top[:q]
    spare = sorted((list(c) for c in pool if len(c) < mx), key=len, reverse=True)
    take = min(q - len(top), len(spare), SPARE_CAP)
    use = top + spare[:take]
    while len(use) < q:
        use.append(top[(len(use) - len(top) - take) % len(top)])
    return use


def picker(pool, uuid, hotkeys):
    """One answer per queried hotkey, in the order the hotkeys were given."""
    assert pool
    assert hotkeys
    use = slots(pool, hotkeys)
    assert len(use) == len(hotkeys)
    offset = _rotation(uuid)
    answers = [list(use[(index + offset) % len(use)])
               for index in range(len(hotkeys))]
    assert len(answers) == len(hotkeys)
    return answers
