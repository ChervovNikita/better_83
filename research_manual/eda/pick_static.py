#!/usr/bin/env python3
"""Hand one clique to each queried hotkey, holding back when omega is scarce.

    picker(pool, uuid, hotkeys) -> list[list[int]], one per hotkey, in order

Same signature as fleet_pick.picker, and the same rotation, so the two can be
swapped and compared.  The one difference is which cliques get chosen.

THE RULE

    P = distinct omega-cliques in the pool
    q = queried hotkeys
    t = q if P >= q else 0

t hotkeys submit distinct omega-cliques; the rest submit distinct maximal
cliques one vertex shorter.  Distinct cliques are exhausted at BOTH sizes before
anything is repeated.

WHY HOLD BACK AT ALL

Reward is optimality*(1+difficulty) + diversity, optimality is exp(-pr/rel) with
pr the fraction of answers STRICTLY LARGER, and diversity is 1/(miners
submitting that identical set).  When only a handful of omega-cliques exist,
everyone who returns omega collides on them and diversity collapses to 1/c,
while an omega-1 answer costs only 1 - exp(-pr/rel) of optimality -- which is
small precisely because few answers are larger.  Submitting a distinct spare
beats duplicating an omega clique whenever

    1 - 1/c  >  (1 - opt(omega-1)) * (1 + difficulty)

Measured on one logged round: 0.102 * 1.8 = 0.184, so it pays from c >= 2.

This is not speculation about the field.  Four coldkeys (5HMevt8h, 5Hg2Ps2L,
5EfHz7fE, 5Eyh8ePM) reach omega on 77% of rounds and are exactly ONE vertex
short on the other 23% -- never two.  On those rounds the median count of
distinct omega-cliques is 2, against 28 on the rounds they go for omega.

WHAT THIS CONTRADICTS

fleet_pick.SPARE_CAP is 2, with a comment recording that converting the whole
shortfall to spares measured WORSE than converting a little.  This rule converts
all of it, but only when P < q.  Both cannot be right unconditionally, and the
existing measurement is not superseded by an argument -- see measure_pick.py,
which scores both against the validator's own calculator.
"""

import hashlib


def _rotation(uuid):
    """Where hotkey 0 starts, same as fleet_pick so the arms differ in one thing."""
    return int(hashlib.sha1(str(uuid).encode()).hexdigest()[:8], 16)


def slots(pool, hotkeys):
    """The multiset of cliques the fleet submits, one entry per hotkey."""
    q = len(hotkeys)
    assert pool and q > 0
    omega = max(len(c) for c in pool)
    top = [list(c) for c in pool if len(c) == omega]
    spare = sorted((list(c) for c in pool if len(c) < omega),
                   key=len, reverse=True)

    use = list(top[:q]) if len(top) >= q else []
    if not use:
        use = list(spare[:q])
        if len(use) < q:                     # not enough spares: fall back to
            use.extend(top[:q - len(use)])   # unused omega cliques, still distinct
    # Nothing distinct left anywhere: repeat, cycling so the repeats spread over
    # every clique rather than piling onto the first.
    base = list(use) or list(top)
    i = 0
    while len(use) < q:
        use.append(list(base[i % len(base)]))
        i += 1
    return use[:q]


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
