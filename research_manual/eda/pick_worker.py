"""The picker, run in its own process.

The dispatcher used to run the picker on threads inside its own process. The
derived picker is pure Python and holds the GIL; with four rounds finishing at
once, four picker threads starved the event loop. Measured with four rounds in
flight at fleet 150/160: rounds whose picker overran were cut off at the
deadline as designed, but the loop could not get the answers out -- 172 and 222
answers missed the miners' 2.5 s wait and scored zero. In separate processes a
picker, finished or abandoned, never competes with the loop for the GIL.

Imported by the dispatcher parent AND by each pool process (spawn), so it
imports nothing heavy at module level.
"""
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in (PARENT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _watch_parent(parent):
    # a pool process outliving a killed dispatcher would hold a core and, with
    # maximin loaded, its OpenMP threads
    while True:
        time.sleep(1.0)
        if os.getppid() != parent:
            os._exit(0)


def init(use_minimax):
    """Pool-process initializer: load the picker now, not inside a round."""
    threading.Thread(target=_watch_parent, args=(os.getppid(),),
                     daemon=True).start()
    import pick_derived  # noqa: F401
    if use_minimax:
        import minimax  # noqa: F401  (loads libminimax.so and OpenMP)


def warm(_=None):
    return os.getpid()


def pick(pool, key, q, n_nodes, hits, n_top_true, n_spare_true, fleet_n,
         deadline, bounded, safety_s):
    """One allocation for the round. Returns (answers, which picker ran).

    `deadline` is a time.monotonic() value; CLOCK_MONOTONIC is system-wide on
    Linux, so it means the same thing in this process as in the dispatcher.
    """
    import pick_derived
    kw = dict(n_nodes=n_nodes, hits=list(hits), n_top_true=n_top_true,
              n_spare_true=n_spare_true, fleet_n=fleet_n)
    if not bounded:
        return pick_derived.picker(pool, key, list(range(q)), **kw), "derived"
    import minimax
    left = (deadline - time.monotonic()) if deadline is not None else 0.0
    kw["deadline_s"] = max(0.0, left - safety_s)
    fell_before = minimax.STATS.get("fallback", 0)
    answers = minimax.picker(pool, key, list(range(q)), **kw)
    fell = minimax.STATS.get("fallback", 0) > fell_before
    return answers, ("maximin->derived" if fell else "maximin")
