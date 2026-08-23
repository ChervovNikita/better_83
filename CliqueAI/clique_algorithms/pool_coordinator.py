"""One shared pool per task, claimed exactly once per hotkey. No network service.

WHY THIS EXISTS. Two measured facts point at the same missing piece:

  * Hash-based assignment costs -0.018/hotkey median at N=40 and -0.097 at N=120,
    because siblings drawing independently from one pool collide by birthday.
  * The omega-1 spread the best field operators run (94.5% of their answers at omega-1
    when a single maximum clique exists) pays +0.34 per round when correctly triggered,
    and CANNOT be triggered by a lone miner: corr(ourTop, nOm) is 0.176-0.233 regardless
    of thread count, and no synapse-observable predicts nOm better than |0.345|.

Both dissolve if the operator's hotkeys share one pool. Assignment becomes exact instead
of probabilistic, and "is the pool shorter than the number of siblings queried" becomes an
observation instead of an estimate -- because the claims themselves count the queried
siblings.

HOW. A directory on tmpfs, one subdirectory per task uuid. The first hotkey to see a task
solves and writes the pool; the others wait for it. Each hotkey then atomically claims an
index by creating a file named for it -- O_CREAT|O_EXCL is atomic on POSIX, so exactly one
claimant wins each index, with no lock and no daemon. A hotkey that already claimed an
index for this task gets the same one back, so a retried request is idempotent.

Claims are ordered, so the first len(top) claimants take distinct maximum cliques and the
overflow takes distinct omega-1 spares. That is precisely the field's observed behaviour,
and it needs no estimate of how many hotkeys will be queried.

DEGRADES SAFELY. Every failure path returns None and the caller falls back to solving
alone. A stale directory is bounded by TTL cleanup. The processes never block on each
other beyond a short bounded wait.
"""
import errno
import json
import os
import time

ROOT = os.environ.get("SN83_POOL_DIR", "/dev/shm/sn83-pool")
# How long a sibling waits for the first solver to publish the pool. The validator's own
# deadline is 6-30 s and the solve takes most of it, so this is generous enough to cover
# a sibling that started late and short enough that waiting never costs the deadline.
WAIT_S = float(os.environ.get("SN83_POOL_WAIT_S", "0.0"))
TTL_S = float(os.environ.get("SN83_POOL_TTL_S", "900"))


def _task_dir(uuid):
    return os.path.join(ROOT, str(uuid).replace("/", "_")[:80])


def _sweep():
    """Drop task directories older than the TTL. Cheap, best-effort, never raises."""
    try:
        now = time.time()
        for name in os.listdir(ROOT):
            d = os.path.join(ROOT, name)
            try:
                if now - os.path.getmtime(d) > TTL_S:
                    for f in os.listdir(d):
                        os.unlink(os.path.join(d, f))
                    os.rmdir(d)
            except OSError:
                pass
    except OSError:
        pass


_pub_count = 0
_claim_count = 0


def publish(uuid, pool):
    """Store this task's pool if nobody has. Returns True if we were the publisher."""
    global _pub_count
    _pub_count += 1
    # Sweep occasionally rather than every call. One task directory is ~9.7 kB across 27
    # files, and this subnet runs ~2500 rounds a day, so an unswept pool directory grows
    # about 24 MB a day -- in tmpfs, which is RAM, on a box min_compute.yml sizes at 4-8
    # cores. Fine for a day and a problem in a month. Every 64th publish keeps the cost
    # negligible while bounding the directory to roughly the TTL's worth of tasks.
    if _pub_count % 64 == 1:
        _sweep()
    d = _task_dir(uuid)
    try:
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, "pool.tmp.%d" % os.getpid())
        with open(tmp, "w") as f:
            json.dump([list(map(int, c)) for c in pool], f)
        final = os.path.join(d, "pool.json")
        try:
            os.link(tmp, final)          # atomic, fails if another process won
            won = True
        except OSError:
            won = False
        os.unlink(tmp)
        return won
    except OSError:
        return False


def fetch(uuid, wait_s=None):
    """The published pool for this task, or None. Waits briefly if asked."""
    d = _task_dir(uuid)
    final = os.path.join(d, "pool.json")
    deadline = time.time() + (WAIT_S if wait_s is None else wait_s)
    while True:
        try:
            with open(final) as f:
                return [tuple(c) for c in json.load(f)]
        except (OSError, ValueError):
            if time.time() >= deadline:
                return None
            time.sleep(0.02)


def claim(uuid, hotkey, n_slots):
    """This hotkey's slot index in [0, n_slots), assigned exactly once.

    Idempotent: a hotkey that already holds an index for this task gets it back, so a
    retried or duplicated request does not consume a second slot.
    """
    d = _task_dir(uuid)
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return None
    mine = os.path.join(d, "hk." + str(hotkey).replace("/", "_")[:80])
    try:
        with open(mine) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        pass
    for i in range(max(1, int(n_slots))):
        slot = os.path.join(d, "slot.%d" % i)
        try:
            fd = os.open(slot, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except OSError as e:
            if e.errno == errno.EEXIST:
                continue
            return None
        try:
            os.write(fd, str(hotkey).encode()[:200])
        finally:
            os.close(fd)
        try:
            with open(mine, "w") as f:
                f.write(str(i))
        except OSError:
            pass
        return i
    return None            # every slot taken: more siblings than the pool holds


def claim_clique(uuid, hotkey, clique):
    """Reserve this exact clique for this hotkey. False if a sibling already holds it.

    This is the inverse of claim(): instead of handing a miner a clique out of one
    shared harvest, it lets each miner solve INDEPENDENTLY -- which is what produces
    basin diversity -- and only deduplicates the results.

    Measured reason for preferring it: a shared pool comes from a single solve_many
    harvest around one incumbent, so when that harvest holds few maximum cliques every
    claim wraps onto the same one. On round n=890 assignment-only coordination scored
    2.0769, identical to every hotkey submitting the same clique, while per-hotkey
    seeding scored 2.1115. Eight independent seeded solves land in eight different
    basins; one harvest does not.

    Idempotent per hotkey, like claim(): a retry re-confirms the same clique.
    """
    key = ",".join(str(int(v)) for v in sorted(clique))
    if not key:
        return False
    size = len(clique)
    d = _task_dir(uuid)
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return True                      # cannot coordinate: do not block the answer
    import hashlib
    # The dedup coordinator never calls publish(), so the TTL sweep that publish()
    # triggers never ran in the path that actually ships. Measured: 103 task directories
    # and 6.1 MB of tmpfs after 103 rounds, growing without bound. Sweeping here every
    # 64th claim covers the dedup design the same way publish() covers the shared-pool one.
    global _claim_count
    _claim_count += 1
    if _claim_count % 64 == 1:
        _sweep()
    # Record participation BEFORE the outcome is known. A hotkey whose clique is already
    # taken still participated, and those are exactly the hotkeys that need the
    # convergence signal -- a fleet that all lands on one clique produces one successful
    # claim and N-1 failures, so counting only successes leaves the denominator at 1 and
    # the signal never fires.
    try:
        with open(os.path.join(d, "hk." + str(hotkey).replace("/", "_")[:80]), "w") as f:
            f.write(str(size))
    except OSError:
        pass
    h = hashlib.sha1(key.encode()).hexdigest()[:24]
    # The size is in the NAME so distinct_claimed can count only max-size claims.
    # Counting every claim makes the signal measure its own output: each omega-1 spread
    # increments it, so after two spreads the agreement threshold is exceeded and the
    # remaining siblings stop spreading. Measured on round n=494 (nOm=1) that capped the
    # gain at +0.1823 with 3 distinct answers, where the boundary analysis says a
    # correctly-spread starved round is worth about +0.57.
    slot = os.path.join(d, "cq.%d." % size + h)
    me = str(hotkey).encode()[:200]
    try:
        fd = os.open(slot, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except OSError as e:
        if e.errno != errno.EEXIST:
            return True                  # unknown failure: answer anyway
        try:
            with open(slot, "rb") as f:
                return f.read().strip() == me     # ours already: idempotent
        except OSError:
            return True
    try:
        os.write(fd, me)
    finally:
        os.close(fd)
    return True


def distinct_claimed(uuid):
    """(distinct max-size cliques reserved, hotkeys that have claimed) for this task.

    BOTH numbers are needed. The distinct count alone is a function of arrival position:
    the second hotkey to arrive always sees 1, however rich the round is, so a trigger on
    it alone fires for early arrivals everywhere. Measured on round n=890 (nOm=3, our
    seeds finding 4 distinct maximum cliques) that cost -0.0152 -- an early sibling spread
    into a round that had cliques to spare.

    Convergence means FEW DISTINCT ACROSS MANY CLAIMANTS, which needs the denominator.

    This is the fleet-level starvation signal, and it is the reason coordination is worth
    more than deduplication alone. A single miner's harvest barely predicts the round's
    omega supply -- corr(ourTop, nOm) is 0.176 at one thread and 0.233 at three, and
    P(nOm<=8 | ourTop<=1) is only 0.53. But the number of DISTINCT cliques that
    independently-seeded siblings converge on tracks it well: corr = +0.679, with
    P(nOm<=8) = 1.00 when siblings agree (d<=2) against 0.06 when they scatter (d>=6),
    over 30 matched rounds.

    Siblings agreeing means they keep landing in the same basin, which is what a round
    with few maximum cliques looks like from the inside.
    """
    d = _task_dir(uuid)
    try:
        best = -1
        counts = {}
        for f in os.listdir(d):
            if not f.startswith("cq."):
                continue
            try:
                sz = int(f.split(".")[1])
            except (IndexError, ValueError):
                continue
            counts[sz] = counts.get(sz, 0) + 1
            if sz > best:
                best = sz
        # (distinct max-size cliques, hotkeys that have claimed at all)
        n_claimants = sum(1 for f in os.listdir(d) if f.startswith("hk."))
        return counts.get(best, 0), n_claimants
    except OSError:
        return 0, 0
