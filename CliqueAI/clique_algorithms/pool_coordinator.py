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


def publish(uuid, pool):
    """Store this task's pool if nobody has. Returns True if we were the publisher."""
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
