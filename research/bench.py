#!/usr/bin/env python3
"""Benchmark runner for solver variants — the autoresearch inner loop.

An experiment is a *variant*: a copy of native/clique.cpp under native/variants/
with one change, built to its own .so and run through the same ctypes ABI. Params
that need no code change are passed as env vars the variant reads.

Sets:
  bigger_val  500 held-out tasks + labels  (the steering set, ~8 min at 14 workers)
  val          42 held-out tasks           (too small to steer on)
  gauntlet   N train tasks, stratified     (verification; never tuned against)
  hard       tasks the champion loses/ties narrowly (fast dev signal, overfits easily)

  python3 bench.py --variant champion --set bigger_val --json runs/champ.json
  python3 bench.py --variant v3_bms --set hard --env SN83_BMS_K=128
"""
import argparse
import collections
import ctypes
import fcntl
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np

from _common import DATA_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(DATA_DIR, "splits")
VARIANTS = os.path.join(HERE, "native", "variants")
TRAIN = os.path.join(SPLITS, "train.jsonl")
TRAIN_INDEX = os.path.join(SPLITS, "train_index.jsonl")
SETS_DIR = os.path.join(DATA_DIR, "sets")

_W = {}


# ---------------------------------------------------------------- variants

def variant_src(name):
    if name == "champion":
        return os.path.join(HERE, "native", "clique.cpp")
    return os.path.join(VARIANTS, f"{name}.cpp")


def variant_lib(name):
    return os.path.join(VARIANTS, f"lib_{name}.so") if name != "champion" \
        else os.path.join(HERE, "native", "libclique.so")


def build(name, extra_flags=""):
    """Compile a variant if its source is newer than its .so. Lock: workers race."""
    src, lib = variant_src(name), variant_lib(name)
    if not os.path.exists(src):
        raise SystemExit(f"no such variant source: {src}")
    os.makedirs(VARIANTS, exist_ok=True)
    with open(os.path.join(VARIANTS, f".{name}.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.exists(lib) and os.path.getmtime(lib) >= os.path.getmtime(src):
            return lib
        cmd = (f"g++ -O3 -march=native -funroll-loops -std=c++17 -pthread "
               f"{extra_flags} -shared -fPIC {src} -o {lib}")
        t0 = time.time()
        subprocess.run(cmd, shell=True, check=True)
        print(f"built {os.path.basename(lib)} in {time.time()-t0:.1f}s", file=sys.stderr)
    return lib


def load(name):
    lib = ctypes.CDLL(variant_lib(name))
    lib.sn83_solve.restype = ctypes.c_int
    lib.sn83_solve.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
                               ctypes.c_double, ctypes.c_uint64, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int32)]
    return lib


# ---------------------------------------------------------------- task sets

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_train_record(off, length):
    with open(TRAIN, "rb") as f:
        f.seek(off)
        return json.loads(f.read(length))


def stratified(idx, n_take, seed):
    """Proportional largest-remainder allocation over (time_limit, difficulty)."""
    import random
    if n_take >= len(idx):
        return list(idx)
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for r in idx:
        by[(r["tl"], r["d"])].append(r)
    keys = sorted(by)
    q = {k: len(by[k]) * n_take / len(idx) for k in keys}
    take = {k: int(v) for k, v in q.items()}
    for k in sorted(keys, key=lambda k: -(q[k] - take[k]))[:n_take - sum(take.values())]:
        take[k] += 1
    out = []
    for k in keys:
        out += rng.sample(by[k], min(take[k], len(by[k])))
    rng.shuffle(out)
    return out


def build_set(name, n, seed=4242, min_v=0, max_v=10**9):
    """A train-derived task set. Train is already disjoint from both held-out
    splits, so these are clean; they exist so that steering on bigger_val (per the
    run directive) still has untouched sets to confirm against.

    `hard` filters to the big graphs where the champion actually loses, giving a
    fast dev signal — but it is a biased sample, so a win there must still be
    confirmed on bigger_val and gauntlet."""
    os.makedirs(SETS_DIR, exist_ok=True)
    idx = [r for r in load_jsonl(TRAIN_INDEX) if min_v <= r["n"] <= max_v]
    picks = stratified(idx, n, seed)
    path = os.path.join(SETS_DIR, f"{name}.jsonl")
    with open(path, "w") as f:
        for r in picks:
            f.write(json.dumps(r) + "\n")
    print(f"{name}: {len(picks)} tasks, {sum(r['tl'] for r in picks):.0f}s of deadline "
          f"-> {path}", file=sys.stderr)
    return path


def get_tasks(setname, limit=0):
    """-> list of dicts with either ('problem', label) inline or a train offset."""
    if setname in ("val", "bigger_val"):
        probs = load_jsonl(os.path.join(SPLITS, f"{setname}_problems.jsonl"))
        labels = {r["uuid"]: r["best_size"] for r in
                  load_jsonl(os.path.join(SPLITS, f"{setname}_labels.jsonl"))}
        tasks = [{"kind": "split", "uuid": p["uuid"], "n": p["n"], "tl": p["time_limit"],
                  "density": p["density"], "b92": p["matrix_b92"],
                  "best": labels[p["uuid"]]} for p in probs]
    else:
        path = os.path.join(SETS_DIR, f"{setname}.jsonl")
        if not os.path.exists(path):
            raise SystemExit(f"unknown set '{setname}' (no {path}) — build it with "
                             f"--build-set {setname} --build-n N")
        recs = load_jsonl(path)
        # Two flavours of set file: train-derived (byte offsets into train.jsonl) and
        # inline (the graph carried in the file, used for subsets of a held-out split).
        tasks = [{"kind": "split", "uuid": r["uuid"], "n": r["n"], "tl": r["tl"],
                  "density": r.get("density"), "b92": r["b92"], "best": r["best"],
                  # carried when the set file has them, so reward replay works on
                  # inline sets too and not only on train-offset sets
                  "best_cliques": r.get("best_cliques"),
                  "best_clique_counts": r.get("best_clique_counts"),
                  "size_hist": r.get("size_hist"), "difficulty": r.get("difficulty"),
                  "any_unique": r.get("any_unique")}
                 if "b92" in r else
                 {"kind": "train", "uuid": r["uuid"], "n": r["n"], "tl": r["tl"],
                  "off": r["o"], "len": r["l"], "best": r["best"]}
                 for r in recs]
    return tasks[:limit] if limit else tasks


# ---------------------------------------------------------------- execution

def effective_cpus():
    """CPUs this container may actually use.

    sched_getaffinity reports the HOST's core count, which on a throttled container
    is a lie: the CFS quota caps total CPU regardless. Measured on this box,
    affinity says 128 while cpu.cfs_quota_us/cpu.cfs_period_us says 15.3. Running
    14 workers x 8 threads under that quota hands each task ~1.1 cores instead of
    8, silently turning every benchmark into a different experiment.
    """
    n = len(os.sched_getaffinity(0))
    try:                                            # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as f:
            q, per = f.read().split()
        if q != "max":
            n = min(n, max(1, int(float(q) / float(per))))
    except (OSError, ValueError):
        try:                                        # cgroup v1
            q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            per = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if q > 0:
                n = min(n, max(1, int(q / per)))
        except (OSError, ValueError):
            pass
    return n


def replay_reward(our_size, collide, size_hist, best_counts, difficulty,
                  any_unique=None):
    """Replay CliqueScoreCalculator.get_scores() with us as one extra responder.

    reward = optimality*(1+difficulty) + diversity, where
      omega_i       = exp(-pr_i / rel_i),  rel = size/max_size,
                      pr = fraction of responders strictly larger
      optimality    = omega / max(omega)          -> anyone at max size scores 1.0
      diversity     = (1/#miners with your exact set), normalised by the best of those

    The asymmetry that matters: tying for best size already maxes the optimality
    term, so the ONLY way to separate from the pack is the diversity term — and that
    term is worth up to 1.0 against an optimality span of about 0.9.
    """
    sizes = []
    for s_, c_ in (size_hist or {}).items():
        sizes.extend([int(s_)] * int(c_))
    if not sizes:
        return None, None, None
    sizes.append(our_size)
    arr = np.array(sizes, dtype=float)
    mx = arr.max()
    if mx <= 0:
        return 0.0, 0.0, 0.0
    rel = arr / mx
    pr = np.array([(arr > s_).sum() / len(arr) for s_ in arr])
    omega = np.where(arr > 0, np.exp(-pr / np.maximum(rel, 1e-9)), 0.0)
    mo = omega.max()
    optim = omega / mo if mo > 0 else omega
    our_opt = float(optim[-1])

    # diversity: our count is collisions+1.
    #
    # max_delta in the validator ranges over EVERY valid answer, not just the
    # best-size ones: `delta = val * (1/count)` and `max_delta = max(delta)`. So a
    # miner who returns a small maximal clique nobody else returned sets max_delta to
    # 1.0. Normalising against only the best-size cliques (as this did) overstates our
    # diversity in exactly the rounds where the best-size cliques are crowded — and
    # `any_unique` in the dataset is computed over all valid answers, so it is the
    # right signal. The term is size-blind by construction: it pays for uniqueness,
    # not for being unique at the top size.
    our_delta = 0.0 if our_size <= 0 else 1.0 / (1 + (collide or 0))
    if any_unique:
        md = 1.0
    else:
        field_best_delta = 0.0
        for c_ in (best_counts or []):
            field_best_delta = max(field_best_delta, 1.0 / int(c_))
        md = max(our_delta, field_best_delta)
    our_div = (our_delta / md) if md > 0 else 0.0
    return float(our_opt * (1 + difficulty) + our_div), our_opt, our_div


def _init(variant, threads, cores, env, offset=0):
    # Be a good citizen on a shared box: never the highest-priority thing running,
    # so an interactive shell still responds while a 500-task sweep is in flight.
    try:
        os.nice(5)
    except OSError:
        pass
    slot = mp.current_process()._identity[0] - 1 if mp.current_process()._identity else 0
    if cores:
        # `offset` lets a second bench run share the box without fighting the first
        # for cores — each run takes a disjoint block, which matters now that the
        # real budget is 15 CPUs, not 128.
        block = cores[(offset + slot * threads) % len(cores):][:threads]
        if len(block) == threads:
            os.sched_setaffinity(0, block)
    os.environ.update(env)
    _W["lib"] = load(variant)
    _W["threads"] = threads


def _run(job):
    from CliqueAI.graph.codec import GraphCodec
    t, time_scale, seed, time_offset = job
    best_cliques = best_counts = size_hist = difficulty = None
    any_unique = None
    if t["kind"] == "split":
        b92 = t["b92"]
        density = t.get("density")
        best_cliques = t.get("best_cliques")
        best_counts = t.get("best_clique_counts")
        size_hist = t.get("size_hist")
        difficulty = t.get("difficulty")
        any_unique = t.get("any_unique")
    else:
        rec = read_train_record(t["off"], t["len"])
        b92, density = rec["matrix_b92"], rec["density"]
        # Train records carry every DISTINCT optimum the field found and how many
        # miners hit each, which is what makes the diversity term measurable.
        best_cliques = rec.get("best_cliques")
        best_counts = rec.get("best_clique_counts")
        size_hist = rec.get("size_hist")
        difficulty = rec.get("difficulty")
        any_unique = rec.get("any_unique")
    A = np.ascontiguousarray(np.array(GraphCodec().decode_matrix(b92), dtype=np.uint8))
    n = A.shape[0]
    out = np.zeros(n, dtype=np.int32)
    tl_eff = max(0.05, t["tl"] - time_offset)
    budget = max(0.01, tl_eff * time_scale - 0.03)
    t0 = time.time()
    size = _W["lib"].sn83_solve(A.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), n,
                                budget, ctypes.c_uint64(seed), _W["threads"], 0, 0,
                                out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
    elapsed = time.time() - t0
    clique = out[:size].tolist()

    ok, why = True, "ok"
    if not clique or len(set(clique)) != len(clique):
        ok, why = False, "empty or repeated vertex"
    else:
        i = np.array(clique, dtype=int)
        if i.min() < 0 or i.max() >= n:
            ok, why = False, "vertex out of range"
        elif A[np.ix_(i, i)].sum() != len(clique) * (len(clique) - 1):
            ok, why = False, "not a clique"
        else:
            cnt = A[i].sum(axis=0)
            inC = np.zeros(n, dtype=bool)
            inC[i] = True
            if np.any((cnt == len(clique)) & (~inC)):
                ok, why = False, "not maximal"
    got = len(clique) if ok else 0
    # Diversity: on chain the reward carries 1/(miners returning this exact vertex
    # set). Matching the field on SIZE but colliding on the SET forfeits it.
    collide = None
    if ok and best_cliques and got == t["best"]:
        key = tuple(sorted(clique))
        collide = 0
        for c, cnt in zip(best_cliques, best_counts or []):
            if tuple(sorted(c)) == key:
                collide = cnt
                break
    reward = optim = divers = None
    if size_hist is not None and difficulty is not None:
        reward, optim, divers = replay_reward(got if ok else 0, collide, size_hist,
                                              best_counts, difficulty, any_unique)
    return dict(uuid=t["uuid"], n=t["n"], tl=t["tl"], density=density, best=t["best"],
                ours=got, delta=(got - t["best"]) if ok else None, ok=ok, why=why,
                elapsed=elapsed, over=elapsed > t["tl"] * 1.02 + 0.25,
                collide=collide, reward=reward, optimality=optim, diversity=divers,
                n_optima=len(best_cliques) if best_cliques else None)


def summarize(rows, label):
    n = len(rows)
    solved = [r for r in rows if r["ok"]]
    deltas = [r["delta"] for r in solved]
    hist = collections.Counter(deltas)
    parity = sum(1 for d in deltas if d >= 0) / n
    print(f"\n=== {label}: {n} tasks | parity {parity:.3%} | mean delta "
          f"{np.mean(deltas):+.3f} | invalid {n - len(solved)} | "
          f"over {sum(r['over'] for r in rows)}")
    for d in range(max(max(hist), 1), min(hist) - 1, -1):
        c = hist.get(d, 0)
        print(f"   {('0' if d == 0 else f'{d:+d}'):>4} {c:>5} {c/n:6.1%} {'#' * round(40*c/n)}")
    col = [r for r in rows if r.get("collide") is not None]
    if True:
        fresh = sum(1 for r in col if r["collide"] == 0)
        shared = [r["collide"] for r in col if r["collide"] > 0]
        div = sum(1.0 / (1 + r["collide"]) for r in col) / len(col)
    rw = [r["reward"] for r in rows if r.get("reward") is not None]
    if rw:
        op = [r["optimality"] for r in rows if r.get("reward") is not None]
        dv = [r["diversity"] for r in rows if r.get("reward") is not None]
        print(f"   REPLAYED REWARD  mean {np.mean(rw):.4f}   "
              f"(optimality {np.mean(op):.4f}, diversity {np.mean(dv):.4f})")
    if col:
        print(f"   diversity: {fresh}/{len(col)} answers matched NO clique the field "
              f"returned ({fresh/len(col):.1%}); of the rest, mean {np.mean(shared) if shared else 0:.1f} "
              f"miners shared it; mean 1/(1+collisions) = {div:.3f}")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["n"] // 100 * 100].append(r)
    print("   by |V|: " + "  ".join(
        f"{k}:{sum(1 for r in v if r['ok'] and r['delta'] >= 0)/len(v):.0%}({len(v)})"
        for k, v in sorted(by.items())))
    byt = collections.defaultdict(list)
    for r in rows:
        byt[r["tl"]].append(r)
    print("   by  tl: " + "  ".join(
        f"{k}:{sum(1 for r in v if r['ok'] and r['delta'] >= 0)/len(v):.0%}({len(v)})"
        for k, v in sorted(byt.items())))
    return parity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="champion")
    ap.add_argument("--set", dest="setname", default="bigger_val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--core-offset", type=int, default=0,
                    help="first core to pin to; use a disjoint offset to run two "
                         "benchmarks side by side without contention")
    ap.add_argument("--reserve-cores", type=int, default=16,
                    help="cores left free for the OS and everything else. Total cores "
                         "touched is capped at (available - reserve), so a sweep can "
                         "never take the whole machine")
    ap.add_argument("--time-scale", type=float, default=0.88)
    ap.add_argument("--time-offset", type=float, default=0.0,
                    help="seconds subtracted from every deadline BEFORE scaling, to "
                         "model the network round trip a live miner pays to receive "
                         "the task and submit the answer. 2.0 is a realistic worst "
                         "case and bites hardest on the 6 s tier")
    ap.add_argument("--seed", type=int, default=0, help="solver RNG seed")
    ap.add_argument("--env", action="append", default=[], help="K=V passed to the solver")
    ap.add_argument("--flags", default="", help="extra g++ flags for this variant")
    ap.add_argument("--json", help="write per-task rows here")
    ap.add_argument("--build-set", help="build a train-derived set and exit")
    ap.add_argument("--build-n", type=int, default=1200)
    ap.add_argument("--build-min-v", type=int, default=0)
    ap.add_argument("--build-max-v", type=int, default=10**9)
    args = ap.parse_args()

    if args.build_set:
        build_set(args.build_set, args.build_n, min_v=args.build_min_v,
                  max_v=args.build_max_v)
        return 0

    build(args.variant, args.flags)
    tasks = get_tasks(args.setname, args.limit)
    env = dict(kv.split("=", 1) for kv in args.env)
    cores = sorted(os.sched_getaffinity(0))
    eff = effective_cpus()
    # Reserve is relative to what we actually have, not the host core count —
    # reserving 16 out of 15 would leave nothing to run on.
    reserve = min(args.reserve_cores, max(1, eff // 8))
    usable = max(args.threads, eff - reserve)
    workers = min(args.workers, max(1, usable // args.threads))
    budget = sum(t["tl"] for t in tasks)
    print(f"{args.variant} on {args.setname}: {len(tasks)} tasks, {budget:.0f}s of deadline, "
          f"{workers}x{args.threads} = {workers*args.threads} cores of {eff} usable "
          f"(host reports {len(cores)}; {reserve} reserved), env={env}", file=sys.stderr)
    if workers * args.threads > eff:
        print("  WARNING: oversubscribed — per-task CPU is below the requested thread "
              "count, so timings do not reflect a real miner", file=sys.stderr)

    t0 = time.time()
    with mp.Pool(workers, initializer=_init,
                 initargs=(args.variant, args.threads, cores, env,
                           args.core_offset)) as pool:
        rows = list(pool.imap_unordered(
            _run, [(t, args.time_scale, args.seed, args.time_offset) for t in tasks],
            chunksize=1))
    wall = time.time() - t0
    parity = summarize(rows, f"{args.variant} / {args.setname}")
    print(f"   wall {wall:.0f}s", file=sys.stderr)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"variant": args.variant, "set": args.setname, "env": env,
                       "seed": args.seed, "threads": args.threads, "parity": parity,
                       "time_offset": args.time_offset,
                       "wall_s": wall, "rows": rows}, f)
        print(f"   -> {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    sys.exit(main())
