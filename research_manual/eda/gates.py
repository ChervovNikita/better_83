#!/usr/bin/env python3
"""Stage gates from gpu_clique_design.md §8.

    python3 gates.py stage0   per-walker steps/s, and whether one walker reaches omega
    python3 gates.py stage1   device cand0/cand1 vs a brute-force host reference
    python3 gates.py stage2   one-walker trajectory diff, and answer validity
    python3 gates.py stage3   distinct cliques vs the CPU arm at matched wall time
    python3 gates.py stage4   queue health and the §6 stall detector
    python3 gates.py all

These count distinct cliques, which is a proxy for the diversity term. Reward
comes from research_manual/simulate.py or research/fleet_sim.py.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
ROOT = os.path.dirname(PARENT)
for _p in (HERE, PARENT, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gpu_lib
from fleet_solver_gpu import fleet_solver

ROUNDS = os.path.join(PARENT, "rounds.json")
STAGE0_SIZES = (290, 490, 690, 890)


def random_graph(n, density=0.9, seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < density).astype(np.uint8)
    A = np.triu(A, 1)
    return A + A.T


def load_rounds(stratum=None, count=8, seed=11):
    """Draws rounds randomly inside an n-stratum, never by deadline order.

    Returns [(n, time_limit, A, field_omega, field_distinct)].
    """
    from CliqueAI.graph.codec import GraphCodec
    with open(ROUNDS) as handle:
        payload = json.load(handle)
    recs = [r for r in payload.values() if r.get("answers")]
    if stratum:
        lo, hi = stratum
        recs = [r for r in recs if lo <= r["number_of_nodes"] <= hi]
    assert recs, "no rounds in stratum %r" % (stratum,)
    codec = GraphCodec()
    out = []
    for i in np.random.default_rng(seed).permutation(len(recs))[:count]:
        rec = recs[int(i)]
        A = np.array(codec.decode_matrix(rec["encoded_matrix"]), dtype=np.uint8)
        omega = max(len(a[3]) for a in rec["answers"])
        distinct = len({tuple(sorted(a[3])) for a in rec["answers"]
                        if len(a[3]) == omega})
        out.append((A.shape[0], float(rec["time_limit"]), A, omega, distinct))
    return out


def arms(both):
    """(lanes, prefix) pairs to measure."""
    return [(32, False), (32, True), (64, False)] if both else [(32, False)]


def tag_of(lanes, prefix):
    return "lanes=%d %s" % (lanes, "prefix/suffix" if prefix else "carry")


def stage0(args):
    """Per-walker depth. Aggregate throughput is not the question: a single job
    has to reach omega inside its own slice or the harvest is empty."""
    print("=== stage 0: per-walker steps/s, and does ONE walker reach omega ===")
    rounds = [load_rounds((n - 15, n + 15), count=1, seed=n)[0]
              for n in STAGE0_SIZES]
    ok = True
    for lanes, prefix in arms(args.both_arms):
        for n, _tl, A, omega, _fd in rounds:
            with gpu_lib.GpuClique(A, lanes=lanes, prefix=prefix) as gpu:
                if n == rounds[0][0] and lanes == 32 and not prefix:
                    print("  " + gpu.info())
                solo_s, solo_steps, _ = gpu.probe(1, args.steps, seed=3)
                full_s, full_steps, best = gpu.probe(0, args.steps, seed=3)
                solo = solo_steps.mean() / solo_s
                full = full_steps.mean() / full_s
                print("  %-22s n=%3d omega=%3d | solo %7.0f steps/s | %4d walkers "
                      "%7.0f steps/s each (%.2fM total) | best=%3d reach_omega=%4.0f%%"
                      % (tag_of(lanes, prefix), n, omega, solo, len(best), full,
                         full * len(best) / 1e6, best.max(),
                         100 * float(np.mean(best >= omega))))
                if best.max() < omega:
                    ok = False
                    print("       FAIL: no walker reached omega=%d in %d steps"
                          % (omega, args.steps))
    print("stage0: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def stage1(args):
    """Candidate sets vs brute force. Both arms always: the carry arm's whole
    claim is that it produces the same sets as the prefix/suffix one."""
    print("=== stage 1: candidate sets vs brute force ===")
    ok = True
    checked_total = 0
    stage_arms = [(32, False), (32, True)]
    if args.both_arms:
        stage_arms.append((64, False))
    for lanes, prefix in stage_arms:
        bad_total = 0
        checked = 0
        for gi in range(args.graphs):
            n = int(np.random.default_rng(gi).integers(64, 900))
            A = random_graph(n, 0.5 + 0.45 * ((gi % 5) / 4.0), seed=1000 + gi)
            with gpu_lib.GpuClique(A, lanes=lanes, prefix=prefix) as gpu:
                for n_bans in (0, 1, 4):
                    got, bad = gpu.check_cands(args.trials, 48, n_bans,
                                               seed=7 + gi * 13 + n_bans)
                    checked += got
                    bad_total += bad
        checked_total += checked
        print("  %-22s %d graphs, %d states checked, %d mismatched"
              % (tag_of(lanes, prefix), args.graphs, checked, bad_total))
        ok = ok and not bad_total
    print("stage1: %s (%d states)" % ("PASS" if ok else "FAIL", checked_total))
    return 0 if ok else 1


def stage2(args):
    """Trajectory diff against the host mirror, and validity of every answer.

    The trajectory hash folds in every add and drop in order, so a match is
    move-for-move agreement and not just the same final size.
    """
    print("=== stage 2: trajectory diff + answer validity ===")
    ok = True
    for lanes, prefix in arms(args.both_arms):
        mismatched = 0
        runs = 0
        for gi in range(args.graphs):
            n = int(np.random.default_rng(500 + gi).integers(64, 400))
            A = random_graph(n, 0.85 + 0.1 * ((gi % 3) / 2.0), seed=2000 + gi)
            with gpu_lib.GpuClique(A, lanes=lanes, prefix=prefix) as gpu:
                for seed in range(1, args.trajectories + 1):
                    for n_bans in (0, 3):
                        r = gpu.check_trajectory(seed, args.steps, n_bans)
                        runs += 1
                        if (r["dev_traj"] != r["ref_traj"] or not r["same_clique"]
                                or r["dev_size"] != r["ref_size"]):
                            mismatched += 1
                            if mismatched <= 3:
                                print("       MISMATCH n=%d seed=%d bans=%d %r"
                                      % (n, seed, n_bans, r))

        A = random_graph(500, seed=99)
        rng = np.random.default_rng(4)
        bans = [rng.integers(0, 500, size=int(rng.integers(0, 5))).tolist()
                for _ in range(args.jobs)]
        with gpu_lib.GpuClique(A, lanes=lanes, prefix=prefix) as gpu:
            cliques, _ = gpu.solve_batch(np.arange(1, args.jobs + 1, dtype=np.uint64),
                                         bans=bans, max_steps=args.steps)
        checks = [gpu_lib.verify(A, c) for c in cliques]
        bad_clique = sum(1 for is_c, _ in checks if not is_c)
        bad_max = sum(1 for is_c, is_m in checks if is_c and not is_m)
        print("  %-22s %d trajectories, %d mismatched | %d answers, %d not cliques, "
              "%d not maximal in full G"
              % (tag_of(lanes, prefix), runs, mismatched, len(cliques), bad_clique,
                 bad_max))
        ok = ok and not (mismatched or bad_clique or bad_max)
    print("stage2: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def stage3(args):
    """Distinct cliques vs the CPU arm: same rounds, same k, same wall clock.

    The metric is distinct per second, never cliques per second: many walkers
    under the same guided rules fall into the same basins.
    """
    print("=== stage 3: distinct cliques at matched wall time ===")
    rows = []
    for n, tl, A, f_omega, f_distinct in load_rounds(args.stratum, args.rounds,
                                                     args.seed):
        budget = max(0.5, tl - 2.0)

        started = time.time()
        cpu = fleet_solver.solve_many(A, budget, args.k)
        cpu_s = time.time() - started
        cpu_omega = max(len(c) for c in cpu)
        cpu_d = len({tuple(c) for c in cpu if len(c) == cpu_omega})

        started = time.time()
        with gpu_lib.GpuClique(A, lanes=args.lanes, prefix=args.prefix) as gpu:
            init = None
            if not args.gpu_only:
                init = sorted(fleet_solver._solve_one(
                    A, budget * fleet_solver.CHAMPION_SHARE, seed=1))
            left = budget - (time.time() - started)
            pool, ctr, _hits = gpu.harvest(max(0.05, left), seed=1,
                                    max_steps=args.steps,
                                    boot_steps=args.boot_steps,
                                    init_clique=init, max_out=8192)
        good = [c for c in pool if all(gpu_lib.verify(A, c))]
        gpu_s = time.time() - started
        gpu_omega = max(len(c) for c in good) if good else 0
        gpu_d = len({tuple(c) for c in good if len(c) == gpu_omega})

        rows.append((n, cpu_omega, gpu_omega, cpu_d, gpu_d))
        print("  n=%3d tl=%4.1f | CPU %5.1fs omega=%3d distinct=%3d | GPU %5.1fs "
              "omega=%3d distinct=%4d | field omega=%d distinct=%d | jobs=%5d "
              "stall=%.2f"
              % (n, tl, cpu_s, cpu_omega, cpu_d, gpu_s, gpu_omega, gpu_d, f_omega,
                 f_distinct, ctr["jobs"], gpu_lib.stall(ctr)), flush=True)

    won = sum(1 for r in rows if r[4] > r[3])
    tied = sum(1 for r in rows if r[4] == r[3])
    print("  distinct: GPU better on %d, tied %d, worse on %d of %d rounds"
          % (won, tied, len(rows) - won - tied, len(rows)))
    lost = [r[0] for r in rows if r[2] < r[1]]
    if lost:
        print("  NOTE: GPU returned a smaller omega on n=%s -- per-walker depth, "
              "not a diversity result" % lost)
    print("stage3: reported (direction, not reward)")
    return 0


def stage4(args):
    """Queue health, and the §6 stall detector.

    A stall near 1 while the card is busy is the predicted ceiling: a ban
    excludes cliques containing the banned vertex, so it excludes the parent but
    never its siblings.
    """
    print("=== stage 4: queue health and the §6 stall detector ===")
    ok = True
    for n, tl, A, _f_omega, _f_distinct in load_rounds(args.stratum, args.rounds,
                                                       args.seed):
        budget = max(0.5, min(args.budget, tl - 2.0))
        started = time.time()
        with gpu_lib.GpuClique(A, lanes=args.lanes, prefix=args.prefix) as gpu:
            pool, ctr, _hits = gpu.harvest(budget, seed=1, max_steps=args.steps,
                                    boot_steps=args.boot_steps, max_out=8192)
        elapsed = time.time() - started
        bad = sum(1 for c in pool if not all(gpu_lib.verify(A, c)))
        omega = max(len(c) for c in pool) if pool else 0
        distinct = len({tuple(c) for c in pool if len(c) == omega})
        print("  n=%3d %.1fs | omega=%3d distinct_omega=%4d (%6.1f/s) | jobs=%6d "
              "synth=%6d | stall=%.2f banback=%.3f | drop=%d overflow=%d kmax=%d "
              "| invalid=%d"
              % (n, elapsed, omega, distinct, distinct / elapsed, ctr["jobs"],
                 ctr["synth"], gpu_lib.stall(ctr),
                 ctr["banback"] / max(1, ctr["jobs"]), ctr["drop"],
                 ctr["overflow"], ctr["kmaxhit"], bad), flush=True)
        if bad:
            ok = False
            print("       FAIL: %d invalid answers" % bad)
    print("stage4: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


STAGES = {"stage0": stage0, "stage1": stage1, "stage2": stage2,
          "stage3": stage3, "stage4": stage4}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=sorted(STAGES) + ["all"])
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: proves the path runs, measures nothing")
    ap.add_argument("--both-arms", action="store_true",
                    help="also measure prefix/suffix (§2) and lanes=64 (§3)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--boot-steps", type=int, default=60000)
    ap.add_argument("--graphs", type=int, default=8)
    ap.add_argument("--trials", type=int, default=512)
    ap.add_argument("--trajectories", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--prefix", action="store_true")
    ap.add_argument("--gpu-only", action="store_true")
    ap.add_argument("--stratum", type=int, nargs=2, default=None,
                    metavar=("N_LO", "N_HI"))
    args = ap.parse_args()

    if args.quick:
        args.graphs, args.trials, args.trajectories = 2, 64, 2
        args.jobs, args.rounds, args.budget = 32, 2, 2.0
        args.steps, args.boot_steps = 4000, 8000
        print("*** --quick: smoke test, measures no reward ***\n")

    if args.stage != "all":
        return STAGES[args.stage](args)
    rc = 0
    for name in ("stage1", "stage2", "stage0", "stage3", "stage4"):
        rc |= STAGES[name](args)
        print()
    return rc


if __name__ == "__main__":
    sys.exit(main())
