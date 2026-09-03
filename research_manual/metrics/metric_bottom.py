"""Share of OUR hotkeys in the bottom 10%, cut at the FIELD-only percentile.

Deregistration takes the worst miners, so what matters is not the mean edge but how
many of ours sit in the tail.  The cut must come from the field alone: our own
hotkeys are part of the population being ranked, so including them lets a fleet that
is uniformly bad move the threshold down under itself and score well.
"""
import json, statistics, collections, sys, os

def bottom_share(path, pct=10.0):
    d = json.load(open(path))
    ours = collections.defaultdict(list); field = collections.defaultdict(list)
    per_round = []
    for rec in d.values():
        o = []; f = []
        for a, s in zip(rec["answers"], rec["scores"]):
            (ours if a[1].startswith("our_") else field)[a[1]].append(s)
            (o if a[1].startswith("our_") else f).append(s)
        if o and f:
            cut = _pct(sorted(f), pct)
            per_round.append(sum(1 for x in o if x <= cut) / len(o))
    om = {k: statistics.mean(v) for k, v in ours.items() if v}
    fm = {k: statistics.mean(v) for k, v in field.items() if v}
    cut = _pct(sorted(fm.values()), pct)
    hk = sum(1 for v in om.values() if v <= cut) / max(1, len(om))
    return hk, (statistics.mean(per_round) if per_round else 0.0), cut, len(om), len(fm)

def _pct(sorted_vals, pct):
    if not sorted_vals: return 0.0
    i = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(i); hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (i - lo) * (sorted_vals[hi] - sorted_vals[lo])

if __name__ == "__main__":
    # run directory holding the simulate.py outputs; override with SN83_RUN_DIR
    # or argv[2] rather than editing this file.
    R = os.environ.get("SN83_RUN_DIR",
                       sys.argv[2] if len(sys.argv) > 2
                       else "/home/dev/autoresearch-runs/sn83-picker/")
    if not R.endswith(os.sep):
        R += os.sep
    variants = [("baseline", "final-base-N%d.json"), ("picker", "e2-blind-N%d.json"),
                ("absolute", "abs-blind-N%d.json"),
                ("TAIL", "tail-blind-N%d.json"),
                ("partial", "a3-partial-N%d.json"), ("oracle", "a3-oracle-N%d.json")]
    pct = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    print("  Share of OUR hotkeys at or below the FIELD's %.0fth percentile" % pct)
    print("  (hotkey-level: each hotkey's mean score across rounds. lower is better)\n")
    print("   N  |" + "".join("%12s" % n for n, _ in variants) + "   | our hotkeys")
    tot = collections.defaultdict(list)
    for N in range(10, 170, 10):
        cells = []; nh = 0
        for name, pat in variants:
            p = R + pat % N
            if not os.path.exists(p): cells.append("%12s" % "-"); continue
            hk, pr, cut, no, nf = bottom_share(p, pct)
            tot[name].append(hk); nh = no
            cells.append("%11.1f%%" % (100 * hk))
        print("  %-4d|" % N + "".join(cells) + "   | %d" % nh)
    print("  ----+" + "-" * (12 * len(variants)))
    print("  mean|" + "".join("%11.1f%%" % (100 * statistics.mean(tot[n])) if tot[n] else "%12s" % "-"
                              for n, _ in variants))
    print("\n  a fleet with the field's own score distribution would sit at %.0f%%" % pct)
