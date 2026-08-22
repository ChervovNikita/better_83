"""Turn the fleet sweep into a decision. The simulator reports GROSS alpha/day.

fleet_sim prints `a/day` = weight_share x MINER_ALPHA_DAY and says so plainly:
"a/day is GROSS - no registration burn, no rate limit, so it is not a break-even
figure." This supplies the missing half, so the sweep can be read as pay/don't-pay.

Costs a fleet of N actually carries:
  * N x recycle, burned at registration, not recoverable
  * the chain admits ~1 registration per 360-block interval (~20/day), so standing
    up N hotkeys takes ceil(N/20) days during which they earn nothing
  * anything evicted must be re-registered to keep the slot

Everything here is a live chain value or comes from the sweep's own JSON.
"""
import json, os, sys
import numpy as np

RECYCLE_TAO = 0.142934258     # measured from chain, subnet 83
ALPHA_TAO   = 0.0106          # matches fleet_sim's ALPHA_TAO
REGS_PER_DAY = 20.0

res_path = sys.argv[1] if len(sys.argv) > 1 else "data/fleet_sim_results.json"
if not os.path.exists(res_path):
    print(f"no sweep yet at {res_path}; run the pipeline first"); sys.exit(2)
# Refuse a results file older than the cache it should have been produced from.
# The pre-fix run left one behind and it read as a plausible sweep -- the same
# stale-artifact hazard flagged in the simulator audit.
cache = os.path.join(os.path.dirname(res_path) or ".", "sim_ts.jsonl")
if os.path.exists(cache) and os.path.getmtime(res_path) < os.path.getmtime(cache):
    import time as _t
    print(f"STALE: {res_path} ({_t.ctime(os.path.getmtime(res_path))}) predates the "
          f"solve cache ({_t.ctime(os.path.getmtime(cache))}).")
    print("It is from an earlier run; refusing to report it. Re-run the simulate stage.")
    sys.exit(3)
rows = json.load(open(res_path))

print(f"{'N':>4} {'share':>9} {'a/day':>9} {'TAO/day':>9} {'reg cost':>10} "
      f"{'payback':>9} {'informative':>12}")
for r in rows:
    N = r["N"]; share = r["share"]
    a_day = share * 2951.6
    tao_day = a_day * ALPHA_TAO
    cost = N * RECYCLE_TAO
    days = cost / tao_day if tao_day > 0 else float("inf")
    pay = f"{days:8.1f}d" if days < 1e4 else "   never"
    flag = "yes" if r.get("informative") else "INSIDE NULL"
    print(f"{N:>4} {share:>8.3%} {a_day:>9.1f} {tao_day:>9.4f} "
          f"t{cost:>9.4f} {pay:>9} {flag:>12}")

print(f"\nchain: recycle t{RECYCLE_TAO}/hotkey, alpha t{ALPHA_TAO}, "
      f"~{REGS_PER_DAY:.0f} registrations/day network-wide")
print("payback ignores the ramp: the chain admits ~1 registration per 360-block")
print("interval, so a fleet of N takes ~N/20 days to stand up, earning nothing")
print("meanwhile. Any row marked INSIDE NULL has no measured skill and its payback")
print("is not a real number -- the share it rests on is a noise draw.")
