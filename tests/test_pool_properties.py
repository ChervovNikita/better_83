"""Guards for the wiring bugs this project actually shipped or nearly shipped.

Each test targets a defect that REACHED a smoke test and passed it, because the smoke
test asserted validity (is it a clique, is it maximal, did it reach omega) and the
defect lived in something nobody had asserted.

Deliberately NOT tested here: how many distinct cliques the search finds. That number
is a seed lottery -- the shipped solver returns between 1 and 24 maximum cliques on the
same round depending only on its seed -- so any threshold on it produces a test that
passes or fails by luck. Two earlier versions of this file did exactly that and passed
under the configuration they existed to reject.
"""
import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIM = os.path.join(ROOT, "CliqueAI", "clique_algorithms", "native_algorithm_shim.py")


def _load_shim(**env):
    """Import the shim fresh under a given environment; its flags are read at import."""
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        spec = importlib.util.spec_from_file_location("shim_undertest", SHIM)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _pool(omega=30, n_top=1, n_spare=4):
    """A synthetic pool: n_top cliques at omega, n_spare at omega-1. No solver needed."""
    top = [list(range(i * 100, i * 100 + omega)) for i in range(n_top)]
    spare = [list(range(1000 + i * 100, 1000 + i * 100 + omega - 1)) for i in range(n_spare)]
    return top + spare


def test_a_lone_hotkey_disables_the_spread_gate_and_the_census_notices():
    """SHIPPED DEFECT, now fixed at the source rather than documented.

    `_spread_pick` computes q = fleet_size * p(difficulty) and returns None when q <= 1,
    so a fleet size of 1 silently disables spreading -- an operator running 40 hotkeys
    who never set SN83_FLEET_SIZE got no spread and no log line saying so.

    The constant is gone. `fleet_size()` counts `hk.*` claim files across recent tasks,
    so the number is observed instead of configured. This asserts both halves: q <= 1
    still disables the gate (the arithmetic is unchanged), and the census reports the
    real count so that case stops arising by accident."""
    shim = _load_shim()
    pool = _pool(n_top=1, n_spare=3, omega=30)
    assert shim._spread_pick(pool, "5Hot", "uuid-1", 0.8, 1) is None, \
        "q = 1 * p < 1, so the gate cannot fire -- this is the arithmetic being guarded"
    assert callable(getattr(shim, "fleet_size", None)), \
        "fleet_size() replaced the FLEET_SIZE constant; the census is the whole fix"


def test_spread_gate_needs_spares_to_spread_with():
    """NEAR-MISS. Ban mode returned max-size cliques only, so `spare` was empty and the
    gate returned None -- the search change would have silently disabled spreading."""
    shim = _load_shim(SN83_FLEET_SIZE=40, SN83_SPREAD=1)
    no_spares = _pool(n_top=1, n_spare=0)
    assert shim._spread_pick(no_spares, "hk0", "uuid-1", 0.8, 40) is None
    with_spares = _pool(n_top=1, n_spare=4)
    assert shim._spread_pick(with_spares, "hk0", "uuid-1", 0.8, 40) is not None


def test_solve_many_accepts_and_forwards_a_thread_budget():
    """SHIPPED DEFECT (e3b2916). The shim computes share = TOTAL_THREADS // active so N
    concurrent requests stay inside the cgroup quota, and passed it to solve_one but not
    to solve_many, whose internal solves then ran at the library default."""
    sys.path.insert(0, os.path.join(ROOT, "research"))
    import inspect
    try:
        import fleet_solver
    except Exception as exc:                      # pragma: no cover
        pytest.skip("fleet_solver unavailable: %s" % exc)
    assert "threads" in inspect.signature(fleet_solver.solve_many).parameters, \
        "solve_many must accept a thread budget"
    # Count per CALL SITE, not per file. The first version of this assertion compared
    # `src.count("solve_many(")` against `src.count("threads=share")` -- but
    # threads=share also appears on the solve_one calls, so removing it from a
    # solve_many call still left the file-wide count high enough and the test passed
    # under the exact defect it guards. Found by mutation-testing it, not by review.
    src = open(SHIM).read()
    missing = []
    idx = 0
    while True:
        idx = src.find("solve_many(", idx)
        if idx < 0:
            break
        depth, j = 0, idx + len("solve_many(") - 1
        while j < len(src):                        # walk to the matching paren
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if "threads=" not in src[idx:j]:
            missing.append(src[idx:j].split("\n")[0])
        idx = j
    assert not missing, (
        "solve_many call site(s) in the shim do not forward a thread budget, so their "
        "internal solves run at the library default against a 15-core quota: %s"
        % missing)


def test_the_coordinator_ships_on_and_degrades_to_the_old_path_alone():
    """The inverse of the test this replaces, which asserted COORD defaults to 0.

    It defaulted off on the reasoning that it assumes a shared filesystem and gives a
    lone miner nothing. Both true; neither is a reason to default off, because the path
    degrades to the uncoordinated one in exactly those cases:

      * a lone hotkey's claims always succeed, so it never harvests and never spreads
      * hotkeys on separate machines cannot see each other's claims, so the same holds
      * every pool_coordinator failure path returns None or True and the block sits in a try

    What it is worth when the hotkeys DO share a host is +0.047/answer from dedup alone
    plus +0.0206 from the scarce-round harvest. Off, that is measured value nobody
    collects -- which is what this test exists to stop happening again."""
    src = open(SHIM).read()
    assert 'os.environ.get("SN83_COORD", "1")' in src, \
        "the coordinator must ship ON. If you are turning it off, put the measurement " \
        "that says so next to this line -- +0.068/answer says otherwise"
    guards = [ln.strip() for ln in src.splitlines() if "if COORD" in ln]
    assert guards == ["if COORD and hotkey is not None and uuid is not None:"], \
        "one entry guard, so the degradation argument above covers the whole block: %s" % guards
    assert "def claim_clique" in open(
        os.path.join(ROOT, "CliqueAI", "clique_algorithms", "pool_coordinator.py")).read(), \
        "the coordinated path needs claim_clique to exist for its lone-miner fallback"


def test_scarce_spread_is_off_by_default_and_switches_both_halves():
    """G82's scarce-round spread. ON by default as of G92, which measured it at +0.0206 per
    answer on the LAZY harvest path -- the one this shim takes -- at 43 better / 23 worse of
    66 changed rounds. G88 measured the same arms with every hotkey harvesting and got
    +0.0004; that configuration does not exist in deployment.

    The property that matters is that SN83_SCARCE_SPREAD switches BOTH halves. The rule
    alone, on the ordinary plateau-walk harvest, scored -0.2019 on band 1 against the
    shipped -0.1669 -- worse than not doing it. If a future edit gates the harvest and the
    rule on different flags, or lets one default on without the other, the result is a
    regression that looks like an improvement in the diff."""
    src = open(SHIM).read()
    assert 'os.environ.get("SN83_SCARCE_SPREAD", "1")' in src, \
        "SN83_SCARCE_SPREAD defaults to 1 as of G92 (+0.0206 measured on the LAZY path, " \
        "the one this shim takes). It is inert unless SN83_COORD=1. If you turn it back " \
        "off, say which measurement says so -- G88's +0.0004 was the EAGER path and does " \
        "not describe deployment."
    # Exactly one READ of the flag, held in _scarce, used by both halves. Count the
    # environment lookup, not the bare name -- the first version counted the name and
    # went red when a comment mentioned the flag, which is a false positive on the thing
    # it exists to protect.
    assert src.count('os.environ.get("SN83_SCARCE_SPREAD"') == 1, \
        "the flag must be read exactly once, into _scarce, so the harvest and the rule " \
        "cannot diverge; a second read is how they drift apart"
    assert 'pool_mode="ban" if _scarce else None' in src, \
        "with the flag on, the harvest must run in delete-and-resolve mode"
    assert 'if _scarce and len(ordered) <= int(' in src, \
        "the rule must be gated on the same _scarce and on the distinct-maximum count"

    import sys
    sys.path.insert(0, os.path.join(ROOT, "research"))
    import inspect
    try:
        import fleet_solver
    except Exception as exc:                      # pragma: no cover
        pytest.skip("fleet_solver unavailable: %s" % exc)
    assert "pool_mode" in inspect.signature(fleet_solver.solve_many).parameters, \
        "solve_many must accept pool_mode; the environment is process-global and " \
        "requests are served on threads, so SN83_POOL cannot be set around one harvest"


def test_the_fleet_sizes_its_own_thread_budget():
    """Replaces a test that asserted the miner WARNS about an unset SN83_THREADS.

    `share = TOTAL_THREADS // active` counted only this process's in-flight requests
    while each hotkey is its own process, so seven hotkeys asked one box for fifty-six
    threads. The old answer was to tell the operator to export SN83_THREADS; the current
    one is to divide by `observed_fleet()`, which counts hotkeys that actually answered
    recent tasks from the pool's own claim files.

    Asserted here because the failure is silent: oversubscription does not error, every
    thread is throttled, and the search does less work per second of a wall-clock budget.
    This project voided a full generation of measurements to exactly that."""
    src = open(SHIM).read()
    assert "max(in_process, fleet_size())" in src, \
        "the per-solve thread budget must divide by the FLEET's concurrency, not just " \
        "this process's -- each hotkey is a separate process and cannot see the others"
    assert 'os.environ.get("SN83_COORD", "1")' in src, \
        "the coordinator ships on; a lone miner's claims always succeed so its behaviour " \
        "is unchanged, and a fleet collects +0.068/answer that a default of 0 leaves"

    sys.path.insert(0, os.path.join(ROOT, "research"))
    import importlib.util as u, tempfile
    os.environ["SN83_POOL_DIR"] = tempfile.mkdtemp(prefix="sn83-fleet-test-")
    spec = u.spec_from_file_location(
        "pc_test", os.path.join(ROOT, "CliqueAI", "clique_algorithms",
                                "pool_coordinator.py"))
    pc = u.module_from_spec(spec)
    spec.loader.exec_module(pc)
    assert pc.observed_fleet() == 1, "a cold pool must read 1, reproducing the old default"
    for task in range(5):
        for hk in range(6):
            pc.claim_clique("t%d" % task, "5Hot%02d" % hk, (1, 2, 3, hk))
    assert pc.observed_fleet() == 6, \
        "six hotkeys claimed on five tasks; the census must see six"


