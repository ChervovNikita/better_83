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


def test_fleet_size_one_silently_disables_the_spread_gate():
    """SHIPPED DEFECT. SN83_FLEET_SIZE defaults to 1, which makes q=1, which makes
    `len(top) >= q` always true, so _spread_pick returns None on every round. An
    operator running 40 hotkeys gets no spread and no log line saying so."""
    shim = _load_shim(SN83_FLEET_SIZE=1, SN83_SPREAD=1)
    got = shim._spread_pick(_pool(), "hk0", "uuid-1", 0.8, shim.FLEET_SIZE)
    assert got is None, "unexpected: the gate fired at fleet_size=1"

    shim40 = _load_shim(SN83_FLEET_SIZE=40, SN83_SPREAD=1)
    picks = {tuple(shim40._spread_pick(_pool(), "hk%d" % h, "uuid-1", 0.8,
                                       shim40.FLEET_SIZE) or ())
             for h in range(8)}
    assert len(picks) > 1, (
        "at fleet_size=40 on a pool holding ONE maximum clique, eight hotkeys must not "
        "all return the same vertex set -- that is diversity 1/N")
    assert any(len(p) == 29 for p in picks), (
        "no hotkey dropped to omega-1; the spread gate is not selecting spares")


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


def test_the_coordinator_is_off_unless_three_variables_are_exported():
    """DORMANT MECHANISM, found 2026-08-24 by auditing the deployed path rather than the
    research harness. `COORD = int(os.environ.get("SN83_COORD", "0"))` and that one flag
    guards the WHOLE coordinator block -- the lazy harvest, the claim dedup, the
    agreement gate, and the partial spread with SN83_PARTIAL_THR inside it. Measured at
    +0.0725/answer at seven hotkeys and +0.0486 at fourteen, and nothing in the repo,
    start_miner.sh, .env or the shell sets it, so none of it has ever run in deployment.

    This test cannot make the operator export the variables. It makes the requirement
    executable, so a reader who changes the default -- in either direction -- has to come
    here and say so."""
    src = open(SHIM).read()
    assert 'os.environ.get("SN83_COORD", "0")' in src, \
        "SN83_COORD no longer defaults to 0; update RESULTS.md step 5b, which tells the " \
        "operator the coordinator is dead code until they export it"
    assert 'os.environ.get("SN83_FLEET_SIZE", "1")' in src, \
        "SN83_FLEET_SIZE no longer defaults to 1; see the FLEET_SIZE=1 test above"
    # The guard must be a single `if COORD and ...` -- if the block is ever split so that
    # part of it runs without COORD, the activation instructions become wrong.
    guards = [ln.strip() for ln in src.splitlines() if "if COORD" in ln]
    assert guards == ["if COORD and hotkey is not None and uuid is not None:"], \
        "the coordinator's entry guard changed; RESULTS.md documents exactly one: %s" % guards

    # And nothing in the repo turns it on, which is the actual finding.
    import subprocess
    hits = subprocess.run(
        ["git", "grep", "-l", "SN83_COORD", "--", ".", ":!research", ":!tests"],
        cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert hits == ["CliqueAI/clique_algorithms/native_algorithm_shim.py"], \
        "something outside research/ and tests/ now mentions SN83_COORD -- if a launcher " \
        "sets it, RESULTS.md step 5b should stop saying nothing does: %s" % hits


def test_scarce_spread_is_off_by_default_and_switches_both_halves():
    """G82's scarce-round spread is worth +0.0297 over the shipped policy on the 17.4% of
    rounds that are scarce (95 better / 5 worse of 100). It is OFF by default because it
    is measured only on bands 1 and 2-3 and costs -0.56 if it fires on band 8+.

    The property that matters is that SN83_SCARCE_SPREAD switches BOTH halves. The rule
    alone, on the ordinary plateau-walk harvest, scored -0.2019 on band 1 against the
    shipped -0.1669 -- worse than not doing it. If a future edit gates the harvest and the
    rule on different flags, or lets one default on without the other, the result is a
    regression that looks like an improvement in the diff."""
    src = open(SHIM).read()
    assert 'os.environ.get("SN83_SCARCE_SPREAD", "0")' in src, \
        "SN83_SCARCE_SPREAD must default to 0; it is measured on scarce rounds only"
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
