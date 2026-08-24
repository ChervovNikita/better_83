"""Exercise the DEPLOYED path in the configuration RESULTS.md recommends.

Rule 6 of the standing orders: verify the deployed path before optimising the research
path. Every coordinator number in this project comes from a Python re-implementation of
the claim logic in `research/`. This calls `native_algorithm` -- the function
`CliqueAI/miner.py` calls -- with the coordinator on, a real fleet size, a shared pool
directory and the scarce-round spread enabled, on logged graphs.

It found a real defect on its first run: the shim passed `pool_mode="ban"` but left
`ban_n` at its default of 3, so the re-solve aimed three vertices below omega, the harvest
returned almost no omega-1 spares, and the spread rule fell through to repeating a
sibling's answer. Three hotkeys returned ONE distinct clique on a round with a single
maximum. With `ban_n=1` and `champion_share=0.35` -- the values G82 actually measured --
the same round returns sizes [24, 23, 23], three distinct answers: one hotkey at omega and
two spread below it, which is the mechanism working.
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "research", "data", "sim_rounds.jsonl")


def _load_shim():
    """By PATH, not by package. `from CliqueAI.clique_algorithms import ...` runs the
    package __init__, which imports the GNN module and therefore torch -- an optional
    dependency this solver does not need and CI may not have."""
    import importlib.util as u
    spec = u.spec_from_file_location(
        "shim_under_test",
        os.path.join(ROOT, "CliqueAI", "clique_algorithms", "native_algorithm_shim.py"))
    mod = u.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _is_maximal_clique(A, verts):
    vs = list(verts)
    if not vs or len(set(vs)) != len(vs):
        return False
    for a in range(len(vs)):
        for b in range(a + 1, len(vs)):
            if not A[vs[a]][vs[b]]:
                return False
    s = set(vs)
    return not any(v not in s and all(A[v][u] for u in vs) for v in range(A.shape[0]))


@pytest.mark.slow
def test_deployed_path_answers_validly_with_the_coordinator_on():
    if not os.path.exists(DATA):
        pytest.skip("logged rounds not present")
    sys.path.insert(0, os.path.join(ROOT, "research"))
    for k, v in (("SN83_COORD", "1"), ("SN83_FLEET_SIZE", "7"),
                 ("SN83_SCARCE_SPREAD", "1"), ("SN83_SCARCE_MAX_ND", "2"),
                 ("SN83_THREADS", "2")):
        os.environ[k] = v
    os.environ["SN83_POOL_DIR"] = tempfile.mkdtemp(prefix="sn83-test-")
    try:
        from CliqueAI.graph.codec import GraphCodec
    except Exception as exc:                       # pragma: no cover
        pytest.skip("codec unavailable: %s" % exc)
    shim = _load_shim()

    done = 0
    for i, line in enumerate(open(DATA)):
        if done >= 2:
            break
        rec = json.loads(line)
        if rec["n"] > 320:
            continue
        done += 1
        A = np.array(GraphCodec().decode_matrix(rec["matrix_b92"]), dtype=np.uint8)
        adj = {v: {u for u in range(rec["n"]) if A[v][u]} for v in range(rec["n"])}
        uuid = "test-%d" % i
        answers = []
        for h in range(2):
            hk = "5TestHotkey%02d" % h
            c = shim.native_algorithm(
                rec["n"], adj, adjacency_matrix=A, timeout=3.0, fallback=lambda: [0],
                seed=shim.solver_seed(hk, uuid), hotkey=hk, uuid=uuid,
                difficulty=rec["difficulty"])
            assert _is_maximal_clique(A, c), \
                "round %d hotkey %s returned a non-maximal clique of size %d" % (i, hk, len(c))
            answers.append(tuple(sorted(c)))
        # Not asserted: that the two differ. Dedup only acts when they collide, and on a
        # rich round two independent solves usually differ anyway -- an assertion here
        # would pass for the wrong reason. The distinctness claim is measured in G74/G82,
        # not asserted on two rounds.
    assert done == 2, "expected two small rounds in the logged data"
