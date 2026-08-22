#!/usr/bin/env python3
"""Ground truth: enumerate the COMPLETE set of maximum cliques with Cliquer.

Our LSCC pool holds ~18.7 maxima and 41-47% of what the coordinated operators submit
lies outside it. That means neither side sees the whole set. Cliquer's Russian-doll
search with `--min w --max w` returns EVERY clique of size exactly w, so on instances
where it terminates we get the true count -- turning "we reach 31% of theirs" from an
estimate into an exact measurement, and handing us every clique our search misses.

Caveat measured upstream: enumeration cost explodes with density (a DIMACS instance at
omega=44 exceeded 180 s), so this will time out on our harder rounds. A timeout is
itself informative -- it bounds where ground truth is affordable.
"""
import os
import subprocess
import tempfile

import numpy as np

CL = os.environ.get("SN83_CLIQUER", "/tmp/mc/b/Cliquer/src/cl")


def to_dimacs(A, path):
    n = A.shape[0]
    e = np.transpose(np.nonzero(np.triu(A, 1)))
    with open(path, "w") as f:
        f.write(f"p edge {n} {len(e)}\n")
        f.writelines(f"e {i+1} {j+1}\n" for i, j in e)


def all_maximum(A, omega, timeout_s):
    """Every clique of size exactly omega. Returns (cliques, complete)."""
    A = np.ascontiguousarray(np.asarray(A, dtype=np.uint8))
    fd, path = tempfile.mkstemp(suffix=".clq")
    os.close(fd)
    try:
        to_dimacs(A, path)
        try:
            out = subprocess.run(
                [CL, "-a", "-x", "--min", str(omega), "--max", str(omega), path],
                capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return [], False
        got = set()
        for line in out.stdout.splitlines():
            if line.startswith("size="):
                try:
                    vs = tuple(sorted(int(x) - 1 for x in line.split(":")[1].split()))
                except (IndexError, ValueError):
                    continue
                got.add(vs)
        return [list(v) for v in got], True
    finally:
        if os.path.exists(path):
            os.unlink(path)
