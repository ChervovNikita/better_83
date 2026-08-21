"""Shared plumbing for the SN83 research tooling."""
import collections
import json
import os
import sys

PROJECT = os.environ.get("SN83_WANDB_PROJECT", "toptensor-ai/CliqueAI")
DEFAULT_VERSIONS = ["0.0.17"]

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESEARCH = os.path.join(REPO, "research")
DATA_DIR = os.environ.get("SN83_DATA_DIR", os.path.join(RESEARCH, "data"))

if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Every field we need. Never request `adjacency_list`: it is redundant with
# encoded_matrix and large rows make the history endpoint return HTTP 500.
KEYS = ["_step", "uuid", "number_of_nodes", "difficulty", "time_limit",
        "encoded_matrix", "miner_ans", "miner_uids", "miner_hotkeys",
        "miner_coldkeys", "miner_optimality", "miner_diversity", "miner_rewards"]

# wandb >= 0.20 reroutes Run.scan_history through a service API that pulls the
# whole run history before yielding anything. Our runs hold ~23k steps of
# adjacency lists, so it hangs indefinitely — measured against 18 rows/s on
# 0.17.0. Fail loudly at startup instead of letting cron time out forever.
WANDB_MAX_MINOR = (0, 20)


def check_wandb_version():
    import wandb
    try:
        major, minor = (int(x) for x in wandb.__version__.split(".")[:2])
    except ValueError:
        return
    if (major, minor) >= WANDB_MAX_MINOR:
        raise RuntimeError(
            f"wandb {wandb.__version__} is too new for scan_history: it downloads "
            f"the entire run history and never returns. Install the pin: "
            f"pip install -r research/requirements.txt")


def popcount_edges(b92, n):
    """Edge count read straight off the base92 payload, no n x n matrix."""
    from CliqueAI.graph.codec import GraphCodec
    c = GraphCodec()
    body = b92[c.HDR_DIGITS:]
    total_bits = n * (n - 1) // 2
    full, rem = divmod(total_bits, c.CHUNK_BITS)
    pos, bits = 0, 0
    for _ in range(full):
        bits += c._dec_fixed_to_int(body[pos:pos + c.CHUNK_DIGITS]).bit_count()
        pos += c.CHUNK_DIGITS
    if rem:
        m = c._min_digits_for_bits(rem)
        bits += c._dec_fixed_to_int(body[pos:pos + m]).bit_count()
    return bits


def row_to_record(row, keep_answers=False):
    """One logged round -> one dataset instance.

    The label is `best_size`, the largest valid clique any miner returned.
    `best_cliques` is the set of DISTINCT optima the field converged on, which
    is the signal an anti-collision picker needs: our answer should come from
    outside that cluster.
    """
    ans = row.get("miner_ans") or []
    opt = row.get("miner_optimality") or []
    n = row["number_of_nodes"]
    valid = [a for a, o in zip(ans, opt) if o > 0]
    best = max((len(a) for a in valid), default=0)
    counts = collections.Counter(tuple(sorted(a)) for a in valid)
    best_sets = sorted(t for t in counts if len(t) == best)
    size_hist = collections.Counter(len(a) for a in valid)
    edges = popcount_edges(row["encoded_matrix"], n)
    rec = {
        "uuid": row["uuid"],
        "n": n,
        "edges": edges,
        "density": round(2 * edges / (n * (n - 1)), 5) if n > 1 else 0.0,
        "time_limit": row["time_limit"],
        "difficulty": row["difficulty"],
        "matrix_b92": row["encoded_matrix"],
        "best_size": best,
        "n_responders": len(ans),
        "n_valid": len(valid),
        "n_at_best": sum(1 for a in valid if len(a) == best),
        "best_cliques": [list(t) for t in best_sets],
        # sufficient statistics to replay validator scoring for a new answer
        "best_clique_counts": [counts[t] for t in best_sets],
        "size_hist": {str(k): v for k, v in sorted(size_hist.items())},
        # {duplicate_count: n_distinct_valid_cliques}. Needed to normalise the
        # diversity term exactly: the normaliser is the best delta over ALL valid
        # answers, so it depends on the full count multiset, not just best-size ones.
        "count_hist": {str(k): v for k, v in
                       sorted(collections.Counter(counts.values()).items())},
        "any_unique": any(c == 1 for c in counts.values()),
    }
    if keep_answers:
        # hotkey identifies the NEURON; a hotkey change at a uid is a
        # re-registration, which is how the simulator tracks live churn.
        hks = row.get("miner_hotkeys") or [None] * len(ans)
        rec["answers"] = [
            {"uid": u, "hk": h, "ck": c, "clique": a, "opt": o, "div": d, "reward": r}
            for u, h, c, a, o, d, r in zip(
                row.get("miner_uids", []), hks, row.get("miner_coldkeys", []), ans, opt,
                row.get("miner_diversity", []), row.get("miner_rewards", []))
        ]
    return rec


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)          # atomic, so a killed cron run can't corrupt state


def discover_runs(api, versions):
    """Runs matching `versions`, filtered SERVER-SIDE.

    Never enumerate the project unfiltered: `api.runs()` materialises every
    run's summary, and each summary carries that run's last logged row —
    adjacency_list included. Unfiltered discovery measured 107.6s against 1.7s
    with the filter.
    """
    runs = list(api.runs(PROJECT, filters={"config.version": {"$in": list(versions)}},
                         per_page=100))
    return sorted(runs, key=lambda r: r.created_at)
