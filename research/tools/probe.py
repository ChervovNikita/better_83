"""Probe a solve_many variant: pool size, validity, and field-rarity of what it finds."""
import ctypes, json, os, sys, collections
import numpy as np
sys.path.insert(0, '/workspace/better_83')
sys.path.insert(0, '/workspace/better_83/research')
from CliqueAI.graph.codec import GraphCodec

def load(so):
    lib = ctypes.CDLL(so)
    lib.sn83_solve_many.restype = ctypes.c_int
    lib.sn83_solve_many.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ctypes.c_double, ctypes.c_uint64, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
    return lib

def solve(lib, M, tl, threads, want, seed=12345):
    n = M.shape[0]
    buf = np.ascontiguousarray(M, dtype=np.uint8)
    out = (ctypes.c_int32 * (want * n))()
    szs = (ctypes.c_int32 * want)()
    k = lib.sn83_solve_many(buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), n,
                            tl, seed, threads, want, out, szs)
    res, off = [], 0
    for i in range(k):
        s = szs[i]; res.append(tuple(sorted(out[off:off+s]))); off += s
    return res

def valid(M, c):
    for i in range(len(c)):
        for j in range(i+1, len(c)):
            if not M[c[i], c[j]]: return False
    return True

def maximal(M, c):
    s = set(c)
    for v in range(M.shape[0]):
        if v in s: continue
        if all(M[v, u] for u in c): return False
    return True

if __name__ == "__main__":
    so, env, nrounds, tl_cap = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    for kv in env.split(","):
        if "=" in kv:
            k, v = kv.split("="); os.environ[k] = v
    lib = load(so)
    cd = GraphCodec()
    rounds = []
    for line in open('data/sim_rounds.jsonl'):
        r = json.loads(line)
        if r.get('answers'): rounds.append(r)
    rounds.sort(key=lambda r: r['uuid'])
    rounds = rounds[:nrounds]
    tot_pool = tot_bad = tot_nonmax = 0
    sole = held = 0
    pools = {}
    for r in rounds:
        M = np.array(cd.decode_matrix(r['matrix_b92']), dtype=np.uint8)
        tl = min(float(r.get('time_limit') or 10.0), tl_cap)
        cl = solve(lib, M, tl, int(os.environ.get('SN83_THREADS', '7')), 40)
        pools[r['uuid']] = cl
        mx = max((len(c) for c in cl), default=0)
        for c in cl:
            if not valid(M, c): tot_bad += 1
            elif not maximal(M, c): tot_nonmax += 1
        tot_pool += len({c for c in cl if len(c) == mx})
        cnt = collections.Counter()
        for a in r['answers']:
            t = tuple(sorted(map(int, a['clique'])))
            if t: cnt[t] += 1
        fmx = max((len(c) for c in cnt), default=0)
        for c in {c for c in cl if len(c) == mx}:
            if c in cnt:
                held += 1; sole += (cnt[c] == 1)
    print(json.dumps({"so": os.path.basename(so), "env": env, "rounds": len(rounds),
        "maxsize_pool_per_round": round(tot_pool/max(len(rounds),1), 2),
        "invalid": tot_bad, "non_maximal": tot_nonmax,
        "found_by_field": held, "of_those_sole_held": sole,
        "sole_share": round(sole/held, 4) if held else None}))
    json.dump({u: [list(c) for c in v] for u, v in pools.items()},
              open(os.environ.get('PROBE_OUT', '/dev/null'), 'w'))
