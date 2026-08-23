"""Score saved pools with the reward reference, PAIRED against the first file.

Pool ORDER is the point: a solver holding the right cliques but submitting them in the
wrong order earns nothing extra, so this scores the first-K prefix. Comparisons are
paired within round and tested with a two-sided sign test over rounds where the
prefix actually CHANGED -- unpaired means across arms are not admissible here.
"""
import collections, json, math, statistics as st, sys
sys.path.insert(0, '/workspace/better_83/research')
from reward_reference import replay_reward, count_hist_from_answers

def sign_p(diffs):
    m = len(diffs)
    if m == 0:
        return 1.0
    b = sum(1 for x in diffs if x > 0)
    lo = min(b, m - b)
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(lo + 1)) / 2 ** m)

K = int(sys.argv[1])
rounds = {}
for line in open('/workspace/better_83/research/data/sim_rounds.jsonl'):
    r = json.loads(line)
    if r.get('answers'):
        rounds[r['uuid']] = r

def score(path):
    out = {}
    for u, cl in json.load(open(path)).items():
        r = rounds.get(u)
        if not r or not cl:
            continue
        ans = r['answers']
        cnt = collections.Counter()
        for a in ans:
            if a.get('opt', 0) > 0:
                cnt[tuple(sorted(map(int, a['clique'])))] += 1
        if not cnt:
            continue
        sh = collections.Counter(len(a['clique']) for a in ans if a.get('opt', 0) > 0)
        ch = count_hist_from_answers(ans)
        ninv = sum(1 for a in ans if a.get('opt', 0) <= 0)
        mx = max(len(c) for c in cnt)
        ours = [tuple(sorted(c)) for c in cl if len(c) == mx]
        if not ours:
            continue
        pre = ours[:K]
        rw = [replay_reward(mx, cnt.get(c, 0), sh, ch, r['difficulty'], ninv)[0]
              for c in pre]
        out[u] = (st.mean(rw), sum(1 for c in ours if c not in cnt) / len(ours),
                  len(ours), frozenset(pre))
    return out

base_path = sys.argv[2]
base = score(base_path)
print("K=%d, paired within round, sign test over CHANGED prefixes only\n" % K)
print("%-30s %6s %8s %9s %7s %9s %8s"
      % ("pool", "rnds", "mean", "delta", "novel%", "better/chg", "sign p"))
for path in sys.argv[2:]:
    cur = score(path)
    common = sorted(set(cur) & set(base))
    if not common:
        print("%-30s   no overlap" % path.split('/')[-1]); continue
    m = st.mean(cur[u][0] for u in common)
    nv = 100 * st.mean(cur[u][1] for u in common)
    if path == base_path:
        print("%-30s %6d %8.4f %9s %6.1f%% %9s %8s"
              % (path.split('/')[-1], len(common), m, "--", nv, "--", "--"))
        continue
    diffs = [cur[u][0] - base[u][0] for u in common if cur[u][3] != base[u][3]]
    diffs = [d for d in diffs if abs(d) > 1e-12]
    b = sum(1 for d in diffs if d > 0)
    print("%-30s %6d %8.4f %+9.4f %6.1f%% %5d/%-4d %8.4f"
          % (path.split('/')[-1], len(common), m,
             m - st.mean(base[u][0] for u in common), nv, b, len(diffs), sign_p(diffs)))
