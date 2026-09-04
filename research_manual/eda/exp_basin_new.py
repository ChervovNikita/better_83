#!/usr/bin/env python3
"""Reorder the omega pool by BASIN and measure, on the new field.

allocate() spreads one hotkey per clique over the FRONT of the pool, and the pool
arrives ordered by basin descending -- most findable first. Measured on this batch
our answers sit at median rank 0.403 of our own pool while every rival sits at
0.52-0.59, so we alone crowd the most-contested region.

Assignment is held fixed (canonical order from the answer SET plus the round id),
so the comparison isolates SELECTION -- reordering also permutes which hotkey
sends which answer, and that permutation is finite-window noise.
"""
import argparse, hashlib, json, os, sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.dirname(HERE)); sys.path.insert(0,os.path.dirname(os.path.dirname(HERE)))
import paths, simulate, solver, pick_derived
ORDER="none"
_orig=pick_derived.picker

def canonical(uuid,hotkeys,answers,index):
    keyed=sorted(answers,key=lambda a:(index.get(tuple(sorted(a)),1<<30),tuple(sorted(a))))
    off=int(hashlib.sha1(str(uuid).encode()).hexdigest()[:8],16)
    return [list(keyed[(i+off)%len(keyed)]) for i in range(len(hotkeys))]

def picker(pool,uuid,hotkeys,**kw):
    index={tuple(sorted(c)):i for i,c in enumerate(pool)}
    if ORDER!="none":
        hits=kw.get("hits") or [0]*len(pool)
        w=max(len(c) for c in pool)
        hm={tuple(sorted(c)):(hits[i] if i<len(hits) else 0) for i,c in enumerate(pool)}
        tops=[list(c) for c in pool if len(c)==w]; rest=[list(c) for c in pool if len(c)!=w]
        key=lambda c: hm[tuple(sorted(c))]
        if ORDER=="asc": tops=sorted(tops,key=key)
        elif ORDER=="desc": tops=sorted(tops,key=key,reverse=True)
        elif ORDER=="random":
            rng=np.random.RandomState(int(hashlib.sha1(str(uuid).encode()).hexdigest()[:8],16)%2**31)
            tops=[tops[i] for i in rng.permutation(len(tops))]
        elif ORDER.startswith("skip"):
            f=float(ORDER[4:])/100.0
            tops=sorted(tops,key=key,reverse=True)
            k=int(f*len(tops))
            tops=tops[k:]+tops[:k]
        pool=tops+rest
    return canonical(uuid,hotkeys,_orig(pool,uuid,hotkeys,**kw),index)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("-N",type=int,required=True); ap.add_argument("--order",default="none")
    ap.add_argument("--data",required=True); ap.add_argument("--only",required=True)
    ap.add_argument("--pool-cache",required=True); ap.add_argument("--pool-k-mult",type=int,default=8)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    global ORDER; ORDER=a.order
    pick_derived.picker=picker
    solver.configure(fleet_n=a.N,pool_cache=a.pool_cache,pool_k_mult=a.pool_k_mult)
    meta=json.load(open(a.data+"/data/metagraph.json"))
    victims=simulate.pick_victims(meta,a.N)
    rows=simulate.load_rounds(a.data+"/data/rounds.json",100000,a.only)
    out,*_=simulate.run(rows,victims)
    json.dump(out,open(a.out,"w"))

if __name__=="__main__": main()
