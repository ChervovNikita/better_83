# research_manual

The solution, and the harness that measures it.

    simulate.py     the only source of truth for a number (runs the validator's own scorer)
    solver.py       dispatch: picks a picker (SN83_PICKER) and a solver (SN83_SOLVER)
    paths.py        where the data lives -- change the layout here, not in every script

    pick_derived.py    the shipped picker, derived from the reward algebra
    pick_value.py      previous production picker (the baseline in every comparison)
    pick_static.py     fixed-rule picker
    fleet_pick.py      legacy picker

    fleet_solver_gpu.py  GPU harvest (default)
    gpu_lib.py           build + ctypes bridge for clique_gpu.cu
    clique_gpu.cu        the CUDA solver
    fleet_solver.py      CPU harvest (SN83_SOLVER=cpu)
    clique.cpp           the CPU solver

    metrics/    scoring a run: edge, bottom-10%, emission share (see metrics/README.md)
    eda/        exploration: probes, one-off analyses, notebooks, dispatcher, docs
    artifacts/  everything that is data rather than code

## artifacts/

    data/     rounds.json, metagraph.json -- the inputs
    cache/    pool caches keyed (uuid, k); pin the harvest so picker runs are paired
    pools/    raw harvest dumps
    rounds/   round-id lists for `--only`
    sim_out/  simulator outputs
    plots/    figures
    misc/     tuning sets, timing, field features

Nothing under artifacts/ is tracked (`*.json`, `*.jsonl`, `*.so`, `*.png` are ignored).

## Running

    SN83_FLEET_N=70 SN83_POOL_CACHE=research_manual/artifacts/cache/cache_latest100.jsonl \
      SN83_PICKER=pick_derived:picker .venv/bin/python research_manual/simulate.py \
      -N 70 --rounds 100000 --only research_manual/artifacts/rounds/latest100.txt --out /tmp/o.json

The first run without `SN83_POOL_CACHE` harvests on the GPU and is slow; with a cache
every picker sees an identical pool, which is what makes the comparison paired.
