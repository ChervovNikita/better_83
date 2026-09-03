# research_manual

The solution, and the harness that measures it.

    simulate.py     the only source of truth for a number (runs the validator's own scorer)
    solver.py       dispatch: picks a picker (SN83_PICKER) and a solver (SN83_SOLVER)
    paths.py        where the data lives -- change the layout here, not in every script

    pick_derived.py    the picker: picker() plus picker_oracle()/picker_partial(),
                       the position-aware bounds used in the comparison tables

    fleet_solver_gpu.py  GPU harvest (default)
    gpu_lib.py           build + ctypes bridge for clique_gpu.cu
    clique_gpu.cu        the CUDA solver
    fleet_solver.py      CPU harvest (SN83_SOLVER=cpu)
    clique.cpp           the CPU solver

    metrics/    scoring a run: edge, bottom-10%, emission share (see metrics/README.md)
    outdated/   superseded pickers and falsified probes; nothing imports them
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

    .venv/bin/python research_manual/simulate.py -N 70 --rounds 100000 \
      --only latest100.txt \
      --pool-cache research_manual/artifacts/cache/cache_latest100.jsonl \
      --out /tmp/o.json

`-N` is the fleet size, and `--only` takes a bare name resolved against
artifacts/rounds/.  The first run without `--pool-cache` harvests on the GPU and is
slow; with a cache every run sees an identical pool, which is what makes a comparison
paired.  There are no SN83_* environment variables any more -- simulate.py asserts if
one is set, so an old script fails loudly instead of silently running unconfigured.
