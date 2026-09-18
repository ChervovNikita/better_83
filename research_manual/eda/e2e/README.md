# End-to-end harness

`simulate.py` measures the reward. This measures everything around it: the
miner process, its axon, the HTTP hop to the dispatcher, sibling batching, the
picker's deadline, and the rare-query alert -- on **production code**, with the
validator's own transport sending the rounds.

It exists because an audit found the losses were never in the solver. At fleet
100 the dispatcher's waiting siblings exhausted anyio's thread pool and whole
rounds missed their deadline; at 150+ the picker starved the event loop. Both
scored fine in `simulate.py` and failed here.

## Run

    K=100 research_manual/eda/e2e/run_e2e.sh
    .venv/bin/python research_manual/eda/e2e/e2e_analyze.py ../better_83_e2e/run1

~20 min for K=160. Needs a GPU, the built native libraries (the dispatcher's
start script builds them), and `research_manual/artifacts/data/rounds.json`
plus `metagraph.json`. `SN83_WORKERS` defaults to the GPU count, so a one-GPU
pod works; the 4-GPU numbers in the repo came from a 4xA4000 box.

Output goes OUTSIDE the repo (`OUT`, default `../better_83_e2e`): a K=160 run
writes ~160 miner logs plus per-round events.

| env | default | |
|---|---|---|
| `K` | 100 | miner processes, one hotkey each |
| `RUN` | run1 | output subdirectory |
| `OUT` | `../better_83_e2e` | where runs and wallets live |
| `PORT0` | 19000 | first axon port; K consecutive ports are used |
| `NREAL` / `NSTRESS` / `NSCEN` | 100 / 60 / 3 | rounds per phase |
| `GAPREAL` / `GAPSTRESS` | 8 / 1.5 | mean seconds between rounds |
| `SN83_*` | see `deploy/sn83.env` | passed to the dispatcher |

## Pieces

| file | |
|---|---|
| `e2e_miner.py` | one production `CliqueAI.miner.Miner` behind a real `bt.Axon`. Only the chain is stubbed: no subtensor, a one-entry metagraph holding the test validator, so the production blacklist/priority/forward paths run unchanged. Instrumentation wraps, never edits. |
| `e2e_validator.py` | sends `MaximumCliqueOfLambdaGraph` through `CliqueAI.transport.axon_requester`, signed by a real dendrite. Rounds arrive with exponential gaps so they overlap; how many of our hotkeys a round queries is drawn Binomial(K, selection_p(D)) as `MinerSelector` draws it. `--scenarios` adds the scripted alert cases (S1-S6). |
| `run_e2e.sh` | starts the dispatcher and K miners, runs the three phases, stops everything. |
| `e2e_analyze.py` | reduces a run: validity by the validator's own `is_valid_maximum_clique`, latency per phase and time limit, local fallbacks, sibling duplicates, dispatcher counters, and the alert contract (expected vs fired vs early). |
| `make_wallets.py` | throwaway local keys, idempotent. Never registered on chain. |

## Phases

- **realistic** -- rounds at the field's own rate. Nothing should fall back to
  a local solve; latency stays inside the round's budget.
- **stress** -- rounds several deep, past anything observed, to find where the
  service degrades and confirm it degrades by rejecting rather than by
  answering late.
- **scenarios** -- the rare-query alert: fires once per round, only on
  `tl >= 10`, only when the binomial left tail is below `RARE_QUERY_P`, and
  only after every sibling of that round has finished. S6 (a sibling arriving
  2.6 s late, after the owner answered) is a known early fire; the fleet's own
  arrival spread was measured at <= 0.13 s.

## Reading the output

What a clean run looks like (fleet 160, 4 GPUs, after the audit fixes):

    empty dispatcher answer -> local fallback: 0
    latency > 3.0s: 0
    invalid (validator check): 0
    expected ... 13   fired: 13   missed: 0   unexpected: 0

`below field-best size` is not a defect: it counts answers smaller than the
best clique anyone found that round, which is normal for the hotkeys the
picker deliberately places at omega-1.
