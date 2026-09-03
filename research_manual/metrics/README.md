# metrics

Scoring a simulator run.  Each script reads the JSON that `simulate.py --out` writes
and reduces it to one number; none of them re-implement the validator's scorer, which
is why the numbers agree with the simulator's own output.

    metric_edge.py    our_average - field_average, the mean-score objective
    metric_bottom.py  share of our hotkeys at or below the FIELD-only 10th percentile
                      (the cut excludes our hotkeys: including them lets a uniformly
                      bad fleet drag the threshold down under itself)
    metric_share.py   emission share, via simulate.validator_weights -- imported, not
                      reimplemented, because the weighting is a sigmoid then a power
                      transform and share is RELATIVE to the total

`simulate.py` stays at the top of research_manual/: it is the run, not a reading of it.
