Superseded pickers, kept for reference only. Nothing imports them.

    pick_value.py   previous default; the baseline in the comparison tables
    pick_static.py  fixed-rule picker
    fleet_pick.py   the original picker

solver.py no longer selects a picker: it calls pick_derived directly.

    pick_probes.py  research probes from the floor study (lambda sweep, maximin,
                    clique selection, free-cap, feedback assignment). All falsified
                    against picker(); numbers in ~/autoresearch-runs/sn83-floor.
                    Archival only: they call the pre-simplification allocate()/eval_J
                    signatures, so they import but will not run. Working versions are
                    at commit 76cf42c.
