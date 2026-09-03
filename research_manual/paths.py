"""Where the data lives.

The solution code sits at the top of research_manual/; everything it reads is under
artifacts/, split by kind.  Keeping the layout in one module means a reorganisation
touches this file rather than every script that opens a dataset.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, "artifacts")

DATA = os.path.join(ARTIFACTS, "data")      # rounds.json, metagraph.json
CACHE = os.path.join(ARTIFACTS, "cache")    # pool caches keyed (uuid, k)
POOLS = os.path.join(ARTIFACTS, "pools")    # raw harvest dumps
ROUNDS = os.path.join(ARTIFACTS, "rounds")  # round-id lists for --only
SIM_OUT = os.path.join(ARTIFACTS, "sim_out")
PLOTS = os.path.join(ARTIFACTS, "plots")
MISC = os.path.join(ARTIFACTS, "misc")

ROUNDS_JSON = os.path.join(DATA, "rounds.json")
METAGRAPH_JSON = os.path.join(DATA, "metagraph.json")


def rounds_list(name):
    """Resolve a --only round-id list by bare name or by path."""
    return name if os.path.sep in name else os.path.join(ROUNDS, name)
