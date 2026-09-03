"""Where the data lives."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, "artifacts")

DATA = os.path.join(ARTIFACTS, "data")
CACHE = os.path.join(ARTIFACTS, "cache")
POOLS = os.path.join(ARTIFACTS, "pools")
ROUNDS = os.path.join(ARTIFACTS, "rounds")
SIM_OUT = os.path.join(ARTIFACTS, "sim_out")
PLOTS = os.path.join(ARTIFACTS, "plots")
MISC = os.path.join(ARTIFACTS, "misc")

ROUNDS_JSON = os.path.join(DATA, "rounds.json")
METAGRAPH_JSON = os.path.join(DATA, "metagraph.json")


def rounds_list(name):
    """Resolve a --only round-id list by bare name or by path."""
    return name if os.path.sep in name else os.path.join(ROUNDS, name)
