"""Miner-side client for the solve dispatcher.

Every function here returns None on ANY failure. Nothing in this module may
raise into the axon request path: an exception there costs the round, whereas
None costs a worse clique from the miner's local fallback.
"""

import json
import os
import urllib.error
import urllib.request

URL = os.environ.get("SN83_DISPATCH_URL", "http://127.0.0.1:8899")


def solve(uuid, hotkey, n_nodes, matrix, time_limit, url=None,
          encoded_matrix="", timeout=None):
    """A clique from the dispatcher, or None if it cannot supply one.

    Pass `encoded_matrix` in preference to `matrix`: it is the base92 string the
    synapse already carries, so the miner skips a decode the dispatcher has to
    do anyway, and the localhost payload drops from ~2.4 MB to ~62 KB at n=900.

    `timeout` is the HTTP deadline and defaults to `time_limit`. Set it SHORTER
    than the round's budget when the caller wants time left to solve locally
    after a dispatcher that never answers.
    """
    url = (url or URL).rstrip("/") + "/solve"
    body = {
        "uuid": str(uuid),
        "hotkey": str(hotkey),
        "number_of_nodes": int(n_nodes),
        "time_limit": float(time_limit),
    }
    if encoded_matrix:
        body["encoded_matrix"] = str(encoded_matrix)
    else:
        body["adjacency_matrix"] = [list(map(int, row)) for row in (matrix or [])]
    try:
        payload = json.dumps(body).encode()
    except (TypeError, ValueError):
        return None
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"})
    deadline = float(timeout if timeout is not None else time_limit)
    try:
        with urllib.request.urlopen(req, timeout=max(0.1, deadline)) as resp:
            body = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    clique = body.get("clique")
    if not clique:
        return None
    try:
        return [int(v) for v in clique]
    except (TypeError, ValueError):
        return None


def health(url=None):
    """Dispatcher status, or None. For monitoring, never in the request path."""
    url = (url or URL).rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
