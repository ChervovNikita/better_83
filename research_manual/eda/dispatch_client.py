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


def _post_solve(uuid, hotkey, n_nodes, matrix, time_limit, url=None,
                encoded_matrix="", timeout=None):
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
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def solve(uuid, hotkey, n_nodes, matrix, time_limit, url=None,
          encoded_matrix="", timeout=None):
    body = _post_solve(uuid, hotkey, n_nodes, matrix, time_limit, url=url,
                       encoded_matrix=encoded_matrix, timeout=timeout)
    if not body:
        return None
    clique = body.get("clique")
    if not clique:
        return None
    try:
        return [int(v) for v in clique]
    except (TypeError, ValueError):
        return None


def solve_source(uuid, hotkey, n_nodes, matrix, time_limit, url=None,
                 encoded_matrix="", timeout=None):
    """Like solve(), plus whether this hotkey was the owner."""
    body = _post_solve(uuid, hotkey, n_nodes, matrix, time_limit, url=url,
                       encoded_matrix=encoded_matrix, timeout=timeout)
    if not body:
        return None, None
    clique = body.get("clique")
    if not clique:
        return None, body.get("source")
    try:
        return [int(v) for v in clique], body.get("source")
    except (TypeError, ValueError):
        return None, body.get("source")


async def solve_source_async(uuid, hotkey, n_nodes, matrix, time_limit, url=None,
                             encoded_matrix="", timeout=None):
    """solve_source() as a coroutine: the axon's loop waits, no thread does.

    `timeout` bounds the WHOLE exchange (connect, send, response), where the
    urllib form's timeout applies per socket operation.
    """
    import aiohttp  # bittensor's own transport dependency

    url = (url or URL).rstrip("/") + "/solve"
    body = {
        "uuid": str(uuid),
        "hotkey": str(hotkey),
        "number_of_nodes": int(n_nodes),
        "time_limit": float(time_limit),
    }
    try:
        if encoded_matrix:
            body["encoded_matrix"] = str(encoded_matrix)
        else:
            body["adjacency_matrix"] = [list(map(int, row)) for row in (matrix or [])]
        deadline = float(timeout if timeout is not None else time_limit)
        client_timeout = aiohttp.ClientTimeout(total=max(0.1, deadline))
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(url, json=body) as resp:
                payload = await resp.json(content_type=None)
    except Exception:  # noqa: BLE001 -- nothing may raise into the axon path
        return None, None
    if not isinstance(payload, dict):
        return None, None
    clique = payload.get("clique")
    if not clique:
        return None, payload.get("source")
    try:
        return [int(v) for v in clique], payload.get("source")
    except (TypeError, ValueError):
        return None, payload.get("source")


def task_progress(uuid, url=None):
    """claims = started waiting; finished = already left /solve. None on miss."""
    url = (url or URL).rstrip("/") + "/task/" + str(uuid)
    try:
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            body = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    try:
        return {
            "claims": int(body.get("claims") or 0),
            "finished": int(body.get("finished") or 0),
            "done": bool(body.get("done")),
        }
    except (TypeError, ValueError):
        return None


def started_waiting(uuid, url=None):
    """How many of our miners have already posted /solve for this uuid."""
    info = task_progress(uuid, url=url)
    if info is None:
        return None
    return info["claims"]


def health(url=None):
    """Dispatcher status, or None. For monitoring, never in the request path."""
    url = (url or URL).rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
