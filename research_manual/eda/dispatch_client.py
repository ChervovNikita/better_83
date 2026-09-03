import json
import os
import urllib.error
import urllib.request

URL = os.environ.get("SN83_DISPATCH_URL", "http://127.0.0.1:8899")


def solve(uuid, hotkey, n_nodes, matrix, time_limit, url=None):
    """A clique from the dispatcher, or None if it cannot supply one."""
    url = (url or URL).rstrip("/") + "/solve"
    payload = json.dumps({
        "uuid": str(uuid),
        "hotkey": str(hotkey),
        "number_of_nodes": int(n_nodes),
        "time_limit": float(time_limit),
        "adjacency_matrix": [list(map(int, row)) for row in matrix],
    }).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=max(0.1, float(time_limit))) as resp:
            body = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    clique = body.get("clique")
    if not clique:
        return None
    return [int(v) for v in clique]


def health(url=None):
    """Dispatcher status, or None. For monitoring, never in the request path."""
    url = (url or URL).rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
