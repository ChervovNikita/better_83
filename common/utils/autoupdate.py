import subprocess
import traceback
import time

import bittensor as bt


def run_cmd(*args) -> str:
    """Run a command synchronously and return stdout as string."""
    try:
        result = subprocess.run(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        return result.stdout.decode().strip()
    except subprocess.CalledProcessError as e:
        raise Exception(f"Command {' '.join(args)} failed: {e.stderr.decode().strip()}")


def get_local_hash() -> str:
    return run_cmd("git", "rev-parse", "HEAD")


def get_tracking_ref() -> str:
    """The upstream this checkout actually tracks, e.g. origin/main.

    Hardcoding origin/main makes autoupdate a restart loop on ANY other branch: the
    hashes can never match, so update_repo_if_needed returns True every pass, the run
    loop raises KeyboardInterrupt, the process exits, the supervisor restarts it, and it
    repeats every 12 seconds without ever serving a request. Anyone deploying a fork, a
    patched miner, or a release branch hits this.
    """
    try:
        return run_cmd("git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    except Exception:
        return "origin/main"


def get_remote_hash(branch=None) -> str:
    run_cmd("git", "fetch")
    return run_cmd("git", "rev-parse", branch or get_tracking_ref())


def update_repo():
    run_cmd("git", "pull")


def update_repo_if_needed() -> bool:
    """Returns True if repo was updated and restart is needed, else False."""
    try:
        local_hash = get_local_hash()
        remote_hash = get_remote_hash()

        # A detached or untracked checkout has nothing to compare against; updating it
        # would discard whatever it was deliberately pinned to.
        if not remote_hash:
            bt.logging.info("No upstream branch; skipping autoupdate.")
            return False

        if local_hash != remote_hash:
            bt.logging.info(f"Update available: {local_hash} -> {remote_hash}")
            bt.logging.info("Updating repository...")
            update_repo()
            bt.logging.info("Repository updated. Please restart the process.")
            return True
        else:
            bt.logging.info("No updates available.")
            return False

    except Exception as e:
        bt.logging.error(f"Error checking for updates: {e}")
        bt.logging.error(traceback.format_exc())
        time.sleep(60) # sleep 1 min
        return True
