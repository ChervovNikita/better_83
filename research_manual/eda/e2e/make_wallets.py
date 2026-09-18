#!/usr/bin/env python3
"""Throwaway wallets for an end-to-end run: one validator, K miner hotkeys.

    python make_wallets.py <wallet-dir> <K>

Idempotent -- only missing hotkeys are created, so run_e2e.sh can call it on
every run and a larger K reuses what is already there. These keys are for the
local harness only; nothing here is ever registered on chain.

`create_new_hotkey` PROMPTS when the hotkey already exists, which hangs a
non-interactive run, so existing ones are skipped by path rather than by
catching the overwrite.
"""
import os
import sys

import bittensor as bt


def main(path, k):
    val = bt.Wallet(name="e2e_val", hotkey="default", path=path)
    val.create_if_non_existent(coldkey_use_password=False,
                               hotkey_use_password=False, suppress=True)
    miner = bt.Wallet(name="e2e_miner", hotkey="hk000", path=path)
    miner.create_if_non_existent(coldkey_use_password=False,
                                 hotkey_use_password=False, suppress=True)
    hotkeys = os.path.join(path, "e2e_miner", "hotkeys")
    made = 0
    for i in range(1, k):
        name = "hk%03d" % i
        if os.path.exists(os.path.join(hotkeys, name)):
            continue
        bt.Wallet(name="e2e_miner", hotkey=name, path=path).create_new_hotkey(
            use_password=False, overwrite=False, suppress=True)
        made += 1
    print("validator %s; %d hotkeys (%d new)"
          % (val.hotkey.ss58_address, k, made), file=sys.stderr)
    return val.hotkey.ss58_address


if __name__ == "__main__":
    print(main(sys.argv[1], int(sys.argv[2])))
