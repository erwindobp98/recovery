#!/usr/bin/env python3
"""
Streaming multiprocessing BIP39 recovery (up to 4 missing words).
- Avoids MemoryError by streaming batches (no huge lists).
- Checkpointing/resume by offset index.
- Uses ProcessPoolExecutor for CPU-bound work.
WARNING: Prints private keys. Use only for wallets you OWN.
"""

import os
import time
import json
import math
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from typing import List
from pathlib import Path

from mnemonic import Mnemonic
from bip_utils import Bip39SeedGenerator, Bip44, Bip44Coins, Bip44Changes
from eth_account import Account

from colorama import init as colorama_init, Fore, Style

# ---------------- CONFIG ----------------
# partial phrase: 12 tokens, use "?" for missing words
partial_phrase = "also naive cream notice lounge mask olympic grocery slush arrive spoil ?"

# target address (lowercase)
target_address = "0xf7c1e38dbc725a69d18922a1ce0c3c0ae62c154d".lower()

# runtime tuning
PROCESS_WORKERS = max(1, (os.cpu_count() or 4) - 1)   # leave one core free
BATCH_SIZE = 5000           # candidate phrases per process job (tune)
MAX_IN_FLIGHT = PROCESS_WORKERS * 2  # max submitted but not completed jobs
CHECKPOINT_FILE = "recover_checkpoint.json"
REPORT_INTERVAL = 5.0       # seconds between progress prints

# derivation (standard m/44'/60'/0'/0/0)
ACCOUNT_INDEX = 0
CHANGE = Bip44Changes.CHAIN_EXT
ADDRESS_INDEX = 0
# --------------------------------------

colorama_init(autoreset=True)

mnemo = Mnemonic("english")
bip39_words = mnemo.wordlist  # 2048 words

# --- prepare words / missing positions ---
words = partial_phrase.split()
if len(words) != 12:
    raise SystemExit("[ERROR] partial_phrase harus 12 kata (pakai '?' untuk kata hilang).")

missing_indexes = [i for i, w in enumerate(words) if w == "?"]
num_missing = len(missing_indexes)
if num_missing == 0:
    raise SystemExit("[ERROR] Tidak ada '?' dalam partial_phrase.")
if num_missing > 4:
    raise SystemExit("[ERROR] Skrip dioptimalkan hingga 4 kata hilang. Untuk >4 kata gunakan solusi khusus/C++.")

# use full BIP39 for each missing position
clues_lists = [bip39_words] * num_missing

# total combinations
radices = [len(lst) for lst in clues_lists]
total_combinations = 1
for r in radices:
    total_combinations *= r

print(f"[INFO] Missing positions: {missing_indexes}")
print(f"[INFO] Kandidat per posisi: {radices}")
print(f"[INFO] Total kombinasi: {total_combinations:,}")

# precompute radices_prod: factor = product(radices[i+1:])
radices_prod = []
for i in range(len(radices)):
    prod = 1
    for r in radices[i+1:]:
        prod *= r
    radices_prod.append(prod)  # same length as radices

# ---------------- checkpoint helpers ----------------
def load_checkpoint(path: str):
    if not os.path.exists(path):
        return {"offset": 0, "checked": 0}
    with open(path, "r") as f:
        return json.load(f)

def save_checkpoint(path: str, state: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)

# ---------------- worker (subprocess) ----------------
# top-level function so it can be pickled
def process_batch(start_index: int, count: int,
                  words_list: List[str], missing_idx: List[int],
                  clues: List[List[str]], target_addr: str):
    """
    Process a batch of 'count' candidate indices starting at start_index.
    Returns (found_flag, found_data_or_None, checked_count).
    """
    from mnemonic import Mnemonic as _Mnemonic
    from bip_utils import Bip39SeedGenerator as _Gen, Bip44 as _B44, Bip44Coins as _Coins, Bip44Changes as _Changes
    from eth_account import Account as _Account

    local_mnemo = _Mnemonic("english")
    checked = 0

    # local references for speed
    radices_local = [len(l) for l in clues]
    radices_prod_local = []
    for i in range(len(radices_local)):
        prod = 1
        for r in radices_local[i+1:]:
            prod *= r
        radices_prod_local.append(prod)

    for offset in range(count):
        idx = start_index + offset
        if idx >= math.prod(radices_local):
            break
        # compute digits for mixed-radix index
        digits = []
        rem = idx
        for r, factor in zip(radices_local, radices_prod_local):
            digit = (rem // factor) % r if factor != 0 else rem % r
            digits.append(digit)
        # build phrase
        phrase_words = words_list[:]
        for pos, digit in zip(missing_idx, digits):
            phrase_words[pos] = clues[missing_idx.index(pos)][digit]
        phrase = " ".join(phrase_words)
        checked += 1

        # BIP39 checksum check
        if not local_mnemo.check(phrase):
            continue

        # derive address from mnemonic (BIP44 ETH)
        seed_bytes = _Gen(phrase).Generate()
        acc = _B44.FromSeed(seed_bytes, _Coins.ETHEREUM).Purpose().Coin().Account(ACCOUNT_INDEX).Change(_Changes.CHAIN_EXT).AddressIndex(ADDRESS_INDEX)
        priv_hex = acc.PrivateKey().Raw().ToHex()
        addr = _Account.from_key(bytes.fromhex(priv_hex)).address.lower()

        if addr == target_addr:
            return True, {"phrase": phrase, "priv": priv_hex, "address": addr, "index": idx}, checked

    return False, None, checked

# ---------------- inline progress printer ----------------
def format_time(s: float) -> str:
    if s == float("inf"):
        return "inf"
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"

def print_progress_inline(checked: int, total: int, rate: float, elapsed: float, eta: float):
    """
    Print a single-line, in-place progress with percentage, bar, counts, rate, ETA.
    Uses colorama for colors and pads/clears the rest of line.
    """
    pct = (checked / total) * 100 if total > 0 else 0.0
    # progress bar (width 30)
    bar_width = 30
    filled = int((pct / 100.0) * bar_width)
    bar = "[" + ("#" * filled).ljust(bar_width) + "]"

    # choose color by pct
    if pct < 50:
        color = Fore.YELLOW
    else:
        color = Fore.GREEN

    checked_str = f"{checked:,}/{total:,}"
    rate_str = f"{rate:.1f}/s"
    elapsed_str = format_time(elapsed)
    eta_str = format_time(eta)

    msg = (f"{color}{bar} {pct:6.2f}%{Style.RESET_ALL} "
           f"checked={checked_str} rate={rate_str} elapsed={elapsed_str} ETA={eta_str}")
    # ensure we overwrite previous content: \r then pad with spaces
    term_width = 120
    padded = msg.ljust(term_width)
    print("\r" + padded, end="", flush=True)

# ---------------- driver ----------------
def main():
    checkpoint = load_checkpoint(CHECKPOINT_FILE)
    offset = checkpoint.get("offset", 0)
    checked_total = checkpoint.get("checked", 0)
    start_time = time.time()
    last_report = 0.0  # force immediate print
    print(f"[INFO] Resuming from offset {offset:,}, checked so far {checked_total:,}")

    found = None

    # we'll iterate start positions lazily to avoid huge lists
    with ProcessPoolExecutor(max_workers=PROCESS_WORKERS) as exe:
        futures_map = {}  # fut -> start_index
        # function to submit next job
        def submit_job(start):
            fut = exe.submit(process_batch, start, BATCH_SIZE, words, missing_indexes, clues_lists, target_address)
            futures_map[fut] = start
            return fut

        # seed initial jobs
        next_start = offset
        while len(futures_map) < MAX_IN_FLIGHT and next_start < total_combinations:
            submit_job(next_start)
            next_start += BATCH_SIZE

        # main loop: wait for any to complete, then submit more
        while futures_map:
            done, _ = wait(futures_map.keys(), return_when=FIRST_COMPLETED)
            for fut in list(done):
                start_idx = futures_map.pop(fut)
                try:
                    ok, data, cnt = fut.result()
                except Exception as e:
                    # print warning on its own line (force newline first)
                    print()  # move to new line before WARN
                    print(f"{Fore.RED}[WARN]{Style.RESET_ALL} Job at {start_idx} failed with exception: {e}")
                    ok, data, cnt = False, None, 0

                checked_total += cnt
                # persist checkpoint: next offset is start_idx + BATCH_SIZE (best-effort)
                checkpoint = {"offset": max(next_start, start_idx + BATCH_SIZE), "checked": checked_total}
                save_checkpoint(CHECKPOINT_FILE, checkpoint)

                now = time.time()
                elapsed = now - start_time
                rate = checked_total / elapsed if elapsed > 0 else 0
                remaining = max(0, total_combinations - checked_total)
                eta = remaining / rate if rate > 0 else float("inf")

                # print inline progress if interval passed (or immediate first time)
                if now - last_report >= REPORT_INTERVAL or last_report == 0.0:
                    print_progress_inline(checked_total, total_combinations, rate, elapsed, eta)
                    last_report = now

                if ok:
                    found = data
                    # reset checkpoint supaya tidak lanjut lagi next run
                    save_checkpoint(CHECKPOINT_FILE, {"offset": 0, "checked": 0})
                    # move to new line for FOUND block
                    print()
                    print(f"{Fore.CYAN}[FOUND]{Style.RESET_ALL} Seed phrase ditemukan!")
                    # cancel pending futures
                    for pending in list(futures_map.keys()):
                        pending.cancel()
                    break

                # submit next job if available
                if next_start < total_combinations:
                    submit_job(next_start)
                    next_start += BATCH_SIZE

            if found:
                break
    
    elapsed_total = time.time() - start_time
    print(f"[DONE] elapsed {elapsed_total:.1f}s - checked {checked_total:,}")
    if found:
        print("Seed phrase :", found['phrase'])
        print("Address     :", found['address'])
        print("Private key :", found['priv'])
        print("\n*** Immediately verify manually on an offline/secure machine.")
    else:
        print("Not found in provided search space (or not reached yet).")

if __name__ == "__main__":
    main()
