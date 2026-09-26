"""
Candidate generation (blocking), shared by train.py, predict.py and
test_blocking.py.

generate_candidates(s1, target) takes the normalized frames from
data.load_normalized() and returns two aligned int arrays (s_idx, t_idx):
row positions into s1 and target, sorted by s_idx then t_idx.
"""
import time
from collections import defaultdict

import numpy as np

from .normalize import digits, trigrams

FUZZY_BLOCK_THRESHOLD = 0.60
SCAN_CHUNK = 200_000


def _columns(frame, start=0, stop=None):
    part = frame.iloc[start:stop]
    return zip(*(part[c].tolist() for c in ("country", "name", "core", "addr")))


def _keys(country, name, core, addr):
    # Empty values are not keys: they would put every record missing that
    # field into one giant block.
    keys = [(kind, country, value) for kind, value in (("name", name), ("core", core), ("addr", addr)) if value]
    keys += [("num", country, n) for n in digits(addr)]
    keys += [("tok", country, t) for t in set(name.split()) if len(t) >= 4]
    return keys


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _fuzzy_fallback(queries, target):
    """Same-country name-trigram Jaccard >= threshold, one pass over target."""
    by_country = defaultdict(list)
    for i, country, grams in queries:
        by_country[country].append((i, grams))

    found = defaultdict(list)
    for start in range(0, len(target), SCAN_CHUNK):
        for off, (country, name, _, _) in enumerate(_columns(target, start, start + SCAN_CHUNK)):
            qs = by_country.get(country)
            if not qs:
                continue
            grams = trigrams(name)
            if not grams:
                continue
            for i, s_grams in qs:
                if _jaccard(s_grams, grams) >= FUZZY_BLOCK_THRESHOLD:
                    found[i].append(start + off)
    return found


def generate_candidates(s1, target, verbose=True):
    start_time = time.perf_counter()

    s_rows = list(_columns(s1))
    s_keys = [_keys(*r) for r in s_rows]
    wanted = {k for keys in s_keys for k in keys}

    # Inverted index restricted to keys that some S1 record actually uses.
    postings = defaultdict(list)
    for start in range(0, len(target), SCAN_CHUNK):
        for off, rec in enumerate(_columns(target, start, start + SCAN_CHUNK)):
            for k in _keys(*rec):
                if k in wanted:
                    postings[k].append(start + off)
    postings = {k: np.asarray(v, dtype=np.int32) for k, v in postings.items()}

    empty = np.empty(0, dtype=np.int32)
    hits = []
    fallback = []
    for i, keys in enumerate(s_keys):
        arrays = [postings[k] for k in keys if k in postings]
        h = np.unique(np.concatenate(arrays)) if arrays else empty
        hits.append(h)
        if not len(h):
            grams = trigrams(s_rows[i][1])
            if grams:
                fallback.append((i, s_rows[i][0], grams))

    if fallback:
        for i, found in _fuzzy_fallback(fallback, target).items():
            hits[i] = np.asarray(sorted(found), dtype=np.int32)

    s_idx = np.repeat(np.arange(len(hits), dtype=np.int32), [len(h) for h in hits])
    t_idx = np.concatenate(hits) if hits else empty

    if verbose:
        print(
            f"Blocking: {len(s_idx):,} candidate pairs for {len(s1):,} S1 records "
            f"({len(fallback):,} used the fuzzy fallback) in {time.perf_counter() - start_time:.0f}s",
            flush=True,
        )
    return s_idx, t_idx
