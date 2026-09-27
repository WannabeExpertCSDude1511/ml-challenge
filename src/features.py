"""
Pair features, computed in parallel chunks over aligned candidate arrays.

Inputs are the normalized frames from data.load_normalized(), so no text is
re-normalized per pair.

Three groups:
- pair features: string similarity of the two records (computed in workers);
- blocking features: the candidate's blocking rank and re-rank similarities;
- group features: how a candidate compares with the other candidates of the
  same S1 record (gap to the best, rank, group size). They tell the model
  whether a candidate stands out, which matters for precision.
"""
import re

import numpy as np
from joblib import Parallel, delayed
from rapidfuzz.fuzz import partial_ratio, ratio, token_set_ratio, token_sort_ratio
from rapidfuzz.process import cpdist

from .data import N_JOBS
from .normalize import digits, trigrams

PAIR_FEATURES = [
    "country_equal",
    "name_exact",
    "name_core_exact",
    "name_ratio",
    "name_core_ratio",
    "name_token_set_ratio",
    "name_jaccard",
    "name_char_jaccard",
    "address_exact",
    "address_ratio",
    "address_jaccard",
    "address_char_jaccard",
    "numeric_overlap",
    "name_len_diff",
    "address_len_diff",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "postcode_equal",
    "postcode_conflict",
    "house_number_equal",
    "house_number_conflict",
    "s1_address_empty",
    "target_address_empty",
]

BLOCKING_FEATURES = ["block_rank", "rerank_name_sim", "rerank_addr_sim", "rerank_sum"]

GROUP_FEATURES = [
    "name_sim_gap",
    "addr_sim_gap",
    "sum_gap",
    "sum_rank_in_group",
    "group_size",
]

FEATURE_NAMES = PAIR_FEATURES + BLOCKING_FEATURES + GROUP_FEATURES

FEATURE_CHUNK = 25_000
POSTCODE = re.compile(r"\b\d{5,6}\b")
NUMBER = re.compile(r"\d+")


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _memo(fn):
    """Per-chunk cache: S1 values repeat for all of a record's candidates."""
    cache = {}

    def get(value):
        if value not in cache:
            cache[value] = fn(value)
        return cache[value]

    return get


def _postcode(addr):
    found = POSTCODE.findall(addr)
    return found[-1] if found else ""


def _house_number(addr):
    found = NUMBER.search(addr)
    return found.group() if found else ""


def _equal_conflict(a_values, b_values):
    equal = [bool(a) and a == b for a, b in zip(a_values, b_values)]
    conflict = [bool(a) and bool(b) and a != b for a, b in zip(a_values, b_values)]
    return equal, conflict


def _chunk_features(sc, tc, sn, tn, sco, tco, sa, ta):
    X = np.empty((len(sn), len(PAIR_FEATURES)), dtype=np.float32)
    s_tokens, s_grams, s_digits = _memo(lambda v: set(v.split())), _memo(trigrams), _memo(digits)
    s_post, s_house = _memo(_postcode), _memo(_house_number)

    X[:, 0] = [a == b for a, b in zip(sc, tc)]
    X[:, 1] = [a == b for a, b in zip(sn, tn)]
    X[:, 2] = [a == b for a, b in zip(sco, tco)]
    X[:, 3] = cpdist(sn, tn, scorer=ratio) / 100.0
    X[:, 4] = cpdist(sco, tco, scorer=ratio) / 100.0
    X[:, 5] = cpdist(sn, tn, scorer=token_set_ratio) / 100.0
    X[:, 6] = [jaccard(s_tokens(a), set(b.split())) for a, b in zip(sn, tn)]
    X[:, 7] = [jaccard(s_grams(a), trigrams(b)) for a, b in zip(sn, tn)]
    X[:, 8] = [a == b for a, b in zip(sa, ta)]
    X[:, 9] = cpdist(sa, ta, scorer=ratio) / 100.0
    X[:, 10] = [jaccard(s_tokens(a), set(b.split())) for a, b in zip(sa, ta)]
    X[:, 11] = [jaccard(s_grams(a), trigrams(b)) for a, b in zip(sa, ta)]
    X[:, 12] = [jaccard(s_digits(a), digits(b)) for a, b in zip(sa, ta)]
    X[:, 13] = [abs(len(a) - len(b)) for a, b in zip(sn, tn)]
    X[:, 14] = [abs(len(a) - len(b)) for a, b in zip(sa, ta)]
    X[:, 15] = cpdist(sn, tn, scorer=partial_ratio) / 100.0
    X[:, 16] = cpdist(sn, tn, scorer=token_sort_ratio) / 100.0
    X[:, 17], X[:, 18] = _equal_conflict([s_post(a) for a in sa], [_postcode(b) for b in ta])
    X[:, 19], X[:, 20] = _equal_conflict([s_house(a) for a in sa], [_house_number(b) for b in ta])
    X[:, 21] = [not a for a in sa]
    X[:, 22] = [not b for b in ta]
    return X


def _gather(frame, idx, column):
    return frame[column].take(idx).tolist()


def _group_features(s_idx, name_sim, addr_sim):
    """Features relative to the other candidates of the same S1 record (s_idx sorted)."""
    total = name_sim + addr_sim
    starts = np.flatnonzero(np.r_[True, s_idx[1:] != s_idx[:-1]])
    sizes = np.diff(np.r_[starts, len(s_idx)])

    def gap(values):
        return np.repeat(np.maximum.reduceat(values, starts), sizes) - values

    order = np.lexsort((-total, s_idx))
    rank = np.empty(len(s_idx), dtype=np.float32)
    rank[order] = np.arange(len(s_idx)) - np.repeat(starts, sizes)
    return np.column_stack([gap(name_sim), gap(addr_sim), gap(total), rank, np.repeat(sizes, sizes)])


def compute_features(s1, target, s_idx, t_idx, blocking, n_jobs=N_JOBS):
    """
    Return a float32 matrix (len(s_idx) x len(FEATURE_NAMES)).

    s_idx must be sorted (as returned by BlockingIndex.query). blocking is the
    dict of per-pair arrays from BlockingIndex.query(..., return_details=True).
    """
    if not len(s_idx):
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    def args(a, b):
        si, ti = s_idx[a:b], t_idx[a:b]
        out = []
        for column in ("country", "name", "core", "addr"):
            out += [_gather(s1, si, column), _gather(target, ti, column)]
        return out

    # Filled in place as chunks arrive, so peak memory is ~1x the matrix.
    X = np.empty((len(s_idx), len(FEATURE_NAMES)), dtype=np.float32)
    starts = range(0, len(s_idx), FEATURE_CHUNK)
    chunks = Parallel(n_jobs=n_jobs, pre_dispatch="n_jobs", return_as="generator")(
        delayed(_chunk_features)(*args(a, a + FEATURE_CHUNK)) for a in starts
    )
    n_pair = len(PAIR_FEATURES)
    for a, chunk in zip(starts, chunks):
        X[a:a + len(chunk), :n_pair] = chunk

    name_sim, addr_sim = blocking["name_sim"], blocking["addr_sim"]
    n_block = n_pair + len(BLOCKING_FEATURES)
    X[:, n_pair:n_block] = np.column_stack([blocking["rank"], name_sim, addr_sim, name_sim + addr_sim])
    X[:, n_block:] = _group_features(s_idx, name_sim, addr_sim)
    return X
