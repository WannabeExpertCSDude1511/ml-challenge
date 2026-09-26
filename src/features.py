"""
Pair features, computed in parallel chunks over aligned candidate arrays.

Inputs are the normalized frames from data.load_normalized(), so no text is
re-normalized per pair.
"""
import numpy as np
from joblib import Parallel, delayed
from rapidfuzz.fuzz import ratio, token_set_ratio
from rapidfuzz.process import cpdist

from .normalize import digits, trigrams

FEATURE_NAMES = [
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
]

FEATURE_CHUNK = 100_000


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _chunk_features(sc, tc, sn, tn, sco, tco, sa, ta):
    X = np.empty((len(sn), len(FEATURE_NAMES)), dtype=np.float32)
    X[:, 0] = [a == b for a, b in zip(sc, tc)]
    X[:, 1] = [a == b for a, b in zip(sn, tn)]
    X[:, 2] = [a == b for a, b in zip(sco, tco)]
    X[:, 3] = cpdist(sn, tn, scorer=ratio) / 100.0
    X[:, 4] = cpdist(sco, tco, scorer=ratio) / 100.0
    X[:, 5] = cpdist(sn, tn, scorer=token_set_ratio) / 100.0
    X[:, 6] = [jaccard(set(a.split()), set(b.split())) for a, b in zip(sn, tn)]
    X[:, 7] = [jaccard(trigrams(a), trigrams(b)) for a, b in zip(sn, tn)]
    X[:, 8] = [a == b for a, b in zip(sa, ta)]
    X[:, 9] = cpdist(sa, ta, scorer=ratio) / 100.0
    X[:, 10] = [jaccard(set(a.split()), set(b.split())) for a, b in zip(sa, ta)]
    X[:, 11] = [jaccard(trigrams(a), trigrams(b)) for a, b in zip(sa, ta)]
    X[:, 12] = [jaccard(digits(a), digits(b)) for a, b in zip(sa, ta)]
    X[:, 13] = [abs(len(a) - len(b)) for a, b in zip(sn, tn)]
    X[:, 14] = [abs(len(a) - len(b)) for a, b in zip(sa, ta)]
    return X


def _gather(frame, idx, column):
    return frame[column].take(idx).tolist()


def compute_features(s1, target, s_idx, t_idx, n_jobs=-1):
    """Return a float32 matrix (len(s_idx) x len(FEATURE_NAMES))."""
    if not len(s_idx):
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    def args(a, b):
        si, ti = s_idx[a:b], t_idx[a:b]
        out = []
        for column in ("country", "name", "core", "addr"):
            out += [_gather(s1, si, column), _gather(target, ti, column)]
        return out

    chunks = Parallel(n_jobs=n_jobs)(
        delayed(_chunk_features)(*args(a, a + FEATURE_CHUNK))
        for a in range(0, len(s_idx), FEATURE_CHUNK)
    )
    return np.vstack(chunks)
