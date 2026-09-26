import numpy as np


def fbeta(precision, recall, beta=0.5):
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom else 0.0


def entity_fbeta(predicted, truth, beta=0.5):
    p = set(predicted)
    t = set(truth)
    if not p and not t:
        return 1.0
    if not p:
        return 0.0
    precision = len(p & t) / len(p)
    recall = len(p & t) / len(t) if t else 0.0
    return fbeta(precision, recall, beta)


def macro_fbeta(predictions, truth, beta=0.5):
    scores = [entity_fbeta(predictions[k], truth.get(k, []), beta) for k in truth]
    return sum(scores) / len(scores) if scores else 0.0


# =========================================================
# Helpers on aligned candidate arrays (s_idx, t_idx)
# =========================================================

def label_pairs(s_idx, t_idx, truth_pos):
    """1 where target t_idx is a true match of S1 s_idx (truth_pos: list of sets)."""
    return np.fromiter(
        (t in truth_pos[s] for s, t in zip(s_idx.tolist(), t_idx.tolist())),
        dtype=np.int8,
        count=len(s_idx),
    )


def blocking_stats(s_idx, y, n_true, n_target):
    """Recall ceiling and candidate-set size. n_true: true matches per S1 record."""
    n_s1 = len(n_true)
    counts = np.bincount(s_idx, minlength=n_s1)
    total_true = int(n_true.sum())
    return {
        "s1_records": n_s1,
        "true_pairs": total_true,
        "recall": int(y.sum()) / total_true if total_true else 0.0,
        "entities_all_found": float(np.mean(np.bincount(s_idx[y == 1], minlength=n_s1) == n_true)),
        "candidate_pairs": int(len(s_idx)),
        "avg_candidates": float(counts.mean()),
        "median_candidates": float(np.median(counts)),
        "p99_candidates": float(np.percentile(counts, 99)),
        "max_candidates": int(counts.max()),
        "zero_candidate_rate": float((counts == 0).mean()),
        "reduction": 1 - len(s_idx) / (n_s1 * n_target),
    }


def macro_f05_arrays(keep, s_idx, y, n_true):
    """
    Vectorized macro F0.5 over all S1 records (singletons included).

    keep: bool mask of predicted pairs. y: pair labels. n_true: true matches
    per S1 record (including ones blocking missed).
    """
    n = len(n_true)
    pred = np.bincount(s_idx[keep], minlength=n)
    tp = np.bincount(s_idx[keep & (y == 1)], minlength=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(pred > 0, tp / np.maximum(pred, 1), 0.0)
        r = np.where(n_true > 0, tp / np.maximum(n_true, 1), 0.0)
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    f = np.where((pred == 0) & (n_true == 0), 1.0, f)
    return float(f.mean()) if n else 0.0


THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2)


def select_threshold(s_idx, probs, y, n_true, seed=0):
    """
    Returns (best_threshold, f05_at_best, crossfit_f05).

    crossfit_f05 is unbiased: split the S1 records in half, pick the best
    threshold on one half, score it on the other, and average both ways.
    """
    def scan(mask_s):
        sel = mask_s[s_idx]
        new_pos = np.cumsum(mask_s) - 1
        si = new_pos[s_idx[sel]]
        return {
            th: macro_f05_arrays(probs[sel] >= th, si, y[sel], n_true[mask_s])
            for th in THRESHOLDS
        }

    full = scan(np.ones(len(n_true), dtype=bool))
    best = max(full, key=full.get)

    half = np.random.default_rng(seed).random(len(n_true)) < 0.5
    a, b = scan(half), scan(~half)
    crossfit = (b[max(a, key=a.get)] * (~half).sum() + a[max(b, key=b.get)] * half.sum()) / len(n_true)
    return float(best), full[best], float(crossfit)


def print_stats(title, stats):
    print(f"\n{title}")
    print("-" * 60)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k:<24}{v:>20,.4f}")
        else:
            print(f"{k:<24}{v:>20,}")
