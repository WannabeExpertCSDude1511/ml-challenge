import argparse
import json
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .blocking import DEFAULT_K, BlockingIndex
from .data import load_split, load_truth, split_s1, truth_positions
from .evaluate import blocking_stats, label_pairs, print_stats, select_threshold
from .features import FEATURE_NAMES, compute_features


def create_model(model_type, y_train):
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError(
                "The 'xgboost' package is not installed. "
                "Install it via 'pip install xgboost' or run with '--model-type histgb'."
            )

        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except ImportError:
            pass

        pos_count = float(np.sum(y_train))
        neg_count = float(len(y_train) - pos_count)
        scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0

        return XGBClassifier(
            n_estimators=250,
            learning_rate=0.08,
            max_depth=6,
            reg_lambda=1.0,
            tree_method="hist",
            device=device,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
        )
    elif model_type == "histgb":
        return HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Choose 'histgb' or 'xgboost'.")


def sample_negatives(s_idx, y, per_s1, seed=0):
    """Keep every positive and at most per_s1 random negatives per S1 record."""
    order = np.lexsort((np.random.default_rng(seed).random(len(s_idx)), y, s_idx))
    s_sorted, y_sorted = s_idx[order], y[order]
    neg = y_sorted == 0
    # rank of each negative within its S1 group
    group_start = np.r_[0, np.flatnonzero(np.diff(s_sorted)) + 1]
    first = np.repeat(group_start, np.diff(np.r_[group_start, len(s_sorted)]))
    rank = np.arange(len(s_sorted)) - first
    keep = ~neg | (rank < per_s1)
    return np.sort(order[keep])


QUERY_CHUNK = 25_000


def candidates_for(s1, positions, target, truth, index, k):
    part = s1.iloc[positions].reset_index(drop=True)
    ids = part["entity_id"].tolist()
    # Query in chunks so blocking memory stays flat for large samples.
    s_parts, t_parts, detail_parts = [], [], []
    for start in range(0, len(part), QUERY_CHUNK):
        s, t, details = index.query(part.iloc[start:start + QUERY_CHUNK].reset_index(drop=True), k,
                                    return_details=True)
        s_parts.append(s + start)
        t_parts.append(t)
        detail_parts.append(details)
    s_idx = np.concatenate(s_parts)
    t_idx = np.concatenate(t_parts)
    details = {key: np.concatenate([d[key] for d in detail_parts]) for key in detail_parts[0]}
    truth_pos, n_true = truth_positions(ids, truth, target)
    y = label_pairs(s_idx, t_idx, truth_pos)
    return part, s_idx, t_idx, details, y, n_true


def main():
    ap = argparse.ArgumentParser(description="Train and evaluate the entity resolution classifier.")
    ap.add_argument("--data", default="dataset/train", help="Path to training data directory")
    ap.add_argument("--model", default="model.joblib", help="Output model joblib path")
    ap.add_argument("--model-type", choices=["histgb", "xgboost"], default="histgb")
    ap.add_argument("--holdout", type=float, default=0.2, help="Fraction of S1 entities held out for evaluation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-size", type=int, default=10000, help="S1 records used (train + holdout); 0 = all")
    ap.add_argument("--neg-per-s1", type=int, default=100, help="Max negative candidates per S1 used for training")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="Blocking: top-k per ranking (saved with the model)")
    args = ap.parse_args()

    started = time.perf_counter()
    s1, target = load_split(args.data, "train")
    train_pos, hold_pos = split_s1(len(s1), args.holdout, args.seed, args.sample_size)
    truth = load_truth(args.data, s1["entity_id"].take(np.r_[train_pos, hold_pos]).tolist())
    print(f"S1 train: {len(train_pos):,}  S1 holdout: {len(hold_pos):,}  targets: {len(target):,}", flush=True)

    index = BlockingIndex(target)

    # ---------------- train ----------------
    part, s_idx, t_idx, details, y, n_true = candidates_for(s1, train_pos, target, truth, index, args.k)
    # Features on the full candidate sets (group features compare a candidate
    # with all of its S1 record's candidates), then sample negatives.
    keep = sample_negatives(s_idx, y, args.neg_per_s1, args.seed)
    X = compute_features(part, target, s_idx, t_idx, details)
    if len(keep) < len(X):
        X = X[keep]
    y_train = y[keep]
    if y_train.sum() == 0:
        raise RuntimeError("No positive training pairs were found inside the candidate set.")
    print(f"Training on {len(X):,} pairs ({int(y_train.sum()):,} positive)", flush=True)
    model = create_model(args.model_type, y_train)
    model.fit(X, y_train)
    del X

    report = {"train_blocking": blocking_stats(s_idx, y, n_true, len(target))}
    del part, s_idx, t_idx, details, y, keep, y_train

    # ---------------- evaluate on holdout ----------------
    threshold = 0.5
    if len(hold_pos):
        part, s_idx, t_idx, details, y, n_true = candidates_for(s1, hold_pos, target, truth, index, args.k)
        X = compute_features(part, target, s_idx, t_idx, details)
        probs = model.predict_proba(X)[:, 1] if len(X) else np.empty(0)
        del X
        threshold, f05_best, f05_crossfit = select_threshold(s_idx, probs, y, n_true, args.seed)
        report["holdout_blocking"] = blocking_stats(s_idx, y, n_true, len(target))
        report["holdout_f05"] = {
            "threshold": threshold,
            "f05_at_threshold": f05_best,
            "f05_crossfit": f05_crossfit,
        }

    report["runtime_seconds"] = round(time.perf_counter() - started, 1)

    for name, stats in report.items():
        if isinstance(stats, dict):
            print_stats(name, stats)
    print(f"\nruntime_seconds: {report['runtime_seconds']}")

    joblib.dump(
        {
            "model": model,
            "features": FEATURE_NAMES,
            "model_type": args.model_type,
            "threshold": threshold,
            "k": args.k,
            "report": report,
        },
        args.model,
    )
    print(json.dumps({"model": args.model, "threshold": threshold}))


if __name__ == "__main__":
    main()
