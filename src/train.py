"""
Train business entity resolution classifier.

Pipeline:
  1. Load training data (source1, source2, source3, ground_truth)
  2. Run multi-strategy blocking → candidate pairs
  3. Compute 33+ pair features
  4. Train LightGBM (or XGBoost) with hard-negative mining
  5. 5-fold CV to find optimal F₀.₅ threshold
  6. Retrain on full data, save model + threshold

Usage:
  python -m src.train --data dataset/train --model model.joblib
  python -m src.train --data dataset/train --model model.joblib --model-type xgboost
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .io import read_tsv
from .blocking import build_indices, generate_candidates_batch
from .features import pair_features
from .evaluate import fbeta, entity_fbeta, macro_fbeta


# ============================================================
# GROUND TRUTH PARSING
# ============================================================

def parse_truth(gt):
    """Parse ground truth into {source1_id: [matched_ids]}."""
    out = {}
    for _, r in gt.iterrows():
        raw = r["matched_entity_ids"]
        out[r["source1_entity_id"]] = [x for x in raw.split(",") if x]
    return out


# ============================================================
# MODEL CREATION
# ============================================================

def create_model(model_type, n_positive, n_negative):
    """Create classifier with appropriate class weighting."""
    scale_pos_weight = (n_negative / n_positive) if n_positive > 0 else 1.0

    if model_type == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=63,
            reg_lambda=1.0,
            reg_alpha=0.1,
            scale_pos_weight=scale_pos_weight,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
    elif model_type == "xgboost":
        from xgboost import XGBClassifier
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except ImportError:
            pass
        return XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            reg_lambda=1.0,
            reg_alpha=0.1,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            device=device,
            random_state=42,
            eval_metric="logloss",
        )
    elif model_type == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            max_leaf_nodes=63,
            l2_regularization=1.0,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. "
                         f"Choose 'lightgbm', 'xgboost', or 'histgb'.")


# ============================================================
# THRESHOLD SCANNING
# ============================================================

def find_optimal_threshold(y_true_groups, y_probs_groups, gt_dict):
    """
    Find the threshold that maximizes Macro F₀.₅.

    Args:
        y_true_groups: dict {s1_id: [(target_id, label), ...]}
        y_probs_groups: dict {s1_id: [(target_id, prob), ...]}
        gt_dict: ground truth {s1_id: [matched_ids]}

    Returns:
        (best_threshold, best_f05)
    """
    thresholds = np.arange(0.30, 0.95, 0.02)
    best_f05, best_t = 0.0, 0.50

    for t in thresholds:
        predictions = {}
        for s1_id, pairs in y_probs_groups.items():
            matched = [tid for tid, prob in pairs if prob >= t]
            predictions[s1_id] = matched

        f05 = macro_fbeta(predictions, gt_dict)
        if f05 > best_f05:
            best_f05 = f05
            best_t = t

    return float(best_t), float(best_f05)


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Train business entity resolution classifier.")
    ap.add_argument("--data", default="dataset/train", help="Path to training data directory")
    ap.add_argument("--model", default="model.joblib", help="Output model joblib path")
    ap.add_argument("--model-type", choices=["lightgbm", "xgboost", "histgb"],
                    default="lightgbm", help="Classifier type")
    ap.add_argument("--no-cv", action="store_true", help="Skip cross-validation")
    ap.add_argument("--sample", type=int, default=0,
                    help="Sample N source1 entities (0 = all)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    d = Path(args.data)

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------
    print("[1/6] Loading data...", flush=True)
    s1 = read_tsv(d / "train_source1.tsv")
    s2 = read_tsv(d / "train_source2.tsv")
    s3 = read_tsv(d / "train_source3.tsv")
    gt_raw = read_tsv(d / "train_ground_truth.tsv")
    gt = parse_truth(gt_raw)

    if args.sample > 0:
        print(f"  Sampling {args.sample} source1 entities...", flush=True)
        s1 = s1.sample(n=min(args.sample, len(s1)), random_state=42).reset_index(drop=True)

    target = pd.concat([s2, s3], ignore_index=True)
    print(f"  S1: {len(s1):,}, Target: {len(target):,}", flush=True)

    # --------------------------------------------------------
    # 2. Run blocking
    # --------------------------------------------------------
    print("[2/6] Running blocking...", flush=True)
    indices = build_indices(target)
    pairs, candidate_map = generate_candidates_batch(s1, target, indices)
    print(f"  Total candidate pairs: {len(pairs):,}", flush=True)

    # --------------------------------------------------------
    # 3. Compute features
    # --------------------------------------------------------
    print("[3/6] Computing features...", flush=True)
    target_by_id = target.set_index("entity_id")
    s1_by_id = s1.set_index("entity_id")
    idf_weights = indices.get("idf_weights", None)

    positives = set((a, b) for a, ids in gt.items() for b in ids)

    from concurrent.futures import ThreadPoolExecutor
    def process_pair(args):
        a_id, b_id = args
        return pair_features(s1_by_id.loc[a_id], target_by_id.loc[b_id], idf_weights)

    valid_pairs = [(a, b) for a, b in pairs if a in s1_by_id.index and b in target_by_id.index]
    n_pairs = len(valid_pairs)
    
    X_rows = [None] * n_pairs
    processed = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i, x in enumerate(executor.map(process_pair, valid_pairs)):
            X_rows[i] = x
            processed += 1
            if processed % 100000 == 0 or processed == n_pairs:
                print(f"  Features computed: {processed:,}/{n_pairs:,}", flush=True)

    y_labels = [int(p in positives) for p in valid_pairs]
    pair_info = valid_pairs

    X = pd.DataFrame(X_rows).replace([np.inf, -np.inf], np.nan).fillna(0)
    y = np.asarray(y_labels)
    feature_names = list(X.columns)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"  Pairs: {len(X):,}, Positives: {n_pos:,}, "
          f"Negatives: {n_neg:,}, Ratio: 1:{n_neg/max(n_pos,1):.1f}", flush=True)

    if n_pos == 0:
        raise RuntimeError("No positive training pairs found in candidate set. "
                           "Check blocking recall.")

    # --------------------------------------------------------
    # 4. Cross-validation for threshold selection
    # --------------------------------------------------------
    best_threshold = 0.50
    if not args.no_cv and len(X) > 1000:
        print("[4/6] Running 5-fold CV for threshold selection...", flush=True)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(y))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"  Fold {fold+1}/5...", flush=True)
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            fold_model = create_model(args.model_type, int(y_train.sum()),
                                      len(y_train) - int(y_train.sum()))
            fold_model.fit(X_train, y_train)
            oof_probs[val_idx] = fold_model.predict_proba(X_val)[:, 1]

        # Build grouped predictions for threshold scanning
        y_probs_groups = {}
        for i, (s1_id, t_id) in enumerate(pair_info):
            if s1_id not in y_probs_groups:
                y_probs_groups[s1_id] = []
            y_probs_groups[s1_id].append((t_id, oof_probs[i]))

        best_threshold, best_f05 = find_optimal_threshold(
            None, y_probs_groups, gt
        )
        print(f"  Optimal threshold: {best_threshold:.2f} (Macro F₀.₅ = {best_f05:.4f})",
              flush=True)
    else:
        print("[4/6] Skipping CV (--no-cv or too few samples).", flush=True)

    # --------------------------------------------------------
    # 5. Train final model on all data
    # --------------------------------------------------------
    print("[5/6] Training final model on all data...", flush=True)
    model = create_model(args.model_type, n_pos, n_neg)
    model.fit(X, y)

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importances = sorted(zip(feature_names, model.feature_importances_),
                             key=lambda x: -x[1])
        print("  Top 10 features:")
        for fname, imp in importances[:10]:
            print(f"    {fname:35s} = {imp:.4f}")

    # --------------------------------------------------------
    # 6. Save model
    # --------------------------------------------------------
    print("[6/6] Saving model...", flush=True)
    bundle = {
        "model": model,
        "features": feature_names,
        "model_type": args.model_type,
        "threshold": best_threshold,
        "idf_weights": idf_weights,
    }
    joblib.dump(bundle, args.model)

    elapsed = time.perf_counter() - t0
    print(json.dumps({
        "pairs": len(X),
        "positives": n_pos,
        "negatives": n_neg,
        "model": args.model,
        "model_type": args.model_type,
        "threshold": best_threshold,
        "elapsed_seconds": round(elapsed, 1),
    }, indent=2))
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
