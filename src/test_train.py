"""
Evaluate model on training data with ground truth.

Reports:
  - Blocking recall (candidate generation quality)
  - Macro F₀.₅ (the competition metric)
  - Per-threshold scan to find optimal F₀.₅
  - Precision / recall breakdown

Usage:
  python -m src.test_train --data dataset/train --model model.joblib --threshold 0.80
  python -m src.test_train --data dataset/train --model model.joblib --scan-thresholds
  python -m src.test_train --data dataset/train --model model.joblib --sample 10000
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .io import read_tsv
from .blocking import build_indices, generate_candidates_for_row
from .features import pair_features
from .evaluate import fbeta, entity_fbeta, macro_fbeta


def parse_truth(gt):
    out = {}
    for _, r in gt.iterrows():
        raw = r["matched_entity_ids"]
        out[r["source1_entity_id"]] = [x for x in raw.split(",") if x]
    return out


def main():
    ap = argparse.ArgumentParser(description="Evaluate entity resolution on training data.")
    ap.add_argument("--data", default="dataset/train")
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--scan-thresholds", action="store_true",
                    help="Scan thresholds from 0.30 to 0.95")
    ap.add_argument("--sample", type=int, default=0,
                    help="Sample N source1 entities (0 = all)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    d = Path(args.data)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------
    print("[1/5] Loading data...", flush=True)
    s1 = read_tsv(d / "train_source1.tsv")
    s2 = read_tsv(d / "train_source2.tsv")
    s3 = read_tsv(d / "train_source3.tsv")
    gt_raw = read_tsv(d / "train_ground_truth.tsv")
    gt = parse_truth(gt_raw)

    if args.sample > 0:
        s1 = s1.sample(n=min(args.sample, len(s1)), random_state=42).reset_index(drop=True)

    target = pd.concat([s2, s3], ignore_index=True)

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_names = bundle["features"]
    threshold = args.threshold if args.threshold is not None else bundle.get("threshold", 0.50)
    idf_weights = bundle.get("idf_weights", None)

    print(f"  S1: {len(s1):,}, Target: {len(target):,}", flush=True)
    print(f"  Model: {bundle.get('model_type', 'unknown')}, Threshold: {threshold:.2f}", flush=True)

    # --------------------------------------------------------
    # Build indices
    # --------------------------------------------------------
    print("[2/5] Building blocking indices...", flush=True)
    indices = build_indices(target)

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------
    print("[3/5] Running blocking + classification...", flush=True)

    all_probs = {}   # {s1_id: [(target_id, prob), ...]}
    blocking_recall_data = {"total_gt": 0, "found_gt": 0}
    n_total = len(s1)

    for i, (_, row) in enumerate(s1.iterrows()):
        s1_id = row["entity_id"]
        true_ids = set(gt.get(s1_id, []))

        # Blocking
        candidates = generate_candidates_for_row(row, target, indices)
        candidate_ids = set(candidates["entity_id"].tolist())

        # Blocking recall
        blocking_recall_data["total_gt"] += len(true_ids)
        blocking_recall_data["found_gt"] += len(true_ids & candidate_ids)

        # Classification
        if len(candidates) > 0:
            X = pd.DataFrame([
                pair_features(row, r, idf_weights)
                for _, r in candidates.iterrows()
            ])
            X = X.reindex(columns=feature_names, fill_value=0)
            probs = model.predict_proba(X)[:, 1]
            all_probs[s1_id] = list(zip(candidates["entity_id"].tolist(), probs.tolist()))
        else:
            all_probs[s1_id] = []

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,}/{n_total:,}", flush=True)

    # --------------------------------------------------------
    # Blocking recall
    # --------------------------------------------------------
    print("\n[4/5] Results:", flush=True)
    total_gt = blocking_recall_data["total_gt"]
    found_gt = blocking_recall_data["found_gt"]
    block_recall = found_gt / total_gt if total_gt > 0 else 0
    print(f"\n  === Blocking Recall ===")
    print(f"  GT matches:       {total_gt:,}")
    print(f"  Found by blocker: {found_gt:,}")
    print(f"  Recall:           {block_recall*100:.2f}%")

    # --------------------------------------------------------
    # Threshold evaluation
    # --------------------------------------------------------
    if args.scan_thresholds:
        print(f"\n  === Threshold Scan ===")
        print(f"  {'Threshold':>10s}  {'Macro F0.5':>10s}  {'Precision':>10s}  {'Recall':>10s}")
        print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

        best_f05, best_t = 0, 0.50
        for t in np.arange(0.30, 0.95, 0.02):
            predictions = {}
            total_predicted = 0
            total_correct = 0
            total_true = 0

            for s1_id, pairs in all_probs.items():
                matched = [tid for tid, prob in pairs if prob >= t]
                predictions[s1_id] = matched
                true = set(gt.get(s1_id, []))
                total_predicted += len(matched)
                total_correct += len(set(matched) & true)
                total_true += len(true)

            f05 = macro_fbeta(predictions, gt)
            prec = total_correct / total_predicted if total_predicted > 0 else 0
            rec = total_correct / total_true if total_true > 0 else 0

            marker = " <<<" if f05 > best_f05 else ""
            print(f"  {t:10.2f}  {f05:10.4f}  {prec:10.4f}  {rec:10.4f}{marker}")

            if f05 > best_f05:
                best_f05 = f05
                best_t = t

        print(f"\n  Best: threshold={best_t:.2f}, Macro F₀.₅={best_f05:.4f}")
    else:
        # Single threshold evaluation
        predictions = {}
        total_predicted = 0
        total_correct = 0
        total_true = 0

        for s1_id, pairs in all_probs.items():
            matched = [tid for tid, prob in pairs if prob >= threshold]
            predictions[s1_id] = matched
            true = set(gt.get(s1_id, []))
            total_predicted += len(matched)
            total_correct += len(set(matched) & true)
            total_true += len(true)

        f05 = macro_fbeta(predictions, gt)
        prec = total_correct / total_predicted if total_predicted > 0 else 0
        rec = total_correct / total_true if total_true > 0 else 0

        print(f"\n  === Classification @ threshold={threshold:.2f} ===")
        print(f"  Macro F₀.₅:    {f05:.4f}")
        print(f"  Precision:      {prec:.4f}")
        print(f"  Recall:         {rec:.4f}")
        print(f"  Total predicted: {total_predicted:,}")
        print(f"  Correct:         {total_correct:,}")
        print(f"  Total true:      {total_true:,}")

    elapsed = time.perf_counter() - t0
    print(f"\n[5/5] Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
