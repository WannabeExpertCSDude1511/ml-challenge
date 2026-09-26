"""
Evaluate model on training data with ground truth (Multithreaded).
"""

import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def process_row(args):
    i, row, gt_dict, target, indices, model, feature_names, idf_weights = args
    s1_id = row["entity_id"]
    true_ids = set(gt_dict.get(s1_id, []))

    # Blocking
    candidates = generate_candidates_for_row(row, target, indices)
    candidate_ids = set(candidates["entity_id"].tolist())

    # Classification
    probs_list = []
    if len(candidates) > 0:
        X = pd.DataFrame([
            pair_features(row, r, idf_weights)
            for _, r in candidates.iterrows()
        ])
        X = X.reindex(columns=feature_names, fill_value=0)
        probs = model.predict_proba(X)[:, 1]
        probs_list = list(zip(candidates["entity_id"].tolist(), probs.tolist()))

    return {
        "i": i,
        "s1_id": s1_id,
        "true_ids": true_ids,
        "candidate_ids": candidate_ids,
        "probs_list": probs_list,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate entity resolution on training data.")
    ap.add_argument("--data", default="dataset/train")
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--scan-thresholds", action="store_true", help="Scan thresholds from 0.30 to 0.95")
    ap.add_argument("--sample", type=int, default=0, help="Sample N source1 entities (0 = all)")
    ap.add_argument("--workers", type=int, default=8, help="Number of threads")
    args = ap.parse_args()

    t0 = time.perf_counter()
    d = Path(args.data)

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

    print("[2/5] Building blocking indices...", flush=True)
    indices = build_indices(target)

    print(f"[3/5] Running blocking + classification ({args.workers} threads)...", flush=True)

    all_probs = {}
    blocking_recall_data = {"total_gt": 0, "found_gt": 0}
    n_total = len(s1)
    
    # Prepare arguments for multiprocessing map
    tasks = [
        (i, row, gt, target, indices, model, feature_names, idf_weights)
        for i, (_, row) in enumerate(s1.iterrows())
    ]

    processed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for res in executor.map(process_row, tasks):
            processed += 1
            all_probs[res["s1_id"]] = res["probs_list"]
            blocking_recall_data["total_gt"] += len(res["true_ids"])
            blocking_recall_data["found_gt"] += len(res["true_ids"] & res["candidate_ids"])
            
            if processed % 500 == 0 or processed == n_total:
                print(f"  Processed {processed:,}/{n_total:,}", flush=True)

    print("\n[4/5] Results:", flush=True)
    total_gt = blocking_recall_data["total_gt"]
    found_gt = blocking_recall_data["found_gt"]
    block_recall = found_gt / total_gt if total_gt > 0 else 0
    print(f"\n  === Blocking Recall ===")
    print(f"  GT matches:       {total_gt:,}")
    print(f"  Found by blocker: {found_gt:,}")
    print(f"  Recall:           {block_recall*100:.2f}%")

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
