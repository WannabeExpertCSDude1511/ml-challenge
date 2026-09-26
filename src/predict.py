"""
Predict: Generate matching_results.tsv and candidate_pairs.tsv for test data.
(Multithreaded)
"""

import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
import pandas as pd

from .io import read_tsv, write_tsv
from .blocking import build_indices, generate_candidates_for_row
from .features import pair_features


def process_row(args):
    i, row, target, indices, model, feature_names, threshold, idf_weights = args
    candidates = generate_candidates_for_row(row, target, indices)
    candidate_ids = list(dict.fromkeys(candidates["entity_id"].tolist()))

    cand_out = {
        "source1_entity_id": row["entity_id"],
        "candidate_entity_ids": ",".join(candidate_ids),
    }

    if len(candidates) > 0:
        X = pd.DataFrame([
            pair_features(row, r, idf_weights)
            for _, r in candidates.iterrows()
        ])
        X = X.reindex(columns=feature_names, fill_value=0)
        probs = model.predict_proba(X)[:, 1]
        matches = [eid for eid, p in zip(candidate_ids, probs) if p >= threshold]
    else:
        matches = []

    res_out = {
        "source1_entity_id": row["entity_id"],
        "matched_entity_ids": ",".join(dict.fromkeys(matches)),
    }
    
    return i, cand_out, res_out


def main():
    ap = argparse.ArgumentParser(description="Predict entity matches on test data.")
    ap.add_argument("--data", default="dataset/test")
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--output", default="output")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--workers", type=int, default=8, help="Number of threads")
    args = ap.parse_args()

    t0 = time.perf_counter()
    d = Path(args.data)
    out = Path(args.output)
    out.mkdir(exist_ok=True, parents=True)

    print("[1/4] Loading data...", flush=True)
    s1 = read_tsv(d / "test_source1.tsv")
    s2 = read_tsv(d / "test_source2.tsv")
    s3 = read_tsv(d / "test_source3.tsv")
    target = pd.concat([s2, s3], ignore_index=True)
    print(f"  S1: {len(s1):,}, Target: {len(target):,}", flush=True)

    print("[2/4] Loading model...", flush=True)
    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_names = bundle["features"]
    threshold = args.threshold if args.threshold is not None else bundle.get("threshold", 0.50)
    idf_weights = bundle.get("idf_weights", None)
    print(f"  Model type: {bundle.get('model_type', 'unknown')}, Threshold: {threshold:.2f}", flush=True)

    print("[3/4] Building blocking indices...", flush=True)
    indices = build_indices(target)

    print(f"[4/4] Predicting ({args.workers} threads)...", flush=True)
    
    tasks = [
        (i, row, target, indices, model, feature_names, threshold, idf_weights)
        for i, (_, row) in enumerate(s1.iterrows())
    ]
    
    n_total = len(s1)
    results = [None] * n_total
    candidates_out = [None] * n_total
    
    processed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i, cand_out, res_out in executor.map(process_row, tasks):
            results[i] = res_out
            candidates_out[i] = cand_out
            processed += 1
            if processed % 500 == 0 or processed == n_total:
                print(f"  Processed {processed:,}/{n_total:,} ({(processed)/n_total*100:.1f}%)", flush=True)

    write_tsv(pd.DataFrame(results), out / "matching_results.tsv")
    write_tsv(pd.DataFrame(candidates_out), out / "candidate_pairs.tsv")

    elapsed = time.perf_counter() - t0
    n_matched = sum(1 for r in results if r["matched_entity_ids"])
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  S1 entities with ≥1 match: {n_matched:,}/{n_total:,}")


if __name__ == "__main__":
    main()
