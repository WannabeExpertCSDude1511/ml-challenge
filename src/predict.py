"""
Predict: Generate matching_results.tsv and candidate_pairs.tsv for test data.

Usage:
  python -m src.predict --data dataset/test --model model.joblib --output output
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .io import read_tsv, write_tsv
from .blocking import build_indices, generate_candidates_for_row
from .features import pair_features


def main():
    ap = argparse.ArgumentParser(description="Predict entity matches on test data.")
    ap.add_argument("--data", default="dataset/test")
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--output", default="output")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override saved threshold (default: use threshold from training)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    d = Path(args.data)
    out = Path(args.output)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    print("[1/4] Loading data...", flush=True)
    s1 = read_tsv(d / "test_source1.tsv")
    s2 = read_tsv(d / "test_source2.tsv")
    s3 = read_tsv(d / "test_source3.tsv")
    target = pd.concat([s2, s3], ignore_index=True)
    print(f"  S1: {len(s1):,}, Target: {len(target):,}", flush=True)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    print("[2/4] Loading model...", flush=True)
    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_names = bundle["features"]
    threshold = args.threshold if args.threshold is not None else bundle.get("threshold", 0.50)
    idf_weights = bundle.get("idf_weights", None)
    print(f"  Model type: {bundle.get('model_type', 'unknown')}", flush=True)
    print(f"  Threshold: {threshold:.2f}", flush=True)
    print(f"  Features: {len(feature_names)}", flush=True)

    # --------------------------------------------------------
    # Build blocking indices
    # --------------------------------------------------------
    print("[3/4] Building blocking indices...", flush=True)
    indices = build_indices(target)

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------
    print("[4/4] Predicting...", flush=True)
    results = []
    candidates_out = []
    n_total = len(s1)

    for i, (_, row) in enumerate(s1.iterrows()):
        # Generate candidates
        candidates = generate_candidates_for_row(row, target, indices)
        candidate_ids = list(dict.fromkeys(candidates["entity_id"].tolist()))

        candidates_out.append({
            "source1_entity_id": row["entity_id"],
            "candidate_entity_ids": ",".join(candidate_ids),
        })

        if len(candidates) > 0:
            # Compute features
            X = pd.DataFrame([
                pair_features(row, r, idf_weights)
                for _, r in candidates.iterrows()
            ])
            X = X.reindex(columns=feature_names, fill_value=0)

            # Predict probabilities
            probs = model.predict_proba(X)[:, 1]

            # Apply threshold
            matches = [eid for eid, p in zip(candidate_ids, probs) if p >= threshold]
        else:
            matches = []

        results.append({
            "source1_entity_id": row["entity_id"],
            "matched_entity_ids": ",".join(dict.fromkeys(matches)),
        })

        if (i + 1) % 50000 == 0:
            print(f"  Processed {i+1:,}/{n_total:,} ({(i+1)/n_total*100:.1f}%)", flush=True)

    # --------------------------------------------------------
    # Write output
    # --------------------------------------------------------
    write_tsv(pd.DataFrame(results), out / "matching_results.tsv")
    write_tsv(pd.DataFrame(candidates_out), out / "candidate_pairs.tsv")

    elapsed = time.perf_counter() - t0
    n_matched = sum(1 for r in results if r["matched_entity_ids"])
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  S1 entities with ≥1 match: {n_matched:,}/{n_total:,}")
    print(f"  Output: {out / 'matching_results.tsv'}")
    print(f"  Output: {out / 'candidate_pairs.tsv'}")


if __name__ == "__main__":
    main()
