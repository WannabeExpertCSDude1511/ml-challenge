import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .blocking import generate_candidates
from .data import load_split
from .features import compute_features
from .io import write_tsv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/test")
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--output", default="output")
    ap.add_argument("--threshold", type=float, default=None, help="Defaults to the threshold chosen in train.py")
    ap.add_argument("--s1-chunk", type=int, default=100_000, help="S1 records scored per batch (bounds memory)")
    args = ap.parse_args()

    s1, target = load_split(args.data, "test")
    bundle = joblib.load(args.model)
    model = bundle["model"]
    threshold = args.threshold if args.threshold is not None else bundle.get("threshold", 0.5)
    print(f"S1: {len(s1):,}  targets: {len(target):,}  threshold: {threshold}", flush=True)

    results, candidates_out = [], []
    for start in range(0, len(s1), args.s1_chunk):
        part = s1.iloc[start:start + args.s1_chunk].reset_index(drop=True)
        s_idx, t_idx = generate_candidates(part, target)
        X = compute_features(part, target, s_idx, t_idx)
        probs = model.predict_proba(X)[:, 1] if len(X) else np.empty(0)

        # candidate_pairs.tsv is exactly the set of pairs the model scored.
        t_ids = np.asarray(target["entity_id"].take(t_idx).tolist(), dtype=object)
        bounds = np.searchsorted(s_idx, np.arange(len(part) + 1))
        for i, s_id in enumerate(part["entity_id"].tolist()):
            a, b = bounds[i], bounds[i + 1]
            ids = t_ids[a:b]
            candidates_out.append((s_id, ",".join(ids)))
            results.append((s_id, ",".join(ids[probs[a:b] >= threshold])))
        print(f"  scored S1 {start + len(part):,}/{len(s1):,}", flush=True)

    out = Path(args.output)
    write_tsv(pd.DataFrame(results, columns=["source1_entity_id", "matched_entity_ids"]), out / "matching_results.tsv")
    write_tsv(pd.DataFrame(candidates_out, columns=["source1_entity_id", "candidate_entity_ids"]), out / "candidate_pairs.tsv")


if __name__ == "__main__":
    main()
