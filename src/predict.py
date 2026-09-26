import argparse
from pathlib import Path
import joblib
import pandas as pd
from .io import read_tsv, write_tsv
from .blocking import build_indices, generate_candidates_for_row
from .features import pair_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/test")
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--output", default="output")
    ap.add_argument("--threshold", type=float, default=0.80)
    args = ap.parse_args()

    d = Path(args.data)
    out = Path(args.output)
    s1 = read_tsv(d / "test_source1.tsv")
    s2 = read_tsv(d / "test_source2.tsv")
    s3 = read_tsv(d / "test_source3.tsv")
    target = pd.concat([s2, s3], ignore_index=True)
    indices = build_indices(target)
    target_by_idx = target
    bundle = joblib.load(args.model)
    model, feature_names = bundle["model"], bundle["features"]

    results, candidates_out = [], []
    for _, row in s1.iterrows():
        candidates = generate_candidates_for_row(row, target_by_idx, indices)
        candidate_ids = list(dict.fromkeys(candidates["entity_id"].tolist()))
        candidates_out.append({
            "source1_entity_id": row["entity_id"],
            "candidate_entity_ids": ",".join(candidate_ids),
        })
        if len(candidates):
            X = pd.DataFrame([pair_features(row, r) for _, r in candidates.iterrows()])
            X = X.reindex(columns=feature_names, fill_value=0)
            probs = model.predict_proba(X)[:, 1]
            matches = [eid for eid, p in zip(candidate_ids, probs) if p >= args.threshold]
        else:
            matches = []
        results.append({
            "source1_entity_id": row["entity_id"],
            "matched_entity_ids": ",".join(dict.fromkeys(matches)),
        })

    write_tsv(pd.DataFrame(results), out / "matching_results.tsv")
    write_tsv(pd.DataFrame(candidates_out), out / "candidate_pairs.tsv")

if __name__ == "__main__":
    main()
