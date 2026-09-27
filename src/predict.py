import argparse
from pathlib import Path

import joblib
import numpy as np

from .blocking import BlockingIndex
from .data import load_split
from .features import compute_features

MATCHING_HEADER = "source1_entity_id\tmatched_entity_ids\n"
CANDIDATE_HEADER = "source1_entity_id\tcandidate_entity_ids\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/test")
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--output", default="output")
    ap.add_argument("--threshold", type=float, default=None, help="Defaults to the threshold chosen in train.py")
    ap.add_argument("--s1-chunk", type=int, default=25_000, help="S1 records scored per batch (bounds memory)")
    args = ap.parse_args()

    s1, target = load_split(args.data, "test")
    bundle = joblib.load(args.model)
    model = bundle["model"]
    threshold = args.threshold if args.threshold is not None else bundle.get("threshold", 0.5)
    k = bundle["k"]
    print(f"S1: {len(s1):,}  targets: {len(target):,}  threshold: {threshold}  k: {k}", flush=True)
    index = BlockingIndex(target)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # Rows are streamed to disk chunk by chunk; holding ~1.7M candidate lists
    # in memory would cost over a GB.
    with open(out / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as matches, \
         open(out / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as candidates:
        matches.write(MATCHING_HEADER)
        candidates.write(CANDIDATE_HEADER)

        for start in range(0, len(s1), args.s1_chunk):
            part = s1.iloc[start:start + args.s1_chunk].reset_index(drop=True)
            s_idx, t_idx, details = index.query(part, k, return_details=True)
            X = compute_features(part, target, s_idx, t_idx, details)
            probs = model.predict_proba(X)[:, 1] if len(X) else np.empty(0)

            # candidate_pairs.tsv is exactly the set of pairs the model scored;
            # every S1 record gets a row, empty when it has no candidates/matches.
            t_ids = np.asarray(target["entity_id"].take(t_idx).tolist(), dtype=object)
            bounds = np.searchsorted(s_idx, np.arange(len(part) + 1))
            for i, s_id in enumerate(part["entity_id"].tolist()):
                a, b = bounds[i], bounds[i + 1]
                ids = t_ids[a:b]
                candidates.write(f"{s_id}\t{','.join(ids)}\n")
                matches.write(f"{s_id}\t{','.join(ids[probs[a:b] >= threshold])}\n")
            del X, probs, t_ids
            print(f"  scored S1 {start + len(part):,}/{len(s1):,}", flush=True)


if __name__ == "__main__":
    main()
