"""
Evaluate blocking alone: recall ceiling and candidate-set size.

Uses the same generate_candidates() as train.py / predict.py, on the same
seeded S1 sample as train.py (train + holdout parts together).
"""
import argparse
import time

import numpy as np

from .blocking import generate_candidates
from .data import load_split, load_truth, split_s1, truth_positions
from .evaluate import blocking_stats, label_pairs, print_stats


def main():
    ap = argparse.ArgumentParser(description="Evaluate blocking recall and candidate volume.")
    ap.add_argument("--data", default="dataset/train")
    ap.add_argument("--sample-size", type=int, default=10000, help="S1 records to evaluate; 0 = all")
    ap.add_argument("--holdout", type=float, default=0.2, help="Same split as train.py, so the sample matches")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    s1, target = load_split(args.data, "train")
    truth = load_truth(args.data)
    train_pos, hold_pos = split_s1(len(s1), args.holdout, args.seed, args.sample_size)
    part = s1.iloc[np.r_[hold_pos, train_pos]].reset_index(drop=True)
    ids = part["entity_id"].tolist()
    print(f"S1 sample: {len(part):,}  targets: {len(target):,}", flush=True)

    start = time.perf_counter()
    s_idx, t_idx = generate_candidates(part, target)
    runtime = time.perf_counter() - start

    truth_pos, n_true = truth_positions(ids, truth, target)
    y = label_pairs(s_idx, t_idx, truth_pos)

    stats = blocking_stats(s_idx, y, n_true, len(target))
    stats["runtime_seconds"] = round(runtime, 1)
    print_stats("ALL COUNTRIES", stats)

    countries = np.asarray(part["country"].tolist(), dtype=object)
    for country in sorted(set(countries)):
        mask = countries == country
        new_pos = np.cumsum(mask) - 1
        sel = mask[s_idx]
        print_stats(
            f"COUNTRY: {country or '(empty)'}",
            blocking_stats(new_pos[s_idx[sel]], y[sel], n_true[mask], len(target)),
        )


if __name__ == "__main__":
    main()
