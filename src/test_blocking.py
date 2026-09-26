"""
Evaluate blocking alone: recall ceiling and candidate-set size.

Uses the same BlockingIndex as train.py / predict.py, on the same
seeded S1 sample as train.py (train + holdout parts together).
"""
import argparse
import time

import numpy as np

from .blocking import DEFAULT_K, BlockingIndex
from .data import load_split, load_truth, split_s1, truth_positions
from .evaluate import blocking_stats, label_pairs, print_stats


def main():
    ap = argparse.ArgumentParser(description="Evaluate blocking recall and candidate volume.")
    ap.add_argument("--data", default="dataset/train")
    ap.add_argument("--sample-size", type=int, default=10000, help="S1 records to evaluate; 0 = all")
    ap.add_argument("--holdout", type=float, default=0.2, help="Same split as train.py, so the sample matches")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, nargs="+", default=[DEFAULT_K], help="One or more K values to compare")
    args = ap.parse_args()

    s1, target = load_split(args.data, "train")
    train_pos, hold_pos = split_s1(len(s1), args.holdout, args.seed, args.sample_size)
    part = s1.iloc[np.r_[hold_pos, train_pos]].reset_index(drop=True)
    del s1
    ids = part["entity_id"].tolist()
    truth = load_truth(args.data, ids)
    print(f"S1 sample: {len(part):,}  targets: {len(target):,}", flush=True)

    start = time.perf_counter()
    index = BlockingIndex(target)
    built = time.perf_counter()
    s_idx, t_idx, rank = index.query(part, max(args.k), return_ranks=True)
    queried = time.perf_counter()
    del index

    truth_pos, n_true = truth_positions(ids, truth, target)
    y = label_pairs(s_idx, t_idx, truth_pos)

    print(f"\nBlocking runtime: index {built - start:.0f}s, query {queried - built:.0f}s "
          f"({(queried - built) / max(len(part), 1) * 1e6:,.0f} us per S1 record)")
    print(f"\n{'K':>4}{'recall':>10}{'all found':>11}{'avg cand':>10}{'p99 cand':>10}{'zero cand':>11}")
    for k in sorted(args.k):
        # rank = best position over the rankings, so a smaller k is a rank filter.
        keep = rank < k
        st = blocking_stats(s_idx[keep], y[keep], n_true, len(target))
        print(f"{k:>4}{st['recall']:>10.4f}{st['entities_all_found']:>11.4f}"
              f"{st['avg_candidates']:>10.2f}{st['p99_candidates']:>10.1f}{st['zero_candidate_rate']:>11.4f}")

    print_stats(f"ALL COUNTRIES (K={max(args.k)})", blocking_stats(s_idx, y, n_true, len(target)))

    countries = np.asarray(part["country"].tolist(), dtype=object)
    for country in sorted(set(countries)):
        mask = countries == country
        new_pos = np.cumsum(mask) - 1
        sel = mask[s_idx]
        print_stats(
            f"COUNTRY: {country or '(empty)'} (K={max(args.k)})",
            blocking_stats(new_pos[s_idx[sel]], y[sel], n_true[mask], len(target)),
        )


if __name__ == "__main__":
    main()
