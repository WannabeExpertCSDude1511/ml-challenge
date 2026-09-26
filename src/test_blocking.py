"""
Evaluate blocking quality on training data.

Metrics:
  1. Blocking recall (match-level and S1-level)
  2. Average / median / max candidates per S1
  3. Candidate reduction ratio
  4. Per-strategy attribution
  5. Runtime

Usage:
  python -m src.test_blocking --data dataset/train --sample-size 10000
  python -m src.test_blocking --data dataset/train --sample-size 0  # full dataset
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from .io import read_tsv
from .blocking import (
    build_indices,
    generate_candidates_for_row,
    blocking_debug_info,
)


def split_ids(value):
    if not value:
        return set()
    return {x.strip() for x in value.split(",") if x.strip()}


def main():
    ap = argparse.ArgumentParser(description="Evaluate blocking quality.")
    ap.add_argument("--data", default="dataset/train")
    ap.add_argument("--sample-size", type=int, default=10000,
                    help="Number of S1 entities to evaluate (0 = all)")
    ap.add_argument("--show-missed", type=int, default=5,
                    help="Number of missed pairs to display")
    args = ap.parse_args()

    d = Path(args.data)

    print("=" * 70)
    print("BLOCKING EVALUATION (v2 — 8 strategies)")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    print("\n[1/5] Loading data...", flush=True)
    s1 = read_tsv(d / "train_source1.tsv")
    s2 = read_tsv(d / "train_source2.tsv")
    s3 = read_tsv(d / "train_source3.tsv")
    gt = read_tsv(d / "train_ground_truth.tsv")

    target = pd.concat([s2, s3], ignore_index=True)

    if args.sample_size > 0 and args.sample_size < len(s1):
        print(f"  Sampling {args.sample_size} source1 entities...", flush=True)
        s1 = s1.sample(n=args.sample_size, random_state=42).reset_index(drop=True)

    print(f"  S1: {len(s1):,}, Target: {len(target):,}", flush=True)

    # --------------------------------------------------------
    # Build GT lookup
    # --------------------------------------------------------
    print("\n[2/5] Preparing ground truth...", flush=True)
    gt_lookup = {}
    for _, row in gt.iterrows():
        gt_lookup[row["source1_entity_id"]] = split_ids(row["matched_entity_ids"])

    # --------------------------------------------------------
    # Build indices
    # --------------------------------------------------------
    print("\n[3/5] Building blocking indices...", flush=True)
    t_idx = time.perf_counter()
    indices = build_indices(target)
    index_time = time.perf_counter() - t_idx
    print(f"  Index construction: {index_time:.1f}s", flush=True)

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------
    print("\n[4/5] Running blocker...", flush=True)
    t_block = time.perf_counter()

    candidate_counts = []
    total_gt = 0
    found_gt = 0
    s1_with_any = 0
    s1_with_all = 0
    s1_zero_candidates = 0
    total_candidates = 0

    # Strategy attribution
    strategy_counts = {}

    missed_examples = []

    for i, (_, row) in enumerate(s1.iterrows()):
        source_id = row["entity_id"]
        source_row = {
            "entity_id": row["entity_id"],
            "business_name": row["business_name"],
            "business_address": row["business_address"],
            "country": row["country"],
        }

        candidates = generate_candidates_for_row(source_row, target, indices)
        candidate_ids = set(candidates["entity_id"].astype(str))
        n_cands = len(candidate_ids)

        candidate_counts.append(n_cands)
        total_candidates += n_cands

        true_ids = gt_lookup.get(source_id, set())
        total_gt += len(true_ids)
        found_ids = true_ids & candidate_ids
        found_gt += len(found_ids)

        if found_ids:
            s1_with_any += 1
        if true_ids and found_ids == true_ids:
            s1_with_all += 1
        if n_cands == 0:
            s1_zero_candidates += 1

        # Collect missed examples
        missed = true_ids - found_ids
        if missed and len(missed_examples) < args.show_missed:
            missed_examples.append({
                "s1_id": source_id,
                "s1_name": row["business_name"],
                "s1_addr": row["business_address"],
                "s1_country": row["country"],
                "missed_ids": list(missed)[:3],
            })

        # Strategy attribution (sample for first 1000)
        if i < 1000 and true_ids:
            debug = blocking_debug_info(source_row, target, indices)
            for strat, count in debug.items():
                if strat != "total":
                    strategy_counts[strat] = strategy_counts.get(strat, 0) + (1 if count > 0 else 0)

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,}/{len(s1):,}", flush=True)

    block_time = time.perf_counter() - t_block
    total_time = time.perf_counter() - t_idx

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------
    print("\n[5/5] RESULTS")
    print("=" * 70)

    cs = pd.Series(candidate_counts)
    possible_pairs = len(s1) * len(target)
    reduction = 1 - (total_candidates / possible_pairs) if possible_pairs > 0 else 0
    recall = found_gt / total_gt if total_gt > 0 else 0

    print(f"  S1 records:                {len(s1):,}")
    print(f"  Target records:            {len(target):,}")
    print(f"  Possible pairs:            {possible_pairs:,}")
    print(f"  Actual candidate pairs:    {total_candidates:,}")
    print(f"  Candidate reduction:       {reduction*100:.4f}%")

    print(f"\n  --- Candidate statistics ---")
    print(f"  Average per S1:    {cs.mean():.1f}")
    print(f"  Median per S1:     {cs.median():.0f}")
    print(f"  Max per S1:        {cs.max():,}")
    print(f"  S1 with 0 cands:   {s1_zero_candidates:,}")

    print(f"\n  --- Blocking recall ---")
    print(f"  Total GT matches:         {total_gt:,}")
    print(f"  Found by blocker:         {found_gt:,}")
    print(f"  Match-level recall:       {recall*100:.2f}%")
    print(f"  S1 with ≥1 GT found:      {s1_with_any:,}/{len(s1):,}")
    print(f"  S1 with ALL GT found:     {s1_with_all:,}/{len(s1):,}")

    if strategy_counts:
        print(f"\n  --- Strategy attribution (first 1K S1 entities) ---")
        for strat, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            print(f"  {strat:25s}: {count:,} entities had hits")

    if missed_examples:
        print(f"\n  --- Missed pair examples ---")
        for ex in missed_examples:
            print(f"  S1: {ex['s1_id']}")
            print(f"      Name: {ex['s1_name']}")
            print(f"      Addr: {ex['s1_addr']}")
            print(f"      Country: {ex['s1_country']}")
            # Look up missed targets
            for mid in ex["missed_ids"]:
                if mid in target.set_index("entity_id").index:
                    t_row = target.set_index("entity_id").loc[mid]
                    print(f"      MISSED: {mid}")
                    print(f"        Name: {t_row.get('business_name', '?')}")
                    print(f"        Addr: {t_row.get('business_address', '?')}")
            print()

    print(f"\n  --- Runtime ---")
    print(f"  Index:    {index_time:.1f}s")
    print(f"  Blocking: {block_time:.1f}s")
    print(f"  Total:    {total_time:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()