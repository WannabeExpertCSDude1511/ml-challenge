import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .blocking import generate_candidates


# =========================================================
# GROUND TRUTH
# =========================================================

def parse_ground_truth(gt):
    """
    Convert GT into a set of true matching pairs.

    GT format:

        source1_entity_id
        matched_entity_ids

    matched_entity_ids contains comma-separated S2/S3 IDs.

    Returns:

        {
            (source1_id, target_id),
            ...
        }
    """

    truth = set()

    for _, row in gt.iterrows():

        source1_id = str(
            row["source1_entity_id"]
        )

        matched_ids = row["matched_entity_ids"]

        if pd.isna(matched_ids):
            continue

        for target_id in str(
            matched_ids
        ).split(","):

            target_id = target_id.strip()

            if target_id:
                truth.add(
                    (
                        source1_id,
                        target_id,
                    )
                )

    return truth


# =========================================================
# STATISTICS
# =========================================================

def percentile(values, p):
    if len(values) == 0:
        return 0.0

    return float(
        np.percentile(values, p)
    )


def calculate_statistics(
    source,
    target,
    candidates,
    truth,
):
    """
    Calculate all blocking statistics.
    """

    candidate_set = set(candidates)

    # -----------------------------------------------------
    # Ground truth
    # -----------------------------------------------------

    total_true_matches = len(truth)

    surviving_matches = (
        truth & candidate_set
    )

    surviving_count = len(
        surviving_matches
    )

    lost_count = (
        total_true_matches
        - surviving_count
    )

    recall = (
        surviving_count
        / total_true_matches
        if total_true_matches > 0
        else 0.0
    )

    # -----------------------------------------------------
    # Candidate statistics
    # -----------------------------------------------------

    candidate_counts = {}

    for source_id, target_id in candidates:

        candidate_counts[source_id] = (
            candidate_counts.get(
                source_id,
                0,
            )
            + 1
        )

    counts = np.array(
        list(
            candidate_counts.values()
        ),
        dtype=np.int64,
    )

    if len(counts) > 0:

        average_candidates = float(
            counts.mean()
        )

        median_candidates = float(
            np.median(counts)
        )

        p25_candidates = percentile(
            counts,
            25,
        )

        p75_candidates = percentile(
            counts,
            75,
        )

        p90_candidates = percentile(
            counts,
            90,
        )

        p95_candidates = percentile(
            counts,
            95,
        )

        p99_candidates = percentile(
            counts,
            99,
        )

        max_candidates = int(
            counts.max()
        )

        min_candidates = int(
            counts.min()
        )

    else:

        average_candidates = 0.0
        median_candidates = 0.0
        p25_candidates = 0.0
        p75_candidates = 0.0
        p90_candidates = 0.0
        p95_candidates = 0.0
        p99_candidates = 0.0
        max_candidates = 0
        min_candidates = 0

    # -----------------------------------------------------
    # Source coverage
    # -----------------------------------------------------

    source_ids = set(
        source["entity_id"].astype(str)
    )

    sources_with_candidates = len(
        set(candidate_counts.keys())
    )

    sources_without_candidates = (
        len(source_ids)
        - sources_with_candidates
    )

    zero_candidate_rate = (
        sources_without_candidates
        / len(source_ids)
        if len(source_ids) > 0
        else 0.0
    )

    # -----------------------------------------------------
    # Brute-force pairs
    # -----------------------------------------------------

    brute_force_pairs = (
        len(source)
        * len(target)
    )

    total_candidate_pairs = len(
        candidate_set
    )

    candidate_reduction = (
        1
        - (
            total_candidate_pairs
            / brute_force_pairs
        )
        if brute_force_pairs > 0
        else 0.0
    )

    # -----------------------------------------------------
    # Candidate precision
    # -----------------------------------------------------

    false_candidates = (
        candidate_set
        - truth
    )

    false_candidate_count = len(
        false_candidates
    )

    candidate_precision = (
        surviving_count
        / total_candidate_pairs
        if total_candidate_pairs > 0
        else 0.0
    )

    # -----------------------------------------------------
    # Per-source GT recall
    # -----------------------------------------------------

    truth_by_source = {}

    for source_id, target_id in truth:

        if source_id not in truth_by_source:
            truth_by_source[source_id] = set()

        truth_by_source[source_id].add(
            target_id
        )

    candidates_by_source = {}

    for source_id, target_id in candidate_set:

        if source_id not in candidates_by_source:
            candidates_by_source[source_id] = set()

        candidates_by_source[source_id].add(
            target_id
        )

    per_source_recalls = []

    sources_with_gt = 0
    sources_with_all_matches = 0
    sources_with_zero_recall = 0

    for source_id, expected_targets in truth_by_source.items():

        sources_with_gt += 1

        actual_candidates = candidates_by_source.get(
            source_id,
            set(),
        )

        retained = (
            expected_targets
            & actual_candidates
        )

        source_recall = (
            len(retained)
            / len(expected_targets)
        )

        per_source_recalls.append(
            source_recall
        )

        if source_recall == 1.0:
            sources_with_all_matches += 1

        if source_recall == 0.0:
            sources_with_zero_recall += 1

    if per_source_recalls:

        mean_source_recall = float(
            np.mean(
                per_source_recalls
            )
        )

        median_source_recall = float(
            np.median(
                per_source_recalls
            )
        )

        p95_source_recall = percentile(
            np.array(
                per_source_recalls
            ),
            95,
        )

    else:

        mean_source_recall = 0.0
        median_source_recall = 0.0
        p95_source_recall = 0.0

    return {
        "total_true_matches": total_true_matches,
        "surviving_matches": surviving_count,
        "lost_matches": lost_count,
        "recall": recall,

        "total_candidates": total_candidate_pairs,
        "false_candidates": false_candidate_count,
        "candidate_precision": candidate_precision,

        "brute_force_pairs": brute_force_pairs,
        "candidate_reduction": candidate_reduction,

        "average_candidates": average_candidates,
        "median_candidates": median_candidates,
        "p25_candidates": p25_candidates,
        "p75_candidates": p75_candidates,
        "p90_candidates": p90_candidates,
        "p95_candidates": p95_candidates,
        "p99_candidates": p99_candidates,
        "max_candidates": max_candidates,
        "min_candidates": min_candidates,

        "sources_with_candidates": sources_with_candidates,
        "sources_without_candidates": sources_without_candidates,
        "zero_candidate_rate": zero_candidate_rate,

        "sources_with_gt": sources_with_gt,
        "sources_with_all_matches": sources_with_all_matches,
        "sources_with_zero_recall": sources_with_zero_recall,

        "mean_source_recall": mean_source_recall,
        "median_source_recall": median_source_recall,
        "p95_source_recall": p95_source_recall,
    }


# =========================================================
# PRINT REPORT
# =========================================================

def print_report(
    name,
    source,
    target,
    stats,
    runtime,
):
    print()
    print("#" * 72)
    print(f"# {name}")
    print("#" * 72)

    print()
    print("DATASET")
    print("-" * 72)

    print(
        f"S1 records:                  "
        f"{len(source):,}"
    )

    print(
        f"Target records:              "
        f"{len(target):,}"
    )

    print(
        f"Ground-truth matches:        "
        f"{stats['total_true_matches']:,}"
    )

    print()
    print("BLOCKING RECALL")
    print("-" * 72)

    print(
        f"True matches retained:       "
        f"{stats['surviving_matches']:,}"
    )

    print(
        f"True matches lost:            "
        f"{stats['lost_matches']:,}"
    )

    print(
        f"Blocking recall:              "
        f"{stats['recall'] * 100:.6f}%"
    )

    print()
    print("PER-SOURCE RECALL")
    print("-" * 72)

    print(
        f"Sources with GT:              "
        f"{stats['sources_with_gt']:,}"
    )

    print(
        f"Sources with ALL matches:     "
        f"{stats['sources_with_all_matches']:,}"
    )

    print(
        f"Sources with ZERO recall:     "
        f"{stats['sources_with_zero_recall']:,}"
    )

    print(
        f"Mean source recall:           "
        f"{stats['mean_source_recall'] * 100:.6f}%"
    )

    print(
        f"Median source recall:         "
        f"{stats['median_source_recall'] * 100:.6f}%"
    )

    print(
        f"P95 source recall:            "
        f"{stats['p95_source_recall'] * 100:.6f}%"
    )

    print()
    print("CANDIDATE VOLUME")
    print("-" * 72)

    print(
        f"Total candidate pairs:        "
        f"{stats['total_candidates']:,}"
    )

    print(
        f"Average candidates / S1:      "
        f"{stats['average_candidates']:,.2f}"
    )

    print(
        f"Median candidates / S1:       "
        f"{stats['median_candidates']:,.2f}"
    )

    print(
        f"P25 candidates / S1:          "
        f"{stats['p25_candidates']:,.2f}"
    )

    print(
        f"P75 candidates / S1:          "
        f"{stats['p75_candidates']:,.2f}"
    )

    print(
        f"P90 candidates / S1:          "
        f"{stats['p90_candidates']:,.2f}"
    )

    print(
        f"P95 candidates / S1:          "
        f"{stats['p95_candidates']:,.2f}"
    )

    print(
        f"P99 candidates / S1:          "
        f"{stats['p99_candidates']:,.2f}"
    )

    print(
        f"Maximum candidates / S1:      "
        f"{stats['max_candidates']:,}"
    )

    print()
    print("CANDIDATE REDUCTION")
    print("-" * 72)

    print(
        f"Brute-force comparisons:      "
        f"{stats['brute_force_pairs']:,}"
    )

    print(
        f"Generated candidates:          "
        f"{stats['total_candidates']:,}"
    )

    print(
        f"Candidate reduction:           "
        f"{stats['candidate_reduction'] * 100:.6f}%"
    )

    print()
    print("CANDIDATE QUALITY")
    print("-" * 72)

    print(
        f"True candidate pairs:          "
        f"{stats['surviving_matches']:,}"
    )

    print(
        f"False candidate pairs:         "
        f"{stats['false_candidates']:,}"
    )

    print(
        f"Candidate precision:           "
        f"{stats['candidate_precision'] * 100:.6f}%"
    )

    print()
    print("ZERO-CANDIDATE ANALYSIS")
    print("-" * 72)

    print(
        f"S1 with candidates:            "
        f"{stats['sources_with_candidates']:,}"
    )

    print(
        f"S1 without candidates:         "
        f"{stats['sources_without_candidates']:,}"
    )

    print(
        f"Zero-candidate rate:           "
        f"{stats['zero_candidate_rate'] * 100:.6f}%"
    )

    print()
    print("RUNTIME")
    print("-" * 72)

    print(
        f"Blocking runtime:              "
        f"{runtime:.2f} seconds"
    )

    if runtime > 0:

        print(
            f"S1 records / second:           "
            f"{len(source) / runtime:,.2f}"
        )

    print()
    print("#" * 72)


# =========================================================
# EVALUATE ONE TARGET
# =========================================================

def evaluate_target(
    source,
    target,
    truth,
    target_name,
):
    """
    Evaluate S1 against one target dataset.
    """

    print()
    print(
        f"Generating candidates for "
        f"S1 → {target_name}..."
    )

    start = time.perf_counter()

    candidates = generate_candidates(
        source,
        target,
    )

    runtime = (
        time.perf_counter()
        - start
    )

    stats = calculate_statistics(
        source,
        target,
        candidates,
        truth,
    )

    print_report(
        name=f"S1 → {target_name}",
        source=source,
        target=target,
        stats=stats,
        runtime=runtime,
    )

    return stats


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate blocking against "
            "S1, S2, S3 and Ground Truth."
        )
    )

    parser.add_argument(
        "--data",
        default="dataset/train",
        help=(
            "Directory containing "
            "train_source1.tsv, "
            "train_source2.tsv, "
            "train_source3.tsv and "
            "train_ground_truth.tsv"
        ),
    )

    args = parser.parse_args()

    data_dir = Path(
        args.data
    )

    # =====================================================
    # FILE PATHS
    # =====================================================

    s1_path = (
        data_dir
        / "train_source1.tsv"
    )

    s2_path = (
        data_dir
        / "train_source2.tsv"
    )

    s3_path = (
        data_dir
        / "train_source3.tsv"
    )

    gt_path = (
        data_dir
        / "train_ground_truth.tsv"
    )

    # =====================================================
    # CHECK FILES
    # =====================================================

    for path in [
        s1_path,
        s2_path,
        s3_path,
        gt_path,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"File not found: {path}"
            )

    # =====================================================
    # LOAD DATA
    # =====================================================

    print()
    print("=" * 72)
    print("LOADING AMAZON ML CHALLENGE DATA")
    print("=" * 72)

    print()
    print("Loading S1...")

    s1 = pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"S1: {len(s1):,} records"
    )

    print()
    print("Loading S2...")

    s2 = pd.read_csv(
        s2_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"S2: {len(s2):,} records"
    )

    print()
    print("Loading S3...")

    s3 = pd.read_csv(
        s3_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"S3: {len(s3):,} records"
    )

    print()
    print("Loading Ground Truth...")

    gt = pd.read_csv(
        gt_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"GT: {len(gt):,} records"
    )

    # =====================================================
    # VALIDATE COLUMNS
    # =====================================================

    required_s1_columns = {
        "entity_id",
        "business_name",
        "business_address",
        "country",
    }

    required_target_columns = {
        "entity_id",
        "business_name",
        "business_address",
        "country",
    }

    required_gt_columns = {
        "source1_entity_id",
        "matched_entity_ids",
    }

    missing_s1 = (
        required_s1_columns
        - set(s1.columns)
    )

    missing_s2 = (
        required_target_columns
        - set(s2.columns)
    )

    missing_s3 = (
        required_target_columns
        - set(s3.columns)
    )

    missing_gt = (
        required_gt_columns
        - set(gt.columns)
    )

    if missing_s1:
        raise ValueError(
            f"S1 missing columns: {missing_s1}"
        )

    if missing_s2:
        raise ValueError(
            f"S2 missing columns: {missing_s2}"
        )

    if missing_s3:
        raise ValueError(
            f"S3 missing columns: {missing_s3}"
        )

    if missing_gt:
        raise ValueError(
            f"GT missing columns: {missing_gt}"
        )

    # =====================================================
    # PARSE GROUND TRUTH
    # =====================================================

    print()
    print("=" * 72)
    print("PARSING GROUND TRUTH")
    print("=" * 72)

    truth = parse_ground_truth(
        gt
    )

    print(
        f"Total true matching pairs: "
        f"{len(truth):,}"
    )

    # =====================================================
    # SPLIT GT BY TARGET SOURCE
    # =====================================================

    truth_s2 = {
        pair
        for pair in truth
        if pair[1].startswith("S2-")
    }

    truth_s3 = {
        pair
        for pair in truth
        if pair[1].startswith("S3-")
    }

    print(
        f"GT pairs targeting S2:      "
        f"{len(truth_s2):,}"
    )

    print(
        f"GT pairs targeting S3:      "
        f"{len(truth_s3):,}"
    )

    # =====================================================
    # S1 → S2
    # =====================================================

    result_s2 = evaluate_target(
        source=s1,
        target=s2,
        truth=truth_s2,
        target_name="S2",
    )

    # =====================================================
    # S1 → S3
    # =====================================================

    result_s3 = evaluate_target(
        source=s1,
        target=s3,
        truth=truth_s3,
        target_name="S3",
    )

    # =====================================================
    # COMBINED TARGET
    # =====================================================

    print()
    print("=" * 72)
    print("BUILDING COMBINED TARGET: S2 + S3")
    print("=" * 72)

    target = pd.concat(
        [
            s2,
            s3,
        ],
        ignore_index=True,
    )

    print(
        f"Combined target records: "
        f"{len(target):,}"
    )

    # -----------------------------------------------------
    # Combined GT
    # -----------------------------------------------------

    combined_truth = (
        truth_s2
        | truth_s3
    )

    # -----------------------------------------------------
    # Evaluate combined blocker
    # -----------------------------------------------------

    result_combined = evaluate_target(
        source=s1,
        target=target,
        truth=combined_truth,
        target_name="S2 + S3",
    )

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print()
    print()
    print("=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    print()

    print(
        f"{'Metric':<35}"
        f"{'S1 → S2':>15}"
        f"{'S1 → S3':>15}"
        f"{'S1 → S2+S3':>18}"
    )

    print("-" * 83)

    print(
        f"{'Blocking recall':<35}"
        f"{result_s2['recall'] * 100:>14.4f}%"
        f"{result_s3['recall'] * 100:>14.4f}%"
        f"{result_combined['recall'] * 100:>17.4f}%"
    )

    print(
        f"{'Candidate precision':<35}"
        f"{result_s2['candidate_precision'] * 100:>14.4f}%"
        f"{result_s3['candidate_precision'] * 100:>14.4f}%"
        f"{result_combined['candidate_precision'] * 100:>17.4f}%"
    )

    print(
        f"{'Candidate reduction':<35}"
        f"{result_s2['candidate_reduction'] * 100:>14.4f}%"
        f"{result_s3['candidate_reduction'] * 100:>14.4f}%"
        f"{result_combined['candidate_reduction'] * 100:>17.4f}%"
    )

    print(
        f"{'Average candidates / S1':<35}"
        f"{result_s2['average_candidates']:>15.2f}"
        f"{result_s3['average_candidates']:>15.2f}"
        f"{result_combined['average_candidates']:>18.2f}"
    )

    print(
        f"{'Median candidates / S1':<35}"
        f"{result_s2['median_candidates']:>15.2f}"
        f"{result_s3['median_candidates']:>15.2f}"
        f"{result_combined['median_candidates']:>18.2f}"
    )

    print(
        f"{'P95 candidates / S1':<35}"
        f"{result_s2['p95_candidates']:>15.2f}"
        f"{result_s3['p95_candidates']:>15.2f}"
        f"{result_combined['p95_candidates']:>18.2f}"
    )

    print(
        f"{'P99 candidates / S1':<35}"
        f"{result_s2['p99_candidates']:>15.2f}"
        f"{result_s3['p99_candidates']:>15.2f}"
        f"{result_combined['p99_candidates']:>18.2f}"
    )

    print(
        f"{'Maximum candidates / S1':<35}"
        f"{result_s2['max_candidates']:>15,}"
        f"{result_s3['max_candidates']:>15,}"
        f"{result_combined['max_candidates']:>18,}"
    )

    print(
        f"{'Zero-candidate rate':<35}"
        f"{result_s2['zero_candidate_rate'] * 100:>14.4f}%"
        f"{result_s3['zero_candidate_rate'] * 100:>14.4f}%"
        f"{result_combined['zero_candidate_rate'] * 100:>17.4f}%"
    )

    print()
    print("=" * 72)
    print("BLOCKING EVALUATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()