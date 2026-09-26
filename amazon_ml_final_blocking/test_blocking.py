"""
Evaluate blocking quality on the GT-preserving debug dataset.

Metrics:

    1. Blocking recall
    2. Average candidates per S1
    3. Median candidates per S1
    4. Maximum candidates for one S1
    5. Total candidate pairs
    6. Candidate reduction
    7. Number of S1 records with zero candidates
    8. Number of S1 records where at least one GT match
       was found
    9. Number of S1 records where ALL GT matches were found
   10. Runtime
"""

import time
from pathlib import Path

import pandas as pd

from src.blocking import (
    build_indices,
    generate_candidates_for_row,
)


# =========================================================
# PATHS
# =========================================================

DATA_DIR = Path(
    "dataset/debug"
)

S1_FILE = (
    DATA_DIR /
    "sample_s1.tsv"
)

S2_FILE = (
    DATA_DIR /
    "sample_s2.tsv"
)

S3_FILE = (
    DATA_DIR /
    "sample_s3.tsv"
)

GT_FILE = (
    DATA_DIR /
    "sample_gt.tsv"
)


# =========================================================
# HELPERS
# =========================================================

def read_tsv(path):

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )


def split_ids(value):

    if not value:
        return set()

    return {
        x.strip()
        for x in value.split(",")
        if x.strip()
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("BLOCKING EVALUATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    start_total = time.perf_counter()

    print("\n[1/5] Loading debug dataset...")

    s1 = read_tsv(
        S1_FILE
    )

    s2 = read_tsv(
        S2_FILE
    )

    s3 = read_tsv(
        S3_FILE
    )

    gt = read_tsv(
        GT_FILE
    )

    target = pd.concat(
        [
            s2,
            s3,
        ],
        ignore_index=True,
    )

    print(
        f"S1 records : {len(s1):,}"
    )

    print(
        f"S2 records : {len(s2):,}"
    )

    print(
        f"S3 records : {len(s3):,}"
    )

    print(
        f"Target     : {len(target):,}"
    )

    print(
        f"GT records : {len(gt):,}"
    )

    # -----------------------------------------------------
    # Build indexes
    # -----------------------------------------------------

    print(
        "\n[2/5] Building blocking indexes..."
    )

    start_index = time.perf_counter()

    indices = build_indices(
        target
    )

    index_time = (
        time.perf_counter()
        - start_index
    )

    print(
        f"Index construction: "
        f"{index_time:.2f} seconds"
    )

    # -----------------------------------------------------
    # Build GT lookup
    # -----------------------------------------------------

    print(
        "\n[3/5] Preparing GT..."
    )

    gt_lookup = {}

    for row in gt.itertuples(
        index=False
    ):

        gt_lookup[
            row.source1_entity_id
        ] = split_ids(
            row.matched_entity_ids
        )

    # -----------------------------------------------------
    # Evaluate each S1
    # -----------------------------------------------------

    print(
        "\n[4/5] Running blocker..."
    )

    candidate_counts = []

    total_gt_matches = 0
    total_found_gt_matches = 0

    s1_with_any_match = 0
    s1_with_all_matches = 0
    s1_with_zero_candidates = 0

    total_candidate_pairs = 0

    start_blocking = time.perf_counter()

    # -----------------------------------------------------
    # Use itertuples.
    # -----------------------------------------------------

    for row in s1.itertuples(
        index=False
    ):

        source_id = row.entity_id

        source_row = {
            "entity_id": row.entity_id,
            "business_name": row.business_name,
            "business_address": row.business_address,
            "country": row.country,
        }

        candidates = (
            generate_candidates_for_row(
                source_row,
                target,
                indices,
            )
        )

        candidate_ids = set(
            candidates[
                "entity_id"
            ].astype(str)
        )

        candidate_count = len(
            candidate_ids
        )

        candidate_counts.append(
            candidate_count
        )

        total_candidate_pairs += (
            candidate_count
        )

        # -------------------------------------------------
        # GT
        # -------------------------------------------------

        true_ids = gt_lookup.get(
            source_id,
            set(),
        )

        total_gt_matches += len(
            true_ids
        )

        found_ids = (
            true_ids &
            candidate_ids
        )

        total_found_gt_matches += (
            len(found_ids)
        )

        if found_ids:
            s1_with_any_match += 1

        if (
            true_ids
            and found_ids == true_ids
        ):
            s1_with_all_matches += 1

        if candidate_count == 0:
            s1_with_zero_candidates += 1

    blocking_time = (
        time.perf_counter()
        - start_blocking
    )

    total_time = (
        time.perf_counter()
        - start_total
    )

    # =====================================================
    # METRICS
    # =====================================================

    candidate_series = pd.Series(
        candidate_counts
    )

    # -----------------------------------------------------
    # Match-level recall
    # -----------------------------------------------------

    if total_gt_matches > 0:

        recall = (
            total_found_gt_matches
            / total_gt_matches
        )

    else:

        recall = 0.0

    # -----------------------------------------------------
    # Candidate reduction
    #
    # Without blocking, every S1 could theoretically be
    # compared with every target.
    # -----------------------------------------------------

    possible_pairs = (
        len(s1) *
        len(target)
    )

    if possible_pairs > 0:

        reduction = (
            1
            -
            (
                total_candidate_pairs
                /
                possible_pairs
            )
        )

    else:

        reduction = 0.0

    # -----------------------------------------------------
    # S1-level recall
    #
    # "Did we find at least one true match?"
    # -----------------------------------------------------

    if len(s1) > 0:

        s1_any_recall = (
            s1_with_any_match
            /
            len(s1)
        )

    else:

        s1_any_recall = 0.0

    # -----------------------------------------------------
    # Complete-match recall
    #
    # "Did we find ALL GT matches for this S1?"
    # -----------------------------------------------------

    if len(s1) > 0:

        s1_all_recall = (
            s1_with_all_matches
            /
            len(s1)
        )

    else:

        s1_all_recall = 0.0

    # =====================================================
    # RESULTS
    # =====================================================

    print(
        "\n[5/5] RESULTS"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        f"S1 records:                  "
        f"{len(s1):,}"
    )

    print(
        f"Target records:              "
        f"{len(target):,}"
    )

    print(
        f"Total possible pairs:        "
        f"{possible_pairs:,}"
    )

    print(
        f"Actual candidate pairs:      "
        f"{total_candidate_pairs:,}"
    )

    print(
        f"Candidate reduction:         "
        f"{reduction * 100:.4f}%"
    )

    print(
        "\n--- Candidate statistics ---"
    )

    print(
        f"Average candidates / S1:     "
        f"{candidate_series.mean():.2f}"
    )

    print(
        f"Median candidates / S1:      "
        f"{candidate_series.median():.2f}"
    )

    print(
        f"Maximum candidates / S1:     "
        f"{candidate_series.max():,}"
    )

    print(
        f"S1 with zero candidates:     "
        f"{s1_with_zero_candidates:,}"
    )

    print(
        "\n--- Blocking recall ---"
    )

    print(
        f"Total GT matches:             "
        f"{total_gt_matches:,}"
    )

    print(
        f"GT matches found:             "
        f"{total_found_gt_matches:,}"
    )

    print(
        f"Match-level recall:           "
        f"{recall * 100:.4f}%"
    )

    print(
        f"S1 with ≥1 GT match found:    "
        f"{s1_with_any_match:,}"
        f" / {len(s1):,}"
    )

    print(
        f"S1-level recall:              "
        f"{s1_any_recall * 100:.4f}%"
    )

    print(
        f"S1 with ALL GT matches found: "
        f"{s1_with_all_matches:,}"
        f" / {len(s1):,}"
    )

    print(
        f"Complete-match recall:        "
        f"{s1_all_recall * 100:.4f}%"
    )

    print(
        "\n--- Runtime ---"
    )

    print(
        f"Index construction:           "
        f"{index_time:.2f}s"
    )

    print(
        f"Blocking:                     "
        f"{blocking_time:.2f}s"
    )

    print(
        f"Total:                        "
        f"{total_time:.2f}s"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()