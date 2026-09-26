"""
Create a small, GT-preserving dataset for testing blocking.

The full dataset contains millions of records, so running the
blocker directly on everything is expensive.

This script creates:

    dataset/debug/
        sample_s1.tsv
        sample_s2.tsv
        sample_s3.tsv
        sample_gt.tsv

The sample is constructed so that:

    - selected S1 records are included
    - ALL of their GT matches are included
    - additional random S2/S3 records are included as negatives

This means blocking recall can be measured correctly.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================

DATA_DIR = Path("dataset/train")
OUTPUT_DIR = Path("dataset/debug")

S1_FILE = DATA_DIR / "train_source1.tsv"
S2_FILE = DATA_DIR / "train_source2.tsv"
S3_FILE = DATA_DIR / "train_source3.tsv"
GT_FILE = DATA_DIR / "train_ground_truth.tsv"

# Number of S1 records used in the debug dataset.
NUM_S1 = 10_000

# Number of random negative S2/S3 records.
#
# These are additional records which are NOT known GT matches
# for the selected S1 records.
NUM_RANDOM_S2 = 50_000
NUM_RANDOM_S3 = 50_000

RANDOM_SEED = 42


# =========================================================
# HELPERS
# =========================================================

def read_tsv(path, nrows=None):
    """
    Read a TSV while keeping all values as strings.
    """

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
    )


def split_gt_ids(value):
    """
    Convert:

        S2-123,S2-456,S3-789

    into:

        ["S2-123", "S2-456", "S3-789"]
    """

    if not value:
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("CREATING BLOCKING DEBUG DATASET")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # -----------------------------------------------------
    # Load GT first
    #
    # GT is relatively small compared with the 5M source
    # files, so this is the easiest place to determine which
    # S1 records we need.
    # -----------------------------------------------------

    print("\n[1/7] Reading ground truth...")

    gt = read_tsv(
        GT_FILE
    )

    print(
        f"GT rows: {len(gt):,}"
    )

    # -----------------------------------------------------
    # Select S1 IDs
    #
    # We sample from GT rather than blindly sampling S1.
    # This guarantees that every selected S1 has known
    # matching information.
    # -----------------------------------------------------

    gt = gt[
        gt["source1_entity_id"].str.len() > 0
    ].copy()

    if len(gt) < NUM_S1:

        raise ValueError(
            f"GT contains only {len(gt):,} "
            f"usable S1 records, but NUM_S1={NUM_S1:,}."
        )

    selected_gt = gt.sample(
        n=NUM_S1,
        random_state=RANDOM_SEED,
    ).copy()

    selected_s1_ids = set(
        selected_gt[
            "source1_entity_id"
        ]
    )

    print(
        f"Selected S1 records: "
        f"{len(selected_s1_ids):,}"
    )

    # -----------------------------------------------------
    # Collect ALL GT target IDs for those S1 records.
    #
    # A single S1 can have multiple matches.
    # -----------------------------------------------------

    print(
        "\n[2/7] Collecting all GT matches..."
    )

    true_s2_ids = set()
    true_s3_ids = set()

    for value in selected_gt[
        "matched_entity_ids"
    ]:

        for entity_id in split_gt_ids(
            value
        ):

            if entity_id.startswith("S2-"):

                true_s2_ids.add(
                    entity_id
                )

            elif entity_id.startswith("S3-"):

                true_s3_ids.add(
                    entity_id
                )

    print(
        f"True S2 matches: "
        f"{len(true_s2_ids):,}"
    )

    print(
        f"True S3 matches: "
        f"{len(true_s3_ids):,}"
    )

    # -----------------------------------------------------
    # Read S1
    # -----------------------------------------------------

    print(
        "\n[3/7] Reading required S1 records..."
    )

    s1 = read_tsv(
        S1_FILE
    )

    sample_s1 = s1[
        s1["entity_id"].isin(
            selected_s1_ids
        )
    ].copy()

    # Preserve GT order is not necessary, but sorting by
    # entity_id makes debugging reproducible.
    sample_s1 = sample_s1.sort_values(
        "entity_id"
    )

    print(
        f"S1 sample: "
        f"{len(sample_s1):,}"
    )

    # -----------------------------------------------------
    # Read S2
    #
    # We need:
    #
    #   1. all true matches
    #   2. random negatives
    # -----------------------------------------------------

    print(
        "\n[4/7] Reading S2..."
    )

    s2 = read_tsv(
        S2_FILE
    )

    true_s2 = s2[
        s2["entity_id"].isin(
            true_s2_ids
        )
    ].copy()

    # Candidate random negative IDs.
    true_s2_id_set = set(
        true_s2_ids
    )

    available_s2 = s2[
        ~s2["entity_id"].isin(
            true_s2_id_set
        )
    ]

    random_count_s2 = min(
        NUM_RANDOM_S2,
        len(available_s2),
    )

    random_s2 = available_s2.sample(
        n=random_count_s2,
        random_state=RANDOM_SEED,
    )

    sample_s2 = pd.concat(
        [
            true_s2,
            random_s2,
        ],
        ignore_index=True,
    ).drop_duplicates(
        subset=["entity_id"]
    )

    sample_s2 = sample_s2.sort_values(
        "entity_id"
    )

    print(
        f"True S2 rows: "
        f"{len(true_s2):,}"
    )

    print(
        f"Random S2 rows: "
        f"{len(random_s2):,}"
    )

    print(
        f"Final S2 sample: "
        f"{len(sample_s2):,}"
    )

    # -----------------------------------------------------
    # Read S3
    # -----------------------------------------------------

    print(
        "\n[5/7] Reading S3..."
    )

    s3 = read_tsv(
        S3_FILE
    )

    true_s3 = s3[
        s3["entity_id"].isin(
            true_s3_ids
        )
    ].copy()

    true_s3_id_set = set(
        true_s3_ids
    )

    available_s3 = s3[
        ~s3["entity_id"].isin(
            true_s3_id_set
        )
    ]

    random_count_s3 = min(
        NUM_RANDOM_S3,
        len(available_s3),
    )

    random_s3 = available_s3.sample(
        n=random_count_s3,
        random_state=RANDOM_SEED + 1,
    )

    sample_s3 = pd.concat(
        [
            true_s3,
            random_s3,
        ],
        ignore_index=True,
    ).drop_duplicates(
        subset=["entity_id"]
    )

    sample_s3 = sample_s3.sort_values(
        "entity_id"
    )

    print(
        f"True S3 rows: "
        f"{len(true_s3):,}"
    )

    print(
        f"Random S3 rows: "
        f"{len(random_s3):,}"
    )

    print(
        f"Final S3 sample: "
        f"{len(sample_s3):,}"
    )

    # -----------------------------------------------------
    # Create GT restricted to our selected S1 records.
    #
    # We keep only matches that are actually present in the
    # sampled S2/S3 data.
    # -----------------------------------------------------

    print(
        "\n[6/7] Creating sample GT..."
    )

    available_target_ids = set(
        sample_s2["entity_id"]
    )

    available_target_ids.update(
        sample_s3["entity_id"]
    )

    def filter_gt_ids(value):

        ids = split_gt_ids(value)

        ids = [
            entity_id
            for entity_id in ids
            if entity_id in available_target_ids
        ]

        return ",".join(ids)

    sample_gt = selected_gt.copy()

    sample_gt[
        "matched_entity_ids"
    ] = sample_gt[
        "matched_entity_ids"
    ].map(
        filter_gt_ids
    )

    # We should never lose a GT match because we explicitly
    # included all GT target IDs.
    sample_gt = sample_gt[
        sample_gt["matched_entity_ids"].str.len() > 0
    ].copy()

    sample_gt = sample_gt.sort_values(
        "source1_entity_id"
    )

    print(
        f"Final GT rows: "
        f"{len(sample_gt):,}"
    )

    # -----------------------------------------------------
    # Write files
    # -----------------------------------------------------

    print(
        "\n[7/7] Writing debug dataset..."
    )

    sample_s1.to_csv(
        OUTPUT_DIR / "sample_s1.tsv",
        sep="\t",
        index=False,
    )

    sample_s2.to_csv(
        OUTPUT_DIR / "sample_s2.tsv",
        sep="\t",
        index=False,
    )

    sample_s3.to_csv(
        OUTPUT_DIR / "sample_s3.tsv",
        sep="\t",
        index=False,
    )

    sample_gt.to_csv(
        OUTPUT_DIR / "sample_gt.tsv",
        sep="\t",
        index=False,
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"S1 : {len(sample_s1):,}"
    )

    print(
        f"S2 : {len(sample_s2):,}"
    )

    print(
        f"S3 : {len(sample_s3):,}"
    )

    print(
        f"GT : {len(sample_gt):,}"
    )

    print(
        f"\nOutput directory:"
        f" {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()