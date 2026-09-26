"""
Create a small blocking-debug dataset from the ground truth.

Approach
--------
1. Read 10,000 GT rows.
2. Extract all S1 IDs appearing in those GT rows.
3. Extract all S2 IDs and S3 IDs appearing in those GT rows.
4. Search the full source TSV files for those exact IDs.
5. Write the required records to dataset/debug/.
6. If debug files already exist, reuse them and only add missing records.

This allows blocking experiments to be run repeatedly on a small,
fixed dataset instead of repeatedly processing ~5M-row TSV files.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

TRAIN_DIR = Path("dataset/train")
DEBUG_DIR = Path("dataset/debug")

GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"
S1_FILE = TRAIN_DIR / "train_source1.tsv"
S2_FILE = TRAIN_DIR / "train_source2.tsv"
S3_FILE = TRAIN_DIR / "train_source3.tsv"

DEBUG_GT_FILE = DEBUG_DIR / "ground_truth.tsv"
DEBUG_S1_FILE = DEBUG_DIR / "source1.tsv"
DEBUG_S2_FILE = DEBUG_DIR / "source2.tsv"
DEBUG_S3_FILE = DEBUG_DIR / "source3.tsv"

# Number of GT rows to use.
N_GT_ROWS = 10_000


# ============================================================
# HELPERS
# ============================================================

def read_existing_ids(path, id_column):
    """
    Read only the ID column from an existing debug TSV.

    Returns an empty set if the file doesn't exist.
    """

    if not path.exists():
        return set()

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        usecols=[id_column],
    )

    return set(
        df[id_column]
        .dropna()
        .astype(str)
    )


def append_missing_records(
    full_file,
    debug_file,
    id_column,
    required_ids,
):
    """
    Search the full TSV for required IDs and add only records
    that are not already present in the debug TSV.

    The full file is processed in chunks so we don't load the
    entire 5M-row file into RAM.
    """

    required_ids = {
        str(x)
        for x in required_ids
        if pd.notna(x)
    }

    existing_ids = read_existing_ids(
        debug_file,
        id_column,
    )

    missing_ids = required_ids - existing_ids

    print()
    print(f"File        : {full_file.name}")
    print(f"Required    : {len(required_ids):,}")
    print(f"Already have: {len(existing_ids):,}")
    print(f"Missing     : {len(missing_ids):,}")

    if not missing_ids:
        print("Nothing to add.")
        return

    found_parts = []

    # --------------------------------------------------------
    # Read the huge source file in chunks.
    # --------------------------------------------------------

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            full_file,
            sep="\t",
            dtype=str,
            chunksize=100_000,
            keep_default_na=False,
        ),
        start=1,
    ):

        # Convert ID to string.
        ids = chunk[id_column].astype(str)

        mask = ids.isin(missing_ids)

        if mask.any():

            found = chunk.loc[mask].copy()

            found_parts.append(found)

            # Remove IDs we have already found.
            found_ids = set(
                found[id_column].astype(str)
            )

            missing_ids -= found_ids

            print(
                f"  chunk {chunk_number:>3}: "
                f"found {len(found_ids):,} "
                f"| remaining {len(missing_ids):,}"
            )

            # We can stop once everything has been found.
            if not missing_ids:
                break

    # --------------------------------------------------------
    # Check for missing records.
    # --------------------------------------------------------

    if missing_ids:

        print(
            f"WARNING: {len(missing_ids):,} "
            f"required IDs were not found in "
            f"{full_file.name}"
        )

        print(
            "First missing IDs:"
        )

        for value in sorted(
            missing_ids
        )[:20]:

            print(
                f"    {value}"
            )

    if not found_parts:

        print("No new records found.")

        return

    new_records = pd.concat(
        found_parts,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Append or create debug file.
    # --------------------------------------------------------

    if debug_file.exists():

        new_records.to_csv(
            debug_file,
            sep="\t",
            index=False,
            mode="a",
            header=False,
        )

    else:

        new_records.to_csv(
            debug_file,
            sep="\t",
            index=False,
        )

    print(
        f"Added {len(new_records):,} records."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CREATING / UPDATING BLOCKING DEBUG DATASET")
    print("=" * 70)

    DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. READ GT SAMPLE
    # ========================================================

    print()
    print("[1/6] Reading ground truth...")

    gt = pd.read_csv(
        GT_FILE,
        sep="\t",
        dtype=str,
        nrows=N_GT_ROWS,
        keep_default_na=False,
    )

    print(
        f"GT rows selected: {len(gt):,}"
    )

    print(
        f"GT columns: {list(gt.columns)}"
    )

    # --------------------------------------------------------
    # Validate GT schema.
    # --------------------------------------------------------

    required_gt_columns = {
        "source1_entity_id",
        "matched_entity_ids",
    }

    missing = (
        required_gt_columns
        - set(gt.columns)
    )

    if missing:

        raise ValueError(
            "Ground truth is missing columns: "
            f"{sorted(missing)}"
        )

    # ========================================================
    # 2. COLLECT REQUIRED IDS
    # ========================================================

    print()
    print("[2/6] Collecting IDs from GT...")

    # --------------------------------------------------------
    # S1 IDs
    # --------------------------------------------------------

    s1_ids = set(
        gt["source1_entity_id"]
        .dropna()
        .astype(str)
    )

    # --------------------------------------------------------
    # S2 / S3 IDs
    # --------------------------------------------------------

    s2_ids = set()
    s3_ids = set()

    for value in gt["matched_entity_ids"]:

        if not value:
            continue

        # A GT row can contain:
        #
        # S2-123,S2-456,S3-789
        #
        for entity_id in str(
            value
        ).split(","):

            entity_id = entity_id.strip()

            if not entity_id:
                continue

            if entity_id.startswith(
                "S2-"
            ):

                s2_ids.add(
                    entity_id
                )

            elif entity_id.startswith(
                "S3-"
            ):

                s3_ids.add(
                    entity_id
                )

    print(
        f"Unique S1 IDs: {len(s1_ids):,}"
    )

    print(
        f"Unique S2 IDs: {len(s2_ids):,}"
    )

    print(
        f"Unique S3 IDs: {len(s3_ids):,}"
    )

    # ========================================================
    # 3. SAVE / UPDATE GT
    # ========================================================

    print()
    print("[3/6] Updating debug ground truth...")

    # --------------------------------------------------------
    # We keep exactly the selected GT rows.
    #
    # If the file already exists, don't duplicate them.
    # --------------------------------------------------------

    if DEBUG_GT_FILE.exists():

        existing_gt = pd.read_csv(
            DEBUG_GT_FILE,
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )

        existing_keys = set(
            existing_gt.apply(
                lambda row: (
                    row[
                        "source1_entity_id"
                    ],
                    row[
                        "matched_entity_ids"
                    ],
                ),
                axis=1,
            )
        )

        new_gt_rows = []

        for _, row in gt.iterrows():

            key = (
                row[
                    "source1_entity_id"
                ],
                row[
                    "matched_entity_ids"
                ],
            )

            if key not in existing_keys:

                new_gt_rows.append(
                    row
                )

        if new_gt_rows:

            new_gt = pd.DataFrame(
                new_gt_rows
            )

            new_gt.to_csv(
                DEBUG_GT_FILE,
                sep="\t",
                index=False,
                mode="a",
                header=False,
            )

            print(
                f"Added GT rows: "
                f"{len(new_gt):,}"
            )

        else:

            print(
                "No new GT rows."
            )

    else:

        gt.to_csv(
            DEBUG_GT_FILE,
            sep="\t",
            index=False,
        )

        print(
            f"Created GT: "
            f"{len(gt):,} rows"
        )

    # ========================================================
    # 4. FIND S1 RECORDS
    # ========================================================

    print()
    print("[4/6] Updating debug S1...")

    append_missing_records(
        full_file=S1_FILE,
        debug_file=DEBUG_S1_FILE,
        id_column="entity_id",
        required_ids=s1_ids,
    )

    # ========================================================
    # 5. FIND S2 RECORDS
    # ========================================================

    print()
    print("[5/6] Updating debug S2...")

    append_missing_records(
        full_file=S2_FILE,
        debug_file=DEBUG_S2_FILE,
        id_column="entity_id",
        required_ids=s2_ids,
    )

    # ========================================================
    # 6. FIND S3 RECORDS
    # ========================================================

    print()
    print("[6/6] Updating debug S3...")

    append_missing_records(
        full_file=S3_FILE,
        debug_file=DEBUG_S3_FILE,
        id_column="entity_id",
        required_ids=s3_ids,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print()

    for label, path in [
        ("S1", DEBUG_S1_FILE),
        ("S2", DEBUG_S2_FILE),
        ("S3", DEBUG_S3_FILE),
        ("GT", DEBUG_GT_FILE),
    ]:

        if path.exists():

            # Only count lines here.
            #
            # This avoids loading the entire debug file
            # unnecessarily.
            with open(
                path,
                "r",
                encoding="utf-8",
            ) as f:

                row_count = sum(
                    1
                    for _ in f
                ) - 1

            print(
                f"{label:<4}: "
                f"{max(row_count, 0):,}"
            )

        else:

            print(
                f"{label:<4}: 0"
            )

    print()
    print(
        f"Output directory: {DEBUG_DIR}"
    )


if __name__ == "__main__":
    main()