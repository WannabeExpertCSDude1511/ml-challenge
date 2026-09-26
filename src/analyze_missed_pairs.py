import os
import pandas as pd

from .blocking import (
    build_indices,
    generate_candidates_for_row,
)

from .normalize import (
    normalize_country,
    normalize_name,
    normalize_name_core,
    normalize_address,
    tokens,
    char_ngrams,
    jaccard,
)


# ============================================================
# PATHS
# ============================================================

DEBUG_DIR = "dataset/debug"

S1_PATH = (
    f"{DEBUG_DIR}/source1.tsv"
)

S2_PATH = (
    f"{DEBUG_DIR}/source2.tsv"
)

S3_PATH = (
    f"{DEBUG_DIR}/source3.tsv"
)

GT_PATH = (
    f"{DEBUG_DIR}/ground_truth.tsv"
)

OUTPUT_PATH = (
    f"{DEBUG_DIR}/missed_pairs_detailed.tsv"
)


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_file(
    candidates,
):

    for path in candidates:

        if os.path.exists(path):

            return path

    return None


def resolve_paths():

    s1 = find_file(
        [
            S1_PATH,
            f"{DEBUG_DIR}/train_source1.tsv",
            f"{DEBUG_DIR}/source1.tsv",
        ]
    )

    s2 = find_file(
        [
            S2_PATH,
            f"{DEBUG_DIR}/train_source2.tsv",
            f"{DEBUG_DIR}/source2.tsv",
        ]
    )

    s3 = find_file(
        [
            S3_PATH,
            f"{DEBUG_DIR}/train_source3.tsv",
            f"{DEBUG_DIR}/source3.tsv",
        ]
    )

    gt = find_file(
        [
            GT_PATH,
            f"{DEBUG_DIR}/train_ground_truth.tsv",
            f"{DEBUG_DIR}/ground_truth.tsv",
        ]
    )

    return s1, s2, s3, gt


# ============================================================
# LOAD
# ============================================================

def load_data():

    s1_path, s2_path, s3_path, gt_path = (
        resolve_paths()
    )

    if not s1_path:
        raise FileNotFoundError(
            "Could not find debug S1 TSV."
        )

    if not s2_path:
        raise FileNotFoundError(
            "Could not find debug S2 TSV."
        )

    if not s3_path:
        raise FileNotFoundError(
            "Could not find debug S3 TSV."
        )

    if not gt_path:
        raise FileNotFoundError(
            "Could not find debug GT TSV."
        )

    print(
        f"S1: {s1_path}"
    )

    print(
        f"S2: {s2_path}"
    )

    print(
        f"S3: {s3_path}"
    )

    print(
        f"GT: {gt_path}"
    )

    s1 = pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
    ).fillna("")

    s2 = pd.read_csv(
        s2_path,
        sep="\t",
        dtype=str,
    ).fillna("")

    s3 = pd.read_csv(
        s3_path,
        sep="\t",
        dtype=str,
    ).fillna("")

    gt = pd.read_csv(
        gt_path,
        sep="\t",
        dtype=str,
    ).fillna("")

    return (
        s1,
        s2,
        s3,
        gt,
    )


# ============================================================
# GT EXPANSION
# ============================================================

def build_gt_pairs(gt):

    pairs = []

    for _, row in gt.iterrows():

        s1_id = row[
            "source1_entity_id"
        ]

        matched = row[
            "matched_entity_ids"
        ]

        if not matched:
            continue

        for target_id in str(
            matched
        ).split(","):

            target_id = target_id.strip()

            if not target_id:
                continue

            pairs.append(
                (
                    s1_id,
                    target_id,
                )
            )

    return pairs


# ============================================================
# ENTITY MAP
# ============================================================

def build_entity_map(
    s2,
    s3,
):

    result = {}

    for _, row in s2.iterrows():

        result[
            row["entity_id"]
        ] = row

    for _, row in s3.iterrows():

        result[
            row["entity_id"]
        ] = row

    return result


# ============================================================
# ROW MAP
# ============================================================

def build_s1_map(s1):

    return {
        row["entity_id"]: row
        for _, row in s1.iterrows()
    }


# ============================================================
# CHECK BLOCKING KEYS
# ============================================================

def get_shared_key_information(
    s1_row,
    target_row,
    indices,
):

    index_data = indices

    if "indexes" in index_data:

        index_data = index_data[
            "indexes"
        ]

    country = normalize_country(
        s1_row["country"]
    )

    target_country = normalize_country(
        target_row["country"]
    )

    if country != target_country:

        return {
            "same_country": False,
            "shared_keys": [],
        }

    s1_name = normalize_name(
        s1_row["business_name"]
    )

    target_name = normalize_name(
        target_row["business_name"]
    )

    s1_core = normalize_name_core(
        s1_row["business_name"]
    )

    target_core = normalize_name_core(
        target_row["business_name"]
    )

    s1_address = normalize_address(
        s1_row["business_address"]
    )

    target_address = normalize_address(
        target_row["business_address"]
    )

    shared = []

    # --------------------------------------------------------
    # Exact name
    # --------------------------------------------------------

    if s1_name and s1_name == target_name:

        shared.append(
            "exact_name"
        )

    # --------------------------------------------------------
    # Core
    # --------------------------------------------------------

    if s1_core and s1_core == target_core:

        shared.append(
            "name_core"
        )

    # --------------------------------------------------------
    # Address
    # --------------------------------------------------------

    if (
        s1_address
        and s1_address == target_address
    ):

        shared.append(
            "exact_address"
        )

    # --------------------------------------------------------
    # Tokens
    # --------------------------------------------------------

    s1_tokens = tokens(
        s1_name
    )

    target_tokens = tokens(
        target_name
    )

    common_tokens = (
        s1_tokens
        & target_tokens
    )

    if common_tokens:

        shared.append(
            "shared_name_token"
        )

    # --------------------------------------------------------
    # Token pair
    # --------------------------------------------------------

    if len(common_tokens) >= 2:

        shared.append(
            "shared_name_token_pair"
        )

    # --------------------------------------------------------
    # N-gram
    # --------------------------------------------------------

    s1_grams = char_ngrams(
        s1_name,
        3,
    )

    target_grams = char_ngrams(
        target_name,
        3,
    )

    common_grams = (
        s1_grams
        & target_grams
    )

    if len(common_grams) >= 2:

        shared.append(
            "shared_ngrams"
        )

    # --------------------------------------------------------
    # Address tokens
    # --------------------------------------------------------

    s1_address_tokens = set(
        s1_address.split()
    )

    target_address_tokens = set(
        target_address.split()
    )

    common_address = (
        s1_address_tokens
        & target_address_tokens
    )

    if common_address:

        shared.append(
            "shared_address_token"
        )

    return {
        "same_country": True,
        "shared_keys": shared,
    }


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_pair(
    s1_row,
    target_row,
    key_info,
):

    s1_name = normalize_name(
        s1_row["business_name"]
    )

    target_name = normalize_name(
        target_row["business_name"]
    )

    s1_core = normalize_name_core(
        s1_row["business_name"]
    )

    target_core = normalize_name_core(
        target_row["business_name"]
    )

    s1_tokens = tokens(
        s1_name
    )

    target_tokens = tokens(
        target_name
    )

    common_tokens = (
        s1_tokens
        & target_tokens
    )

    # --------------------------------------------------------
    # No country
    # --------------------------------------------------------

    if not key_info["same_country"]:

        return "COUNTRY_MISMATCH"

    # --------------------------------------------------------
    # Shared obvious key
    # --------------------------------------------------------

    if key_info["shared_keys"]:

        return (
            "HAS_SHARED_KEY_CHECK_INDEX"
        )

    # --------------------------------------------------------
    # Multilingual / transliteration issue
    # --------------------------------------------------------

    raw_s1 = str(
        s1_row["business_name"]
    )

    raw_target = str(
        target_row["business_name"]
    )

    s1_ascii = all(
        ord(c) < 128
        for c in raw_s1
        if c.strip()
    )

    target_ascii = all(
        ord(c) < 128
        for c in raw_target
        if c.strip()
    )

    if (
        s1_ascii != target_ascii
    ):

        return (
            "MULTILINGUAL_NAME"
        )

    # --------------------------------------------------------
    # Name exists but no overlap
    # --------------------------------------------------------

    if s1_name and target_name:

        return (
            "NO_NAME_OVERLAP"
        )

    # --------------------------------------------------------
    # Missing address
    # --------------------------------------------------------

    if (
        not s1_row["business_address"]
        or not target_row["business_address"]
    ):

        return (
            "ADDRESS_MISSING"
        )

    return "OTHER"


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "ANALYSE MISSED BLOCKING PAIRS"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print(
        "\n[1/6] Loading data..."
    )

    (
        s1,
        s2,
        s3,
        gt,
    ) = load_data()

    print(
        f"S1 rows: {len(s1):,}"
    )

    print(
        f"S2 rows: {len(s2):,}"
    )

    print(
        f"S3 rows: {len(s3):,}"
    )

    print(
        f"GT rows: {len(gt):,}"
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    print(
        "\n[2/6] Building target..."
    )

    target = pd.concat(
        [
            s2,
            s3,
        ],
        ignore_index=True,
    )

    target = target.drop_duplicates(
        subset=[
            "entity_id"
        ]
    )

    target = target.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # GT
    # --------------------------------------------------------

    print(
        "\n[3/6] Expanding ground truth..."
    )

    gt_pairs = build_gt_pairs(
        gt
    )

    print(
        f"GT pairs: {len(gt_pairs):,}"
    )

    # --------------------------------------------------------
    # Build indexes
    # --------------------------------------------------------

    print(
        "\n[4/6] Building blocking indexes..."
    )

    indices = build_indices(
        target
    )

    s1_map = build_s1_map(
        s1
    )

    target_map = build_entity_map(
        target,
        target.iloc[0:0],
    )

    # target_map above only receives S2-like rows,
    # so rebuild directly for all target rows.

    target_map = {
        row["entity_id"]: row
        for _, row in target.iterrows()
    }

    # --------------------------------------------------------
    # Find missed
    # --------------------------------------------------------

    print(
        "\n[5/6] Finding missed pairs..."
    )

    missed = []

    checked_s1 = {}

    for counter, (
        s1_id,
        target_id,
    ) in enumerate(
        gt_pairs,
        start=1,
    ):

        if counter % 10000 == 0:

            print(
                f"Checked {counter:,} / "
                f"{len(gt_pairs):,}",
                end="\r",
            )

        s1_row = s1_map.get(
            s1_id
        )

        target_row = target_map.get(
            target_id
        )

        if s1_row is None:
            continue

        if target_row is None:
            continue

        # Cache candidate sets per S1.
        if s1_id not in checked_s1:

            candidates = (
                generate_candidates_for_row(
                    s1_row,
                    target,
                    indices,
                )
            )

            checked_s1[s1_id] = set(
                candidates[
                    "entity_id"
                ].astype(str)
            )

        candidate_ids = (
            checked_s1[s1_id]
        )

        if str(target_id) in candidate_ids:

            continue

        key_info = (
            get_shared_key_information(
                s1_row,
                target_row,
                indices,
            )
        )

        reason = classify_pair(
            s1_row,
            target_row,
            key_info,
        )

        s1_name = normalize_name(
            s1_row["business_name"]
        )

        target_name = normalize_name(
            target_row["business_name"]
        )

        s1_core = normalize_name_core(
            s1_row["business_name"]
        )

        target_core = normalize_name_core(
            target_row["business_name"]
        )

        s1_address = normalize_address(
            s1_row["business_address"]
        )

        target_address = normalize_address(
            target_row["business_address"]
        )

        missed.append(
            {
                "s1_entity_id":
                    s1_id,

                "target_entity_id":
                    target_id,

                "country":
                    s1_row["country"],

                "s1_name":
                    s1_row["business_name"],

                "target_name":
                    target_row["business_name"],

                "s1_name_normalized":
                    s1_name,

                "target_name_normalized":
                    target_name,

                "s1_name_core":
                    s1_core,

                "target_name_core":
                    target_core,

                "s1_address":
                    s1_row[
                        "business_address"
                    ],

                "target_address":
                    target_row[
                        "business_address"
                    ],

                "s1_address_normalized":
                    s1_address,

                "target_address_normalized":
                    target_address,

                "s1_tokens":
                    " ".join(
                        sorted(
                            tokens(
                                s1_name
                            )
                        )
                    ),

                "target_tokens":
                    " ".join(
                        sorted(
                            tokens(
                                target_name
                            )
                        )
                    ),

                "shared_keys":
                    "|".join(
                        key_info[
                            "shared_keys"
                        ]
                    ),

                "reason":
                    reason,
            }
        )

    print()

    # --------------------------------------------------------
    # Write
    # --------------------------------------------------------

    print(
        "\n[6/6] Writing results..."
    )

    missed_df = pd.DataFrame(
        missed
    )

    os.makedirs(
        DEBUG_DIR,
        exist_ok=True,
    )

    missed_df.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"Total GT pairs : {len(gt_pairs):,}"
    )

    print(
        f"Missed pairs   : {len(missed_df):,}"
    )

    if len(missed_df):

        print(
            "\nReasons:"
        )

        counts = (
            missed_df[
                "reason"
            ]
            .value_counts()
        )

        for reason, count in counts.items():

            print(
                f"  {reason:<35} {count:,}"
            )

    print()
    print(
        f"Output: {OUTPUT_PATH}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()