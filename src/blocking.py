from collections import defaultdict, Counter
import math

from .normalize import (
    normalize_country,
    normalize_name,
    normalize_name_core,
    normalize_address,
    tokens,
    address_tokens,
    address_location_tokens,
    numeric_tokens,
    char_ngrams,
    token_pairs,
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_NAME_TOKEN_FREQUENCY = 500

MAX_ADDRESS_TOKEN_FREQUENCY = 500

MAX_TOKEN_PAIR_FREQUENCY = 1000

NGRAM_SIZE = 3

MAX_NGRAM_FREQUENCY = 500

MIN_SHARED_RARE_NGRAMS = 2

MAX_ADDRESS_NUMBER_LOCATION_FREQUENCY = 500


# ============================================================
# HELPER
# ============================================================

def _add(
    index,
    key,
    row_idx,
):

    if key is None:
        return

    index[key].add(
        row_idx
    )


# ============================================================
# FREQUENCY BUILDING
# ============================================================

def _build_frequencies(target):

    name_token_frequency = Counter()

    address_token_frequency = Counter()

    token_pair_frequency = Counter()

    ngram_frequency = Counter()

    address_number_location_frequency = Counter()

    for _, row in target.iterrows():

        country = normalize_country(
            row.get("country", "")
        )

        name = normalize_name(
            row.get("business_name", "")
        )

        address = normalize_address(
            row.get("business_address", "")
        )

        # ----------------------------------------------------
        # Name tokens
        # ----------------------------------------------------

        for token in tokens(name):

            if len(token) >= 3:

                name_token_frequency[
                    (
                        country,
                        token,
                    )
                ] += 1

        # ----------------------------------------------------
        # Name pairs
        # ----------------------------------------------------

        for pair in token_pairs(name):

            token_pair_frequency[
                (
                    country,
                    pair[0],
                    pair[1],
                )
            ] += 1

        # ----------------------------------------------------
        # Address tokens
        # ----------------------------------------------------

        for token in address_tokens(
            address
        ):

            address_token_frequency[
                (
                    country,
                    token,
                )
            ] += 1

        # ----------------------------------------------------
        # Name ngrams
        # ----------------------------------------------------

        for gram in char_ngrams(
            name,
            NGRAM_SIZE,
        ):

            ngram_frequency[
                (
                    country,
                    gram,
                )
            ] += 1

        # ----------------------------------------------------
        # Address number + location
        # ----------------------------------------------------

        numbers = numeric_tokens(
            address
        )

        locations = address_location_tokens(
            address
        )

        for number in numbers:

            for location in locations:

                address_number_location_frequency[
                    (
                        country,
                        number,
                        location,
                    )
                ] += 1

    return {
        "name_token":
            name_token_frequency,

        "address_token":
            address_token_frequency,

        "token_pair":
            token_pair_frequency,

        "ngram":
            ngram_frequency,

        "address_num_location":
            address_number_location_frequency,
    }


# ============================================================
# BUILD INDEXES
# ============================================================

def build_indices(target):

    frequencies = _build_frequencies(
        target
    )

    indexes = {

        "country_name":
            defaultdict(set),

        "country_core":
            defaultdict(set),

        "country_address":
            defaultdict(set),

        "rare_name_token":
            defaultdict(set),

        "rare_name_pair":
            defaultdict(set),

        "rare_address_token":
            defaultdict(set),

        "address_num_location":
            defaultdict(set),

        "rare_ngram":
            defaultdict(set),
    }

    for idx, row in target.iterrows():

        country = normalize_country(
            row.get("country", "")
        )

        name = normalize_name(
            row.get("business_name", "")
        )

        core = normalize_name_core(
            row.get("business_name", "")
        )

        address = normalize_address(
            row.get("business_address", "")
        )

        # ====================================================
        # EXACT NAME
        # ====================================================

        if name:

            _add(
                indexes["country_name"],
                (
                    country,
                    name,
                ),
                idx,
            )

        # ====================================================
        # NAME CORE
        # ====================================================

        if core:

            _add(
                indexes["country_core"],
                (
                    country,
                    core,
                ),
                idx,
            )

        # ====================================================
        # EXACT ADDRESS
        # ====================================================

        if address:

            _add(
                indexes["country_address"],
                (
                    country,
                    address,
                ),
                idx,
            )

        # ====================================================
        # RARE NAME TOKEN
        # ====================================================

        for token in tokens(name):

            if len(token) < 3:
                continue

            frequency = frequencies[
                "name_token"
            ].get(
                (
                    country,
                    token,
                ),
                0,
            )

            if frequency <= MAX_NAME_TOKEN_FREQUENCY:

                _add(
                    indexes["rare_name_token"],
                    (
                        country,
                        token,
                    ),
                    idx,
                )

        # ====================================================
        # RARE NAME PAIR
        # ====================================================

        for pair in token_pairs(name):

            frequency = frequencies[
                "token_pair"
            ].get(
                (
                    country,
                    pair[0],
                    pair[1],
                ),
                0,
            )

            if frequency <= MAX_TOKEN_PAIR_FREQUENCY:

                _add(
                    indexes["rare_name_pair"],
                    (
                        country,
                        pair[0],
                        pair[1],
                    ),
                    idx,
                )

        # ====================================================
        # RARE ADDRESS TOKEN
        # ====================================================

        for token in address_tokens(
            address
        ):

            frequency = frequencies[
                "address_token"
            ].get(
                (
                    country,
                    token,
                ),
                0,
            )

            if frequency <= MAX_ADDRESS_TOKEN_FREQUENCY:

                _add(
                    indexes["rare_address_token"],
                    (
                        country,
                        token,
                    ),
                    idx,
                )

        # ====================================================
        # ADDRESS NUMBER + LOCATION
        # ====================================================

        numbers = numeric_tokens(
            address
        )

        locations = address_location_tokens(
            address
        )

        for number in numbers:

            for location in locations:

                frequency = frequencies[
                    "address_num_location"
                ].get(
                    (
                        country,
                        number,
                        location,
                    ),
                    0,
                )

                if (
                    frequency
                    <= MAX_ADDRESS_NUMBER_LOCATION_FREQUENCY
                ):

                    _add(
                        indexes[
                            "address_num_location"
                        ],
                        (
                            country,
                            number,
                            location,
                        ),
                        idx,
                    )

        # ====================================================
        # RARE NGRAM
        # ====================================================

        for gram in char_ngrams(
            name,
            NGRAM_SIZE,
        ):

            frequency = frequencies[
                "ngram"
            ].get(
                (
                    country,
                    gram,
                ),
                0,
            )

            if frequency <= MAX_NGRAM_FREQUENCY:

                _add(
                    indexes["rare_ngram"],
                    (
                        country,
                        gram,
                    ),
                    idx,
                )

    return {
        "indexes": indexes,
        "frequencies": frequencies,
    }


# ============================================================
# EXACT
# ============================================================

def _exact_candidates(
    row,
    indexes,
):

    country = normalize_country(
        row.get("country", "")
    )

    name = normalize_name(
        row.get("business_name", "")
    )

    core = normalize_name_core(
        row.get("business_name", "")
    )

    address = normalize_address(
        row.get("business_address", "")
    )

    hits = set()

    hits |= indexes[
        "country_name"
    ].get(
        (
            country,
            name,
        ),
        set(),
    )

    hits |= indexes[
        "country_core"
    ].get(
        (
            country,
            core,
        ),
        set(),
    )

    hits |= indexes[
        "country_address"
    ].get(
        (
            country,
            address,
        ),
        set(),
    )

    return hits


# ============================================================
# RARE NAME TOKEN
# ============================================================

def _rare_name_token_candidates(
    row,
    indexes,
):

    country = normalize_country(
        row.get("country", "")
    )

    name = normalize_name(
        row.get("business_name", "")
    )

    hits = set()

    for token in tokens(name):

        if len(token) < 3:
            continue

        hits |= indexes[
            "rare_name_token"
        ].get(
            (
                country,
                token,
            ),
            set(),
        )

    return hits


# ============================================================
# NAME PAIRS
# ============================================================

def _rare_name_pair_candidates(
    row,
    indexes,
):

    country = normalize_country(
        row.get("country", "")
    )

    name = normalize_name(
        row.get("business_name", "")
    )

    hits = set()

    for pair in token_pairs(name):

        hits |= indexes[
            "rare_name_pair"
        ].get(
            (
                country,
                pair[0],
                pair[1],
            ),
            set(),
        )

    return hits


# ============================================================
# ADDRESS TOKENS
# ============================================================

def _rare_address_token_candidates(
    row,
    indexes,
):

    country = normalize_country(
        row.get("country", "")
    )

    address = normalize_address(
        row.get("business_address", "")
    )

    hits = set()

    for token in address_tokens(
        address
    ):

        hits |= indexes[
            "rare_address_token"
        ].get(
            (
                country,
                token,
            ),
            set(),
        )

    return hits


# ============================================================
# ADDRESS NUMBER + LOCATION
# ============================================================

def _address_num_location_candidates(
    row,
    indexes,
):

    country = normalize_country(
        row.get("country", "")
    )

    address = normalize_address(
        row.get("business_address", "")
    )

    numbers = numeric_tokens(
        address
    )

    locations = address_location_tokens(
        address
    )

    hits = set()

    for number in numbers:

        for location in locations:

            hits |= indexes[
                "address_num_location"
            ].get(
                (
                    country,
                    number,
                    location,
                ),
                set(),
            )

    return hits


# ============================================================
# NGRAM
# ============================================================

def _rare_ngram_candidates(
    row,
    indexes,
):

    country = normalize_country(
        row.get("country", "")
    )

    name = normalize_name(
        row.get("business_name", "")
    )

    grams = char_ngrams(
        name,
        NGRAM_SIZE,
    )

    if not grams:
        return set()

    counts = Counter()

    for gram in grams:

        rows = indexes[
            "rare_ngram"
        ].get(
            (
                country,
                gram,
            ),
            set(),
        )

        for idx in rows:

            counts[idx] += 1

    if len(grams) <= 4:

        required = 1

    else:

        required = max(
            MIN_SHARED_RARE_NGRAMS,
            math.ceil(
                len(grams) * 0.30
            ),
        )

    return {
        idx
        for idx, count in counts.items()
        if count >= required
    }


# ============================================================
# MAIN API
# ============================================================

def generate_candidates_for_row(
    row,
    target,
    indices,
):

    index_data = indices

    if "indexes" in index_data:

        index_data = index_data[
            "indexes"
        ]

    hits = set()

    # --------------------------------------------------------
    # Every method is country-aware.
    # --------------------------------------------------------

    hits |= _exact_candidates(
        row,
        index_data,
    )

    hits |= _rare_name_token_candidates(
        row,
        index_data,
    )

    hits |= _rare_name_pair_candidates(
        row,
        index_data,
    )

    hits |= _rare_address_token_candidates(
        row,
        index_data,
    )

    hits |= _address_num_location_candidates(
        row,
        index_data,
    )

    hits |= _rare_ngram_candidates(
        row,
        index_data,
    )

    if not hits:

        return target.iloc[
            0:0
        ].copy()

    valid_hits = [
        idx
        for idx in hits
        if idx in target.index
    ]

    if not valid_hits:

        return target.iloc[
            0:0
        ].copy()

    candidates = target.loc[
        sorted(valid_hits)
    ]

    if "entity_id" in candidates.columns:

        candidates = candidates.drop_duplicates(
            subset=["entity_id"]
        )

    return candidates


# ============================================================
# FULL GENERATOR
# ============================================================

def generate_candidates(
    source1,
    target,
):

    indices = build_indices(
        target
    )

    pairs = []

    for _, row in source1.iterrows():

        candidates = generate_candidates_for_row(
            row,
            target,
            indices,
        )

        source_id = row[
            "entity_id"
        ]

        for _, candidate in candidates.iterrows():

            pairs.append(
                (
                    source_id,
                    candidate[
                        "entity_id"
                    ],
                )
            )

    return list(
        dict.fromkeys(
            pairs
        )
    )


# ============================================================
# DEBUG INFORMATION
# ============================================================

def blocking_debug_info(
    row,
    target,
    indices,
):

    index_data = indices

    if "indexes" in index_data:

        index_data = index_data[
            "indexes"
        ]

    exact = _exact_candidates(
        row,
        index_data,
    )

    rare_name_token = (
        _rare_name_token_candidates(
            row,
            index_data,
        )
    )

    rare_name_pair = (
        _rare_name_pair_candidates(
            row,
            index_data,
        )
    )

    rare_address_token = (
        _rare_address_token_candidates(
            row,
            index_data,
        )
    )

    address_combo = (
        _address_num_location_candidates(
            row,
            index_data,
        )
    )

    ngram = (
        _rare_ngram_candidates(
            row,
            index_data,
        )
    )

    all_candidates = (
        exact
        | rare_name_token
        | rare_name_pair
        | rare_address_token
        | address_combo
        | ngram
    )

    return {
        "exact": len(exact),
        "rare_name_token":
            len(rare_name_token),
        "rare_name_pair":
            len(rare_name_pair),
        "rare_address_token":
            len(rare_address_token),
        "address_num_location":
            len(address_combo),
        "rare_ngram":
            len(ngram),
        "total":
            len(all_candidates),
    }


# ============================================================
# MODULE ENTRY
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "COUNTRY-AWARE MULTILINGUAL BLOCKER"
    )

    print(
        "=" * 70
    )

    print(
        "Country-aware exact blocking"
    )

    print(
        "Frequency-aware token blocking"
    )

    print(
        "Frequency-aware token-pair blocking"
    )

    print(
        "Frequency-aware address blocking"
    )

    print(
        "Selective character n-gram blocking"
    )

    print(
        "Multilingual transliteration enabled"
    )

    print(
        "=" * 70
    )