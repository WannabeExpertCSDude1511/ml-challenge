"""Country-aware, frequency-aware blocking for S1 -> S2/S3 entity matching.

Design goals:
- country is mandatory for every blocking key
- high recall without all-pairs comparison
- build target features once; never renormalize target rows during queries
- use integer row positions internally and return a DataFrame for compatibility
- keep several independent blocking families so one weak field does not kill recall
"""

from collections import Counter, defaultdict
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

# ------------------------------------------------------------
# Tuning
# ------------------------------------------------------------
MAX_NAME_TOKEN_FREQUENCY = 500
MAX_ADDRESS_TOKEN_FREQUENCY = 500
MAX_TOKEN_PAIR_FREQUENCY = 1000
MAX_NGRAM_FREQUENCY = 500
MAX_ADDRESS_NUMBER_LOCATION_FREQUENCY = 500
MAX_ADDRESS_PAIR_FREQUENCY = 1000

NGRAM_SIZE = 3
MIN_SHARED_RARE_NGRAMS = 2


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _add(index, key, row_idx):
    if key:
        index[key].add(row_idx)


def _features(row):
    """Compute all normalized blocking features exactly once for a row."""
    raw_name = row.get("business_name", "")
    raw_address = row.get("business_address", "")
    country = normalize_country(row.get("country", ""))
    name = normalize_name(raw_name)
    core = normalize_name_core(raw_name)
    address = normalize_address(raw_address)

    return {
        "country": country,
        "name": name,
        "core": core,
        "address": address,
        "name_tokens": tokens(name),
        "name_pairs": token_pairs(name),
        "address_tokens": address_tokens(address),
        "address_locations": address_location_tokens(address),
        "numbers": numeric_tokens(address),
        "ngrams": char_ngrams(name, NGRAM_SIZE),
    }


def _build_frequencies(features):
    name_token_frequency = Counter()
    address_token_frequency = Counter()
    token_pair_frequency = Counter()
    ngram_frequency = Counter()
    address_number_location_frequency = Counter()
    address_pair_frequency = Counter()

    for f in features:
        c = f["country"]
        for t in f["name_tokens"]:
            if len(t) >= 3:
                name_token_frequency[(c, t)] += 1

        for a, b in f["name_pairs"]:
            token_pair_frequency[(c, a, b)] += 1

        for t in f["address_tokens"]:
            address_token_frequency[(c, t)] += 1

        for g in f["ngrams"]:
            ngram_frequency[(c, g)] += 1

        for n in f["numbers"]:
            for loc in f["address_locations"]:
                address_number_location_frequency[(c, n, loc)] += 1

        # Address pairs are deliberately frequency-limited. They recover
        # partial addresses while avoiding a broad single-token explosion.
        addr = sorted(f["address_tokens"])
        for i in range(len(addr)):
            for j in range(i + 1, len(addr)):
                address_pair_frequency[(c, addr[i], addr[j])] += 1

    return {
        "name_token": name_token_frequency,
        "address_token": address_token_frequency,
        "token_pair": token_pair_frequency,
        "ngram": ngram_frequency,
        "address_num_location": address_number_location_frequency,
        "address_pair": address_pair_frequency,
    }


def build_indices(target):
    """Build all target-side indexes.

    The returned object intentionally contains the same top-level
    ``indexes``/``frequencies`` structure as the previous implementation.
    ``features`` is added for diagnostics and fast reuse.
    """
    # Important: reset to positional indexing because target is usually a
    # concat of S2/S3. The blocker internally uses positions only.
    target = target.reset_index(drop=True)

    features = []
    for row in target.itertuples(index=False):
        features.append(_features(row._asdict()))

    frequencies = _build_frequencies(features)

    indexes = {
        "country_name": defaultdict(set),
        "country_core": defaultdict(set),
        "country_address": defaultdict(set),
        "rare_name_token": defaultdict(set),
        "rare_name_pair": defaultdict(set),
        "rare_address_token": defaultdict(set),
        "rare_address_pair": defaultdict(set),
        "address_num_location": defaultdict(set),
        "rare_ngram": defaultdict(set),
    }

    for idx, f in enumerate(features):
        c = f["country"]

        if f["name"]:
            _add(indexes["country_name"], (c, f["name"]), idx)
        if f["core"]:
            _add(indexes["country_core"], (c, f["core"]), idx)
        if f["address"]:
            _add(indexes["country_address"], (c, f["address"]), idx)

        for t in f["name_tokens"]:
            if len(t) >= 3 and frequencies["name_token"].get((c, t), 0) <= MAX_NAME_TOKEN_FREQUENCY:
                _add(indexes["rare_name_token"], (c, t), idx)

        for a, b in f["name_pairs"]:
            if frequencies["token_pair"].get((c, a, b), 0) <= MAX_TOKEN_PAIR_FREQUENCY:
                _add(indexes["rare_name_pair"], (c, a, b), idx)

        for t in f["address_tokens"]:
            if frequencies["address_token"].get((c, t), 0) <= MAX_ADDRESS_TOKEN_FREQUENCY:
                _add(indexes["rare_address_token"], (c, t), idx)

        for a, b in token_pairs(" ".join(sorted(f["address_tokens"]))):
            if frequencies["address_pair"].get((c, a, b), 0) <= MAX_ADDRESS_PAIR_FREQUENCY:
                _add(indexes["rare_address_pair"], (c, a, b), idx)

        for n in f["numbers"]:
            for loc in f["address_locations"]:
                if frequencies["address_num_location"].get((c, n, loc), 0) <= MAX_ADDRESS_NUMBER_LOCATION_FREQUENCY:
                    _add(indexes["address_num_location"], (c, n, loc), idx)

        for g in f["ngrams"]:
            if frequencies["ngram"].get((c, g), 0) <= MAX_NGRAM_FREQUENCY:
                _add(indexes["rare_ngram"], (c, g), idx)

    return {
        "indexes": indexes,
        "frequencies": frequencies,
        "features": features,
    }


def _get_indexes(indices):
    return indices["indexes"] if "indexes" in indices else indices


def _exact_candidates(f, ix):
    c = f["country"]
    hits = set()
    if f["name"]:
        hits.update(ix["country_name"].get((c, f["name"]), ()))
    if f["core"]:
        hits.update(ix["country_core"].get((c, f["core"]), ()))
    if f["address"]:
        hits.update(ix["country_address"].get((c, f["address"]), ()))
    return hits


def _union_keys(ix, index_name, keys):
    hits = set()
    index = ix[index_name]
    for key in keys:
        rows = index.get(key)
        if rows:
            hits.update(rows)
    return hits


def _rare_name_token_candidates(f, ix):
    c = f["country"]
    return _union_keys(ix, "rare_name_token", ((c, t) for t in f["name_tokens"] if len(t) >= 3))


def _rare_name_pair_candidates(f, ix):
    c = f["country"]
    return _union_keys(ix, "rare_name_pair", ((c, a, b) for a, b in f["name_pairs"]))


def _rare_address_token_candidates(f, ix):
    c = f["country"]
    return _union_keys(ix, "rare_address_token", ((c, t) for t in f["address_tokens"]))


def _rare_address_pair_candidates(f, ix):
    c = f["country"]
    addr = sorted(f["address_tokens"])
    return _union_keys(
        ix,
        "rare_address_pair",
        ((c, addr[i], addr[j]) for i in range(len(addr)) for j in range(i + 1, len(addr))),
    )


def _address_num_location_candidates(f, ix):
    c = f["country"]
    return _union_keys(
        ix,
        "address_num_location",
        ((c, n, loc) for n in f["numbers"] for loc in f["address_locations"]),
    )


def _rare_ngram_candidates(f, ix):
    grams = f["ngrams"]
    if not grams:
        return set()

    c = f["country"]
    counts = Counter()
    index = ix["rare_ngram"]

    for g in grams:
        rows = index.get((c, g))
        if rows:
            for idx in rows:
                counts[idx] += 1

    if len(grams) <= 4:
        required = 1
    else:
        required = max(MIN_SHARED_RARE_NGRAMS, math.ceil(len(grams) * 0.30))

    return {idx for idx, count in counts.items() if count >= required}


def generate_candidate_indices(row, indices):
    """Fast internal API: return integer target positions."""
    f = _features(row)
    ix = _get_indexes(indices)

    hits = set()
    hits.update(_exact_candidates(f, ix))
    hits.update(_rare_name_token_candidates(f, ix))
    hits.update(_rare_name_pair_candidates(f, ix))
    hits.update(_rare_address_token_candidates(f, ix))
    hits.update(_rare_address_pair_candidates(f, ix))
    hits.update(_address_num_location_candidates(f, ix))
    hits.update(_rare_ngram_candidates(f, ix))
    return hits


def generate_candidates_for_row(row, target, indices):
    """Compatibility API used by test_blocking.py."""
    if target.index.start != 0 or not target.index.equals(range(len(target))):
        target = target.reset_index(drop=True)

    hits = generate_candidate_indices(row, indices)
    if not hits:
        return target.iloc[0:0].copy()

    # Do not sort: sorting large candidate sets is unnecessary work.
    positions = list(hits)
    candidates = target.iloc[positions]

    if "entity_id" in candidates.columns:
        candidates = candidates.drop_duplicates(subset=["entity_id"])

    return candidates


def generate_candidates(source1, target):
    indices = build_indices(target)
    pairs = []
    for row in source1.to_dict("records"):
        source_id = row["entity_id"]
        for idx in generate_candidate_indices(row, indices):
            pairs.append((source_id, target.iloc[idx]["entity_id"]))
    return list(dict.fromkeys(pairs))


def blocking_debug_info(row, target, indices):
    f = _features(row)
    ix = _get_indexes(indices)

    exact = _exact_candidates(f, ix)
    name_token = _rare_name_token_candidates(f, ix)
    name_pair = _rare_name_pair_candidates(f, ix)
    addr_token = _rare_address_token_candidates(f, ix)
    addr_pair = _rare_address_pair_candidates(f, ix)
    addr_combo = _address_num_location_candidates(f, ix)
    ngram = _rare_ngram_candidates(f, ix)
    all_candidates = exact | name_token | name_pair | addr_token | addr_pair | addr_combo | ngram

    return {
        "exact": len(exact),
        "rare_name_token": len(name_token),
        "rare_name_pair": len(name_pair),
        "rare_address_token": len(addr_token),
        "rare_address_pair": len(addr_pair),
        "address_num_location": len(addr_combo),
        "rare_ngram": len(ngram),
        "total": len(all_candidates),
    }


if __name__ == "__main__":
    print("Country-aware multilingual blocker")
    print("Use build_indices() / generate_candidates_for_row() from the pipeline.")
