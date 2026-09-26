from collections import defaultdict
from .normalize import (
    normalize_country,
    normalize_name,
    normalize_name_core,
    normalize_address,
    tokens,
    numeric_tokens,
    char_ngrams,
    jaccard,
)


def _add(index, key, row_idx):
    if key:
        index[key].add(row_idx)


def build_indices(target):
    idx = {
        "country_name": defaultdict(set),
        "country_core": defaultdict(set),
        "country_addr": defaultdict(set),
        "country_num": defaultdict(set),
        "token": defaultdict(set),
        "country_ngram": defaultdict(set),
    }

    normalized_ngrams = {}

    for i, r in target.iterrows():
        country = normalize_country(r["country"])
        name = normalize_name(r["business_name"])
        core = normalize_name_core(r["business_name"])
        addr = normalize_address(r["business_address"])

        _add(idx["country_name"], (country, name), i)
        _add(idx["country_core"], (country, core), i)
        _add(idx["country_addr"], (country, addr), i)

        for n in numeric_tokens(addr):
            _add(idx["country_num"], (country, n), i)

        for t in tokens(name):
            if len(t) >= 4:
                _add(idx["token"], (country, t), i)

        # Precompute character n-grams once for each target name
        target_ngrams = char_ngrams(name)
        normalized_ngrams[i] = target_ngrams

        # Inverted index:
        # (country, ngram) -> target row indices
        for gram in target_ngrams:
            _add(idx["country_ngram"], (country, gram), i)

    idx["_normalized_ngrams"] = normalized_ngrams

    return idx


FUZZY_BLOCK_THRESHOLD = 0.60


def fuzzy_name_candidates(row, indices):
    country = normalize_country(row["country"])
    name = normalize_name(row["business_name"])
    source_ngrams = char_ngrams(name)

    # First retrieve only target rows that share at least
    # one character n-gram with the source name.
    possible_hits = set()

    for gram in source_ngrams:
        possible_hits |= indices["country_ngram"].get(
            (country, gram),
            set(),
        )

    # Now perform the actual fuzzy comparison only
    # on those candidates.
    hits = set()

    for i in possible_hits:
        target_ngrams = indices["_normalized_ngrams"][i]

        if jaccard(source_ngrams, target_ngrams) >= FUZZY_BLOCK_THRESHOLD:
            hits.add(i)

    return hits


def generate_candidates_for_row(row, target, indices):
    country = normalize_country(row["country"])
    name = normalize_name(row["business_name"])
    core = normalize_name_core(row["business_name"])
    addr = normalize_address(row["business_address"])
    
    hits = set()
    hits |= indices["country_name"].get((country, name), set())
    hits |= indices["country_core"].get((country, core), set())
    hits |= indices["country_addr"].get((country, addr), set())
    
    for n in numeric_tokens(addr):
        hits |= indices["country_num"].get((country, n), set())
        
    for t in tokens(name):
        if len(t) >= 4:
            hits |= indices["token"].get((country, t), set())
            
    if not hits:
    # Fuzzy fallback using the precomputed n-gram index.
      hits = fuzzy_name_candidates(row, indices)
        
    cands = target.loc[sorted(hits)]
    if "entity_id" in cands.columns:
        cands = cands.drop_duplicates(subset=["entity_id"])
    return cands


def generate_candidates_for_row(row, target, indices):
    country = normalize_country(row["country"])
    name = normalize_name(row["business_name"])
    core = normalize_name_core(row["business_name"])
    addr = normalize_address(row["business_address"])

    hits = set()

    hits |= indices["country_name"].get(
        (country, name),
        set(),
    )

    hits |= indices["country_core"].get(
        (country, core),
        set(),
    )

    hits |= indices["country_addr"].get(
        (country, addr),
        set(),
    )

    for n in numeric_tokens(addr):
        hits |= indices["country_num"].get(
            (country, n),
            set(),
        )

    for t in tokens(name):
        if len(t) >= 4:
            hits |= indices["token"].get(
                (country, t),
                set(),
            )

    if not hits:
        # Fuzzy fallback using the precomputed n-gram index.
        hits = fuzzy_name_candidates(row, indices)

    cands = target.loc[sorted(hits)]

    if "entity_id" in cands.columns:
        cands = cands.drop_duplicates(
            subset=["entity_id"]
        )

    return cands
