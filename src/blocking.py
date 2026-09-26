from collections import defaultdict
from .normalize import normalize_name, normalize_name_core, normalize_address, tokens, numeric_tokens,  char_ngrams, jaccard


def _add(index, key, row_idx):
    if key:
        index[key].add(row_idx)


def build_indices(target):
    idx = {k: defaultdict(set) for k in ("country_name", "country_core", "country_addr", "country_num", "token")}
    for i, r in target.iterrows():
        country = str(r["country"]).lower()
        name = normalize_name(r["business_name"])
        core = normalize_name_core(name)
        addr = normalize_address(r["business_address"])
        _add(idx["country_name"], (country, name), i)
        _add(idx["country_core"], (country, core), i)
        _add(idx["country_addr"], (country, addr), i)
        for n in numeric_tokens(addr):
            _add(idx["country_num"], (country, n), i)
        for t in tokens(name):
            if len(t) >= 4:
                _add(idx["token"], (country, t), i)
    return idx

FUZZY_BLOCK_THRESHOLD = 0.60


def fuzzy_name_candidates(row, target):
    country = str(row["country"]).lower()
    name = normalize_name(row["business_name"])
    source_ngrams = char_ngrams(name)

    hits = set()

    for i, r in target.iterrows():
        if str(r["country"]).lower() != country:
            continue

        target_name = normalize_name(r["business_name"])
        target_ngrams = char_ngrams(target_name)

        if jaccard(source_ngrams, target_ngrams) >= FUZZY_BLOCK_THRESHOLD:
            hits.add(i)

    return hits

def generate_candidates_for_row(row, target, indices):
    country = str(row["country"]).lower()
    name = normalize_name(row["business_name"])
    core = normalize_name_core(name)
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
        # Conservative fallback: same-country records only. This preserves recall
        # for noisy records while keeping inference tractable for typical datasets.
        hits = fuzzy_name_candidates(row, target)
    return target.loc[sorted(hits)]


def generate_candidates(source1, target):
    indices = build_indices(target)
    rows = []
    for _, r in source1.iterrows():
        c = generate_candidates_for_row(r, target, indices)
        for _, t in c.iterrows():
            rows.append((r["entity_id"], t["entity_id"]))
    return rows
