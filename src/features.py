from rapidfuzz.fuzz import ratio, token_set_ratio
from .normalize import normalize_name, normalize_name_core, normalize_address, tokens, numeric_tokens, char_ngrams


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pair_features(a, b):
    na, nb = normalize_name(a["business_name"]), normalize_name(b["business_name"])
    nca, ncb = normalize_name_core(na), normalize_name_core(nb)
    aa, ab = normalize_address(a["business_address"]), normalize_address(b["business_address"])
    nt_a, nt_b = tokens(na), tokens(nb)
    at_a, at_b = tokens(aa), tokens(ab)
    num_a, num_b = numeric_tokens(aa), numeric_tokens(ab)
    cn_a, cn_b = char_ngrams(na), char_ngrams(nb)
    ca_a, ca_b = char_ngrams(aa), char_ngrams(ab)
    return {
        "country_equal": int(str(a["country"]).lower() == str(b["country"]).lower()),
        "name_exact": int(na == nb),
        "name_core_exact": int(nca == ncb),
        "name_ratio": ratio(na, nb) / 100.0,
        "name_core_ratio": ratio(nca, ncb) / 100.0,
        "name_token_set_ratio": token_set_ratio(na, nb) / 100.0,
        "name_jaccard": jaccard(nt_a, nt_b),
        "name_char_jaccard": jaccard(cn_a, cn_b),
        "address_exact": int(aa == ab),
        "address_ratio": ratio(aa, ab) / 100.0,
        "address_jaccard": jaccard(at_a, at_b),
        "address_char_jaccard": jaccard(ca_a, ca_b),
        "numeric_overlap": jaccard(num_a, num_b),
        "name_len_diff": abs(len(na) - len(nb)),
        "address_len_diff": abs(len(aa) - len(ab)),
    }
