from rapidfuzz.fuzz import ratio, token_sort_ratio, token_set_ratio
from rapidfuzz.distance import JaroWinkler
from .normalize import (
    normalize_name, normalize_name_leet, normalize_name_core,
    normalize_address, normalize_address_stripped,
    tokens, tokens_leet, address_tokens, address_location_tokens,
    numeric_tokens, char_ngrams, extract_street_numbers, extract_state,
    normalize_country, consonant_skeleton, sorted_token_signature,
    jaccard, LEGAL_SUFFIX_VARIANTS,
)

def pair_features(a, b, idf_weights=None):
    """
    Compute pair-wise features for entity matching.
    """
    # 0. Safely extract values
    name_a = a.get('business_name', '') or ''
    name_b = b.get('business_name', '') or ''
    addr_a = a.get('business_address', '') or ''
    addr_b = b.get('business_address', '') or ''
    country_a = a.get('country', '') or ''
    country_b = b.get('country', '') or ''

    # 1. Base normalizations
    norm_name_a = normalize_name(name_a)
    norm_name_b = normalize_name(name_b)
    core_name_a = normalize_name_core(name_a)
    core_name_b = normalize_name_core(name_b)
    leet_name_a = normalize_name_leet(name_a)
    leet_name_b = normalize_name_leet(name_b)
    
    norm_addr_a = normalize_address(addr_a)
    norm_addr_b = normalize_address(addr_b)

    # 2. Token sets
    toks_name_a = tokens(name_a)
    toks_name_b = tokens(name_b)
    toks_addr_a = tokens(addr_a)
    toks_addr_b = tokens(addr_b)

    feats = {}

    # Group 1: Name similarity (10 features)
    feats['name_exact'] = int(norm_name_a == norm_name_b and norm_name_a != "")
    feats['name_core_exact'] = int(core_name_a == core_name_b and core_name_a != "")
    
    feats['name_jaro_winkler'] = JaroWinkler.similarity(norm_name_a, norm_name_b) if norm_name_a and norm_name_b else 0.0
    feats['name_ratio'] = ratio(norm_name_a, norm_name_b) / 100.0 if norm_name_a and norm_name_b else 0.0
    feats['name_token_sort_ratio'] = token_sort_ratio(norm_name_a, norm_name_b) / 100.0 if norm_name_a and norm_name_b else 0.0
    feats['name_token_set_ratio'] = token_set_ratio(norm_name_a, norm_name_b) / 100.0 if norm_name_a and norm_name_b else 0.0
    
    feats['name_jaccard'] = jaccard(toks_name_a, toks_name_b)
    feats['name_char_ngram_jaccard_3'] = jaccard(char_ngrams(norm_name_a, 3), char_ngrams(norm_name_b, 3))
    feats['name_char_ngram_jaccard_4'] = jaccard(char_ngrams(norm_name_a, 4), char_ngrams(norm_name_b, 4))
    feats['name_leet_ratio'] = ratio(leet_name_a, leet_name_b) / 100.0 if leet_name_a and leet_name_b else 0.0

    # Group 2: Phonetic similarity (3 features)
    skel_a = consonant_skeleton(name_a)
    skel_b = consonant_skeleton(name_b)
    feats['phonetic_exact'] = int(skel_a == skel_b and skel_a != "")
    feats['phonetic_ratio'] = ratio(skel_a, skel_b) / 100.0 if skel_a and skel_b else 0.0
    
    sig_a = sorted_token_signature(name_a)
    sig_b = sorted_token_signature(name_b)
    feats['sorted_token_sig_exact'] = int(sig_a == sig_b and sig_a != "")

    # Group 3: Name structural (4 features)
    len_a = len(toks_name_a)
    len_b = len(toks_name_b)
    feats['name_len_ratio'] = min(len_a, len_b) / max(len_a, len_b) if max(len_a, len_b) > 0 else 1.0
    feats['name_token_count_diff'] = abs(len_a - len_b)
    
    core_toks_a = core_name_a.split()
    core_toks_b = core_name_b.split()
    feats['name_first_token_match'] = int(core_toks_a[0] == core_toks_b[0]) if core_toks_a and core_toks_b else 0
    
    legal_a = {t for t in leet_name_a.split() if t in LEGAL_SUFFIX_VARIANTS}
    legal_b = {t for t in leet_name_b.split() if t in LEGAL_SUFFIX_VARIANTS}
    feats['legal_suffix_compatible'] = int(len(legal_a & legal_b) > 0)

    # Group 4: Address similarity (10 features)
    feats['address_exact'] = int(norm_addr_a == norm_addr_b and norm_addr_a != "")
    feats['address_ratio'] = ratio(norm_addr_a, norm_addr_b) / 100.0 if norm_addr_a and norm_addr_b else 0.0
    feats['address_token_jaccard'] = jaccard(toks_addr_a, toks_addr_b)
    feats['address_char_jaccard'] = jaccard(char_ngrams(norm_addr_a, 3), char_ngrams(norm_addr_b, 3))
    
    street_a = extract_street_numbers(addr_a)
    street_b = extract_street_numbers(addr_b)
    feats['street_number_match'] = int(len(street_a & street_b) > 0)
    feats['street_number_jaccard'] = jaccard(street_a, street_b)
    
    state_a = extract_state(addr_a, country_a)
    state_b = extract_state(addr_b, country_b)
    feats['state_match'] = int(state_a == state_b and state_a != "")
    
    loc_a = " ".join(sorted(address_location_tokens(addr_a)))
    loc_b = " ".join(sorted(address_location_tokens(addr_b)))
    feats['city_similarity'] = ratio(loc_a, loc_b) / 100.0 if loc_a and loc_b else 0.0
    
    num_a = numeric_tokens(addr_a)
    num_b = numeric_tokens(addr_b)
    feats['address_numeric_jaccard'] = jaccard(num_a, num_b)
    
    feats['address_has_both'] = int(bool(norm_addr_a) and bool(norm_addr_b))

    # Group 5: Cross-field (3 features)
    core_set_a = set(core_toks_a)
    core_set_b = set(core_toks_b)
    
    frac_a_in_b = len(core_set_a & toks_addr_b) / len(core_set_a) if core_set_a else 0.0
    frac_b_in_a = len(core_set_b & toks_addr_a) / len(core_set_b) if core_set_b else 0.0
    feats['name_in_address'] = max(frac_a_in_b, frac_b_in_a)
    
    country_norm_a = normalize_country(country_a)
    country_norm_b = normalize_country(country_b)
    feats['country_match'] = int(country_norm_a == country_norm_b and country_norm_a != "")
    
    union_a = toks_name_a | toks_addr_a
    union_b = toks_name_b | toks_addr_b
    feats['overall_token_overlap'] = jaccard(union_a, union_b)

    # Group 6: Rarity-weighted (3 features)
    if idf_weights:
        name_int = toks_name_a & toks_name_b
        name_uni = toks_name_a | toks_name_b
        if name_uni:
            s_i = sum(idf_weights.get(t, 0.0) for t in name_int)
            s_u = sum(idf_weights.get(t, 0.0) for t in name_uni)
            feats['name_rare_token_overlap'] = s_i / s_u if s_u > 0 else 0.0
        else:
            feats['name_rare_token_overlap'] = 0.0
            
        addr_int = toks_addr_a & toks_addr_b
        addr_uni = toks_addr_a | toks_addr_b
        if addr_uni:
            s_i = sum(idf_weights.get(t, 0.0) for t in addr_int)
            s_u = sum(idf_weights.get(t, 0.0) for t in addr_uni)
            feats['address_rare_token_overlap'] = s_i / s_u if s_u > 0 else 0.0
        else:
            feats['address_rare_token_overlap'] = 0.0
            
        if name_int:
            feats['max_shared_idf'] = max((idf_weights.get(t, 0.0) for t in name_int), default=0.0)
        else:
            feats['max_shared_idf'] = 0.0
    else:
        feats['name_rare_token_overlap'] = 0.0
        feats['address_rare_token_overlap'] = 0.0
        feats['max_shared_idf'] = 0.0

    return feats
