"""
Multi-strategy blocking for business entity resolution.

Uses a union of 8 independent blocking strategies so that a candidate pair
only needs to be found by ANY ONE strategy.

Strategies:
  1. Exact normalized name + country
  2. Core name (no legal suffix) + country
  3. Phonetic (consonant skeleton) + country
  4. TF-IDF top-K name similarity (character n-grams)
  5. Sorted token signature + country
  6. Address number + state (garbled-name rescue)
  7. Street number + rare address token
  8. Rare character 4-gram overlap (≥ 30%)

Memory-efficient: indices built once, S1 processed in batches.
"""

from collections import defaultdict, Counter
import math
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .normalize import (
    normalize_country,
    normalize_name,
    normalize_name_leet,
    normalize_name_core,
    normalize_address,
    normalize_address_stripped,
    tokens,
    tokens_leet,
    address_tokens,
    address_location_tokens,
    numeric_tokens,
    char_ngrams,
    token_pairs,
    consonant_skeleton,
    sorted_token_signature,
    extract_street_numbers,
    extract_state,
    jaccard,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Maximum frequency for a token/key to be considered "rare" enough for blocking
MAX_NAME_TOKEN_FREQ = 500
MAX_ADDRESS_TOKEN_FREQ = 500
MAX_NGRAM_FREQ = 500

# Character n-gram size for Strategy 8
NGRAM_SIZE = 4

# Minimum shared rare n-grams for Strategy 8
MIN_SHARED_RARE_NGRAMS = 2

# TF-IDF blocking: how many top candidates per query
TFIDF_TOP_K = 50

# TF-IDF: batch size for processing S1 entities
TFIDF_BATCH_SIZE = 10000


# ============================================================
# HELPERS
# ============================================================

def _add(index, key, row_idx):
    """Add row_idx to inverted index under key."""
    if key is None or key == "" or key == ("",):
        return
    index[key].add(row_idx)


def _safe_get(row, key, default=""):
    """Safely get a value from a dict or Series."""
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        val = getattr(row, key, default)
        return val if val is not None else default
    except Exception:
        return default


# ============================================================
# PRECOMPUTE TARGET FEATURES (done once)
# ============================================================

def _precompute_target(target):
    """
    Precompute normalized fields for every target row.
    Returns a list of dicts, one per target row, indexed by position.
    """
    precomputed = []
    for idx, row in target.iterrows():
        country = normalize_country(row.get("country", ""))
        name = normalize_name(row.get("business_name", ""))
        name_leet = normalize_name_leet(row.get("business_name", ""))
        core = normalize_name_core(row.get("business_name", ""))
        address = normalize_address(row.get("business_address", ""))
        address_stripped = normalize_address_stripped(row.get("business_address", ""))
        skeleton = consonant_skeleton(row.get("business_name", ""))
        sorted_sig = sorted_token_signature(row.get("business_name", ""))
        street_nums = extract_street_numbers(row.get("business_address", ""))
        state = extract_state(row.get("business_address", ""), row.get("country", ""))

        precomputed.append({
            "idx": idx,
            "country": country,
            "name": name,
            "name_leet": name_leet,
            "core": core,
            "address": address,
            "address_stripped": address_stripped,
            "skeleton": skeleton,
            "sorted_sig": sorted_sig,
            "street_nums": street_nums,
            "state": state,
            "name_tokens": set(name_leet.split()) if name_leet else set(),
            "addr_tokens": address_tokens(row.get("business_address", "")),
        })
    return precomputed


def _precompute_row(row):
    """Precompute normalized fields for a single source row."""
    country = normalize_country(_safe_get(row, "country", ""))
    name = normalize_name(_safe_get(row, "business_name", ""))
    name_leet = normalize_name_leet(_safe_get(row, "business_name", ""))
    core = normalize_name_core(_safe_get(row, "business_name", ""))
    address = normalize_address(_safe_get(row, "business_address", ""))
    address_stripped = normalize_address_stripped(_safe_get(row, "business_address", ""))
    skeleton = consonant_skeleton(_safe_get(row, "business_name", ""))
    sorted_sig = sorted_token_signature(_safe_get(row, "business_name", ""))
    street_nums = extract_street_numbers(_safe_get(row, "business_address", ""))
    state = extract_state(_safe_get(row, "business_address", ""), _safe_get(row, "country", ""))

    return {
        "country": country,
        "name": name,
        "name_leet": name_leet,
        "core": core,
        "address": address,
        "address_stripped": address_stripped,
        "skeleton": skeleton,
        "sorted_sig": sorted_sig,
        "street_nums": street_nums,
        "state": state,
        "name_tokens": set(name_leet.split()) if name_leet else set(),
        "addr_tokens": address_tokens(_safe_get(row, "business_address", "")),
    }


# ============================================================
# BUILD FREQUENCIES
# ============================================================

def _build_frequencies(precomputed):
    """Build frequency counters for rarity-based blocking."""
    name_token_freq = Counter()
    addr_token_freq = Counter()
    ngram_freq = Counter()

    for pc in precomputed:
        country = pc["country"]
        for tok in pc["name_tokens"]:
            if len(tok) >= 3:
                name_token_freq[(country, tok)] += 1
        for tok in pc["addr_tokens"]:
            addr_token_freq[(country, tok)] += 1
        for gram in char_ngrams(pc["name_leet"], NGRAM_SIZE):
            ngram_freq[(country, gram)] += 1

    return {
        "name_token": name_token_freq,
        "addr_token": addr_token_freq,
        "ngram": ngram_freq,
    }


# ============================================================
# BUILD INVERTED INDICES
# ============================================================

def build_indices(target):
    """
    Build all blocking indices for the target (S2+S3) DataFrame.

    Returns dict with:
      - "indexes": the inverted indices
      - "frequencies": token frequency counters
      - "precomputed": precomputed normalization for each target row
      - "idf_weights": IDF weights for name tokens
      - "target_tfidf": (tfidf_matrix, vectorizer) tuple for TF-IDF blocking
    """
    print("  [blocking] Precomputing target normalization...", flush=True)
    precomputed = _precompute_target(target)

    print("  [blocking] Building frequencies...", flush=True)
    frequencies = _build_frequencies(precomputed)

    # --------------------------------------------------------
    # Build inverted indices for strategies 1-3, 5-8
    # --------------------------------------------------------
    indexes = {
        "country_name": defaultdict(set),       # Strategy 1
        "country_core": defaultdict(set),       # Strategy 2
        "country_skeleton": defaultdict(set),   # Strategy 3
        "country_sorted_sig": defaultdict(set), # Strategy 5
        "addr_num_state": defaultdict(set),     # Strategy 6
        "addr_num_rare_tok": defaultdict(set),  # Strategy 7
        "rare_name_token": defaultdict(set),    # (supporting)
        "rare_ngram": defaultdict(set),         # Strategy 8
    }

    print("  [blocking] Building inverted indices...", flush=True)
    for pc in precomputed:
        idx = pc["idx"]
        country = pc["country"]

        # Strategy 1: Exact normalized name + country
        if pc["name"]:
            _add(indexes["country_name"], (country, pc["name"]), idx)

        # Strategy 2: Core name + country
        if pc["core"]:
            _add(indexes["country_core"], (country, pc["core"]), idx)

        # Strategy 3: Phonetic skeleton + country
        if pc["skeleton"]:
            _add(indexes["country_skeleton"], (country, pc["skeleton"]), idx)

        # Strategy 5: Sorted token signature + country
        if pc["sorted_sig"]:
            _add(indexes["country_sorted_sig"], (country, pc["sorted_sig"]), idx)

        # Strategy 6: Address number + state (garbled name rescue)
        if pc["state"] and pc["street_nums"]:
            for num in pc["street_nums"]:
                _add(indexes["addr_num_state"], (country, num, pc["state"]), idx)

        # Strategy 7: Street number + rare address token
        for tok in pc["addr_tokens"]:
            freq = frequencies["addr_token"].get((country, tok), 0)
            if freq <= MAX_ADDRESS_TOKEN_FREQ and pc["street_nums"]:
                for num in pc["street_nums"]:
                    _add(indexes["addr_num_rare_tok"], (country, num, tok), idx)

        # Supporting: Rare name tokens
        for tok in pc["name_tokens"]:
            if len(tok) < 3:
                continue
            freq = frequencies["name_token"].get((country, tok), 0)
            if freq <= MAX_NAME_TOKEN_FREQ:
                _add(indexes["rare_name_token"], (country, tok), idx)

        # Strategy 8: Rare n-grams
        for gram in char_ngrams(pc["name_leet"], NGRAM_SIZE):
            freq = frequencies["ngram"].get((country, gram), 0)
            if freq <= MAX_NGRAM_FREQ:
                _add(indexes["rare_ngram"], (country, gram), idx)

    # --------------------------------------------------------
    # Compute IDF weights for name tokens (for feature engineering)
    # --------------------------------------------------------
    print("  [blocking] Computing IDF weights...", flush=True)
    n_docs = len(precomputed)
    doc_freq = Counter()
    for pc in precomputed:
        for tok in pc["name_tokens"]:
            doc_freq[tok] += 1
    idf_weights = {}
    for tok, df in doc_freq.items():
        idf_weights[tok] = math.log((n_docs + 1) / (df + 1)) + 1

    # --------------------------------------------------------
    # Strategy 4: TF-IDF matrix for name similarity blocking
    # --------------------------------------------------------
    print("  [blocking] Building TF-IDF matrix...", flush=True)
    tfidf_data = _build_tfidf_index(precomputed)

    return {
        "indexes": indexes,
        "frequencies": frequencies,
        "precomputed": precomputed,
        "idf_weights": idf_weights,
        "tfidf_data": tfidf_data,
    }


# ============================================================
# TF-IDF BLOCKING (Strategy 4)
# ============================================================

def _build_tfidf_index(precomputed):
    """
    Build a TF-IDF matrix over normalized names using character n-grams.
    Returns (matrix, vocab, idx_map) where:
      - matrix: sparse CSR matrix [n_targets x n_features]
      - vocab: dict mapping ngram → column index
      - idx_map: list mapping matrix row → target DataFrame index
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Collect texts and their DataFrame indices
    texts = []
    idx_map = []
    for pc in precomputed:
        texts.append(pc["name_leet"] if pc["name_leet"] else " ")
        idx_map.append(pc["idx"])

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=500000,  # Cap vocabulary to control memory
        sublinear_tf=True,
        dtype=np.float32,
    )

    matrix = vectorizer.fit_transform(texts)

    return {
        "matrix": matrix,
        "vectorizer": vectorizer,
        "idx_map": idx_map,
    }


def _tfidf_candidates_batch(query_texts, tfidf_data, top_k=TFIDF_TOP_K):
    """
    Find top-K TF-IDF candidates for a batch of query texts.

    Returns a list of sets, one per query, containing target DataFrame indices.
    """
    vectorizer = tfidf_data["vectorizer"]
    target_matrix = tfidf_data["matrix"]
    idx_map = tfidf_data["idx_map"]

    # Transform queries
    query_matrix = vectorizer.transform(query_texts)

    # Sparse matrix multiplication: [n_queries x n_features] @ [n_features x n_targets]
    scores = query_matrix.dot(target_matrix.T)

    results = []
    for i in range(scores.shape[0]):
        row = scores.getrow(i)
        if row.nnz == 0:
            results.append(set())
            continue

        # Get top-K indices
        data = row.data
        indices = row.indices

        if len(data) <= top_k:
            top_indices = indices
        else:
            # Partial sort for top-K
            top_k_pos = np.argpartition(data, -top_k)[-top_k:]
            top_indices = indices[top_k_pos]

        results.append({idx_map[j] for j in top_indices})

    return results


# ============================================================
# INDIVIDUAL STRATEGY LOOKUPS
# ============================================================

def _strategy_exact_name(pc, indexes):
    """Strategy 1: Exact normalized name + country."""
    if not pc["name"]:
        return set()
    return indexes["country_name"].get((pc["country"], pc["name"]), set()).copy()


def _strategy_core_name(pc, indexes):
    """Strategy 2: Core name (no legal suffixes) + country."""
    if not pc["core"]:
        return set()
    return indexes["country_core"].get((pc["country"], pc["core"]), set()).copy()


def _strategy_phonetic(pc, indexes):
    """Strategy 3: Consonant skeleton + country."""
    if not pc["skeleton"]:
        return set()
    return indexes["country_skeleton"].get((pc["country"], pc["skeleton"]), set()).copy()


def _strategy_sorted_sig(pc, indexes):
    """Strategy 5: Sorted token signature + country."""
    if not pc["sorted_sig"]:
        return set()
    return indexes["country_sorted_sig"].get((pc["country"], pc["sorted_sig"]), set()).copy()


def _strategy_addr_num_state(pc, indexes):
    """Strategy 6: Address number + state (garbled-name rescue)."""
    if not pc["state"] or not pc["street_nums"]:
        return set()
    hits = set()
    for num in pc["street_nums"]:
        hits |= indexes["addr_num_state"].get((pc["country"], num, pc["state"]), set())
    return hits


def _strategy_addr_num_rare_tok(pc, indexes, frequencies):
    """Strategy 7: Street number + rare address token."""
    if not pc["street_nums"]:
        return set()
    hits = set()
    for tok in pc["addr_tokens"]:
        freq = frequencies["addr_token"].get((pc["country"], tok), 0)
        if freq <= MAX_ADDRESS_TOKEN_FREQ:
            for num in pc["street_nums"]:
                hits |= indexes["addr_num_rare_tok"].get(
                    (pc["country"], num, tok), set()
                )
    return hits


def _strategy_rare_name_token(pc, indexes, frequencies):
    """Supporting strategy: Rare name token matching."""
    hits = set()
    for tok in pc["name_tokens"]:
        if len(tok) < 3:
            continue
        freq = frequencies["name_token"].get((pc["country"], tok), 0)
        if freq <= MAX_NAME_TOKEN_FREQ:
            hits |= indexes["rare_name_token"].get((pc["country"], tok), set())
    return hits


def _strategy_rare_ngram(pc, indexes, frequencies):
    """Strategy 8: Rare character n-gram overlap (≥ 30% or MIN_SHARED)."""
    grams = char_ngrams(pc["name_leet"], NGRAM_SIZE)
    if not grams:
        return set()

    counts = Counter()
    for gram in grams:
        freq = frequencies["ngram"].get((pc["country"], gram), 0)
        if freq <= MAX_NGRAM_FREQ:
            for idx in indexes["rare_ngram"].get((pc["country"], gram), set()):
                counts[idx] += 1

    if len(grams) <= 4:
        required = 1
    else:
        required = max(MIN_SHARED_RARE_NGRAMS, math.ceil(len(grams) * 0.30))

    return {idx for idx, count in counts.items() if count >= required}


# ============================================================
# MAIN API: generate candidates for one row
# ============================================================

def generate_candidates_for_row(row, target, indices):
    """
    Generate candidate matches for a single source1 row.

    Args:
        row: dict-like with entity_id, business_name, business_address, country
        target: full target DataFrame (S2+S3)
        indices: output of build_indices()

    Returns:
        DataFrame subset of target containing candidate rows.
    """
    index_data = indices.get("indexes", indices)
    frequencies = indices.get("frequencies", {})

    if isinstance(index_data, dict) and "country_name" not in index_data:
        # Backwards compat: if indices is the old format
        index_data = indices

    pc = _precompute_row(row)

    hits = set()
    hits |= _strategy_exact_name(pc, index_data)
    hits |= _strategy_core_name(pc, index_data)
    hits |= _strategy_phonetic(pc, index_data)
    hits |= _strategy_sorted_sig(pc, index_data)
    hits |= _strategy_addr_num_state(pc, index_data)
    hits |= _strategy_addr_num_rare_tok(pc, index_data, frequencies)
    hits |= _strategy_rare_name_token(pc, index_data, frequencies)
    hits |= _strategy_rare_ngram(pc, index_data, frequencies)

    if not hits:
        return target.iloc[0:0].copy()

    valid_hits = [idx for idx in hits if idx in target.index]
    if not valid_hits:
        return target.iloc[0:0].copy()

    candidates = target.loc[sorted(valid_hits)]
    if "entity_id" in candidates.columns:
        candidates = candidates.drop_duplicates(subset=["entity_id"])
    return candidates


# ============================================================
# BATCH CANDIDATE GENERATION (with TF-IDF)
# ============================================================

def generate_candidates_batch(source1, target, indices, use_tfidf=True):
    """
    Generate candidate pairs for all source1 entities.
    Uses TF-IDF blocking (Strategy 4) in batch mode for efficiency.

    Returns:
        list of (source1_entity_id, candidate_entity_id) tuples
        Also returns candidate_map: {s1_id: [candidate_ids]}
    """
    index_data = indices["indexes"]
    frequencies = indices["frequencies"]
    tfidf_data = indices.get("tfidf_data")

    pairs = []
    candidate_map = {}

    # Process in batches for TF-IDF
    s1_rows = list(source1.iterrows())
    n_total = len(s1_rows)

    for batch_start in range(0, n_total, TFIDF_BATCH_SIZE):
        batch_end = min(batch_start + TFIDF_BATCH_SIZE, n_total)
        batch = s1_rows[batch_start:batch_end]

        # Precompute for batch
        batch_pcs = []
        for _, row in batch:
            batch_pcs.append(_precompute_row(row))

        # TF-IDF candidates for the batch
        tfidf_results = None
        if use_tfidf and tfidf_data is not None:
            query_texts = [pc["name_leet"] if pc["name_leet"] else " " for pc in batch_pcs]
            tfidf_results = _tfidf_candidates_batch(query_texts, tfidf_data)

        # Combine all strategies
        for i, ((_, row), pc) in enumerate(zip(batch, batch_pcs)):
            hits = set()

            # Index-based strategies
            hits |= _strategy_exact_name(pc, index_data)
            hits |= _strategy_core_name(pc, index_data)
            hits |= _strategy_phonetic(pc, index_data)
            hits |= _strategy_sorted_sig(pc, index_data)
            hits |= _strategy_addr_num_state(pc, index_data)
            hits |= _strategy_addr_num_rare_tok(pc, index_data, frequencies)
            hits |= _strategy_rare_name_token(pc, index_data, frequencies)
            hits |= _strategy_rare_ngram(pc, index_data, frequencies)

            # TF-IDF candidates (Strategy 4)
            if tfidf_results is not None:
                hits |= tfidf_results[i]

            # Resolve to entity_ids
            valid_hits = [idx for idx in hits if idx in target.index]
            if valid_hits:
                candidate_eids = target.loc[sorted(valid_hits), "entity_id"].drop_duplicates().tolist()
            else:
                candidate_eids = []

            source_id = row["entity_id"]
            candidate_map[source_id] = candidate_eids
            for ceid in candidate_eids:
                pairs.append((source_id, ceid))

        # Progress
        done = batch_end
        print(f"  [blocking] Processed {done:,}/{n_total:,} source1 entities "
              f"({done/n_total*100:.1f}%)", flush=True)

    # Deduplicate pairs
    pairs = list(dict.fromkeys(pairs))
    return pairs, candidate_map


# ============================================================
# SIMPLE FULL GENERATOR (backwards compatible)
# ============================================================

def generate_candidates(source1, target):
    """
    Generate all candidate pairs. Simple interface for training.
    Returns list of (s1_entity_id, target_entity_id) tuples.
    """
    indices = build_indices(target)
    pairs, _ = generate_candidates_batch(source1, target, indices)
    return pairs


# ============================================================
# DEBUG: per-strategy breakdown for a single row
# ============================================================

def blocking_debug_info(row, target, indices):
    """Return per-strategy candidate counts for debugging."""
    index_data = indices["indexes"]
    frequencies = indices["frequencies"]
    tfidf_data = indices.get("tfidf_data")

    pc = _precompute_row(row)

    exact = _strategy_exact_name(pc, index_data)
    core = _strategy_core_name(pc, index_data)
    phonetic = _strategy_phonetic(pc, index_data)
    sorted_sig = _strategy_sorted_sig(pc, index_data)
    addr_state = _strategy_addr_num_state(pc, index_data)
    addr_rare = _strategy_addr_num_rare_tok(pc, index_data, frequencies)
    rare_tok = _strategy_rare_name_token(pc, index_data, frequencies)
    ngram = _strategy_rare_ngram(pc, index_data, frequencies)

    # TF-IDF for single row
    tfidf_hits = set()
    if tfidf_data is not None:
        text = pc["name_leet"] if pc["name_leet"] else " "
        results = _tfidf_candidates_batch([text], tfidf_data)
        tfidf_hits = results[0] if results else set()

    all_candidates = exact | core | phonetic | sorted_sig | addr_state | addr_rare | rare_tok | ngram | tfidf_hits

    return {
        "exact_name": len(exact),
        "core_name": len(core),
        "phonetic": len(phonetic),
        "tfidf": len(tfidf_hits),
        "sorted_sig": len(sorted_sig),
        "addr_num_state": len(addr_state),
        "addr_num_rare_tok": len(addr_rare),
        "rare_name_token": len(rare_tok),
        "rare_ngram": len(ngram),
        "total": len(all_candidates),
    }


# ============================================================
# MODULE ENTRY
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MULTI-STRATEGY BLOCKER v2")
    print("=" * 70)
    print("Strategy 1: Exact normalized name + country")
    print("Strategy 2: Core name (no legal suffix) + country")
    print("Strategy 3: Phonetic (consonant skeleton) + country")
    print("Strategy 4: TF-IDF top-K name similarity")
    print("Strategy 5: Sorted token signature + country")
    print("Strategy 6: Address number + state (garbled-name rescue)")
    print("Strategy 7: Street number + rare address token")
    print("Strategy 8: Rare character 4-gram overlap")
    print("=" * 70)