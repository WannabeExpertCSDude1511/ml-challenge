"""
Candidate generation (blocking), shared by train.py, predict.py and
test_blocking.py.

Two stages, within the same country:

1. Pool: top-POOL_K nearest neighbours by TF-IDF cosine similarity.
2. Re-rank the pool with exact string similarity (rapidfuzz) and keep the
   top-K. The final candidate set is at most a few times K per S1 record.

Stage 1 vectors:

    name vector    : words and adjacent word pairs of the core name (legal
                     words removed) plus character 3-grams inside each word,
                     so typos and concatenations still share most 3-grams
    address vector : address words, numbers and adjacent word pairs

Word pairs turn combinations of common words ("431 jackson", "shiva anand")
into distinctive features even when each word alone is too common.

Each stage keeps two rankings and unions their top lists:

    name-led    : name similarity    + CROSS_WEIGHT * address similarity
    address-led : address similarity + CROSS_WEIGHT * name similarity

The address-led list finds same-address records whose name differs (trade
names, poor transliterations); the small cross term breaks ties among the many
records sharing a common name.

Stage 2 similarities: name = max(token-set ratio of the core names, ratio of
the names with spaces removed), address = token-set ratio. A candidate is kept
if it is in the top K of either re-ranked list or of the stage-1 ranking.

Features found in more than MAX_DF target records (common words and 3-grams,
city names) are dropped before the vectors are normalized, so similarity is
cosine over the distinctive features only. They carry little IDF weight but
would dominate the cost of the sparse product, and leaving them in the vector
length would make near-identical names score low.

Country is only an open-set grouping key: records are compared with targets
that carry the same normalized country label, whatever the labels are.

The built index is cached under cache/ (keyed by the normalized target files
and INDEX_VERSION), so only the first run pays for building it. Re-ranking workers memory-map
the normalized target cache files, so target strings are shared, not copied.
"""
import hashlib
import json
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from scipy import sparse
from rapidfuzz.fuzz import ratio, token_set_ratio
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import HashingVectorizer

from .data import CACHE_DIR, N_JOBS

DEFAULT_K = 20
POOL_K = 200
CROSS_WEIGHT = 0.1
MAX_DF = 5_000
N_FEATURES = 2 ** 21
VECTOR_CHUNK = 50_000
QUERY_CHUNK = 250
RERANK_CHUNK = 50_000
# Scores below this are never kept; it keeps the per-row top-k selection small.
MIN_SCORE = 0.05

# Bump when the vectors or pruning change; it invalidates cached indexes.
INDEX_VERSION = 7

_HASH = dict(n_features=N_FEATURES, alternate_sign=False, norm=None, binary=True,
             lowercase=False, dtype=np.float32)
_NAME_GRAMS = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), **_HASH)
_WORDS = HashingVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), **_HASH)


def _vectorize(cores, names, addresses):
    # Words are prefixed so a word never shares a hash with an identical 3-gram.
    texts = [core or name for core, name in zip(cores, names)]
    name_raw = _NAME_GRAMS.transform(texts) + _WORDS.transform(
        [" ".join("#" + w for w in t.split()) for t in texts]
    )
    name_raw.data[:] = 1
    return name_raw.tocsr(), _WORDS.transform(addresses)


def _vector_chunks(frame, n_jobs):
    """Yield (start, name_matrix, address_matrix) for consecutive row chunks."""
    starts = range(0, len(frame), VECTOR_CHUNK)
    results = Parallel(n_jobs=n_jobs, return_as="generator", pre_dispatch="n_jobs")(
        delayed(_vectorize)(
            frame["core"].iloc[i:i + VECTOR_CHUNK].tolist(),
            frame["name"].iloc[i:i + VECTOR_CHUNK].tolist(),
            frame["addr"].iloc[i:i + VECTOR_CHUNK].tolist(),
        )
        for i in starts
    )
    for start, (name_raw, addr_raw) in zip(starts, results):
        yield start, name_raw, addr_raw


def _country_codes(frame):
    codes, labels = frame["country"].factorize()
    return np.asarray(codes), list(labels)


def _row_top_k(R, k):
    """Row-wise top-k of a CSR score matrix. Returns (rows, cols, rank)."""
    R = R.tocsr()
    keep = R.data >= MIN_SCORE
    row_of = np.repeat(np.arange(R.shape[0], dtype=np.int32), np.diff(R.indptr))[keep]
    cols, data = R.indices[keep], R.data[keep]
    # Sort by row, then by descending score; rank = position within the row.
    order = np.lexsort((-data, row_of))
    row_of, cols = row_of[order], cols[order]
    starts = np.searchsorted(row_of, row_of)
    rank = (np.arange(len(row_of)) - starts).astype(np.int32)
    top = rank < k
    return row_of[top], cols[top], rank[top]


def _union_top_k(name_led, addr_led, k):
    """Union of two top-k lists; rank is the best rank over both."""
    r1, c1, k1 = _row_top_k(name_led, k)
    r2, c2, k2 = _row_top_k(addr_led, k)
    rows, cols, ranks = np.r_[r1, r2], np.r_[c1, c2], np.r_[k1, k2]
    order = np.lexsort((ranks, cols, rows))
    rows, cols, ranks = rows[order], cols[order], ranks[order]
    first = np.r_[True, (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])]
    return rows[first], cols[first], ranks[first]


def _rerank_text(names, cores, addresses):
    """Strings used for re-ranking: name (core, or full name if empty), without spaces, address."""
    name = [core or full for core, full in zip(cores, names)]
    return name, [n.replace(" ", "") for n in name], addresses


_MAPPED = {}


def _mapped_target(files):
    """The target table memory-mapped from its cache files, opened once per worker."""
    if files not in _MAPPED:
        _MAPPED[files] = pa.concat_tables(
            [pa.ipc.open_file(pa.memory_map(f)).read_all() for f in files]
        ).select(["name", "core", "addr"])
    return _MAPPED[files]


def _similarity_chunk(files, q_name, q_compact, q_addr, q_pos, t_idx):
    """q_* hold only the S1 records in this chunk; q_pos maps each pair to one of them."""
    q_name = [q_name[i] for i in q_pos]
    q_compact = [q_compact[i] for i in q_pos]
    q_addr = [q_addr[i] for i in q_pos]
    t = _mapped_target(files).take(pa.array(t_idx))
    t_name, t_compact, t_addr = _rerank_text(
        t.column("name").to_pylist(), t.column("core").to_pylist(), t.column("addr").to_pylist()
    )
    tokens = cpdist(q_name, t_name, scorer=token_set_ratio)
    compact = cpdist(q_compact, t_compact, scorer=ratio)
    addr = cpdist(q_addr, t_addr, scorer=token_set_ratio)
    return (np.maximum(tokens, compact) / 100).astype(np.float32), (addr / 100).astype(np.float32)


def _string_similarity(s1, target_files, s_idx, t_idx, n_jobs):
    """(name_sim, addr_sim) in [0, 1] for aligned pairs, computed in parallel chunks."""
    q_name, q_compact, q_addr = _rerank_text(
        s1["name"].tolist(), s1["core"].tolist(), s1["addr"].tolist()
    )

    def args(a):
        # Pairs are sorted by S1 record, so a chunk covers few distinct records:
        # send those once plus an index array instead of one string per pair.
        records, q_pos = np.unique(s_idx[a:a + RERANK_CHUNK], return_inverse=True)
        records = records.tolist()
        return (target_files, [q_name[i] for i in records], [q_compact[i] for i in records],
                [q_addr[i] for i in records], q_pos.tolist(), t_idx[a:a + RERANK_CHUNK])

    parts = Parallel(n_jobs=n_jobs, pre_dispatch="n_jobs")(
        delayed(_similarity_chunk)(*args(a)) for a in range(0, len(s_idx), RERANK_CHUNK)
    )
    if not parts:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def _ranks(s_idx, score, tiebreak):
    """Rank of each pair within its S1 record by descending score (s_idx sorted)."""
    order = np.lexsort((tiebreak, -score, s_idx))
    ranks = np.empty(len(order), dtype=np.int32)
    ranks[order] = np.arange(len(order)) - np.searchsorted(s_idx[order], s_idx[order])
    return ranks


def _cache_path(target):
    key = target.attrs.get("cache_key")
    if not key:
        return None
    digest = hashlib.sha1(f"{key}|{INDEX_VERSION}|{MAX_DF}|{N_FEATURES}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"blocking-index-{digest}"


def _free_workers():
    """Shut down idle joblib workers so they release their memory."""
    get_reusable_executor().shutdown(wait=True)


class BlockingIndex:
    """Weighted target vectors grouped by country. Build once, query many times."""

    def __init__(self, target, n_jobs=N_JOBS, verbose=True):
        start_time = time.perf_counter()
        self.n_jobs = n_jobs
        # Cache files behind the target frame (see data.load_split), for re-ranking workers.
        self.target_files = tuple(str(f) for f in target.attrs["cache_files"])
        cache = _cache_path(target)
        if cache is not None and (cache / "meta.json").exists():
            self._load(cache)
            source = "loaded from cache"
        else:
            self._build(target)
            if cache is not None:
                self._save(cache)
                # Swap the in-memory copy for the memory-mapped one.
                self.groups = None
                self._load(cache)
            source = "built"

        if verbose:
            nnz = sum(g[1].nnz for g in self.groups.values())
            print(f"Blocking index {source}: {len(target):,} targets, {len(self.groups)} countries, "
                  f"{nnz:,} nonzeros, {time.perf_counter() - start_time:.0f}s", flush=True)

    def _build(self, target):
        n = len(target)

        # Pass 1: document frequencies (vectors are discarded right away).
        name_df = np.zeros(N_FEATURES, dtype=np.int64)
        addr_df = np.zeros(N_FEATURES, dtype=np.int64)
        for _, name_raw, addr_raw in _vector_chunks(target, self.n_jobs):
            name_df += np.bincount(name_raw.indices, minlength=N_FEATURES)
            addr_df += np.bincount(addr_raw.indices, minlength=N_FEATURES)

        self.name_idf = (np.log((1 + n) / (1 + name_df)) + 1).astype(np.float32)
        self.addr_idf = (np.log((1 + n) / (1 + addr_df)) + 1).astype(np.float32)
        self.name_keep = (name_df <= MAX_DF).astype(np.float32)
        self.addr_keep = (addr_df <= MAX_DF).astype(np.float32)
        del name_df, addr_df

        # Pass 2: weighted, pruned vectors, split by country as they are built.
        codes, labels = _country_codes(target)
        parts = defaultdict(list)
        for start, name_raw, addr_raw in _vector_chunks(target, self.n_jobs):
            V = sparse.hstack(self._weighted(name_raw, addr_raw), format="csr")
            del name_raw, addr_raw
            chunk_codes = codes[start:start + V.shape[0]]
            for c in np.unique(chunk_codes):
                rows = np.flatnonzero(chunk_codes == c)
                parts[c].append((rows + start, V[rows]))
            del V
        del codes
        # The vectorizing workers are idle from here on; free their memory.
        _free_workers()

        # Per country: row positions + one stacked matrix (name features then
        # address features) x targets, so a ranking is a single sparse product.
        self.groups = {}
        for c in list(parts):
            chunk_list = parts.pop(c)
            rows = np.concatenate([p[0] for p in chunk_list]).astype(np.int32)
            stacked = sparse.vstack([p[1] for p in chunk_list], format="csr")
            del chunk_list
            # CSC of (targets x features) is CSR of its transpose: one conversion.
            self.groups[labels[c]] = (rows, stacked.tocsc().T)
            del stacked

    def _save(self, path):
        """One .npy file per array, so loading can memory-map instead of copying."""
        CACHE_DIR.mkdir(exist_ok=True)
        for old in CACHE_DIR.glob("blocking-index-*"):
            shutil.rmtree(old, ignore_errors=True)
        tmp = path.with_name(path.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        arrays = {
            "name_idf": self.name_idf, "addr_idf": self.addr_idf,
            "name_keep": self.name_keep, "addr_keep": self.addr_keep,
        }
        labels = list(self.groups)
        for i, (rows, m) in enumerate(self.groups.values()):
            arrays.update({f"g{i}_rows": rows, f"g{i}_data": m.data, f"g{i}_indices": m.indices, f"g{i}_indptr": m.indptr})
        for name, array in arrays.items():
            np.save(tmp / f"{name}.npy", array)
        shapes = [list(m.shape) for _, m in self.groups.values()]
        (tmp / "meta.json").write_text(json.dumps({"labels": labels, "shapes": shapes}))
        tmp.replace(path)

    def _load(self, path):
        def load(name):
            return np.load(path / f"{name}.npy", mmap_mode="r")

        meta = json.loads((path / "meta.json").read_text())
        self.name_idf, self.addr_idf = load("name_idf"), load("addr_idf")
        self.name_keep, self.addr_keep = load("name_keep"), load("addr_keep")
        self.groups = {}
        for i, (label, shape) in enumerate(zip(meta["labels"], meta["shapes"])):
            m = sparse.csr_matrix((load(f"g{i}_data"), load(f"g{i}_indices"), load(f"g{i}_indptr")),
                                  shape=tuple(shape), copy=False)
            self.groups[label] = (load(f"g{i}_rows"), m)

    def _weighted(self, name_raw, addr_raw):
        """Unit-length TF-IDF name and address vectors over the kept (distinctive) features."""
        blocks = []
        for X, idf, keep in (
            (name_raw, self.name_idf, self.name_keep),
            (addr_raw, self.addr_idf, self.addr_keep),
        ):
            X = (X @ sparse.diags(idf * keep)).tocsr()
            X.eliminate_zeros()
            norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
            scale = np.where(norms > 0, 1 / np.maximum(norms, 1e-12), 0).astype(np.float32)
            X = (sparse.diags(scale) @ X).tocsr()
            blocks.append(X)
        return blocks

    def query(self, s1, k=DEFAULT_K, return_ranks=False, verbose=True):
        """
        Return (s_idx, t_idx[, rank]) for every S1 row: the stage-1 pool
        re-ranked, keeping pairs whose best rank is below k. Sorted by s_idx,
        then rank.
        """
        start_time = time.perf_counter()
        s_idx, t_idx, pool_rank = self._pool(s1, max(k, POOL_K))
        _free_workers()

        name_sim, addr_sim = _string_similarity(s1, self.target_files, s_idx, t_idx, self.n_jobs)
        rank = np.minimum.reduce([
            _ranks(s_idx, name_sim + CROSS_WEIGHT * addr_sim, pool_rank),
            _ranks(s_idx, addr_sim + CROSS_WEIGHT * name_sim, pool_rank),
            pool_rank,
        ])
        del name_sim, addr_sim, pool_rank
        _free_workers()

        keep = rank < k
        s_idx, t_idx, rank = s_idx[keep], t_idx[keep], rank[keep]
        order = np.lexsort((rank, s_idx))
        s_idx, t_idx, rank = s_idx[order], t_idx[order], rank[order]

        if verbose:
            print(f"Blocking: {len(s_idx):,} candidate pairs for {len(s1):,} S1 records "
                  f"(k={k}, pool {max(k, POOL_K)}) in {time.perf_counter() - start_time:.0f}s", flush=True)
        return (s_idx, t_idx, rank) if return_ranks else (s_idx, t_idx)

    def _pool(self, s1, k):
        """Stage 1: union of the TF-IDF name-led and address-led top-k."""
        blocks = [self._weighted(n, a) for _, n, a in _vector_chunks(s1, self.n_jobs)]
        Sn = sparse.vstack([b[0] for b in blocks], format="csr")
        Sa = sparse.vstack([b[1] for b in blocks], format="csr")
        del blocks
        # Weighting the query gives each ranking's score in one product.
        name_led_q = sparse.hstack([Sn, CROSS_WEIGHT * Sa], format="csr")
        addr_led_q = sparse.hstack([CROSS_WEIGHT * Sn, Sa], format="csr")
        del Sn, Sa
        codes, labels = _country_codes(s1)

        jobs = []
        for c, label in enumerate(labels):
            if label not in self.groups:
                continue
            s_rows = np.flatnonzero(codes == c).astype(np.int32)
            for a in range(0, len(s_rows), QUERY_CHUNK):
                jobs.append((s_rows[a:a + QUERY_CHUNK], label))

        def run(job):
            chunk, label = job
            t_rows, matrix_t = self.groups[label]
            r, col, rank = _union_top_k(name_led_q[chunk] @ matrix_t, addr_led_q[chunk] @ matrix_t, k)
            return chunk[r], t_rows[col], rank

        # Threads share the index without copying it; the sparse products
        # partly release the GIL.
        with ThreadPoolExecutor(self.n_jobs) as pool:
            results = list(pool.map(run, jobs))
        del name_led_q, addr_led_q

        if results:
            s_idx, t_idx, ranks = (np.concatenate(p) for p in zip(*results))
        else:
            s_idx, t_idx, ranks = (np.empty(0, np.int32) for _ in range(3))
        order = np.lexsort((ranks, s_idx))
        return s_idx[order], t_idx[order], ranks[order]


def generate_candidates(s1, target, k=DEFAULT_K, index=None):
    """Top-k candidates for every S1 row. Pass a prebuilt index to reuse it."""
    if index is None:
        index = BlockingIndex(target)
    return index.query(s1, k)
