"""
Data loading shared by train.py, predict.py and test_blocking.py.

- Every record is normalized once and cached to cache/*.arrow, keyed by the
  source file (size + mtime) and NORMALIZE_VERSION, so reruns skip the work.
- The S1 holdout split is fixed by the seed and independent of --sample-size:
  a smaller sample is always a prefix of a larger one, so runs are comparable.
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import Parallel, delayed

from .io import read_tsv
from .normalize import NORMALIZE_VERSION, normalize_fields

# Worker processes for parallel steps; kept low so a 16 GB laptop has headroom.
N_JOBS = 6

CACHE_DIR = Path("cache")
NORMALIZE_CHUNK = 25_000
COLUMNS = ["country", "name", "core", "addr"]


def _normalize_chunk(names, addresses, countries):
    return [normalize_fields(n, a, c) for n, a, c in zip(names, addresses, countries)]


def load_normalized(path, n_jobs=N_JOBS):
    """
    Return a DataFrame with entity_id + normalized country/name/core/addr.
    Columns are pyarrow-backed strings backed by a memory-mapped cache file,
    so ~10M records cost file cache (shared, reclaimable), not private memory.
    """
    cache = normalized_cache(path, n_jobs)
    frame = _open_arrow(cache).to_pandas(types_mapper=pd.ArrowDtype)
    frame.attrs["cache_key"] = cache.stem
    return frame


def _open_arrow(path):
    return pa.ipc.open_file(pa.memory_map(str(path))).read_all()


def _write_arrow(table, path):
    tmp = path.with_suffix(".tmp")
    with pa.OSFile(str(tmp), "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table, max_chunksize=1_000_000)
    tmp.replace(path)


def normalized_cache(path, n_jobs=N_JOBS):
    """Path of the normalized Arrow cache for a source file, building it if needed."""
    path = Path(path)
    stat = path.stat()
    stem = f"{path.stem}-{stat.st_size}-{stat.st_mtime_ns}-v{NORMALIZE_VERSION}"
    cache = CACHE_DIR / f"{stem}.arrow"

    if cache.exists():
        return cache

    CACHE_DIR.mkdir(exist_ok=True)
    legacy = CACHE_DIR / f"{stem}.parquet"
    if legacy.exists():
        # Same normalization, older on-disk format: convert instead of recomputing.
        _write_arrow(pq.read_table(legacy), cache)
        legacy.unlink()
        return cache

    start = time.perf_counter()
    print(f"Normalizing {path.name} (cached afterwards)...", flush=True)
    df = read_tsv(path)
    bounds = range(0, len(df), NORMALIZE_CHUNK)
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_normalize_chunk)(
            df["business_name"].iloc[i:i + NORMALIZE_CHUNK].tolist(),
            df["business_address"].iloc[i:i + NORMALIZE_CHUNK].tolist(),
            df["country"].iloc[i:i + NORMALIZE_CHUNK].tolist(),
        )
        for i in bounds
    )
    rows = [r for part in parts for r in part]
    out = pd.DataFrame(rows, columns=COLUMNS)
    out.insert(0, "entity_id", df["entity_id"].to_numpy())
    del df, rows, parts

    for old in CACHE_DIR.glob(f"{path.stem}-*"):
        old.unlink()
    _write_arrow(pa.Table.from_pandas(out, preserve_index=False), cache)
    print(f"  {len(out):,} records in {time.perf_counter() - start:.0f}s", flush=True)
    return cache


def load_split(data_dir, prefix):
    """Return (s1, target) where target is S2 + S3 concatenated."""
    d = Path(data_dir)
    s1 = load_normalized(d / f"{prefix}_source1.tsv")
    caches = [normalized_cache(d / f"{prefix}_source{i}.tsv") for i in (2, 3)]
    # Concatenating Arrow tables is zero-copy; pd.concat would copy ~1.3 GB.
    target = pa.concat_tables([_open_arrow(c) for c in caches]).to_pandas(types_mapper=pd.ArrowDtype)
    # Identifies the normalized target data, e.g. for the blocking index cache.
    target.attrs["cache_key"] = "+".join(c.stem for c in caches)
    target.attrs["cache_files"] = [c.resolve() for c in caches]
    return s1, target


def load_truth(data_dir, s1_ids=None):
    """Return {source1_entity_id: set(matched ids)}, optionally only for s1_ids."""
    gt = read_tsv(Path(data_dir) / "train_ground_truth.tsv")
    if s1_ids is not None:
        gt = gt[gt["source1_entity_id"].isin(set(s1_ids))]
    return {
        s: {x.strip() for x in m.split(",") if x.strip()}
        for s, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"])
    }


def split_s1(n, holdout=0.2, seed=42, sample_size=0):
    """
    Return (train_positions, holdout_positions) into S1.

    The first `holdout` fraction of a seeded permutation is held out. With
    sample_size > 0 the same proportions are taken as prefixes of both parts.
    """
    perm = np.random.default_rng(seed).permutation(n)
    n_hold = int(round(n * holdout))
    hold, train = perm[:n_hold], perm[n_hold:]
    if sample_size:
        k = int(round(sample_size * holdout))
        hold, train = hold[:k], train[:sample_size - k]
    return train, hold


def truth_positions(s1_ids, truth, target):
    """
    For each S1 id, the set of target row positions it truly matches.
    Returns (list_of_sets, n_true array).
    """
    wanted = {t for s in s1_ids for t in truth.get(s, ())}
    rows = np.flatnonzero(target["entity_id"].isin(wanted).to_numpy())
    pos = dict(zip(target["entity_id"].take(rows).tolist(), rows.tolist()))
    sets = [{pos[t] for t in truth.get(s, ()) if t in pos} for s in s1_ids]
    n_true = np.array([len(truth.get(s, ())) for s in s1_ids], dtype=np.int64)
    return sets, n_true
