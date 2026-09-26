"""
Data loading shared by train.py, predict.py and test_blocking.py.

- Every record is normalized once and cached to cache/*.parquet, keyed by the
  source file (size + mtime) and NORMALIZE_VERSION, so reruns skip the work.
- The S1 holdout split is fixed by the seed and independent of --sample-size:
  a smaller sample is always a prefix of a larger one, so runs are comparable.
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .io import read_tsv
from .normalize import NORMALIZE_VERSION, normalize_fields

# Avoids joblib's noisy physical-core detection warning on Windows.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))

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
    Columns are pyarrow-backed strings to keep ~10M records in memory.
    """
    path = Path(path)
    stat = path.stat()
    cache = CACHE_DIR / f"{path.stem}-{stat.st_size}-{stat.st_mtime_ns}-v{NORMALIZE_VERSION}.parquet"

    if cache.exists():
        return pd.read_parquet(cache, dtype_backend="pyarrow")

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

    CACHE_DIR.mkdir(exist_ok=True)
    for old in CACHE_DIR.glob(f"{path.stem}-*.parquet"):
        old.unlink()
    out.to_parquet(cache, index=False)
    print(f"  {len(out):,} records in {time.perf_counter() - start:.0f}s", flush=True)
    return pd.read_parquet(cache, dtype_backend="pyarrow")


def load_split(data_dir, prefix):
    """Return (s1, target) where target is S2 + S3 concatenated."""
    d = Path(data_dir)
    s1 = load_normalized(d / f"{prefix}_source1.tsv")
    target = pd.concat(
        [load_normalized(d / f"{prefix}_source2.tsv"), load_normalized(d / f"{prefix}_source3.tsv")],
        ignore_index=True,
    )
    return s1, target


def load_truth(data_dir):
    """Return {source1_entity_id: set(matched ids)}."""
    gt = read_tsv(Path(data_dir) / "train_ground_truth.tsv")
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
    wanted = sorted({t for s in s1_ids for t in truth.get(s, ())})
    pos = dict(zip(wanted, pd.Index(target["entity_id"]).get_indexer(wanted).tolist()))
    sets = [{pos[t] for t in truth.get(s, ()) if pos[t] >= 0} for s in s1_ids]
    n_true = np.array([len(truth.get(s, ())) for s in s1_ids], dtype=np.int64)
    return sets, n_true
