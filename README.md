# Business Entity Resolution

Finds, for every Source 1 (S1) business record, the matching records in Source 2 and Source 3 (S2/S3). The pipeline is two-stage: **blocking** (a small candidate set per S1 record) followed by a **pairwise classifier** with a tuned threshold. It produces the two submission files, `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

It uses only the supplied training and test data: no external databases, geocoders, APIs or internet services. The classifier is scikit-learn's `HistGradientBoostingClassifier`; no pretrained language models are used.

## Expected data

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

All files are read as tab-separated text, with every column as a string.

## Install

Python 3.12 (tested on Windows 11, 16 GB RAM).

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the submission

One command (Windows PowerShell) runs train → predict → validation and logs everything under `logs/`:

```powershell
powershell -ExecutionPolicy Bypass -File run_submission.ps1                  # in the foreground
powershell -ExecutionPolicy Bypass -File launch_submission.ps1               # detached (scheduled task)
Get-Content logs\submission-status.txt                                       # RUNNING <step> / DONE / FAILED <step>
```

Or step by step (any OS):

```bash
# 1. Train on a seeded sample of S1 (80% train / 20% holdout). Reports holdout F0.5,
#    blocking recall and candidates per S1; saves the model, threshold and K.
python -m src.train --data dataset/train --sample-size 100000 --model model.joblib

# 2. Predict on the full test set; writes output/matching_results.tsv and output/candidate_pairs.tsv.
python -m src.predict --data dataset/test --model model.joblib --output output

# 3. Validate. The official validator checks the matching file; its candidate check
#    needs ~8-10 GB for the full candidate file, so pass a non-existent --candidate path
#    and check candidates with the streaming checker instead.
python <student_resource>/utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/__skip__.tsv --test-dir dataset/test
python -m src.check_candidates --candidates output/candidate_pairs.tsv \
    --matching output/matching_results.tsv --test-dir dataset/test
```

Measured on a 16 GB laptop with 6 worker processes (memory is approximate):

| Step | Time | Memory |
|---|---|---|
| First run on a dataset: normalization cache | ~2–5 min per dataset | ~3 GB |
| First run on a dataset: blocking index | ~4–5 min per dataset | ~4–5 GB |
| `train --sample-size 100000` | 12 min | ~5 GB |
| `train --sample-size 10000` (quick evaluation) | ~1.5 min | ~4 GB |
| `predict` on the full test set (1.73M S1) | 3 h 41 min | ~5 GB |
| Validation + streaming candidate check | ~1.5 min | ~1.5 GB |

## Evaluation tools

```bash
# Blocking only: recall and candidates per S1 for several K, overall and per country
python -m src.test_blocking --data dataset/train --sample-size 10000 --k 5 10 20 50

# Train + holdout report on a smaller sample
python -m src.train --data dataset/train --sample-size 10000 --model model_eval.joblib

# Normalization unit test (set PYTHONIOENCODING=utf-8 on a Windows console)
python -m src.test_train
```

The holdout is 20% of *all* S1 entities (fixed seed); `--sample-size N` takes the same proportions as prefixes, so a smaller sample is always a subset of a larger one and results are comparable across runs. Macro F0.5 is reported at the threshold chosen by the scan and as a **cross-fitted** estimate (tune on half of the holdout, score on the other half), which is not inflated by tuning.

## How it works

1. **Normalization** (`src/normalize.py`). Lowercasing, accent and punctuation removal, and transliteration of all nine Indian scripts to Latin (`indic-transliteration`, ITRANS, plus fixes such as dropping the word-final inherent "a"). Legal and common words are mapped to canonical tokens (`private`/`pvt`/`praivet` → `pvt`, `limited` → `ltd`, French `sarl`/`sas` are legal suffixes), and so are address words (`street` → `st`, `avenue` → `ave`, `boulevard`/`bd` → `blvd`, US state names → codes). Domain-style names keep only the domain label. The *core* name is the name without legal suffixes.
2. **Blocking** (`src/blocking.py`), always within the same country label. Country is an open set; nothing is hard-coded per country:
   - **Pool:** TF-IDF nearest neighbours. Name vectors: words, adjacent word pairs and in-word character 3-grams of the core name. Address vectors: words, numbers and adjacent word pairs. Two rankings (name-led and address-led, each with a small cross term) keep the top 200 each.
   - **Re-rank:** exact string similarity (rapidfuzz) on the pool; each S1 keeps the top K = 20 of the re-ranked name-led, address-led and TF-IDF rankings (~58 candidates per S1).
3. **Features** (`src/features.py`), 32 per pair:
   - string similarities (exact, ratio, token-set/sort, partial, token and 3-gram Jaccard)
   - address numbers (postcode and house-number equal/conflict, digit overlap)
   - empty-field flags
   - the blocking rank and re-rank similarities
   - features relative to the other candidates of the same S1 record (gap to the best candidate, rank, group size)
4. **Model** (`src/train.py`): `HistGradientBoostingClassifier` (250 iterations, 31 leaves). The decision threshold is chosen on the holdout to maximize macro F0.5 and saved with the model.
5. **Prediction** (`src/predict.py`): blocking → features → probabilities → threshold, in chunks of 25,000 S1 records. `candidate_pairs.tsv` is exactly the set of pairs the model scored, so every match is also a candidate.

Final model (100,000-S1 training sample, 20,000 held out): holdout macro F0.5 0.942 (cross-fitted 0.942), blocking recall 96.7% with 57.8 candidates per S1 on average. On the test set, 57.4 candidates per S1 on average (max 80).

## Caches and memory

- `cache/*.arrow`: normalized records per source file, memory-mapped. Rebuilt when a source file or `NORMALIZE_VERSION` (in `normalize.py`) changes.
- `cache/blocking-index-*/`: TF-IDF index per target dataset (train and test side by side), memory-mapped `.npy` files. Rebuilt when `INDEX_VERSION` or the pruning settings (in `blocking.py`) change.
- Parallel steps use at most `N_JOBS = 6` worker processes (`src/data.py`). `src/__init__.py` sets `OPENBLAS_NUM_THREADS=1`, which saves ~1 GB per process.

## Files

| File | Role |
|---|---|
| `src/normalize.py` | Text normalization and transliteration |
| `src/data.py` | Loading, normalization cache, fixed S1 holdout split, ground truth |
| `src/blocking.py` | Candidate generation (`BlockingIndex`) |
| `src/features.py` | Pair features |
| `src/evaluate.py` | Macro F0.5, blocking statistics, threshold selection |
| `src/train.py` | Training and holdout evaluation |
| `src/predict.py` | Test predictions and submission files |
| `src/test_blocking.py` | Blocking evaluation |
| `src/check_candidates.py` | Streaming check of `candidate_pairs.tsv` |
| `src/test_train.py` | Normalization unit test |
| `src/io.py` | TSV reading/writing helpers |
| `run_submission.ps1`, `launch_submission.ps1` | Full submission run (foreground / detached) |
| `run_test_blocking.sh` | Blocking evaluation on all training S1 (slow; prefer `--sample-size`) |
