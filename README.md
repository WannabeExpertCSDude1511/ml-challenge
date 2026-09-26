# Business Entity Resolution

This package implements a two-stage entity-resolution pipeline for the supplied challenge specification: normalization and blocking followed by supervised pair classification. It produces the required `matching_results.tsv` and `candidate_pairs.tsv` files.

The implementation uses only the supplied training/test data. It does not call external entity databases, geocoders, APIs, or internet services.

## Expected data

Place the challenge data under:

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

All files are read with a tab separator.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Normalization cache

The first run of any command normalizes every record once (in parallel) and caches the result under `cache/`. Later runs reuse it; the cache is rebuilt automatically when a source file or `NORMALIZE_VERSION` changes.

## Train and evaluate

`train.py` holds out 20% of S1 entities (fixed seed), trains on the rest and reports, on the holdout: macro F_0.5 (with the threshold chosen by the scan, plus an unbiased cross-fitted estimate), blocking recall, and average / 99th-percentile candidates per S1. The chosen threshold is saved with the model.

```bash
# Evaluation run on a seeded sample of S1 (train + holdout)
python -m src.train --data dataset/train --model model.joblib --sample-size 10000

# All S1 records
python -m src.train --data dataset/train --model model.joblib --sample-size 0

# XGBoost instead of HistGradientBoosting
python -m src.train --model-type xgboost
```

The holdout is the same set of entities for every `--sample-size`; a smaller sample is a prefix of a larger one.

## Predict

```bash
python -m src.predict --data dataset/test --model model.joblib --output output
```

Uses the same blocking and features as training. `candidate_pairs.tsv` contains exactly the pairs the model scored; `--threshold` overrides the saved one.

## Test blocking alone

```bash
python -m src.test_blocking --data dataset/train --sample-size 10000
```

Runs the same `generate_candidates()` as `train.py` / `predict.py` on the same S1 sample and reports recall and candidates per S1, overall and per country.

## Normalization unit test

```bash
python -m src.test_train
```

## Validate submission

From the challenge's `student_resource/` directory, run the supplied validator:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Important modeling notes

Source 1 is treated as the deduplicated reference source. Source 2 and Source 3 are combined as the target population. Candidate generation uses country-aware exact normalized name/address blocks, legal-suffix-stripped names, numeric address tokens, and distinctive name tokens, with a same-country fallback.

Pair features include exact and fuzzy name agreement, legal-suffix-normalized name agreement, token overlap, character n-gram overlap, address similarity, numeric-token overlap, country agreement, and length differences.

The classifier is `HistGradientBoostingClassifier`, which is available in scikit-learn and does not require a large external language model.
