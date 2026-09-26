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

## Train

Run from the project root:

```bash
# Default: HistGradientBoostingClassifier
python -m src.train --data dataset/train --model model.joblib --model-type histgb

# XGBoost Classifier
python -m src.train --data dataset/train --model model.joblib --model-type xgboost
```

## Predict

```bash
python -m src.predict --data dataset/test --model model.joblib --output output --threshold 0.80
```

The threshold is intentionally exposed as a parameter. It should be selected using an entity-level validation split and the competition's macro F_0.5 metric rather than assumed to be optimal.

## Test on Training Data

Evaluate the model against ground truth data (evaluating candidate recall, pair precision/recall, and Macro $F_{0.5}$):

```bash
python -m src.test_train --data dataset/train --model model.joblib --threshold 0.80
```

To scan multiple thresholds to find the optimal Macro $F_{0.5}$ threshold:

```bash
python -m src.test_train --data dataset/train --model model.joblib --scan-thresholds
```

## Test Blocking Stage Alone

Run heavy diagnostics and key attribution benchmarks on candidate generation & blocking:

```bash
# Convenient one-step executable script (auto-handles virtual environment)
./run_test_blocking.sh

# Or directly via Python module (sample of 10,000 entities with failure diagnostics)
python -m src.test_blocking --data dataset/train --sample-size 10000 --show-missed 10

# Direct Python run on full training dataset
python -m src.test_blocking --data dataset/train --sample-size 0
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
