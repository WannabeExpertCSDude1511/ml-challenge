# Methodology

## Methodology used

Two-stage entity resolution: candidate generation/blocking followed by supervised pair classification.

## Candidate generation / blocking strategy

Candidates are generated from normalized country-aware exact name and address keys, legal-suffix-stripped names, numeric address tokens, and distinctive business-name tokens. A same-country fallback is used when no block fires.

## Model architecture and feature engineering

The model is a scikit-learn HistGradientBoostingClassifier. Features capture exact normalized equality, fuzzy string similarity, token Jaccard overlap, character n-gram overlap, numeric address-token overlap, country equality, and string-length differences.

## Evaluation

Threshold selection should be performed using a Source-1-entity-level validation split and macro F_0.5, including singleton Source-1 entities.

## External data / fair play

The pipeline is designed to use only the supplied challenge data. No external business lookup, geocoding, registration database, API, or internet-based entity enrichment is used.
