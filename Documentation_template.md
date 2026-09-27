# Methodology

## Methodology used

Two-stage entity resolution:
1. **Blocking** gives each Source 1 (S1) record a small candidate set from Source 2 and Source 3.
2. A **pairwise classifier** scores each (S1, candidate) pair, and a threshold tuned for macro F0.5 decides the matches.

The pipeline:
1. Normalize every record once (cached): name, core name (legal suffixes removed), address, country.
2. Block: a TF-IDF nearest-neighbour pool, re-ranked by exact string similarity; the top K per S1 record become the candidates.
3. Compute 32 features per candidate pair.
4. Classify with gradient-boosted trees and apply the threshold.

`candidate_pairs.tsv` is exactly the set of pairs the classifier scores, so every predicted match is also a candidate.

All development used a fixed protocol: 20% of S1 entities held out (fixed seed), models trained on a seeded sample of the remaining 80%. Each change was measured by holdout macro F0.5, blocking recall, and candidates per S1 record.

## Candidate generation / blocking strategy

**Normalization.**
- Lowercasing, accent and punctuation removal.
- Transliteration of all nine Indian scripts to Latin with `indic-transliteration` (ITRANS), plus rules for the word-final inherent vowel (`राम` → `ram`), candra vowels, Tamil stops and Malayalam chillu letters.
- Canonical tokens:
  - legal forms: `private`/`pvt`/transliterated `praivet` → `pvt`, `limited` → `ltd`, plus French `sarl`, `sas`, `sa`, …
  - address words: `street` → `st`, `avenue` → `ave`, `boulevard`/`bd` → `blvd`, …
  - US state names → codes
- Domain-style names keep only the domain label.

**Stage 1 — TF-IDF pool** (same country label only):
- Name vector: words, adjacent word pairs and in-word character 3-grams of the core name. This handles typos, word order and concatenations such as `maurewilliamscolombier`.
- Address vector: words, numbers and adjacent word pairs (e.g. `431 jackson`).
- Features are hashed and IDF-weighted. Features occurring in more than 5,000 targets are dropped before normalization, so similarity is cosine over distinctive features and the sparse products stay cheap.
- Two rankings keep their top 200 each:
  - name-led: cos(name) + 0.1·cos(address)
  - address-led: cos(address) + 0.1·cos(name)

  The address-led ranking finds same-address records with a different trading name.

**Stage 2 — re-ranking.** The pool is re-scored with rapidfuzz:
- name similarity = max(token-set ratio of core names, ratio of space-less names)
- address similarity = token-set ratio

A candidate is kept if it ranks in the top **K = 20** of the re-ranked name-led, re-ranked address-led or stage-1 ranking.

**Country** is only an open-set grouping key; nothing is hard-coded per country. On the first 300,000 test S1 records, predicted matches per S1 (and the share with none) were:

| Country | Mean predicted matches | Share with no match |
|---|---|---|
| France (unseen in training) | 3.41 | 4.7% |
| India | 3.08 | 6.6% |
| US | 3.21 | 5.8% |

These are in line with the training data (3.46 true matches per S1, 5.6% singletons).

**Results.**

Training sample (10,000 S1 against all 10.3M training targets):

| K | Recall | Avg / P99 candidates per S1 |
|---|---|---|
| 10 | 95.6% | 26.7 / 37 |
| **20 (used)** | **96.5%** | **57.9 / 73** |
| 50 | 97.1% | 145.6 / 174 |

Test set (`candidate_pairs.tsv`): 1,732,544 S1 records; 57.4 candidates per S1 on average (median 58, 99th percentile 73, max 80); 99.5M pairs in total.

The initial exact-key blocking reached 98.0% recall with ~505,000 candidates per S1. The new blocking uses about 10,000× fewer candidates for ~1.5 points of recall.

**Scalability.** The target index (~10M records) is built once in two streaming passes (~5 min) and memory-mapped. Queries run in bounded-memory chunks; cost grows linearly with S1 and never compares against all targets.

## Model architecture and feature engineering

**Model:** scikit-learn `HistGradientBoostingClassifier` (250 iterations, learning rate 0.08, 31 leaves, L2 1.0). It is trained on all candidate pairs of 80,000 S1 records (4.6M pairs), where positives are pairs listed in the ground truth.

**Features** (32 per pair):

| Group | Features |
|---|---|
| Name | exact (name, core), ratio (name, core), token-set, token-sort and partial ratio, word and 3-gram Jaccard, length difference |
| Address | exact, ratio, word and 3-gram Jaccard, digit-token Jaccard, length difference |
| Address numbers | postcode equal / conflict, house number equal / conflict |
| Data quality | S1 / candidate address empty, country equal |
| Blocking | blocking rank, re-rank name and address similarity, their sum |
| Relative to the S1 record's other candidates | gap to the best name, address and combined similarity; rank by combined similarity; number of candidates |

The relative and blocking features tell the model whether a candidate stands out among its competitors, which matters for precision under F0.5. They raised holdout F0.5 from 0.932 to 0.942.

**Decision rule:** a pair is a match if its probability is at least the threshold. The threshold is chosen on the holdout by scanning 0.02–0.98 for the best macro F0.5 (**0.70**).

## Evaluation

Macro F0.5 over all holdout S1 entities, including singletons and true matches lost by blocking. "Cross-fitted" means the threshold is tuned on one half of the holdout and scored on the other, so the number is not inflated by tuning.

| Version | Holdout F0.5 | Cross-fitted F0.5 | Blocking recall | Avg candidates / S1 |
|---|---|---|---|---|
| Initial pipeline (200-S1 sample) | 0.488 | 0.488 | 98.0% | ~505,000 |
| Two-stage top-K blocking (10,000 S1) | 0.932 | 0.929 | 96.5% | 58 |
| + relative, blocking and address-number features (10,000 S1) | 0.942 | 0.941 | 96.5% | 58 |
| **Final model (100,000 S1; 20,000 held out)** | **0.942** | **0.942** | **96.7%** | **57.8** |

Everything runs on a 16 GB laptop with 6 worker processes:
- Training on 100,000 S1 records took 12 minutes.
- Prediction on the full test set took 3 h 41 min.

Outputs pass the official `validate_submission.py`, and `candidate_pairs.tsv` passes a streaming checker with the same rules (`src/check_candidates.py`). `README.md` and `run_submission.ps1` reproduce the submission end to end.

## External data / fair play

Only the supplied challenge data is used: no external business lookup, geocoding, registration database, API or internet enrichment. Transliteration uses the rule-based `indic-transliteration` and `indic-nlp-library` packages (MIT). The classifier is scikit-learn gradient boosting (BSD-3-Clause); no neural network or pretrained model is used.
