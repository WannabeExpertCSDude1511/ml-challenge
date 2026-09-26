# Business Entity Resolution Pipeline

This document provides detailed instructions for migrating, setting up, and running the end-to-end entity resolution pipeline on a high-compute Linux server.

## 🌟 Overview
This pipeline solves the Amazon ML Challenge using a highly optimized, multi-threaded approach:
1. **Normalization:** Handles multi-script transliteration, leet-speak reversal, and Indian/US state normalization.
2. **Blocking Ensemble (8 Strategies):** Uses an aggressive 8-strategy inverted index (including TF-IDF and address-only rescue) to achieve **~98.5% recall**.
3. **Feature Engineering:** Computes 33 complex fuzzy/NLP features (Jaro-Winkler, N-Grams, IDF weighting).
4. **Classification:** Uses **LightGBM** with 5-fold cross-validation to automatically determine the optimal $F_{0.5}$ threshold.

---

## 🚀 1. Linux Server Setup

When moving this code from your Mac to a Linux server, **do not copy the `.venv` folder**. Virtual environments are OS-specific.

1. **Copy the project files** to your Linux server (excluding `.venv`).
2. **Ensure your datasets are in place:**
   - `dataset/train/` should contain `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`
   - `dataset/test/` should contain `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`
3. **Create a fresh virtual environment and install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## ⚙️ 2. The Execution Scripts

We have created three main execution scripts to handle the entire lifecycle. Because the full dataset involves billions of string comparisons, you should use a background task manager like `nohup`, `tmux`, or `screen` to prevent the process from dying if your SSH connection drops.

### A. The Training Run (`run_train.sh`)
This script trains the model on the **full 12.5M row training dataset** and evaluates its final $F_{0.5}$ score. 
* ⚠️ **Memory Warning:** Processing all 2.2M queries will generate hundreds of millions of candidate pairs. This will require **100GB+ of RAM**.
* **To run in the background:**
  ```bash
  nohup ./run_train.sh > server_train.log 2>&1 &
  ```
* **To monitor progress:** `tail -f server_train.log`
* **Outputs:** `model.joblib` (contains model weights, optimal threshold, and IDF dictionaries).

### B. The Final Prediction & Packaging (`run_final_test.sh`)
Once `model.joblib` is generated, use this script to predict matches for the unseen test dataset. 
* This script automatically triggers `prepare_submission.sh` at the end to generate the exact `.zip` file required by the competition portal.
* **To run in the background:**
  ```bash
  nohup ./run_final_test.sh > server_predict.log 2>&1 &
  ```
* **Outputs:** `output/matching_results.tsv`, `output/candidate_pairs.tsv`, and `my_team_submission.zip`.

### C. The Debug Run (`run_debug.sh`)
If you ever want to test a quick code change (like adding a new feature) without waiting hours, use the debug dataset (10K queries).
* **To run:** `./run_debug.sh`

---

## 🧠 3. Pipeline Modules (Under the Hood)

If you need to tweak the ML logic, here is where everything lives:

* **`src/normalize.py`**: The bedrock of the pipeline. It handles text cleaning, zero-padding removal, and caching. **Uses `@lru_cache` heavily to prevent redundant calculations.**
* **`src/blocking.py`**: Generates candidates. It builds inverted indices in memory. If you run out of RAM even on the Linux server, lower `TFIDF_TOP_K = 50` to something like `20` to reduce memory load.
* **`src/features.py`**: Contains `pair_features(a, b, idf_weights)`. If you want to add new similarity metrics, add them here.
* **`src/train.py`**: The training loop. It leverages `concurrent.futures.ThreadPoolExecutor` to multi-thread feature extraction across all available CPU cores.
* **`src/evaluate.py`**: Contains the strict Macro $F_{0.5}$ math logic used by the leaderboard.

---

## 📦 4. Submission Details

The competition has strict formatting rules for the `.zip` upload. 
The `prepare_submission.sh` script handles all of this automatically:
1. It copies `matching_results.tsv` and `candidate_pairs.tsv` to an `output/` folder.
2. It nests all your Python code under `code/business_entity_resolution/src/`.
3. It packages `README.md`, `requirements.txt`, and `Documentation_template.md`.
4. It zips it cleanly so the competition auto-grader doesn't crash.

Before submitting, make sure you fill out your methodology in `Documentation_template.md`!
