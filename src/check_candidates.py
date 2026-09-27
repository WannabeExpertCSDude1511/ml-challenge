"""
Streaming checker for candidate_pairs.tsv (and its consistency with
matching_results.tsv), for files too large for the official validator's
in-memory candidate check.

Applies the same rules as utils/validate_submission.py:
exact header, one row per test S1 entity (no missing, extra or duplicate rows),
only S2-/S3- IDs, no duplicate IDs within a list, and (warning only) every
matched ID also appears among that S1 entity's candidates.

Memory: only the ~1.7M required S1 IDs are held; candidate lists are checked
line by line. The subset check reads both files in lockstep, which works for
the files written by predict.py (same S1 order). --check-ids additionally
loads all S2/S3 IDs (~1 GB).

Usage:
    python -m src.check_candidates --candidates output/candidate_pairs.tsv \
        --matching output/matching_results.tsv --test-dir dataset/test
"""
import argparse
import os
import sys

import numpy as np

CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
MAX_EXAMPLES = 5


def read_first_column(path):
    with open(path, encoding="utf-8") as f:
        next(f, None)
        return {line.split("\t", 1)[0].strip() for line in f if line.strip()}


def parse(line):
    s1, _, rest = line.rstrip("\n").partition("\t")
    return s1, (rest.split(",") if rest.strip() else [])


class Issues:
    def __init__(self):
        self.found = {}

    def add(self, kind, example):
        examples = self.found.setdefault(kind, [0, []])
        examples[0] += 1
        if len(examples[1]) < MAX_EXAMPLES:
            examples[1].append(example)

    def report(self, label):
        for kind, (count, examples) in self.found.items():
            print(f"  {label}: {kind}: {count:,} (e.g. {', '.join(examples)})")


def main():
    ap = argparse.ArgumentParser(description="Stream-check candidate_pairs.tsv.")
    ap.add_argument("--candidates", default="output/candidate_pairs.tsv")
    ap.add_argument("--matching", default="output/matching_results.tsv")
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--check-ids", action="store_true", help="Also check IDs exist in test S2/S3 (~1 GB)")
    args = ap.parse_args()

    required = read_first_column(os.path.join(args.test_dir, "test_source1.tsv"))
    valid = None
    if args.check_ids:
        valid = set()
        for name in ("test_source2.tsv", "test_source3.tsv"):
            valid |= read_first_column(os.path.join(args.test_dir, name))
    print(f"required S1 entities: {len(required):,}")

    errors, warnings = Issues(), Issues()
    seen, counts = set(), []
    lockstep = os.path.isfile(args.matching)

    with open(args.candidates, encoding="utf-8") as cand, \
         open(args.matching if lockstep else os.devnull, encoding="utf-8") as match:
        header = cand.readline().rstrip("\n").split("\t")
        if [h.strip().lower() for h in header] != CANDIDATE_HEADER:
            print(f"FAIL: candidate header {header} != {CANDIDATE_HEADER}")
            return 1
        if lockstep:
            m_header = match.readline().rstrip("\n").split("\t")
            if [h.strip().lower() for h in m_header] != MATCHING_HEADER:
                print(f"FAIL: matching header {m_header} != {MATCHING_HEADER}")
                return 1

        for line_num, line in enumerate(cand, start=2):
            if "\t" not in line:
                if line.strip():
                    errors.add("malformed row (no tab)", f"line {line_num}")
                continue
            s1, ids = parse(line)
            if s1 in seen:
                errors.add("duplicate S1 row", s1)
            seen.add(s1)
            if s1 not in required:
                errors.add("S1 not in test set", s1)
            counts.append(len(ids))
            id_set = set(ids)
            if len(id_set) != len(ids):
                errors.add("duplicate ID within a list", s1)
            for tid in id_set:
                if not tid.startswith(("S2-", "S3-")):
                    errors.add("ID without S2-/S3- prefix", tid)
                elif valid is not None and tid not in valid:
                    errors.add("ID not in test S2/S3", tid)

            if lockstep:
                m_line = match.readline()
                m_s1, m_ids = parse(m_line) if m_line else (None, [])
                if m_s1 != s1:
                    warnings.add("matching file not in the same S1 order; subset check stopped", f"line {line_num}")
                    lockstep = False
                elif set(m_ids) - id_set:
                    warnings.add("matched IDs not among candidates", s1)

    missing = required - seen
    for s1 in list(missing)[:MAX_EXAMPLES]:
        errors.add("required S1 missing", s1)
    if len(missing) > MAX_EXAMPLES:
        errors.found["required S1 missing"][0] = len(missing)

    counts = np.asarray(counts)
    if len(counts):
        print(f"candidate rows: {len(counts):,} ({int((counts == 0).sum()):,} empty); "
              f"candidates per S1: avg {counts.mean():.2f}, median {np.median(counts):.0f}, "
              f"p99 {np.percentile(counts, 99):.0f}, max {counts.max()}; total pairs {int(counts.sum()):,}")
    warnings.report("WARNING")
    if errors.found:
        print("FAIL:")
        errors.report("ERROR")
        return 1
    print("PASS: candidate_pairs.tsv follows the submission rules.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
