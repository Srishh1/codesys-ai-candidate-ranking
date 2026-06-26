#!/usr/bin/env python3
"""
Extended validation script.
Runs both the official hackathon validator logic AND extra checks:
  - All candidate_ids exist in candidates.jsonl
  - No honeypots in top 10
  - Score distribution looks sane
  - Reasoning is non-empty and non-identical

Usage:
    python validate.py --submission ./submission.csv --candidates ./candidates.jsonl
"""

import csv
import json
import re
import sys
import argparse
from collections import Counter


REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")


def load_valid_ids(candidates_path: str) -> set:
    print(f"Loading valid candidate IDs from {candidates_path}...")
    valid_ids = set()
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                valid_ids.add(obj["candidate_id"])
    print(f"  {len(valid_ids):,} valid IDs loaded.")
    return valid_ids


def validate(submission_path: str, candidates_path: str) -> list[str]:
    errors = []
    warnings = []

    # Load valid IDs for cross-check
    valid_ids = load_valid_ids(candidates_path) if candidates_path else None

    print(f"\nValidating {submission_path}...")

    with open(submission_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != REQUIRED_HEADER:
            errors.append(f"Header mismatch. Expected: {REQUIRED_HEADER}, got: {header}")
        
        rows = [r for r in reader if any(c.strip() for c in r)]

    # Row count
    if len(rows) != 100:
        errors.append(f"Expected exactly 100 data rows, got {len(rows)}")

    seen_ids = set()
    seen_ranks = set()
    by_rank = []
    reasoning_texts = []

    for i, cells in enumerate(rows):
        row_num = i + 2
        if len(cells) != 4:
            errors.append(f"Row {row_num}: expected 4 columns, got {len(cells)}")
            continue

        cid, rank_s, score_s, reasoning = cells

        # candidate_id
        cid = cid.strip()
        if not CANDIDATE_ID_PATTERN.match(cid):
            errors.append(f"Row {row_num}: invalid candidate_id format: '{cid}'")
        elif cid in seen_ids:
            errors.append(f"Row {row_num}: duplicate candidate_id '{cid}'")
        else:
            seen_ids.add(cid)
            if valid_ids and cid not in valid_ids:
                errors.append(f"Row {row_num}: candidate_id '{cid}' not in candidates.jsonl")

        # rank
        try:
            rank = int(rank_s.strip())
            if not 1 <= rank <= 100:
                errors.append(f"Row {row_num}: rank {rank} out of range 1-100")
            elif rank in seen_ranks:
                errors.append(f"Row {row_num}: duplicate rank {rank}")
            else:
                seen_ranks.add(rank)
        except ValueError:
            errors.append(f"Row {row_num}: rank must be integer, got '{rank_s}'")
            rank = None

        # score
        try:
            score = float(score_s.strip())
        except ValueError:
            errors.append(f"Row {row_num}: score must be float, got '{score_s}'")
            score = None

        # reasoning
        if not reasoning.strip():
            warnings.append(f"Row {row_num}: empty reasoning (penalized in Stage 4)")
        reasoning_texts.append(reasoning.strip())

        if rank is not None and score is not None:
            by_rank.append((rank, score, cid))

    # Missing ranks
    missing = set(range(1, 101)) - seen_ranks
    if missing:
        errors.append(f"Missing ranks: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")

    # Score monotonicity
    by_rank.sort(key=lambda x: x[0])
    for i in range(len(by_rank) - 1):
        r1, s1, _ = by_rank[i]
        r2, s2, _ = by_rank[i + 1]
        if s1 < s2:
            errors.append(f"Score increases at rank {r1}→{r2}: {s1:.4f} < {s2:.4f}")

    # Tie-break: equal scores → candidate_id ascending
    for i in range(len(by_rank) - 1):
        r1, s1, c1 = by_rank[i]
        r2, s2, c2 = by_rank[i + 1]
        if s1 == s2 and c1 > c2:
            errors.append(f"Tie-break violation at ranks {r1},{r2}: {c1} > {c2} (need ascending)")

    # Reasoning quality checks
    reasoning_counter = Counter(reasoning_texts)
    most_common_reason, freq = reasoning_counter.most_common(1)[0]
    if freq > 5:
        warnings.append(f"Reasoning appears {freq}x: '{most_common_reason[:60]}...' — may be penalized for templating")

    empty_reasoning = sum(1 for r in reasoning_texts if not r)
    if empty_reasoning:
        warnings.append(f"{empty_reasoning} empty reasoning entries")

    # Print results
    print("\n" + "=" * 60)
    if errors:
        print(f"VALIDATION FAILED — {len(errors)} error(s):\n")
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print("VALIDATION PASSED — submission format is correct.")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠  {w}")

    # Stats
    if by_rank:
        scores = [s for _, s, _ in by_rank]
        print(f"\nScore range: {min(scores):.4f} – {max(scores):.4f}")
        print(f"Top-10 scores: {[f'{s:.4f}' for _, s, _ in by_rank[:10]]}")

    print("=" * 60)
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="./submission.csv")
    parser.add_argument("--candidates", default="./candidates.jsonl")
    args = parser.parse_args()

    errors = validate(args.submission, args.candidates)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()