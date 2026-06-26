#!/usr/bin/env python3
"""
Dataset exploration script.
Run this first to understand what you're working with.

Usage:
    python explore.py --candidates ./candidates.jsonl
"""

import json
import argparse
from collections import Counter
from datetime import datetime, date

REFERENCE_DATE = date(2026, 5, 24)

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "ltimindtree", "ltts", "mindtree", "zensar",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="./candidates.jsonl")
    args = parser.parse_args()

    print(f"Loading {args.candidates}...")
    candidates = []
    with open(args.candidates) as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates):,} candidates\n")

    # Title distribution
    titles = Counter(c["profile"]["current_title"] for c in candidates)
    print("=== TOP 30 TITLES ===")
    for t, n in titles.most_common(30):
        print(f"  {n:6d}  {t}")

    # YoE
    yoe_buckets = Counter(int(c["profile"]["years_of_experience"]) for c in candidates)
    print("\n=== YEARS OF EXPERIENCE ===")
    for k in sorted(yoe_buckets):
        bar = "█" * (yoe_buckets[k] // 200)
        print(f"  {k:2d} yr: {yoe_buckets[k]:5d}  {bar}")

    # Skill uniformity check
    all_skills = []
    for c in candidates:
        for s in c["skills"]:
            all_skills.append(s["name"])
    skill_counts = Counter(all_skills)
    vals = list(skill_counts.values())
    print(f"\n=== SKILL DISTRIBUTION (n={len(skill_counts)} unique skills) ===")
    print(f"  Min appearances: {min(vals)}")
    print(f"  Max appearances: {max(vals)}")
    print(f"  Mean: {sum(vals)/len(vals):.0f}")
    print(f"  ⚠  Skills appearing 10k-13k times: {sum(1 for v in vals if 10000 <= v <= 13000)}")
    print(f"  → Skills are uniformly distributed = raw skill presence is NOISE")

    # Honeypot scan
    honeypots = []
    for c in candidates:
        for s in c["skills"]:
            if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0:
                honeypots.append(c["candidate_id"])
                break
    print(f"\n=== HONEYPOTS ===")
    print(f"  Expert skill + 0 months: {len(honeypots)} candidates")

    # Consulting
    consulting_count = sum(
        1 for c in candidates
        if any(f in c["profile"]["current_company"].lower() for f in CONSULTING_FIRMS)
    )
    print(f"\n=== CONSULTING FIRMS (current company) ===")
    print(f"  {consulting_count:,} candidates currently at consulting firms")

    # Availability
    inactive = sum(
        1 for c in candidates
        if c["redrob_signals"]["last_active_date"] < "2026-01-01"
    )
    not_open = sum(
        1 for c in candidates
        if not c["redrob_signals"]["open_to_work_flag"]
    )
    assessed = sum(
        1 for c in candidates
        if c["redrob_signals"]["skill_assessment_scores"]
    )
    print(f"\n=== BEHAVIORAL SIGNALS ===")
    print(f"  Inactive since before Jan 2026: {inactive:,}")
    print(f"  NOT open to work: {not_open:,}")
    print(f"  Have assessment scores: {assessed:,}")

    # Keyword stuffers
    ai_skill_set = {
        "embeddings", "semantic search", "rag", "pinecone", "faiss",
        "information retrieval", "vector search", "nlp", "lora", "qlora",
    }
    ai_titles_lower = {"ml", "ai", "machine learning", "data scientist", "nlp", "search"}
    stuffers = []
    for c in candidates:
        title = c["profile"]["current_title"].lower()
        is_ai = any(x in title for x in ai_titles_lower)
        cskills = {s["name"].lower() for s in c["skills"]}
        matches = ai_skill_set & cskills
        if not is_ai and len(matches) >= 3:
            stuffers.append(c["candidate_id"])
    print(f"\n=== KEYWORD STUFFERS ===")
    print(f"  Non-AI titled with 3+ AI skills: {len(stuffers):,}")

    print("\n=== SUMMARY ===")
    print(f"  Total: {len(candidates):,}")
    print(f"  Honeypots: {len(honeypots)}")
    print(f"  Consulting (current): {consulting_count:,}")
    print(f"  Keyword stuffers: {len(stuffers):,}")
    print(f"  With assessments (gold signal): {assessed:,}")
    print(f"\nTrue target pool (AI/ML titles, 5-9yr, not consulting): estimate ~1,200-1,500")


if __name__ == "__main__":
    main()