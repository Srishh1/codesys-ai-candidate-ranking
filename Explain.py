#!/usr/bin/env python3
"""
Score explainer v2 — full breakdown for any candidate with v2 weights.

Usage:
    python explain.py --candidates ./candidates.jsonl --id CAND_0018499
    python explain.py --candidates ./candidates.jsonl --top 10
"""

import json
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import Rank as R


def explain_candidate(candidate: dict) -> None:
    cid = candidate["candidate_id"]
    profile = candidate["profile"]
    career = candidate.get("career_history", [])
    rs = candidate["redrob_signals"]

    print(f"\n{'='*70}")
    print(f"  {cid}  —  {profile['current_title']}")
    print(f"  {profile['summary'][:130]}...")
    print(f"{'='*70}")

    is_hp, hp_reason = R.is_honeypot(candidate)
    if is_hp:
        print(f"  ⛔  HONEYPOT: {hp_reason}")
        return

    final, comps = R.score_candidate(candidate)

    print(f"\n  SCORE BREAKDOWN (v2 weights):")
    print(f"  {'Signal':<22} {'Score':>6}  {'Weight':>6}  {'Contrib':>8}  {'Bar'}")
    print(f"  {'-'*58}")
    for k, v in comps.items():
        if k.startswith("_"):
            continue
        w = R.WEIGHTS.get(k, 0)
        bar = "█" * int(v * 12)
        print(f"  {k:<22} {v:>6.3f}  {w:>6.2f}  {v*w:>8.4f}  {bar}")
    print(f"  {'-'*58}")
    print(f"  {'weighted sum':<22} {comps['_weighted_pre']:>6.3f}")
    print(f"  {'disqualifier cap':<22} {comps['_disq_cap']:>6.3f}")
    print(f"  {'× availability mult':<22} {comps['_avail_mult']:>6.3f}")
    print(f"  {'FINAL SCORE':<22} {final:>6.4f}")

    print(f"\n  PROFILE:")
    print(f"  Title      : {profile['current_title']}  (tier score: {R.score_title(profile['current_title']):.2f})")
    print(f"  YoE        : {profile['years_of_experience']:.1f} yr")
    print(f"  Location   : {profile['location']}, {profile['country']}")
    print(f"  Assessments: {rs.get('skill_assessment_scores', {})}")

    print(f"\n  BEHAVIORAL SIGNALS:")
    print(f"  open_to_work={rs.get('open_to_work_flag')}, last_active={rs.get('last_active_date')}")
    print(f"  notice={rs.get('notice_period_days')}d, response_rate={rs.get('recruiter_response_rate',0):.0%}")
    print(f"  saved_30d={rs.get('saved_by_recruiters_30d')}, views_30d={rs.get('profile_views_received_30d')}")
    print(f"  apps_30d={rs.get('applications_submitted_30d')}, offer_acceptance={rs.get('offer_acceptance_rate')}")
    print(f"  interview_completion={rs.get('interview_completion_rate',0):.0%}, github={rs.get('github_activity_score')}")

    print(f"\n  CAREER (top 3):")
    for i, job in enumerate(career[:3]):
        print(f"  [{i}] {job['title']} @ {job['company']} ({job['company_size']}) — {job['duration_months']}mo")
        print(f"      {job['description'][:160]}")

    print(f"\n  REASONING:")
    print(f"  {R.build_reasoning(candidate, final, comps)}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="./candidates.jsonl")
    parser.add_argument("--id", help="Explain specific candidate ID")
    parser.add_argument("--top", type=int, help="Score all candidates, explain top N")
    args = parser.parse_args()

    print(f"Loading {args.candidates}...")
    candidates = []
    by_id = {}
    with open(args.candidates) as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                candidates.append(c)
                by_id[c["candidate_id"]] = c

    if args.id:
        c = by_id.get(args.id)
        if not c:
            print(f"Not found: {args.id}")
            sys.exit(1)
        explain_candidate(c)

    elif args.top:
        print(f"Scoring all {len(candidates):,} candidates...")
        scored = []
        for c in candidates:
            is_hp, _ = R.is_honeypot(c)
            if is_hp:
                continue
            final, comps = R.score_candidate(c)
            scored.append((final, c))
        scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
        for _, c in scored[:args.top]:
            explain_candidate(c)
    else:
        print("Use --id CAND_XXXXXXX or --top N")
        parser.print_help()


if __name__ == "__main__":
    main()