# Redrob Hackathon — Candidate Ranker

**Approach: Semantic Career Scoring with Full Signal Utilization**

No embeddings. No GPU. No API calls. Runs in ~9 seconds on any laptop.

---

## Setup

```bash
# Python 3.10+ required (uses built-in type hints)
python3 --version

# No external packages needed for rank.py / validate.py / explain.py / explore.py
# XLSX export requires:
pip install openpyxl
```

---

## File structure

```
redrob_ranker/
├── rank.py          ← MAIN script — scores all 100K, writes submission.csv
├── validate.py      ← Validates submission format + cross-checks candidate IDs
├── explain.py       ← Debug any candidate's full score breakdown
├── explore.py       ← Dataset statistics and trap analysis
├── requirements.txt
└── README.md
```

Your dataset files (from the challenge):
```
<your_data_dir>/
├── candidates.jsonl
├── job_description.docx
├── redrob_signals_doc.docx
├── candidate_schema.json
├── sample_submission.csv
└── submission_metadata_template.yaml
```

---

## Quick start

```bash
# Step 1: Understand the dataset
python3 explore.py --candidates /path/to/candidates.jsonl

# Step 2: Run the ranker
python3 rank.py \
  --candidates /path/to/candidates.jsonl \
  --out ./submission.csv

# Step 3: Validate your submission
python3 validate.py \
  --submission ./submission.csv \
  --candidates /path/to/candidates.jsonl

# Step 4: Debug any candidate
python3 explain.py \
  --candidates /path/to/candidates.jsonl \
  --id CAND_0046525

# Step 5: Explain top 10
python3 explain.py \
  --candidates /path/to/candidates.jsonl \
  --top 10

# Step 6: Convert to XLSX for final submission
python3 -c "
import csv, openpyxl
rows = list(csv.DictReader(open('submission.csv')))
wb = openpyxl.Workbook(); ws = wb.active
ws.append(['candidate_id','rank','score','reasoning'])
for r in rows:
    ws.append([r['candidate_id'], int(r['rank']), float(r['score']), r['reasoning']])
wb.save('submission.xlsx')
"
```

---

## What the approach does (and why)

### The key insight: skills are noise

Every skill in the dataset appears ~7,000–12,000 times across 100,000 candidates
(133 unique skills total, mean ~7,220 appearances each). That means an HR Manager
and an ML Engineer have roughly equal probability of listing "RAG" as a skill.
Raw skill presence = noise, not signal.

**True signal priority:**
1. **Career description text** (28% weight) — what they actually built, via 7 synonym clusters covering retrieval, ranking, recommendation, embeddings, evaluation, production, and LLM/RAG terminology
2. **Assessment scores** (22% weight) — platform-verified, ~24% of all candidates have these; JD-relevant skills get 1.5× weight in the average
3. **Title** (12% weight, soft modifier not a gate) — Tier A/B/0 system
4. **Trust-weighted skills** (10% weight) — endorsements × duration_months kills keyword stuffers
5. **Company type, YoE, notice, location, GitHub** (remaining 28%)
6. **JD disqualifier cap** — applied via `min()`, not multiplication
7. **Availability multiplier** — 9 behavioral signals, applied last, range 0.20–1.0

### The traps caught (confirmed on local dataset, 100K candidates)

| Trap | Count | How caught |
|------|-------|------------|
| Honeypots (expert skill + 0 months, or timeline mismatch) | 21–54 depending on detector pass | Hard-zeroed in Step 1 |
| Keyword stuffers (non-AI title with 3+ AI skills) | 5,508 | Trust-weighted skills neutralize them |
| Consulting-only career (current employer) | 31,155 | `jd_disqualifier_cap()` → 0.25–0.35 |
| Inactive since before Jan 2026 | 31,472 | Availability multiplier |
| Not open to work | 64,661 | Availability multiplier |
| Have assessment scores (verified signal) | 24,244 (24%) | Weighted at 22%, 1.5× for JD-relevant skills |

### Scoring weights (must sum to 1.0)

| Signal | Weight | Notes |
|--------|--------|-------|
| career_narrative | 0.28 | 7 synonym clusters, log-scaled, recency-weighted; capped at 0.5 if no production evidence |
| assessments | 0.22 | Unassessed candidates get neutral prior 0.40, not 0 |
| title | 0.12 | Tier A = 1.0, Tier B = 0.45, Tier 0 = 0.05 (soft modifier, not a gate) |
| skills_trust | 0.10 | min(endorsements/20,1) × min(duration_months/18,1) |
| company_type | 0.08 | Product company 0.80–1.0; consulting 0.20–0.50 |
| yoe | 0.07 | 5–9yr = 1.0; decays outside that band |
| notice | 0.05 | ≤15d = 1.0; >90d = 0.15 |
| location | 0.04 | Pune/Noida = 1.0; other target cities = 0.85 |
| github | 0.04 | activity_score/100; no account = 0.25 (not penalized to 0) |

**Applied after the weighted sum:**
- `jd_disqualifier_cap()` — caps (not multiplies) the score: all-consulting career → 0.25, current-consulting-no-product-history → 0.35, pure-research signal → 0.40, CV/robotics-only with no NLP → 0.45, title-chaser pattern → 0.55
- `availability_multiplier()` — multiplies the capped score by 0.20–1.0, combining last_active recency, open_to_work flag, recruiter response rate, interview completion rate, applications submitted, saved_by_recruiters, profile views, offer acceptance rate, profile completeness, and verification signals

`final_score = min(weighted_sum, disqualifier_cap) × availability_multiplier`

---

## Tuning the weights

Edit `WEIGHTS` dict in `rank.py` (must sum to 1.0):

```python
WEIGHTS = {
    "career_narrative": 0.28,
    "assessments":      0.22,
    "title":            0.12,
    ...
}
```

Then re-run `rank.py` and `validate.py` to check the output. Use
`explain.py --top 20` to inspect what changed before committing a weight change
to a real submission — you only get a limited number of submission attempts.

---

## Expected output (local run)

```
Loading candidates from candidates.jsonl...
Loaded 100,000 candidates. Scoring...
  Honeypots detected: 54
  Tier-0 titles (fast-pathed): 65,641
  Fully scored: 34,305

Top 10 candidates:
  # 1  CAND_0018499  0.7900  Senior Machine Learning Engineer (7.2yr)
  # 2  CAND_0046525  0.7848  Senior Machine Learning Engineer (6.1yr)
  # 3  CAND_0077337  0.7822  Staff Machine Learning Engineer (7.0yr)
  # 4  CAND_0011687  0.7395  Senior NLP Engineer (7.8yr)
  # 5  CAND_0053591  0.7343  AI Engineer (5.3yr)
  ...

Done. Score range: 0.7182–0.7900
```

Runtime on a MacBook Air M2: **~9 seconds** for the full 100K candidates.

---

## Reasoning format

Every row's `reasoning` column has two parts:

**Strengths** — title, YoE, current company, what they built (drawn from the same
7 synonym clusters used for scoring), and their top assessment score if present.

**Concerns** — explicit, specific reasons for the rank: notice period above
preference, not marked open to work, inactive N days, currently at a named
consulting firm, no production-scale evidence in career text, below-threshold
assessment average, or location/visa mismatch.

This means even rank #100 has a reasoning that justifies its position rather
than just restating strengths — required for the rank-consistency check at
Stage 4 evaluation.

---

## Submission checklist

- [ ] `submission.csv` (and `submission.xlsx`) has exactly 100 rows (+ header)
- [ ] Columns: `candidate_id, rank, score, reasoning`
- [ ] Ranks 1–100, no duplicates
- [ ] Scores monotonically non-increasing (higher rank = higher or equal score)
- [ ] Ties broken by candidate_id ascending
- [ ] All `candidate_id`s are valid (exist in `candidates.jsonl`)
- [ ] All reasoning entries are non-empty and non-identical
- [ ] `validate.py` returns "VALIDATION PASSED"
- [ ] Zero honeypots in top 100 (run `explain.py --top 100` and spot-check, or
      grep the reasoning column — honeypot candidates are excluded before scoring)
- [ ] XLSX file generated for final form submission (form requires XLSX, spec
      sample is CSV — provide both)