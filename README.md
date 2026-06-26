# Redrob Hackathon — Candidate Ranker

**Approach: Title-Gated Career Narrative Scoring**

No embeddings. No GPU. No API calls. Runs in ~5–10 seconds on any laptop.

---

## Setup

```bash
# Python 3.10+ required (uses built-in type hints)
python3 --version

# No external packages needed. Optionally:
pip install tqdm   # for progress bars (optional)
```

---

## File structure

```
redrob_ranker/
├── rank.py          ← MAIN script — scores all 100K, writes submission.csv
├── validate.py      ← Validates submission format + cross-checks candidate IDs
├── explain.py       ← Debug any candidate's score breakdown
├── explore.py       ← Dataset statistics and trap analysis
├── requirements.txt ← No external deps needed
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
python explore.py --candidates /path/to/candidates.jsonl

# Step 2: Run the ranker
python rank.py \
  --candidates /path/to/candidates.jsonl \
  --out ./submission.csv

# Step 3: Validate your submission
python validate.py \
  --submission ./submission.csv \
  --candidates /path/to/candidates.jsonl

# Step 4: Debug any candidate
python explain.py \
  --candidates /path/to/candidates.jsonl \
  --id CAND_0046525

# Step 5: Explain top 10
python explain.py \
  --candidates /path/to/candidates.jsonl \
  --top 10
```

---

## What the approach does (and why)

### The key insight: skills are noise

Every skill in the dataset appears ~12,000 times across 100,000 candidates.
That means an HR Manager and an ML Engineer have equal probability of listing
"RAG" as a skill. Semantic search on skill tags = random noise.

**True signal priority:**
1. **Job title** (decisive gate — 65K+ candidates with wrong titles are soft-excluded)
2. **Career description text** (what they actually built, not what they claimed)
3. **Assessment scores** (platform-verified, ~300 candidates have these — gold)
4. **Trust-weighted skills** (endorsements × duration kills keyword stuffers)
5. **Behavioral availability** (applied as multiplier, not additive)

### The traps caught

| Trap | Count | How caught |
|------|-------|------------|
| Honeypots (expert skill + 0 months) | 54 | Hard-zeroed in Step 1 |
| Keyword stuffers (non-AI with 3+ AI skills) | 2,772 | Trust-weighted skills |
| Consulting-only career | ~29,930 | Consulting multiplier |
| Inactive candidates | ~31,000 | Availability multiplier |
| Not open to work | ~64,000 | Availability multiplier |

### Scoring weights

| Signal | Weight | Notes |
|--------|--------|-------|
| title | 0.28 | Tier A (ML Eng, Search Eng...) = 1.0; wrong title = 0.0 |
| career_narrative | 0.22 | Keyword hits in actual job descriptions |
| assessments | 0.16 | Platform-verified scores / 100 |
| skills_trust | 0.10 | endorsements × duration_months |
| company_type | 0.08 | Product > consulting |
| yoe | 0.07 | 5–9 yr sweet spot |
| notice | 0.04 | ≤30d ideal |
| location | 0.03 | Pune/Noida/NCR preferred |
| github | 0.02 | Activity score |
| × consulting_mult | — | 0.2 if all consulting; 0.7 if current-only |
| × availability_mult | — | 0.25–1.0 based on activity + open_to_work |

---

## Tuning the weights

Edit `WEIGHTS` dict in `rank.py` (must sum to 1.0):

```python
WEIGHTS = {
    "title":            0.28,
    "career_narrative": 0.22,
    "assessments":      0.16,
    ...
}
```

Then re-run `rank.py` and `validate.py` to check the output.

Use `explain.py --top 20` to inspect what changed.

---

## Expected output

```
Loading candidates from candidates.jsonl...
Loaded 100,000 candidates. Scoring...
  Honeypots detected: 54
  Title-gated (score≈0): 65,641
  Fully scored: 34,305

Top 10 candidates:
  # 1  CAND_0046525  0.8843  Senior Machine Learning Engineer (6.1yr)
  # 2  CAND_0053591  0.8745  AI Engineer (5.3yr)
  # 3  CAND_0064326  0.8605  Search Engineer (7.6yr)
  ...

Done. Written 100 ranked candidates to submission.csv

Score stats — top-10 avg: 0.8461, top-100 avg: 0.7689
```

Runtime on a 2021 MacBook Pro: **~6 seconds**.

---

## Submission checklist

- [ ] `submission.csv` has exactly 100 rows (+ header)
- [ ] Columns: `candidate_id, rank, score, reasoning`
- [ ] Ranks 1–100, no duplicates
- [ ] Scores monotonically non-increasing (higher rank = higher score)
- [ ] All `candidate_id`s are valid (exist in `candidates.jsonl`)
- [ ] All reasoning entries are non-empty and non-identical
- [ ] `validate.py` returns "VALIDATION PASSED"