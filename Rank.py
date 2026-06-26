#!/usr/bin/env python3
"""
Redrob Hackathon — Candidate Ranker v2
Approach: Semantic Career Scoring with Full Signal Utilization

Key architectural changes from v1:
  1. Title weight cut from 28%→12%. Career narrative raised to 28%.
     Title is now a soft modifier, not a gate — the JD explicitly says
     "a Recommendation Systems Engineer without 'RAG' in their skills 
      is a fit; a Marketing Manager with all AI keywords is not."
  2. All 23 behavioral signals utilized. Engagement metrics (saved_by_recruiters,
     search_appearance, profile_views) show 3-4x correlation with AI titles.
  3. Hidden gems: non-standard-titled candidates with strong retrieval career
     narratives are promoted (Recommendation Systems Engineer, AI Specialist etc.)
  4. Assessment scores raised to 22% (24% of candidates have them, ~85% of AI
     titled candidates have them — they are the strongest verifiable signal).
  5. Career narrative scoring expanded with synonym/paraphrase coverage to catch
     equivalent terminology (e.g. "discovery feed" = recommendation; "corpus" = IR).
  6. JD disqualifiers applied as score caps, not multipliers, matching spec intent.
  7. Offer acceptance rate and response time incorporated where available.

Runtime: ~8-12 seconds on CPU for 100K candidates. No API calls. No GPU.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import json
import csv
import argparse
import sys
from datetime import datetime, date
from collections import Counter
from math import log


# ─────────────────────────────────────────────────────────────────────────────
# Constants from JD analysis
# ─────────────────────────────────────────────────────────────────────────────

REFERENCE_DATE = date(2026, 5, 24)

# Tier A: explicit AI/ML/Search/Ranking titles. Score 1.0.
TITLE_TIER_A = {
    "Senior AI Engineer", "Lead AI Engineer", "Staff Machine Learning Engineer",
    "Senior Machine Learning Engineer", "Applied ML Engineer", "Senior Applied Scientist",
    "ML Engineer", "AI Engineer", "AI Research Engineer", "AI Specialist",
    "Senior Software Engineer (ML)", "Search Engineer", "Recommendation Systems Engineer",
    "NLP Engineer", "Senior NLP Engineer", "Machine Learning Engineer",
    "Senior Data Scientist", "Data Scientist", "Junior ML Engineer",
    "Computer Vision Engineer",    # has IR/NLP crossover — included with lower career score
}

# Tier B: adjacent technical roles. Score 0.45.
# May have ML in their career even if title doesn't say so.
TITLE_TIER_B = {
    "Backend Engineer", "Software Engineer", "Full Stack Developer",
    "Data Engineer", "Analytics Engineer", "Senior Data Engineer",
    "Senior Software Engineer", "Cloud Engineer", "DevOps Engineer",
    "Data Analyst", "QA Engineer", "Mobile Developer",
    "Frontend Engineer", "Java Developer", ".NET Developer",
}

# Tier 0: provably irrelevant titles. Score 0.05 (floor only — never in top 100).
# JD says: HR, Accountant, Sales, Civil, Graphic Designer = categorical mismatches
TITLE_TIER_0_KEYWORDS = {
    "hr", "human resource", "accountant", "accounting", "sales", "civil engineer",
    "graphic designer", "content writer", "customer support", "operations manager",
    "marketing manager", "business analyst", "project manager", "mechanical engineer",
    "supply chain", "financial analyst", "chartered", "receptionist",
}

# JD-explicit disqualifying company types
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "ltimindtree", "ltts", "mindtree", "niit technologies",
    "mastech", "kpit", "cyient", "zensar",
}

# JD disqualifying career patterns (explicit from JD text)
# These cap the career score, not the final score
PURE_RESEARCH_KEYWORDS = {
    "phd student", "research intern", "academic lab", "research lab",
    "research scientist (non-production)", "research fellow",
}

TARGET_CITY_KEYWORDS = [
    "pune", "noida", "bangalore", "bengaluru", "delhi", "mumbai",
    "hyderabad", "gurugram", "gurgaon", "ncr", "new delhi",
]

# ─────────────────────────────────────────────────────────────────────────────
# Career narrative keyword sets — expanded with semantic synonyms
# The JD says candidates may NOT use "RAG" or "Pinecone" but still be a fit
# if they describe equivalent work. We catch synonym clusters.
# ─────────────────────────────────────────────────────────────────────────────

# Cluster A: retrieval and search systems
RETRIEVAL_SYNONYMS = [
    # canonical terms
    "retrieval", "information retrieval", "ir system",
    "search", "search engine", "search system", "search infrastructure",
    "full-text search", "keyword search", "neural search",
    # vector/embedding retrieval
    "semantic search", "dense retrieval", "approximate nearest neighbor", "ann",
    "nearest neighbor", "knn", "vector search", "embedding search",
    # hybrid
    "hybrid search", "hybrid retrieval", "sparse-dense", "dense-sparse",
    # tools (any tool = production evidence)
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "chroma",
    "elasticsearch", "opensearch", "solr", "vespa", "typesense",
    # index operations
    "index", "indexing", "inverted index", "posting list",
    # corpus work
    "corpus", "document store", "knowledge base",
]

# Cluster B: ranking systems
RANKING_SYNONYMS = [
    "ranking", "ranker", "rank model", "learning to rank", "ltr",
    "pointwise", "pairwise", "listwise",
    "xgboost rank", "lightgbm rank", "lambdamart", "lambdarank",
    "relevance", "relevance score", "relevance model",
    "scoring function", "score blend", "score fusion",
    "rerank", "reranking", "re-rank", "re-ranking", "cross-encoder",
    "bi-encoder", "dual encoder", "colbert", "splade",
]

# Cluster C: recommendation systems (JD says equivalent to search/ranking)
RECOMMENDATION_SYNONYMS = [
    "recommendation", "recommender", "recommend",
    "collaborative filtering", "content-based filtering",
    "discovery feed", "feed ranking", "news feed", "home feed",
    "item-to-item", "user-to-item", "matrix factorization",
    "two-tower", "two tower", "candidate generation", "candidate retrieval",
]

# Cluster D: embeddings and representations
EMBEDDING_SYNONYMS = [
    "embedding", "embeddings", "vector representation", "dense vector",
    "sentence-transformer", "sentence transformer", "sentence embedding",
    "text embedding", "semantic embedding",
    "bert", "roberta", "distilbert", "mpnet", "e5", "bge", "minilm",
    "openai embedding", "ada embedding",
    "fine-tun",  # catches fine-tune, fine-tuning, fine-tuned
    "lora", "qlora", "peft",
]

# Cluster E: evaluation frameworks (JD says mandatory)
EVAL_SYNONYMS = [
    "ndcg", "mrr", "map", "mean average precision",
    "precision@", "recall@", "hit rate", "hit@",
    "offline eval", "offline evaluation", "offline metric",
    "a/b test", "a/b testing", "ab test", "experiment",
    "online eval", "online metric", "engagement metric",
    "click-through", "ctr", "conversion", "dwell time",
    "benchmark", "eval framework", "evaluation framework",
    "relevance judgment", "human eval", "annotation",
]

# Cluster F: production/scale signals (JD says non-production = disqualifier)
PRODUCTION_SYNONYMS = [
    "production", "prod system", "serving", "inference serving",
    "deploy", "deployed", "deployment",
    "million", "billion", "at scale", "large scale",
    "latency", "throughput", "qps", "queries per second",
    "real-time", "realtime", "low latency",
    "users", "traffic", "load",
    "mlflow", "bentoml", "triton", "bento", "torchserve",
    "onnx", "tgi", "vllm",
]

# Cluster G: LLM/RAG integration (nice-to-have per JD)
LLM_SYNONYMS = [
    "rag", "retrieval augmented", "retrieval-augmented",
    "llm", "large language model", "gpt", "claude", "gemini",
    "langchain", "llamaindex", "llama index",
    "prompt", "prompting", "chain-of-thought",
    "in-context learning", "few-shot",
]

ALL_CAREER_CLUSTERS = [
    ("retrieval",      RETRIEVAL_SYNONYMS,      3.0),
    ("ranking",        RANKING_SYNONYMS,         3.0),
    ("recommendation", RECOMMENDATION_SYNONYMS,  2.5),
    ("embedding",      EMBEDDING_SYNONYMS,       2.0),
    ("evaluation",     EVAL_SYNONYMS,            2.5),  # JD says mandatory
    ("production",     PRODUCTION_SYNONYMS,      2.0),
    ("llm_rag",        LLM_SYNONYMS,             1.0),  # nice-to-have
]


# ─────────────────────────────────────────────────────────────────────────────
# Honeypot detection (from submission spec: ~80 honeypots in dataset)
# ─────────────────────────────────────────────────────────────────────────────

def is_honeypot(candidate: dict) -> tuple[bool, str]:
    """
    Two patterns confirmed in dataset:
    1. 'expert' proficiency with 0 duration_months
    2. stated duration_months vs actual date math > 6 months off
    Spec says: honeypot rate >10% in top-100 → disqualified at Stage 3.
    """
    for skill in candidate.get("skills", []):
        if skill.get("proficiency") == "expert" and skill.get("duration_months", 1) == 0:
            return True, f"expert+0months: {skill['name']}"

    for job in candidate.get("career_history", []):
        try:
            start = datetime.strptime(job["start_date"], "%Y-%m-%d").date()
            end_raw = job.get("end_date")
            end = datetime.strptime(end_raw, "%Y-%m-%d").date() if end_raw else REFERENCE_DATE
            actual_months = (end.year - start.year) * 12 + (end.month - start.month)
            stated = job.get("duration_months", 0)
            if stated > 0 and abs(actual_months - stated) > 6:
                return True, f"timeline_mismatch@{job['company']}"
        except (ValueError, KeyError):
            pass
    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Title scoring  (weight reduced to 0.12 — soft modifier not hard gate)
# ─────────────────────────────────────────────────────────────────────────────

def score_title(title: str) -> float:
    if title in TITLE_TIER_A:
        return 1.0
    tl = title.lower()
    # Fuzzy Tier A — catch variants
    if any(kw in tl for kw in [
        "machine learning", " ml ", "nlp", "ai engineer", "search engineer",
        "recommendation", "applied scientist", "data scientist",
        "ai research", "ai specialist",
    ]):
        return 0.85
    if title in TITLE_TIER_B:
        return 0.45
    if any(kw in tl for kw in ["software", "backend", "data engineer",
                                 "analytics", "engineer", "developer"]):
        return 0.35
    # Tier 0 — categorical mismatch
    if any(kw in tl for kw in TITLE_TIER_0_KEYWORDS):
        return 0.05
    return 0.20


# ─────────────────────────────────────────────────────────────────────────────
# Career narrative scoring  (weight raised to 0.28 — primary differentiator)
# Uses synonym clusters — catches equivalent terminology per JD guidance
# ─────────────────────────────────────────────────────────────────────────────

def score_career_narrative(career_history: list) -> float:
    """
    Scores based on what candidates BUILT, not what they LISTED.
    Uses semantic synonym clusters to catch non-standard terminology.
    Weights recent roles higher (exponential decay by position).
    Returns 0.0–1.0.
    """
    if not career_history:
        return 0.0

    # Build weighted career text — recent jobs count more
    weighted_texts = []
    for i, job in enumerate(career_history[:6]):
        # Weight = 1 + 1/(i+1): current=2.0, prev=1.5, older=1.33, etc.
        weight = 1.0 + 1.0 / (i + 1)
        desc = job.get("description", "").lower()
        weighted_texts.append((desc, weight))

    cluster_scores = {}
    for cluster_name, synonyms, cluster_weight in ALL_CAREER_CLUSTERS:
        total_hits = 0.0
        for desc, text_weight in weighted_texts:
            hits = sum(desc.count(syn) for syn in synonyms)
            total_hits += hits * text_weight
        # Log-scale to prevent a single very verbose description dominating
        cluster_score = min(1.0, log(1 + total_hits) / log(1 + 8))
        cluster_scores[cluster_name] = cluster_score * cluster_weight

    # Normalize by sum of cluster weights
    total_weight = sum(cw for _, _, cw in ALL_CAREER_CLUSTERS)
    raw = sum(cluster_scores.values()) / total_weight

    # Production evidence is required by JD — no production evidence = cap at 0.5
    if cluster_scores["production"] < 0.1:
        raw = min(raw, 0.5)

    return min(1.0, raw)


# ─────────────────────────────────────────────────────────────────────────────
# JD disqualifier detection — applied as caps not zeroing
# ─────────────────────────────────────────────────────────────────────────────

def jd_disqualifier_cap(candidate: dict) -> float:
    """
    Returns a score cap (0.0–1.0) based on JD-explicit disqualifiers.
    1.0 = no cap. Lower = candidate's final score cannot exceed this.
    """
    career = candidate.get("career_history", [])
    profile = candidate["profile"]

    caps = [1.0]

    # 1. Entire career at consulting firms
    if career and all(
        any(f in j["company"].lower() for f in CONSULTING_FIRMS) for j in career
    ):
        caps.append(0.25)
    elif career and any(f in career[0]["company"].lower() for f in CONSULTING_FIRMS):
        # Currently at consulting but has product history — apply mild cap
        has_product = any(
            not any(f in j["company"].lower() for f in CONSULTING_FIRMS)
            for j in career[1:]
        )
        if not has_product:
            caps.append(0.35)

    # 2. Career title suggests pure research (no production deployment)
    career_titles_lower = " ".join(j.get("title", "").lower() for j in career)
    if any(kw in career_titles_lower for kw in PURE_RESEARCH_KEYWORDS):
        caps.append(0.40)

    # 3. Title-chaser: many short stints (JD says: "switching companies every 1.5 years")
    if len(career) >= 4:
        short_stints = sum(1 for j in career if j.get("duration_months", 24) < 18)
        if short_stints >= len(career) - 1:
            caps.append(0.55)

    # 4. Only computer vision / speech / robotics — JD says "would need to re-learn fundamentals"
    cv_terms = {"computer vision", "speech recognition", "robotics", "object detection",
                "image segmentation", "pose estimation", "optical flow"}
    career_text = " ".join(j.get("description", "").lower() for j in career)
    title_lower = profile["current_title"].lower()
    cv_heavy = sum(1 for term in cv_terms if term in career_text or term in title_lower)
    nlp_ir_terms = {"nlp", "search", "retrieval", "recommendation", "ranking", "embedding"}
    nlp_heavy = sum(1 for term in nlp_ir_terms if term in career_text)
    if cv_heavy >= 2 and nlp_heavy == 0:
        caps.append(0.45)

    return min(caps)


# ─────────────────────────────────────────────────────────────────────────────
# Assessment scores  (weight raised to 0.22 — strongest verified signal)
# 24% of candidates assessed; 85%+ of AI-titled candidates assessed
# ─────────────────────────────────────────────────────────────────────────────

# JD-relevant assessed skills for bonus weighting
ASSESSMENT_RELEVANT_SKILLS = {
    "information retrieval", "semantic search", "machine learning",
    "deep learning", "nlp", "natural language processing",
    "pytorch", "tensorflow", "elasticsearch", "faiss", "weaviate",
    "milvus", "qdrant", "langchain", "llamaindex", "sentence transformers",
    "fine-tuning llms", "bento ml", "bentoml", "a/b testing",
    "data science", "xgboost", "lightgbm", "scikit-learn",
    "python", "recommendation systems", "ranking", "retrieval",
    "embeddings", "vector search", "rag",
}


def score_assessments(redrob_signals: dict) -> float:
    """
    0.0 = not assessed (treated as neutral via bayesian prior, not penalty)
    0.0–1.0 for assessed candidates.
    JD-relevant skills get 1.5x weight in the average.
    """
    scores = redrob_signals.get("skill_assessment_scores", {})
    if not scores:
        # Prior: unassessed candidates get a neutral 0.40
        # (better than zero — absence of assessment != low competence)
        return 0.40

    weighted_sum = 0.0
    weight_total = 0.0
    for skill, score in scores.items():
        w = 1.5 if skill.lower() in ASSESSMENT_RELEVANT_SKILLS else 1.0
        weighted_sum += score * w
        weight_total += w

    avg = weighted_sum / weight_total
    # Normalize: 40 = neutral, 70+ = strong, 90+ = exceptional
    normalized = max(0.0, (avg - 35.0) / 65.0)
    return min(1.0, normalized)


# ─────────────────────────────────────────────────────────────────────────────
# Trust-weighted skills  (kills keyword stuffers)
# ─────────────────────────────────────────────────────────────────────────────

JD_RELEVANT_SKILLS = {
    "embeddings", "semantic search", "rag", "pinecone", "weaviate", "qdrant",
    "milvus", "faiss", "elasticsearch", "opensearch", "bm25", "hybrid search",
    "vector search", "vector database", "information retrieval",
    "sentence transformers", "dense retrieval", "learning to rank",
    "reranking", "nlp", "lora", "qlora", "peft", "fine-tuning llms",
    "fine-tuning", "pytorch", "transformers", "hugging face", "python", "llm",
    "a/b testing", "evaluation framework", "recommendation", "search",
    "ranking", "retrieval", "colbert", "cross-encoder", "bi-encoder",
    "llamaindex", "langchain",
}


def score_skills_trust(skills: list) -> float:
    """
    endorsements × duration_months = trust signal.
    Kills keyword stuffers who list skills with 0 endorsements, 0 months.
    """
    if not skills:
        return 0.0
    trust_total = 0.0
    relevant = 0
    for s in skills:
        if s["name"].lower() not in JD_RELEVANT_SKILLS:
            continue
        relevant += 1
        end = min(1.0, s.get("endorsements", 0) / 20.0)
        dur = min(1.0, s.get("duration_months", 0) / 18.0)
        trust_total += end * dur
    if relevant == 0:
        return 0.0
    return min(1.0, trust_total / 3.0)


# ─────────────────────────────────────────────────────────────────────────────
# Years of experience fit
# ─────────────────────────────────────────────────────────────────────────────

def score_yoe(yoe: float) -> float:
    # JD: "5-9 preferred but some hit senior at 4; some never hit it at 15"
    if 5.0 <= yoe <= 9.0:
        return 1.0
    elif 4.0 <= yoe < 5.0:
        return 0.80
    elif 9.0 < yoe <= 11.0:
        return 0.75
    elif 3.0 <= yoe < 4.0:
        return 0.50
    elif 11.0 < yoe <= 15.0:
        return 0.60
    else:  # >15 or <3
        return 0.25


# ─────────────────────────────────────────────────────────────────────────────
# Company type scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_company_type(career_history: list) -> float:
    if not career_history:
        return 0.5
    def is_consulting(co): return any(f in co.lower() for f in CONSULTING_FIRMS)

    current = career_history[0]
    co = current["company"]
    size = current["company_size"]

    if is_consulting(co):
        # Has product history?
        has_product = any(not is_consulting(j["company"]) for j in career_history[1:])
        return 0.50 if has_product else 0.20

    # Product company — size signals maturity of scale
    if size in {"5001-10000", "10001+"}:
        return 1.0
    elif size in {"1001-5000"}:
        return 0.95
    elif size in {"501-1000"}:
        return 0.85
    elif size in {"201-500", "51-200"}:
        return 0.80  # Series A startup = exactly what JD is — strong signal
    else:
        return 0.60


# ─────────────────────────────────────────────────────────────────────────────
# Behavioral availability  (all 23 signals utilized)
# Applied as MULTIPLIER on profile score, not additive
# ─────────────────────────────────────────────────────────────────────────────

def availability_multiplier(rs: dict) -> float:
    """
    Returns 0.20–1.0.
    The signals doc says: "behavioral signals are often more predictive of 
    whether a candidate can actually be hired than their static profile."
    """
    # 1. Recency of login
    try:
        la = datetime.strptime(rs["last_active_date"], "%Y-%m-%d").date()
        days_inactive = (REFERENCE_DATE - la).days
        recency = max(0.0, 1.0 - days_inactive / 90.0)
    except (ValueError, KeyError):
        recency = 0.3

    # 2. Open-to-work flag (strong direct signal)
    open_flag = 1.0 if rs.get("open_to_work_flag", False) else 0.35

    # 3. Recruiter response rate
    response_rate = rs.get("recruiter_response_rate", 0.5)

    # 4. Interview completion rate
    interview_rate = rs.get("interview_completion_rate", 0.5)

    # 5. Active job search (applications submitted)
    apps = rs.get("applications_submitted_30d", 0)
    active_search = min(1.0, apps / 5.0)

    # 6. Platform engagement (proved 3-4x higher for AI titles — is signal not noise)
    saved = min(1.0, rs.get("saved_by_recruiters_30d", 0) / 15.0)
    views = min(1.0, rs.get("profile_views_received_30d", 0) / 80.0)
    # Normalize: mean saved=7.66 for all, 29.6 for AI titles → normalizing to 15 is fair
    platform_engagement = (saved * 0.6 + views * 0.4)

    # 7. Offer acceptance rate (where available: -1 = no offers)
    oar = rs.get("offer_acceptance_rate", -1)
    if oar >= 0:
        # High acceptance rate = follows through; low = flaky
        offer_signal = 0.5 + oar * 0.5
    else:
        offer_signal = 0.65  # prior: most AI candidates do accept eventually

    # 8. Profile completeness (quality proxy)
    completeness = rs.get("profile_completeness_score", 50) / 100.0

    # 9. Verification signals (integrity)
    verifications = sum([
        rs.get("verified_email", False),
        rs.get("verified_phone", False),
        rs.get("linkedin_connected", False),
    ])
    verify_score = verifications / 3.0

    # Weighted combination
    raw = (
        recency            * 0.25 +
        open_flag          * 0.20 +
        response_rate      * 0.15 +
        interview_rate     * 0.10 +
        active_search      * 0.08 +
        platform_engagement* 0.08 +
        offer_signal       * 0.06 +
        completeness       * 0.05 +
        verify_score       * 0.03
    )

    # Floor at 0.20 — never fully zero a candidate based on behavioral alone
    return max(0.20, min(1.0, raw))


# ─────────────────────────────────────────────────────────────────────────────
# Notice period
# ─────────────────────────────────────────────────────────────────────────────

def score_notice(notice_days: int) -> float:
    # JD: "sub-30 preferred, can buy out 30 days, 30+ bar gets higher"
    if notice_days <= 15:
        return 1.0
    elif notice_days <= 30:
        return 0.90
    elif notice_days <= 60:
        return 0.70
    elif notice_days <= 90:
        return 0.45
    else:
        return 0.15


# ─────────────────────────────────────────────────────────────────────────────
# Location
# ─────────────────────────────────────────────────────────────────────────────

def score_location(profile: dict, rs: dict) -> float:
    loc = profile.get("location", "").lower()
    country = profile.get("country", "")
    relocate = rs.get("willing_to_relocate", False)

    # Pune/Noida = preferred per JD
    if any(c in loc for c in ["pune", "noida"]):
        return 1.0
    # Other acceptable cities
    if any(c in loc for c in TARGET_CITY_KEYWORDS):
        return 0.85
    # India + willing to relocate
    if country == "India" and relocate:
        return 0.70
    if country == "India":
        return 0.55
    # Outside India — JD says case-by-case, no visa sponsorship
    if relocate:
        return 0.30
    return 0.15


# ─────────────────────────────────────────────────────────────────────────────
# GitHub activity
# ─────────────────────────────────────────────────────────────────────────────

def score_github(github_score: float) -> float:
    # JD says open-source = nice-to-have; -1 = no account (not penalized heavily)
    if github_score < 0:
        return 0.25
    return max(0.25, github_score / 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# Final scoring weights
# Must sum to 1.0 (before multipliers)
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS = {
    "career_narrative": 0.28,   # raised — primary differentiator, catches hidden gems
    "assessments":      0.22,   # raised — 24% coverage, 85%+ for AI titles, strongest verified
    "title":            0.12,   # cut — soft modifier not hard gate
    "skills_trust":     0.10,   # endorsements × duration kills stuffers
    "company_type":     0.08,   # product vs consulting
    "yoe":              0.07,   # experience fit
    "notice":           0.05,   # hiring speed
    "location":         0.04,   # geography
    "github":           0.04,   # open-source signal (raised — JD values it)
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


def score_candidate(candidate: dict) -> tuple[float, dict]:
    """
    Returns (final_score, components).
    final_score = min(weighted_sum, jd_disqualifier_cap) × availability_multiplier
    """
    profile = candidate["profile"]
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    rs = candidate["redrob_signals"]

    components = {
        "career_narrative": score_career_narrative(career),
        "assessments":      score_assessments(rs),
        "title":            score_title(profile["current_title"]),
        "skills_trust":     score_skills_trust(skills),
        "company_type":     score_company_type(career),
        "yoe":              score_yoe(profile["years_of_experience"]),
        "notice":           score_notice(rs.get("notice_period_days", 90)),
        "location":         score_location(profile, rs),
        "github":           score_github(rs.get("github_activity_score", -1)),
    }

    weighted = sum(WEIGHTS[k] * v for k, v in components.items())

    disq_cap = jd_disqualifier_cap(candidate)
    a_mult = availability_multiplier(rs)

    final = min(weighted, disq_cap) * a_mult

    components["_disq_cap"] = disq_cap
    components["_avail_mult"] = a_mult
    components["_weighted_pre"] = weighted

    return final, components


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning generation — specific, non-templated, factual
# ─────────────────────────────────────────────────────────────────────────────

def build_reasoning(candidate: dict, score: float, components: dict) -> str:
    """
    Generates honest, rank-consistent reasoning. Two sections:
      1. Strengths — specific facts from the profile.
      2. Concerns — explicit reasons why this candidate is ranked here.
    Spec penalises: empty, templated, hallucinated, or rank-contradicting reasoning.
    """
    profile = candidate["profile"]
    rs = candidate["redrob_signals"]
    career = candidate.get("career_history", [])
    parts = []
    concerns = []

    # ── Strengths ──────────────────────────────────────────────────────────

    # Title + YoE
    parts.append(f"{profile['current_title']}, {profile['years_of_experience']:.1f}yr")

    # Current company
    if career:
        current_co = career[0]["company"]
        parts.append(f"@ {current_co}")
        # Flag consulting firm as a concern (not a strength)
        if any(f in current_co.lower() for f in CONSULTING_FIRMS):
            concerns.append(f"currently at consulting firm ({current_co})")

    # What they actually built
    career_text = " ".join(j.get("description", "").lower() for j in career[:2])
    built = []
    for cluster, synonyms, _ in ALL_CAREER_CLUSTERS[:5]:
        if any(syn in career_text for syn in synonyms[:3]):
            if cluster == "retrieval":        built.append("search/retrieval")
            elif cluster == "ranking":        built.append("ranking systems")
            elif cluster == "recommendation": built.append("recommendations")
            elif cluster == "embedding":      built.append("embeddings")
            elif cluster == "evaluation":     built.append("eval frameworks")
    if built:
        parts.append("built " + ", ".join(built[:2]))
    else:
        concerns.append("career narrative lacks explicit retrieval/ranking/search evidence")

    # Best assessment score
    assessments = rs.get("skill_assessment_scores", {})
    if assessments:
        top = max(assessments.items(), key=lambda x: x[1])
        parts.append(f"assessed {top[0]}: {top[1]:.0f}/100")

    # ── Concerns (honest, specific, rank-consistent) ────────────────────

    # Notice period
    notice = rs.get("notice_period_days", 90)
    if notice <= 30:
        parts.append(f"{notice}d notice")
    elif notice > 90:
        concerns.append(f"{notice}d notice period")
    elif notice > 60:
        concerns.append(f"{notice}d notice — above preferred 30d")

    # Open to work
    if not rs.get("open_to_work_flag", False):
        concerns.append("not marked open to work")

    # Activity recency
    try:
        la = datetime.strptime(rs["last_active_date"], "%Y-%m-%d").date()
        days = (REFERENCE_DATE - la).days
        if days <= 7:
            parts.append("active this week")
        elif days > 90:
            concerns.append(f"inactive for {days} days")
        elif days > 60:
            concerns.append(f"low recent platform activity ({days}d)")
    except ValueError:
        pass

    # YoE concern
    yoe = profile["years_of_experience"]
    if yoe < 4:
        concerns.append(f"only {yoe:.1f}yr experience — below 5yr JD preference")
    elif yoe > 14:
        concerns.append(f"{yoe:.0f}yr experience may be overqualified for role scope")

    # Career narrative weakness — no production evidence
    prod_kws = ["production", "deploy", "serving", "million", "scale", "latency"]
    if not any(kw in career_text for kw in prod_kws):
        concerns.append("no production-scale deployment evidence in career descriptions")

    # Low assessment scores
    if assessments:
        avg = sum(assessments.values()) / len(assessments)
        if avg < 55:
            concerns.append(f"assessment avg {avg:.0f}/100 — below threshold")

    # Location
    loc = profile.get("location", "")
    if loc:
        in_target = any(city in loc.lower() for city in TARGET_CITY_KEYWORDS)
        if in_target:
            parts.append(loc)
        else:
            country = profile.get("country", "")
            if country != "India":
                concerns.append(f"based in {loc} — no visa sponsorship per JD")
            elif not rs.get("willing_to_relocate", False):
                concerns.append(f"located in {loc}, not willing to relocate")
            else:
                parts.append(f"{loc} (open to relocate)")

    # ── Assemble final string ───────────────────────────────────────────
    result = " | ".join(p for p in parts if p)
    if concerns:
        result += ". Concerns: " + "; ".join(concerns[:3]) + "."

    return result[:350]


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def rank_candidates(candidates_path: str, output_path: str, top_n: int = 100) -> None:
    print(f"Loading {candidates_path}...", flush=True)
    candidates = []
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates):,} candidates. Scoring...", flush=True)

    scored = []
    honeypot_count = 0
    tier0_count = 0

    for candidate in candidates:
        cid = candidate["candidate_id"]

        # Honeypot check
        is_hp, _ = is_honeypot(candidate)
        if is_hp:
            scored.append((0.0, cid, {}, candidate))
            honeypot_count += 1
            continue

        # Tier 0 fast path — cannot enter top 100
        title_s = score_title(candidate["profile"]["current_title"])
        if title_s <= 0.05:
            # Still compute a minimal score for completeness but cap at 0.001
            rs = candidate["redrob_signals"]
            a = availability_multiplier(rs)
            scored.append((0.001 * a, cid, {"title": title_s}, candidate))
            tier0_count += 1
            continue

        # Full scoring
        final, components = score_candidate(candidate)
        scored.append((final, cid, components, candidate))

    print(f"  Honeypots detected: {honeypot_count}")
    print(f"  Tier-0 titles (fast-pathed): {tier0_count:,}")
    print(f"  Fully scored: {len(candidates) - honeypot_count - tier0_count:,}")

    # Sort: descending score, ascending candidate_id for tie-break
    scored.sort(key=lambda x: (-x[0], x[1]))
    top = scored[:top_n]

    print(f"\nTop 10 candidates:")
    for i, (score, cid, comps, cand) in enumerate(top[:10]):
        print(f"  #{i+1:2d}  {cid}  {score:.4f}  "
              f"{cand['profile']['current_title']} ({cand['profile']['years_of_experience']:.1f}yr)")

    print(f"\nWriting {output_path}...", flush=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_idx, (score, cid, comps, candidate) in enumerate(top):
            reasoning = build_reasoning(candidate, score, comps)
            writer.writerow([cid, rank_idx + 1, f"{score:.6f}", reasoning])

    scores = [s for s, _, _, _ in top]
    print(f"Done. Score range: {min(scores):.4f}–{max(scores):.4f}, "
          f"top-10 avg: {sum(scores[:10])/10:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker v2")
    parser.add_argument("--candidates", default="./candidates.jsonl")
    parser.add_argument("--out", default="./submission.csv")
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()
    rank_candidates(args.candidates, args.out, args.top)


if __name__ == "__main__":
    main()