"""Real analytical SQL (window functions, aggregations) rather than pulling everything into
pandas — this is what the dashboard's Analytics tab renders."""
from app.db import fetch_all


def skill_demand() -> list[dict]:
    """How many candidates have each skill vs. how many open JDs require it — a supply/demand gap."""
    return fetch_all(
        """
        SELECT
            skill_name,
            COUNT(DISTINCT resume_id) AS candidate_count
        FROM resume_skills
        GROUP BY skill_name
        ORDER BY candidate_count DESC
        LIMIT 25
        """
    )


def jd_skill_gap(jd_id: int) -> list[dict]:
    """For one JD: each required skill and how many candidates in the pool actually have it."""
    return fetch_all(
        """
        SELECT
            req.skill_name,
            COUNT(DISTINCT rs.resume_id) AS candidates_with_skill
        FROM jd_required_skills req
        LEFT JOIN resume_skills rs ON rs.skill_name = req.skill_name
        WHERE req.jd_id = %s
        GROUP BY req.skill_name
        ORDER BY candidates_with_skill ASC
        """,
        (jd_id,),
    )


def score_distribution(jd_id: int) -> list[dict]:
    """Histogram buckets of LLM fit scores for a given JD's match results."""
    return fetch_all(
        """
        SELECT
            width_bucket(llm_fit_score, 0, 100, 10) AS bucket,
            COUNT(*) AS count,
            MIN(llm_fit_score) AS bucket_min,
            MAX(llm_fit_score) AS bucket_max
        FROM match_results
        WHERE jd_id = %s
        GROUP BY bucket
        ORDER BY bucket
        """,
        (jd_id,),
    )


def top_candidates(jd_id: int, limit: int = 10) -> list[dict]:
    """Ranked shortlist using RANK() so ties share a position, the way a recruiter would read it."""
    return fetch_all(
        """
        SELECT
            c.full_name,
            c.email,
            mr.llm_fit_score,
            mr.embedding_similarity,
            mr.explanation,
            RANK() OVER (ORDER BY mr.llm_fit_score DESC) AS rank
        FROM match_results mr
        JOIN resumes r ON r.id = mr.resume_id
        JOIN candidates c ON c.id = r.candidate_id
        WHERE mr.jd_id = %s
        ORDER BY rank
        LIMIT %s
        """,
        (jd_id, limit),
    )


def avg_experience_by_skill() -> list[dict]:
    return fetch_all(
        """
        SELECT
            rs.skill_name,
            ROUND(AVG(r.total_years_experience)::numeric, 1) AS avg_years_experience,
            COUNT(DISTINCT rs.resume_id) AS candidate_count
        FROM resume_skills rs
        JOIN resumes r ON r.id = rs.resume_id
        WHERE r.total_years_experience IS NOT NULL
        GROUP BY rs.skill_name
        HAVING COUNT(DISTINCT rs.resume_id) >= 2
        ORDER BY avg_years_experience DESC
        LIMIT 20
        """
    )


def pipeline_funnel() -> dict:
    """Parsed -> embedded -> scored counts, i.e. where candidates currently sit in the pipeline."""
    return fetch_all(
        """
        SELECT
            (SELECT COUNT(*) FROM resumes) AS parsed,
            (SELECT COUNT(*) FROM resume_embeddings) AS embedded,
            (SELECT COUNT(DISTINCT resume_id) FROM match_results) AS scored
        """
    )[0]


def extraction_cost_summary() -> dict:
    """Average latency/token cost split by extraction method — the evidence behind the
    hybrid-pipeline design decision, surfaced directly in the dashboard."""
    return fetch_all(
        """
        SELECT
            extraction_method,
            COUNT(*) AS runs,
            ROUND(AVG(rule_based_latency_ms)::numeric, 1) AS avg_rule_latency_ms,
            ROUND(AVG(llm_latency_ms)::numeric, 1) AS avg_llm_latency_ms,
            ROUND(AVG(llm_tokens_used)::numeric, 0) AS avg_tokens_used
        FROM extraction_runs
        GROUP BY extraction_method
        """
    )
