"""End-to-end matching: embed JD -> shortlist via pgvector -> LLM rerank the shortlist ->
persist. This is what the dashboard's "JD Matching" tab calls."""
from app.config import settings
from app.db import execute, execute_returning_id, fetch_all
from app.matching.llm_score import score_candidate
from app.matching.vector_match import embed_job_description, shortlist_candidates
from app.parsing.schema import MatchResult


def store_job_description(title: str, company: str | None, raw_text: str,
                           required_skills: list[str], min_years_experience: float | None) -> int:
    jd_id = execute_returning_id(
        """INSERT INTO job_descriptions (title, company, raw_text, min_years_experience)
           VALUES (%s, %s, %s, %s) RETURNING id""",
        (title, company, raw_text, min_years_experience),
    )
    for skill in required_skills:
        execute(
            "INSERT INTO jd_required_skills (jd_id, skill_name) VALUES (%s, %s)",
            (jd_id, skill),
        )
    embedding = embed_job_description(raw_text)
    execute(
        "INSERT INTO jd_embeddings (jd_id, embedding) VALUES (%s, %s)",
        (jd_id, embedding),
    )
    return jd_id


def run_matching(jd_id: int, jd_text: str, shortlist_size: int | None = None) -> list[MatchResult]:
    shortlist_size = shortlist_size or settings.SHORTLIST_SIZE
    jd_embedding = embed_job_description(jd_text)

    candidates = shortlist_candidates(jd_embedding, limit=shortlist_size)
    results: list[MatchResult] = []

    for candidate in candidates:
        resume_id = candidate["resume_id"]
        summary_row = fetch_all(
            "SELECT summary, raw_text FROM resumes WHERE id = %s", (resume_id,)
        )[0]
        resume_text_for_llm = summary_row["summary"] or summary_row["raw_text"][:4000]

        match = score_candidate(
            resume_id=resume_id,
            jd_id=jd_id,
            resume_summary=resume_text_for_llm,
            jd_text=jd_text,
            embedding_similarity=candidate["similarity"],
        )
        _store_match_result(match)
        results.append(match)

    results.sort(key=lambda m: m.llm_fit_score or 0, reverse=True)
    return results


def _store_match_result(match: MatchResult) -> None:
    execute(
        """
        INSERT INTO match_results
            (resume_id, jd_id, embedding_similarity, llm_fit_score, matched_skills,
             missing_skills, explanation, risk_flags)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (resume_id, jd_id) DO UPDATE SET
            embedding_similarity = EXCLUDED.embedding_similarity,
            llm_fit_score = EXCLUDED.llm_fit_score,
            matched_skills = EXCLUDED.matched_skills,
            missing_skills = EXCLUDED.missing_skills,
            explanation = EXCLUDED.explanation,
            risk_flags = EXCLUDED.risk_flags
        """,
        (
            match.resume_id,
            match.jd_id,
            match.embedding_similarity,
            match.llm_fit_score,
            match.matched_skills,
            match.missing_skills,
            match.explanation,
            match.risk_flags,
        ),
    )
