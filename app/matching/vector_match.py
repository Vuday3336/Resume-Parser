"""Stage 1 of matching: fast approximate-nearest-neighbor search over resume embeddings using
pgvector's cosine-distance operator (<=>). This is what lets the system scale — running the LLM
rerank (stage 2) against every candidate for every JD would be prohibitively slow and expensive
at more than a few dozen resumes.
"""
from app.db import fetch_all
from app.embeddings.embedder import embed_batch


def embed_job_description(jd_text: str) -> list[float]:
    return embed_batch([jd_text])[0].tolist()


def shortlist_candidates(jd_embedding: list[float], limit: int = 10) -> list[dict]:
    """Returns candidates ranked by cosine similarity, most similar first.
    `1 - cosine_distance` converts pgvector's distance metric into an intuitive 0-1 similarity."""
    rows = fetch_all(
        """
        SELECT
            r.id AS resume_id,
            c.full_name,
            c.email,
            r.total_years_experience,
            1 - (re.embedding <=> %s::vector) AS similarity
        FROM resume_embeddings re
        JOIN resumes r ON r.id = re.resume_id
        JOIN candidates c ON c.id = r.candidate_id
        ORDER BY re.embedding <=> %s::vector
        LIMIT %s
        """,
        (jd_embedding, jd_embedding, limit),
    )
    return rows


def get_resume_skills(resume_id: int) -> list[str]:
    rows = fetch_all(
        "SELECT DISTINCT skill_name FROM resume_skills WHERE resume_id = %s",
        (resume_id,),
    )
    return [r["skill_name"] for r in rows]
