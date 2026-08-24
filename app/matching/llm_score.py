"""Stage 2 of matching: LLM-generated fit score + human-readable explanation, run only against
the stage-1 shortlist (see vector_match.shortlist_candidates). This two-stage design is the
project's main cost-control decision: embeddings are ~100x cheaper than a chat completion, so we
spend the expensive call only where it can change the outcome — ranking the top candidates
against each other — rather than scoring the entire candidate pool.
"""
import json

from openai import OpenAI

from app.config import settings
from app.parsing.schema import MatchResult

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "fit_score": {"type": "number", "description": "0-100 fit score"},
        "matched_skills": {"type": "array", "items": {"type": "string"}},
        "missing_skills": {"type": "array", "items": {"type": "string"}},
        "explanation": {"type": "string", "description": "2-3 sentence justification"},
        "risk_flags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "e.g. 'no direct leadership experience', 'employment gap 2022-2023'",
        },
    },
    "required": ["fit_score", "matched_skills", "missing_skills", "explanation", "risk_flags"],
    "additionalProperties": False,
}

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        settings.require_openai()
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def score_candidate(resume_id: int, jd_id: int, resume_summary: str, jd_text: str,
                     embedding_similarity: float) -> MatchResult:
    client = _get_client()
    response = client.chat.completions.create(
        model=settings.CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a technical recruiter assistant. Score how well a candidate fits a "
                    "job description based only on the provided text. Be specific about which "
                    "required skills are present or missing. Do not invent experience."
                ),
            },
            {
                "role": "user",
                "content": f"JOB DESCRIPTION:\n{jd_text}\n\nCANDIDATE PROFILE:\n{resume_summary}",
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "fit_score", "schema": _RESPONSE_SCHEMA, "strict": True},
        },
    )
    payload = json.loads(response.choices[0].message.content)

    return MatchResult(
        resume_id=resume_id,
        jd_id=jd_id,
        embedding_similarity=embedding_similarity,
        llm_fit_score=payload["fit_score"],
        matched_skills=payload["matched_skills"],
        missing_skills=payload["missing_skills"],
        explanation=payload["explanation"],
        risk_flags=payload["risk_flags"],
    )
