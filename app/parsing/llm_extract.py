"""LLM-based structured extraction for the fields regex can't reliably handle: education,
work-experience narratives, and skills not covered by the static taxonomy.

Uses OpenAI's structured-output (JSON schema) mode so the response is guaranteed to parse into
our Pydantic models rather than relying on prompt-engineered JSON that occasionally breaks.
"""
import json
import time

from openai import OpenAI

from app.config import settings
from app.parsing.schema import Education, Experience, ExtractionSource, SkillMention

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": ["string", "null"]},
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "institution": {"type": "string"},
                    "degree": {"type": ["string", "null"]},
                    "field_of_study": {"type": ["string", "null"]},
                    "start_year": {"type": ["integer", "null"]},
                    "end_year": {"type": ["integer", "null"]},
                },
                "required": ["institution"],
                "additionalProperties": False,
            },
        },
        "experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "title": {"type": "string"},
                    "start_date": {"type": ["string", "null"]},
                    "end_date": {"type": ["string", "null"]},
                    "is_current": {"type": "boolean"},
                    "bullets": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["company", "title"],
                "additionalProperties": False,
            },
        },
        "additional_skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Skills, tools, or technologies mentioned that a static keyword list would likely miss.",
        },
    },
    "required": ["summary", "education", "experience", "additional_skills"],
    "additionalProperties": False,
}

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        settings.require_openai()
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def run_llm_extraction(resume_text: str) -> dict:
    """Returns {education, experience, summary, additional_skills, latency_ms, tokens_used}."""
    start = time.perf_counter()
    client = _get_client()
    response = client.chat.completions.create(
        model=settings.CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract structured data from resumes. Only use information explicitly "
                    "present in the text — never invent employers, dates, or schools."
                ),
            },
            {"role": "user", "content": resume_text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "resume_extraction", "schema": _RESPONSE_SCHEMA, "strict": True},
        },
    )
    latency_ms = (time.perf_counter() - start) * 1000
    payload = json.loads(response.choices[0].message.content)

    return {
        "summary": payload.get("summary"),
        "education": [Education(**e) for e in payload.get("education", [])],
        "experience": [Experience(**e) for e in payload.get("experience", [])],
        "additional_skills": [
            SkillMention(name=s, source=ExtractionSource.LLM) for s in payload.get("additional_skills", [])
        ],
        "latency_ms": latency_ms,
        "tokens_used": response.usage.total_tokens if response.usage else None,
    }
