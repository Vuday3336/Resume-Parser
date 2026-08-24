"""Orchestrates the hybrid extraction pipeline and persists results.

Pipeline: raw text -> rule-based extraction (fast, deterministic) -> LLM extraction (narrative
fields + skills the taxonomy missed) -> merge -> embed -> store. Every stage's latency/token cost
is recorded in `extraction_runs` so the evaluation report and analytics dashboard can show real
cost/latency tradeoffs instead of hand-waving them.
"""
from app.db import execute, execute_returning_id
from app.embeddings.embedder import embed_resume_text
from app.parsing.llm_extract import run_llm_extraction
from app.parsing.rule_based import run_rule_based_extraction
from app.parsing.schema import ExtractionSource, ParsedResume


def parse_resume(raw_text: str, use_llm: bool = True) -> ParsedResume:
    rule_result = run_rule_based_extraction(raw_text)

    if not use_llm:
        return ParsedResume(
            contact=rule_result["contact"],
            skills=rule_result["skills"],
            total_years_experience=rule_result["total_years_experience"],
            raw_text=raw_text,
            extraction_method=ExtractionSource.RULE_BASED,
            rule_based_latency_ms=rule_result["latency_ms"],
        )

    llm_result = run_llm_extraction(raw_text)

    # Merge: union skills by name, keep track of which method(s) found each one.
    skill_names_seen = {s.name.lower() for s in rule_result["skills"]}
    merged_skills = list(rule_result["skills"])
    for skill in llm_result["additional_skills"]:
        if skill.name.lower() not in skill_names_seen:
            merged_skills.append(skill)
            skill_names_seen.add(skill.name.lower())

    return ParsedResume(
        contact=rule_result["contact"],
        summary=llm_result["summary"],
        education=llm_result["education"],
        experience=llm_result["experience"],
        skills=merged_skills,
        total_years_experience=rule_result["total_years_experience"],
        raw_text=raw_text,
        extraction_method=ExtractionSource.MERGED,
        rule_based_latency_ms=rule_result["latency_ms"],
        llm_latency_ms=llm_result["latency_ms"],
        llm_tokens_used=llm_result["tokens_used"],
    )


def store_resume(parsed: ParsedResume, source_filename: str | None = None) -> int:
    candidate_id = execute_returning_id(
        """
        INSERT INTO candidates (full_name, email, phone, linkedin_url, github_url)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (email) DO UPDATE SET full_name = EXCLUDED.full_name
        RETURNING id
        """,
        (
            parsed.contact.full_name,
            parsed.contact.email,
            parsed.contact.phone,
            parsed.contact.linkedin_url,
            parsed.contact.github_url,
        ),
    )

    resume_id = execute_returning_id(
        """
        INSERT INTO resumes (candidate_id, source_filename, summary, raw_text,
                              total_years_experience, extraction_method)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            candidate_id,
            source_filename,
            parsed.summary,
            parsed.raw_text,
            parsed.total_years_experience,
            parsed.extraction_method.value,
        ),
    )

    for edu in parsed.education:
        execute(
            """INSERT INTO education (resume_id, institution, degree, field_of_study, start_year, end_year)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (resume_id, edu.institution, edu.degree, edu.field_of_study, edu.start_year, edu.end_year),
        )

    for exp in parsed.experience:
        execute(
            """INSERT INTO experience (resume_id, company, title, start_date, end_date, is_current, bullets)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (resume_id, exp.company, exp.title, exp.start_date, exp.end_date, exp.is_current, exp.bullets),
        )

    for skill in parsed.skills:
        execute(
            """INSERT INTO resume_skills (resume_id, skill_name, source)
               VALUES (%s, %s, %s) ON CONFLICT DO NOTHING""",
            (resume_id, skill.name, skill.source.value),
        )

    execute(
        """INSERT INTO extraction_runs (resume_id, extraction_method, rule_based_latency_ms,
                                         llm_latency_ms, llm_tokens_used)
           VALUES (%s, %s, %s, %s, %s)""",
        (
            resume_id,
            parsed.extraction_method.value,
            parsed.rule_based_latency_ms,
            parsed.llm_latency_ms,
            parsed.llm_tokens_used,
        ),
    )

    embedding_text = _build_embedding_text(parsed)
    vector = embed_resume_text(embedding_text)
    execute(
        """INSERT INTO resume_embeddings (resume_id, embedding) VALUES (%s, %s)
           ON CONFLICT (resume_id) DO UPDATE SET embedding = EXCLUDED.embedding""",
        (resume_id, vector),
    )

    return resume_id


def _build_embedding_text(parsed: ParsedResume) -> str:
    """What actually gets embedded matters a lot for match quality — we bias toward skills and
    experience titles/bullets rather than the raw text, since headers/formatting noise hurts
    cosine similarity."""
    parts = [s.name for s in parsed.skills]
    parts += [f"{e.title} at {e.company}" for e in parsed.experience]
    for e in parsed.experience:
        parts.extend(e.bullets)
    if parsed.summary:
        parts.append(parsed.summary)
    return "\n".join(parts) or parsed.raw_text


def ingest_resume_file(raw_text: str, source_filename: str | None = None, use_llm: bool = True) -> int:
    parsed = parse_resume(raw_text, use_llm=use_llm)
    return store_resume(parsed, source_filename=source_filename)
