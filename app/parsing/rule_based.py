"""Deterministic, regex/NER-based extraction.

Used as the first stage of the hybrid pipeline (see app/ingestion.py). Rule-based extraction is
preferred for fields where regex is more *reliable* than an LLM (exact strings like emails/phones
never get paraphrased), and it's free and instant, which matters when this needs to be re-run
across thousands of resumes.
"""
import re
import time

import spacy

from app.parsing.schema import ContactInfo, ExtractionSource, SkillMention
from app.parsing.skills_taxonomy import ALL_SKILLS

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}")
_LINKEDIN_RE = re.compile(r"(https?://)?(www\.)?linkedin\.com/in/[A-Za-z0-9\-_%]+/?", re.IGNORECASE)
_GITHUB_RE = re.compile(r"(https?://)?(www\.)?github\.com/[A-Za-z0-9\-_]+/?", re.IGNORECASE)

_YEARS_EXPERIENCE_RE = re.compile(r"(\d+(?:\.\d+)?)\+?\s*years?\s+(?:of\s+)?experience", re.IGNORECASE)

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_contact_info(text: str) -> ContactInfo:
    email_match = _EMAIL_RE.search(text)
    phone_match = _PHONE_RE.search(text)
    linkedin_match = _LINKEDIN_RE.search(text)
    github_match = _GITHUB_RE.search(text)

    name = _guess_name(text)

    return ContactInfo(
        full_name=name,
        email=email_match.group(0) if email_match else None,
        phone=phone_match.group(0).strip() if phone_match else None,
        linkedin_url=linkedin_match.group(0) if linkedin_match else None,
        github_url=github_match.group(0) if github_match else None,
    )


def _guess_name(text: str) -> str | None:
    """Heuristic: the first PERSON entity spaCy finds in the top few lines of the resume."""
    header = "\n".join(text.strip().splitlines()[:5])
    doc = _get_nlp()(header)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None


def extract_skills(text: str) -> list[SkillMention]:
    found: list[SkillMention] = []
    lower_text = text.lower()
    for skill in ALL_SKILLS:
        pattern = r"(?<![\w+#.]){}(?![\w+#])".format(re.escape(skill.lower()))
        if re.search(pattern, lower_text):
            found.append(SkillMention(name=skill, source=ExtractionSource.RULE_BASED))
    return found


def extract_total_years_experience(text: str) -> float | None:
    match = _YEARS_EXPERIENCE_RE.search(text)
    if match:
        return float(match.group(1))
    return None


def run_rule_based_extraction(text: str) -> dict:
    """Returns a dict (not the full ParsedResume — education/experience narrative parsing is
    left to the LLM stage, since it requires understanding free-text structure rule-based
    extraction can't reliably do) plus a latency measurement for the evaluation harness."""
    start = time.perf_counter()
    result = {
        "contact": extract_contact_info(text),
        "skills": extract_skills(text),
        "total_years_experience": extract_total_years_experience(text),
    }
    latency_ms = (time.perf_counter() - start) * 1000
    result["latency_ms"] = latency_ms
    return result
