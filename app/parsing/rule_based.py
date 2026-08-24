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

_NAME_LINE_RE = re.compile(r"^[A-Za-z][A-Za-z.'\-]*(?:\s+[A-Za-z][A-Za-z.'\-]*){0,4}$")

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


def _looks_like_name(candidate: str) -> bool:
    """Rejects anything that couldn't plausibly be a person's name — URLs, emails, and
    contact-info lines in particular, which is what this exists to guard against."""
    candidate = candidate.strip()
    if not candidate or len(candidate) > 60:
        return False
    if any(ch in candidate for ch in ("@", "|", "/", "\\", "http", ".com")):
        return False
    if re.search(r"\d", candidate):
        return False
    return bool(_NAME_LINE_RE.match(candidate))


def _normalize_case(name: str) -> str:
    # Many resume headers put the name in ALL CAPS ("VARDHINEEDI UDAY KIRAN") — title-case it
    # for display. Leave mixed-case names untouched since capitalization may be intentional.
    return name.title() if name.isupper() else name


def _guess_name(text: str) -> str | None:
    """Tries the most common resume convention first — name is the very first line — since
    spaCy's statistical PERSON tagger is unreliable on ALL-CAPS text, which many resume headers
    use and which the first-line heuristic handles directly instead. Falls back to spaCy NER
    over the remaining header lines (with obvious contact-info lines filtered out first) for
    layouts where the name isn't on line one."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if lines and _looks_like_name(lines[0]):
        return _normalize_case(lines[0])

    header_lines = [
        line for line in lines[:5]
        if not any(marker in line for marker in ("@", "http", ".com", "|"))
    ]
    doc = _get_nlp()("\n".join(header_lines))
    for ent in doc.ents:
        if ent.label_ == "PERSON" and _looks_like_name(ent.text):
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
