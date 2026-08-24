"""Fast, offline tests for the rule-based extractor — no DB or API calls."""
from app.parsing.rule_based import (
    extract_contact_info,
    extract_skills,
    extract_total_years_experience,
)

ALL_CAPS_HEADER = (
    "VARDHINEEDI UDAY KIRAN\n"
    "+91 9992922121 | vuday3336@gmail.com | linkedin.com/in/vardhineedi-uday-kiran | "
    "github.com/Vuday3336\nProfessional Summary"
)

SAMPLE_RESUME = """
Priya Nair
priya.nair.dev@example.com | +1 555-201-3344 | linkedin.com/in/priyanairdev | github.com/priyanair

Backend engineer with 4 years of experience.
Skilled in Python, PostgreSQL, Docker, and AWS. Built REST API services with FastAPI.
"""


def test_extract_email():
    contact = extract_contact_info(SAMPLE_RESUME)
    assert contact.email == "priya.nair.dev@example.com"


def test_extract_phone():
    contact = extract_contact_info(SAMPLE_RESUME)
    assert contact.phone is not None
    assert "555" in contact.phone


def test_extract_linkedin_and_github():
    contact = extract_contact_info(SAMPLE_RESUME)
    assert "linkedin.com/in/priyanairdev" in contact.linkedin_url
    assert "github.com/priyanair" in contact.github_url


def test_extract_skills_finds_known_taxonomy_terms():
    skills = {s.name for s in extract_skills(SAMPLE_RESUME)}
    assert {"Python", "PostgreSQL", "Docker", "AWS", "REST API"}.issubset(skills)


def test_extract_skills_does_not_false_positive_on_substrings():
    # "R" (the language) must not match inside unrelated words like "PostgreSQL" or "Order".
    text = "We use PostgreSQL and process customer orders efficiently."
    skills = {s.name for s in extract_skills(text)}
    assert "R" not in skills


def test_extract_total_years_experience():
    assert extract_total_years_experience(SAMPLE_RESUME) == 4.0


def test_extract_total_years_experience_returns_none_when_absent():
    assert extract_total_years_experience("No experience statement here.") is None


def test_extract_name_from_all_caps_header():
    # Regression test: spaCy's PERSON tagger is unreliable on ALL-CAPS text, which previously
    # caused the extractor to fall through and grab the contact-info line instead of the name.
    contact = extract_contact_info(ALL_CAPS_HEADER)
    assert contact.full_name == "Vardhineedi Uday Kiran"


def test_extract_name_does_not_grab_contact_info_line():
    contact = extract_contact_info(ALL_CAPS_HEADER)
    assert "linkedin.com" not in contact.full_name
    assert "@" not in contact.full_name
