"""Derives total years of experience from work-history date ranges — a fallback for when the
resume never states a total explicitly (see rule_based.extract_total_years_experience, which
only catches literal phrases like "3 years of experience").

Sums each role's individual duration rather than taking the earliest-to-latest span. Span would
overstate experience for anyone with gaps between roles — very common for students and
early-career candidates with several short, non-contiguous internships — so summing is the more
honest number. The tradeoff is double-counting genuinely overlapping roles (e.g. a part-time job
held alongside a full-time one), which is rare enough on a resume to accept as a simplification.
"""
from datetime import datetime

from dateutil import parser as date_parser

from app.parsing.schema import Experience

_CURRENT_MARKERS = {"present", "current", "now", "ongoing", "till date", "to date"}


def _parse_date(text: str | None) -> datetime | None:
    if not text or not text.strip():
        return None
    if text.strip().lower() in _CURRENT_MARKERS:
        return datetime.now()
    try:
        # Missing day/month default to Jan 1 of the current year rather than "today", so a
        # bare "2022" doesn't silently inherit today's month/day.
        return date_parser.parse(text, fuzzy=True, default=datetime(datetime.now().year, 1, 1))
    except (ValueError, OverflowError):
        return None


def estimate_years_from_experience(experience: list[Experience]) -> float | None:
    now = datetime.now()
    total_days = 0.0
    found_any = False

    for exp in experience:
        start = _parse_date(exp.start_date)
        if start is None:
            continue
        end = now if exp.is_current else (_parse_date(exp.end_date) or now)
        if end < start:
            continue  # unparseable or contradictory date pair — skip rather than guess
        total_days += (end - start).days
        found_any = True

    if not found_any or total_days <= 0:
        return None
    return round(total_days / 365.25, 1)
