"""Offline tests for deriving total years of experience from work-history dates — no DB or API
calls, following the same pattern as test_rule_based.py."""
from app.parsing.experience_years import estimate_years_from_experience
from app.parsing.schema import Experience


def _exp(start, end=None, is_current=False):
    return Experience(company="Acme", title="Engineer", start_date=start, end_date=end,
                       is_current=is_current, bullets=[])


def test_single_role_sums_correctly():
    years = estimate_years_from_experience([_exp("Jan 2020", "Jan 2022")])
    assert 1.9 < years < 2.1


def test_multiple_roles_sum_not_span():
    # Regression case: three short internships years apart should sum their real durations
    # (~1 year total), not span from the first start to the last end (~4+ years) — a candidate
    # with gaps between roles hasn't been "working" the whole span.
    roles = [
        _exp("May 2022", "Aug 2022"),
        _exp("Mar 2024", "Aug 2024"),
        _exp("May 2026", is_current=True),
    ]
    years = estimate_years_from_experience(roles)
    assert years is not None
    assert years < 2.0  # summed durations, not the ~4.3 year span


def test_current_role_uses_today():
    years = estimate_years_from_experience([_exp("Jan 2020", is_current=True)])
    assert years is not None
    assert years > 0


def test_present_marker_in_end_date_treated_as_current():
    years = estimate_years_from_experience([_exp("Jan 2020", "Present")])
    assert years is not None
    assert years > 0


def test_returns_none_when_no_parseable_dates():
    assert estimate_years_from_experience([_exp(None, None)]) is None
    assert estimate_years_from_experience([]) is None


def test_returns_none_for_contradictory_dates():
    # end before start — skip rather than guess
    assert estimate_years_from_experience([_exp("Jan 2023", "Jan 2020")]) is None
