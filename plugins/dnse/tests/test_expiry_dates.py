"""#118 — the expiry arithmetic the GTD clamp falls back on (``pynecore_dnse.expiry``).

Pure functions, no clock and no venue: every case passes its own ``today``, so these
verdicts cannot drift with the wall clock (a real-``now`` test would silently re-pin
itself after the next roll).

The month table is CONFIRMED from ``plugins/dnse/testing/dnse_fixtures.json``
(2026-08-04 ``/market/instruments``): ``41I1G8000``=VN30F1M in August, ``41I1G9000``
=VN30F2M (September), ``41I1GC000``=VN30F1Q (December), ``41I1H3000``=VN30F2Q (March);
plus the 2026-09-14 live read where ``41I1G9000`` had become the September front month.
"""
from datetime import date

import pytest

from pynecore_dnse import expiry


# --- dated contract code -> month -------------------------------------------

@pytest.mark.parametrize("code, month", [
    ("41I1G8000", 8),    # VN30F1M on 2026-08-04 (fixture)
    ("41I1G9000", 9),    # VN30F2M then, VN30F1M on 2026-09-14 (live)
    ("41I1GA000", 10),   # 'A' is October — the letters start where the digits run out
    ("41I1GC000", 12),   # VN30F1Q (fixture)
    ("41I1H3000", 3),    # VN30F2Q (fixture) — a different cycle letter, same rule
])
def __test_contract_month_reads_the_krx_month_char__(code, month):
    assert expiry.contract_month(code) == month


@pytest.mark.parametrize("not_a_contract", [
    "VN30F1M",    # the ALIAS — unresolved, must not be mistaken for a dated code
    "HPG",        # a stock
    "",
    None,
    "41I1GD000",  # 'D' is not a month char
])
def __test_contract_month_is_none_for_anything_not_a_dated_contract__(not_a_contract):
    assert expiry.contract_month(not_a_contract) is None, \
        "a non-dated code must answer None, never a month, or the clamp invents an expiry"


# --- 3rd Thursday ------------------------------------------------------------

@pytest.mark.parametrize("year, month, expected", [
    (2026, 8, date(2026, 8, 20)),    # the venue-served finalTradeDate of 2026-08-14
    (2026, 9, date(2026, 9, 17)),    # the live #118 contract's real final trade date
    (2026, 10, date(2026, 10, 15)),
    (2026, 1, date(2026, 1, 15)),    # month starting on a Thursday
    (2026, 2, date(2026, 2, 19)),    # month starting on a Sunday (the Tet case below)
])
def __test_third_thursday__(year, month, expected):
    got = expiry.third_thursday(year, month)
    assert got == expected and got.weekday() == 3


# --- venue date parsing (was a bare ``except: pass``) ------------------------

@pytest.mark.parametrize("raw, expected", [
    ("2026-08-20", date(2026, 8, 20)),                 # bare ISO
    ("2026-08-20T00:00:00Z", date(2026, 8, 20)),       # RFC3339 — the live secdef shape
    (20260416, date(2026, 4, 16)),                     # the compact form in the API docs
    ("20260416", date(2026, 4, 16)),
])
def __test_parse_venue_date_accepts_every_known_shape__(raw, expected):
    assert expiry.parse_venue_date(raw) == expected


@pytest.mark.parametrize("absent", [None, "", "   ", "null", "None"])
def __test_parse_venue_date_reports_absence_as_none__(absent):
    """An ABSENT expiry is a fact (the field is intermittent), not a failure."""
    assert expiry.parse_venue_date(absent) is None


@pytest.mark.parametrize("unknown", ["17/09/2026", "Sep 17 2026", "2026-13-99",
                                     "202604", "1755648000"])
def __test_parse_venue_date_raises_loudly_on_an_unknown_format__(unknown):
    """A NEW venue format must fail LOUD here — the old ``except: pass`` swallowed it
    into a silent fail-open to ``now + 7d``, which is the #118 outage."""
    with pytest.raises(ValueError):
        expiry.parse_venue_date(unknown)


# --- working-day walks -------------------------------------------------------

@pytest.mark.parametrize("day, expected", [
    (date(2026, 9, 19), date(2026, 9, 18)),   # Saturday -> Friday
    (date(2026, 9, 20), date(2026, 9, 18)),   # Sunday   -> Friday
    (date(2026, 9, 18), date(2026, 9, 18)),   # already open
    (date(2026, 9, 2), date(2026, 9, 1)),     # National Day (fixed) -> the day before
])
def __test_last_open_day_on_or_before_walks_back_off_weekends_and_holidays__(day, expected):
    assert expiry.last_open_day_on_or_before(day) == expected


@pytest.mark.parametrize("day, expected", [
    (date(2026, 9, 14), date(2026, 9, 15)),   # Monday -> Tuesday
    (date(2026, 9, 18), date(2026, 9, 21)),   # Friday -> Monday
    (date(2026, 9, 1), date(2026, 9, 3)),     # skips National Day (09-02)
])
def __test_next_open_day_after_is_the_gtd_floor__(day, expected):
    assert expiry.next_open_day_after(day) == expected


# --- the computed final trade date ------------------------------------------

def __test_computed_final_trade_date_for_the_live_118_contract__():
    """2026-09-14, front month ``41I1G9000`` -> Thu 2026-09-17 (the real expiry)."""
    assert expiry.computed_final_trade_date(
        "41I1G9000", today=date(2026, 9, 14)) == date(2026, 9, 17)


def __test_computed_final_trade_date_takes_the_nearest_FUTURE_occurrence__():
    """The cycle letter is never decoded: an already-passed month rolls to next year.

    Reading ``41I1G9000`` (September) on 2026-10-01 must not answer a date in the past —
    it answers September 2027, which the caller then loses to the plain +7d window.
    """
    assert expiry.computed_final_trade_date(
        "41I1G9000", today=date(2026, 10, 1)) == date(2027, 9, 16)


def __test_computed_final_trade_date_includes_its_own_expiry_day__():
    """On the final day the contract still trades, so ``today`` itself is a valid answer."""
    assert expiry.computed_final_trade_date(
        "41I1G9000", today=date(2026, 9, 17)) == date(2026, 9, 17)


def __test_february_2026_walks_back_out_of_the_tet_closure__():
    """THE documented holiday case (#118): Feb 2026's 3rd Thursday is inside Tet.

    Tet 2026 ran 2026-02-14..2026-02-22, swallowing Thu 2026-02-19. A naive
    3rd-Thursday rule would hand the venue a date the exchange is shut on; the walk-back
    crosses the whole closure plus the preceding weekend and lands on Fri 2026-02-13.
    This is the case that makes the hand-entered ``DATED_HOLIDAYS`` table load-bearing —
    lunar holidays are not derivable from month/day.
    """
    assert expiry.third_thursday(2026, 2) == date(2026, 2, 19), "inside the Tet closure"
    assert expiry.computed_final_trade_date(
        "41I1G2000", today=date(2026, 1, 5)) == date(2026, 2, 13)


def __test_computed_final_trade_date_is_none_for_an_unresolved_alias__():
    """No dated code -> no computed expiry (the caller keeps the plain window + warns)."""
    assert expiry.computed_final_trade_date("VN30F1M", today=date(2026, 9, 14)) is None
    assert expiry.computed_final_trade_date("HPG", today=date(2026, 9, 14)) is None


# --- the coverage horizon (the honest part) ----------------------------------

def __test_holiday_coverage_horizon_is_declared__():
    """Past the verified horizon the weekend walk-back holds and the holiday one may not.

    The horizon is what turns "we might be a day late" into a logged fact instead of a
    silent one — ``broker._warn_computed_expiry_once`` reads exactly this.
    """
    assert expiry.holiday_coverage_is_verified(date(2026, 9, 17)) is True
    assert expiry.holiday_coverage_is_verified(
        expiry.HOLIDAY_TABLE_VERIFIED_THROUGH) is True
    assert expiry.holiday_coverage_is_verified(date(2027, 2, 18)) is False, \
        "Tet 2027 is not in the table — a computed date there must NOT claim coverage"
