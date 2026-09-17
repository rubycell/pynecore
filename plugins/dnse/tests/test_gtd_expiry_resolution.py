"""#118 — how ``broker._gtd`` resolves and bounds a conditional's expiry.

Companion to ``test_gtd_expiry_clamp_starved.py`` (the RED baseline, which pins the
no-venue-expiry case end to end). This file pins the rest of the resolution order and
both bounds:

* the venue's own ``finalTradeDate`` still WINS over the computed one, in every shape
  it is served in — including the compact ``20260416`` integer form from the docs;
* an UNKNOWN date shape warns loudly and degrades to the computed expiry, instead of
  the old bare ``except: pass`` silently falling open to ``now + 7d``;
* the GTD is FLOORED at the next open day, so a stale/past expiry can never put a date
  in the past on the wire (``broker.py`` ``_clamp_gtd_to_expiry``).

Same fake-client seam as ``test_broker_orders.py`` — no live venue, no real files — and
the clock is frozen the same way (a ``datetime`` subclass swapped into the broker
module) so no verdict here can drift with the wall clock.
"""
from datetime import datetime, timezone

import pytest

import pynecore.lib as lib

lib.bar_index = 0  # let the [BROKER] log formatter render during broker._emit()

from pynecore_dnse import broker

#: 2026-09-14, a Monday inside expiry week — the day #118 was measured on prod.
FROZEN_NOW = datetime(2026, 9, 14, 3, 0, 0, tzinfo=timezone.utc)

#: The September 2026 front month's real final trade date (3rd Thursday).
FINAL_TRADE_DATE = "2026-09-17"

#: The next open day after :data:`FROZEN_NOW` — the floor the GTD can never go below.
NEXT_OPEN_DAY = "2026-09-15"

FRONT_CONTRACT = "41I1G9000"

_INSTRUMENTS = (200, {"data": [
    {"symbol": FRONT_CONTRACT, "symbolType": "VN30F1M", "securityGroupId": "FU"},
]})
_LOAN_OK = (200, {"loanPackages": [{"id": 42}]})


class _FrozenDatetime(datetime):
    """``datetime`` with a fixed ``now()`` — everything else intact."""

    @classmethod
    def now(cls, tz=None):                                   # noqa: D102
        return FROZEN_NOW if tz is None else FROZEN_NOW.astimezone(tz)


@pytest.fixture
def frozen_clock(monkeypatch):
    monkeypatch.setattr(broker, "datetime", _FrozenDatetime)
    return FROZEN_NOW


@pytest.fixture
def warnings(monkeypatch):
    """Capture the broker's WARNING lines — a fallback MUST be audible (#118)."""
    captured = []
    monkeypatch.setattr(broker.log, "broker_warning",
                        lambda message, *args: captured.append(message % args))
    return captured


def _broker_with_expiry(fake_client, tmp_path, final_trade_date):
    """A derivative broker whose secdef carries ``final_trade_date`` (verbatim)."""
    row = {"ceilingPrice": "1550", "floorPrice": "1450", "securityGroupId": "FU",
           "symbol": FRONT_CONTRACT}
    if final_trade_date is not None:
        row["finalTradeDate"] = final_trade_date
    config = broker.DNSEBrokerConfig(
        api_key="k", api_secret="s", account_no="ACC001", trading_token="tok-A",
        token_file=str(tmp_path / "missing_token.json"))
    b = broker.DNSEBroker(symbol="VN30F1M", timeframe="15", config=config)
    b._client = fake_client(get_security_definition=(200, [row]),
                            get_instruments=_INSTRUMENTS, get_loan_packages=_LOAN_OK)
    return b


def _gtd_day(b) -> str:
    return b._gtd(days=7)[:10]


# --- the venue's own value wins ---------------------------------------------

@pytest.mark.parametrize("served", [
    "2026-09-17",                # bare ISO
    "2026-09-17T00:00:00Z",      # RFC3339 — the shape measured on 2026-08-14
    20260917,                    # the compact integer form the API docs describe
    "20260917",
])
def __test_a_served_final_trade_date_still_clamps_the_gtd__(
        fake_client, tmp_path, frozen_clock, served):
    """The venue value is the authority — the computed fallback never overrides it.

    Regression pin for the clamp #118 must NOT break: with a real ``finalTradeDate``
    the GTD is that date, not ``now + 7d`` (2026-09-21, which DNSE refuses with
    CO-ORD-006).
    """
    b = _broker_with_expiry(fake_client, tmp_path, served)

    assert _gtd_day(b) == FINAL_TRADE_DATE


def __test_a_served_expiry_is_cached_under_the_dated_code_not_the_alias__(
        fake_client, tmp_path, frozen_clock):
    """The last-known-good expiry is keyed by the DATED contract code (ties #113).

    The existing ``_secdef`` cache is alias-keyed AND permanent, so an alias key here
    would survive a roll and clamp to a dead contract. Keying on ``41I1G9000`` means the
    next contract simply misses the cache.
    """
    b = _broker_with_expiry(fake_client, tmp_path, "2026-09-17T00:00:00Z")

    _gtd_day(b)

    assert set(b._final_trade_date_cache) == {FRONT_CONTRACT}, \
        "the expiry cache must be keyed by the dated contract code, never 'VN30F1M'"


# --- an unknown date shape is loud, not swallowed ----------------------------

def __test_an_unreadable_final_trade_date_warns_and_falls_back_to_the_computed_one__(
        fake_client, tmp_path, frozen_clock, warnings):
    """A NEW venue date format must never become a silent ``now + 7d``.

    The old ``except: pass`` (``broker.py:897``) swallowed exactly this. Now the parse
    failure is logged and the computed 3rd-Thursday expiry takes over, so the order is
    still placeable.
    """
    b = _broker_with_expiry(fake_client, tmp_path, "17/09/2026")

    assert _gtd_day(b) == FINAL_TRADE_DATE, \
        "an unparsable expiry must degrade to the COMPUTED date, not to +7d"
    assert any("unreadable finalTradeDate" in line for line in warnings), \
        f"the parse failure must be audible; got {warnings}"


def __test_the_computed_fallback_announces_itself__(
        fake_client, tmp_path, frozen_clock, warnings):
    """Every fallback taken logs once — a silent one is how #118 lived for six weeks."""
    b = _broker_with_expiry(fake_client, tmp_path, None)

    assert _gtd_day(b) == FINAL_TRADE_DATE
    assert any("COMPUTED" in line and FRONT_CONTRACT in line for line in warnings), \
        f"the computed fallback must name itself and the contract; got {warnings}"

    warnings.clear()
    b._gtd(days=7)
    assert warnings == [], "the fallback warns ONCE per contract, not once per order"


# --- the floor (the bug with no test before #118) ----------------------------

def __test_a_past_final_trade_date_is_floored_at_the_next_open_day__(
        fake_client, tmp_path, frozen_clock, warnings):
    """``min(target, last)`` with a STALE expiry used to emit a GTD IN THE PAST.

    2026-08-20 was the previous contract's final trade date; an alias-keyed permanent
    secdef cache (#113) can still be serving it after the roll. Unclamped, the plugin
    would put 2026-08-20 on the wire on 2026-09-14 — refused just as hard as a GTD that
    is too late, and far harder to read in a log.
    """
    b = _broker_with_expiry(fake_client, tmp_path, "2026-08-20T00:00:00Z")

    day = _gtd_day(b)

    assert day == NEXT_OPEN_DAY, f"a past expiry must be floored, got {day}"
    assert datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc) > FROZEN_NOW
    assert any("GTD floored" in line for line in warnings), \
        f"flooring means the expiry is unusable — it must be audible; got {warnings}"


def __test_on_the_final_trade_date_the_gtd_is_that_days_session_close__(
        fake_client, tmp_path, frozen_clock, warnings):
    """On the final trade date the GTD must stay ON that date, at the close.

    INVERTED 2026-09-17. This test previously asserted the opposite — that
    expiry day floors to the NEXT open day — and its own docstring conceded
    the premise was "UNVERIFIED at the venue: whether DNSE accepts a GTD one
    day past the final trade date ON that date has not been measured".

    It was measured, on prod, and the venue refused: four conditional
    placements (STOP buy/sell, STOP-LIMIT buy/sell) all rejected
    ``CO-ORD-006 http=400 "Validate Order Failed"``, each immediately preceded
    by the floor warning this test used to require. So the old assertion
    pinned a guess that turned out to be wrong, and pinning it kept the bug
    stable: the clamp was "correct" by its own test while making every native
    STOP unplaceable for a whole session.

    The cause was the ceiling's TIME, not its date. Midnight UTC of the final
    trade date is 07:00 ICT on that date — before the session opens — so by
    mid-session it is already in the past, the floor takes over, and the GTD
    lands one day PAST the final trade date. The ceiling is now the session
    close (14:45 ICT = 07:45Z), which is inside the final date in ICT and
    still ahead of any in-session order.
    """
    b = _broker_with_expiry(fake_client, tmp_path, "2026-09-14")   # == frozen today

    assert _gtd_day(b) == "2026-09-14", (
        "the GTD must stay ON the final trade date — emitting the next day is "
        "exactly what DNSE refuses with CO-ORD-006"
    )
    assert b._gtd(days=7) == "2026-09-14T07:30:00Z", (
        "and at the END OF THE CONTINUOUS SESSION (14:30 ICT), not the "
        "14:45 close and not midnight UTC — 07:45Z was REFUSED four times "
        "on prod, while the operator's 14:30-expiry order was accepted"
    )
    assert not any("GTD floored" in line for line in warnings), (
        "the final trade date is USABLE, so nothing was floored and there is "
        "nothing to warn about; warning here would train the operator to "
        "ignore the line that matters when a secdef really is stale"
    )



# --- #118 (2026-09-17): the ceiling is a TIME, not just a date ---------------

def __test_the_day_before_expiry_also_uses_the_session_close__(
        fake_client, tmp_path, frozen_clock, warnings):
    """The ceiling moved from 00:00Z to 07:45Z on EVERY day, not just expiry day.

    Frozen now is 2026-09-14; a final trade date of 2026-09-17 is three days
    out, so the ceiling is comfortably in the future either way and the DATE
    is unchanged. What changes is the time-of-day, and pinning it here stops
    a future edit from quietly reverting the ceiling while the date-only
    assertions elsewhere stay green. 07:30Z is the end of the CONTINUOUS
    session (14:30 ICT); 07:45Z (the close) was measured REFUSED.
    """
    b = _broker_with_expiry(fake_client, tmp_path, "2026-09-17")

    assert b._gtd(days=7) == "2026-09-17T07:30:00Z"
    assert not any("GTD floored" in line for line in warnings)


def __test_a_stale_expiry_still_floors_and_still_warns__(
        fake_client, tmp_path, frozen_clock, warnings):
    """The floor path survives — it is for a STALE secdef, which is real (#113).

    Only a final trade date genuinely in the PAST should reach it. This is the
    control for the inversion above: if the fix had simply deleted the floor,
    this goes red, and a rolled-away contract would put a GTD in the past on
    the wire.
    """
    b = _broker_with_expiry(fake_client, tmp_path, "2026-08-20")   # previous contract

    assert _gtd_day(b) == NEXT_OPEN_DAY
    assert any("GTD floored" in line for line in warnings)
