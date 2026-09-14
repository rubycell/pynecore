"""Derivative expiry (final trade date) arithmetic for DNSE conditional GTDs (#118).

A DNSE conditional (``orderCategory=STOP``) rides on ``durationType=GTD`` +
``durationDateTime`` — and the venue refuses any GTD that reaches past the
contract's **final trade date** with ``CO-ORD-006``. The venue's own
``finalTradeDate`` field is the authority, but it is **INTERMITTENT**: the
2026-08-04 fixture already carries ``finalTradeDate: null`` and the 2026-09-14
live read carried no such key at all, while 2026-08-14 served ``2026-08-20``.
So the plugin needs a derived answer for the windows where the venue serves none.

VN30 futures expire on the **3rd Thursday** of their contract month (measured:
2026-08-20 and 2026-09-17 are both 3rd Thursdays), and the dated KRX code names
that month — see :data:`MONTH_CODES`. Everything here is a **pure function of a
supplied date**: no clock, no I/O, so the caller (``broker._gtd``) owns the clock
and tests can freeze it.

**KNOWN LIMITATION — holiday coverage is partial (read before trusting a date).**
Walking back off a closed 3rd Thursday needs the VN exchange calendar, and the
lunar holidays (Tet, Hung Kings) are not derivable from month/day — they exist
here only as hand-entered dated rows with a verification horizon
(:data:`HOLIDAY_TABLE_VERIFIED_THROUGH`). Past that horizon the weekend walk-back
is guaranteed and the holiday walk-back is NOT: a computed date can overshoot an
unlisted closure. That is survivable — it is only ever used when the venue serves
no ``finalTradeDate``, and the venue's own value overrides it the moment it
reappears — but the caller MUST log it (``broker._final_trade_date`` does) rather
than present a computed date as if it were venue-served.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

__all__ = (
    "MONTH_CODES", "contract_month", "third_thursday", "parse_venue_date",
    "is_exchange_closed", "last_open_day_on_or_before", "next_open_day_after",
    "computed_final_trade_date", "holiday_coverage_is_verified",
    "HOLIDAY_TABLE_VERIFIED_THROUGH",
)

#: KRX month character -> calendar month. CONFIRMED from
#: ``plugins/dnse/testing/dnse_fixtures.json`` (2026-08-04 ``/market/instruments``):
#: ``41I1G8000``=VN30F1M in August, ``41I1G9000``=VN30F2M (September),
#: ``41I1GC000``=VN30F1Q (December), ``41I1H3000``=VN30F2Q (March) — and the
#: 2026-09-14 live read where ``41I1G9000`` had become the September front month.
MONTH_CODES = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9, "A": 10, "B": 11, "C": 12,
}

#: A dated VN30 futures contract: ``41I1`` + cycle letter + month char + ``000``.
#: The cycle letter (``G``/``H``) is deliberately NOT decoded — the month alone
#: identifies the contract once you take the nearest FUTURE occurrence, which is
#: what :func:`computed_final_trade_date` does.
_CONTRACT_RE = re.compile(r"^41I1(?P<cycle>[A-Z])(?P<month>[1-9ABC])000$")

#: VN Labour Code fixed-date public holidays (Art. 112). Adjacent "day off in
#: lieu" days (e.g. the second National Day day, which moves year to year) are
#: NOT encoded — add them as dated rows when announced.
FIXED_HOLIDAYS_MMDD = frozenset({
    (1, 1),     # New Year's Day
    (4, 30),    # Reunification Day
    (5, 1),     # Labour Day
    (9, 2),     # National Day
})

#: Dated closures that cannot be derived from month/day. Each row names its
#: source; the table is the whole reason :data:`HOLIDAY_TABLE_VERIFIED_THROUGH`
#: exists. Lunar-calendar holidays MUST be entered here by hand.
DATED_HOLIDAYS = frozenset({
    # Tet (Lunar New Year) 2026 — announced break 2026-02-14..2026-02-22;
    # these are its WEEKDAYS (the surrounding days are weekends anyway).
    # This is the case that breaks a naive 3rd-Thursday rule: Feb 2026's 3rd
    # Thursday (02-19) falls INSIDE the closure, so the final trade date walks
    # back to Fri 2026-02-13.
    date(2026, 2, 16), date(2026, 2, 17), date(2026, 2, 18),
    date(2026, 2, 19), date(2026, 2, 20),
    # Measured closed live (#70) — the day the session-phase classifier called
    # 'continuous' while the exchange was shut. Mirrors the entry in
    # plugins/dnse/testing/live_test/level0_venue_semantics/l0_order_semantics.py.
    date(2026, 8, 31),
})

#: Last day this table is claimed to be COMPLETE. Beyond it, only weekends and
#: the fixed-date holidays are handled — an unlisted lunar closure will be
#: missed. Callers must warn when they rely on a computed date past this.
HOLIDAY_TABLE_VERIFIED_THROUGH = date(2026, 12, 31)

#: Bound on any walk, so a pathological table can never spin forever.
_MAX_WALK_DAYS = 31


def contract_month(contract_code: str | None) -> int | None:
    """Calendar month of a DATED VN30 futures code, or ``None``.

    ``None`` for anything that is not a dated contract — a stock (``HPG``), an
    unresolved alias (``VN30F1M``), an empty string — so the caller can tell
    "not a derivative contract code" from "month 1".
    """
    match = _CONTRACT_RE.match((contract_code or "").strip().upper())
    return MONTH_CODES[match.group("month")] if match else None


def third_thursday(year: int, month: int) -> date:
    """The 3rd Thursday of ``year``/``month`` — VN30 futures' expiry rule."""
    first = date(year, month, 1)
    first_thursday_day = 1 + (3 - first.weekday()) % 7     # Monday == 0
    return date(year, month, first_thursday_day + 14)


def parse_venue_date(raw: object) -> date | None:
    """Parse a venue-served date field. ``None`` when ABSENT, raises when UNKNOWN.

    Three shapes are accepted, all of them observed or documented:

    * ``"2026-08-20"``                — bare ISO date
    * ``"2026-08-20T00:00:00Z"``      — RFC3339 (the live secdef shape)
    * ``20260416`` / ``"20260416"``   — the compact integer form in the API docs

    An absent value (``None``, ``""``, JSON ``null``) returns ``None`` — that is
    a fact, not a failure. Anything else raises :class:`ValueError` **loudly**:
    the previous ``except: pass`` turned a new venue format into a silent
    fail-open to ``now + 7d``, which is exactly the #118 outage.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        head = text[:10]
        try:
            return date(int(head[:4]), int(head[5:7]), int(head[8:10]))
        except ValueError as exc:
            raise ValueError(f"unparsable DNSE date {raw!r}: {exc}") from exc
    if len(text) == 8 and text.isdigit():
        try:
            return date(int(text[:4]), int(text[4:6]), int(text[6:]))
        except ValueError as exc:
            raise ValueError(f"unparsable DNSE date {raw!r}: {exc}") from exc
    raise ValueError(
        f"unrecognised DNSE date format {raw!r} — expected 'YYYY-MM-DD', "
        f"'YYYY-MM-DDThh:mm:ssZ' or the compact 'YYYYMMDD' form")


def is_exchange_closed(day: date) -> bool:
    """Weekend, fixed-date holiday, or a listed dated closure.

    Only as complete as :data:`DATED_HOLIDAYS` — see the module docstring and
    :func:`holiday_coverage_is_verified`.
    """
    return (day.weekday() >= 5
            or (day.month, day.day) in FIXED_HOLIDAYS_MMDD
            or day in DATED_HOLIDAYS)


def last_open_day_on_or_before(day: date) -> date:
    """Walk BACK from ``day`` to the first day the exchange is open."""
    for _ in range(_MAX_WALK_DAYS):
        if not is_exchange_closed(day):
            return day
        day -= timedelta(days=1)
    raise ValueError(f"no open day within {_MAX_WALK_DAYS} days before {day}")


def next_open_day_after(day: date) -> date:
    """Walk FORWARD from the day AFTER ``day`` to the first open day."""
    day += timedelta(days=1)
    for _ in range(_MAX_WALK_DAYS):
        if not is_exchange_closed(day):
            return day
        day += timedelta(days=1)
    raise ValueError(f"no open day within {_MAX_WALK_DAYS} days after {day}")


def computed_final_trade_date(contract_code: str | None, *, today: date) -> date | None:
    """Final trade date derived from a DATED contract code, or ``None``.

    3rd Thursday of the code's month, taken at its **nearest future occurrence**
    (so the cycle letter never has to be decoded), then walked back to the last
    open day. ``None`` when ``contract_code`` is not a dated contract.

    The result can equal ``today`` — the contract still trades on its final day.
    """
    month = contract_month(contract_code)
    if month is None:
        return None
    candidate = last_open_day_on_or_before(third_thursday(today.year, month))
    if candidate < today:
        candidate = last_open_day_on_or_before(third_thursday(today.year + 1, month))
    return candidate


def holiday_coverage_is_verified(day: date) -> bool:
    """Is the holiday table claimed complete through ``day``?

    ``False`` means a computed date near ``day`` may overshoot an unlisted
    (lunar) closure — the caller must say so out loud.
    """
    return day <= HOLIDAY_TABLE_VERIFIED_THROUGH
