"""#132 W1 core — did price trade THROUGH a resting order's level without a fill?

Pure decision module: no I/O, no clock of its own, no venue. The caller feeds
it a resting level, the prints it has seen, the latest venue record of the
order, and the current time; it answers with one of four verdicts. That shape
is deliberate (card #132, frozen design): the alarm direction can only be
tested with REAL prices and a MOCKED order record, because a correct venue
cannot be made to slip on command — so the decision must be separable from
the reads.

Verdicts
--------
QUIET                   price never crossed the level, or the order did its job
                        (its own fill, or its normal-book child filled).
PENDING                 price crossed, the record still shows the order
                        un-activated / unfilled, but less than ``grace_s`` has
                        elapsed since the cross — the venue and the engine are
                        allowed a beat (measured ~1.1 s fill -> bracket).
TRADED_THROUGH_UNFILLED price crossed, grace elapsed, and the record read AFTER
                        the cross still shows no fill: a stop that never
                        activated (S1), or one that activated whose child rests
                        unfilled (S2, "triggered, unfilled, still exposed").
                        For a protective stop this is a naked position wearing
                        a protection order's name.
UNDETERMINED            the evidence cannot answer: no print newer than
                        ``stale_s`` (a dead price feed must never read as
                        quiet), or the record predates the cross.

What "through" means
--------------------
* A STOP activates when price REACHES its trigger: buy-stop at any print
  ``>= level``, sell-stop at any print ``<= level``.
* A LIMIT is only owed a fill by a print STRICTLY beyond it: sell-limit
  ``> level``, buy-limit ``< level``. A print AT the limit is a touch — others
  may be ahead in the queue — and is not evidence of a slip.

Prints may be per-trade or a bar extreme stamped at the bar's END (the
conservative reading: the cross can only have happened earlier).

The OCO stop-loss leg
---------------------
A bracket's SL lives on the OCO umbrella (``stopPrice``); the only child the
venue shows is the TP. Judge the SL as its own ``RestingLevel`` from the
umbrella record with ``child_is_for_level=False`` unless the venue named an SL
child — the umbrella being ``Activated`` (it is, from birth) proves nothing
about the SL. ``position_open=False`` with a filled SL child is the satisfied
form.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

QUIET = "QUIET"
PENDING = "PENDING"
TRADED_THROUGH_UNFILLED = "TRADED_THROUGH_UNFILLED"
UNDETERMINED = "UNDETERMINED"

_FILLED_STATUSES = {"filled", "partiallyfilled"}


@dataclass(frozen=True)
class RestingLevel:
    order_id: str
    side: str            # "buy" | "sell"
    kind: str            # "stop" | "limit"
    level: float
    book: str            # "STOP" | "OCO" | "NORMAL" — informational
    resting_since: float # epoch s; prints before this cannot cross a level that did not exist


@dataclass(frozen=True)
class Print:
    ts: float
    price: float


@dataclass(frozen=True)
class OrderRecord:
    order_id: str
    status: str                      # venue orderStatus: New / Activated / Filled / Canceled …
    child_id: Optional[str]
    child_status: Optional[str]
    child_filled_qty: float
    read_at: float                   # epoch s when this record was read from the venue
    child_is_for_level: bool = True  # False when the only child named is another leg (OCO TP)
    position_open: Optional[bool] = None
    own_filled_qty: float = 0.0      # for a NORMAL-book order (limit): its own fillQuantity


@dataclass(frozen=True)
class Verdict:
    kind: str
    order_id: str
    level: float
    detail: str
    crossed_at: Optional[float] = None


def _satisfied(level: RestingLevel, record: OrderRecord) -> bool:
    """The order did what a resting order is for — a fill exists for THIS level."""
    if record.own_filled_qty > 0 or str(record.status).lower() in _FILLED_STATUSES:
        return True
    if level.kind == "stop":
        return bool(record.child_is_for_level and record.child_id
                    and str(record.child_status or "").lower() in _FILLED_STATUSES
                    and record.child_filled_qty > 0)
    return False


def _crossing(level: RestingLevel, prints: Iterable[Print]) -> tuple[Optional[float], Optional[float]]:
    """(ts of the first print through the level, the extreme print that did it)."""
    first_ts: Optional[float] = None
    extreme: Optional[float] = None
    for p in sorted(prints, key=lambda x: x.ts):
        if p.ts < level.resting_since:
            continue
        if level.kind == "stop":
            through = p.price >= level.level if level.side == "buy" else p.price <= level.level
        else:
            through = p.price < level.level if level.side == "buy" else p.price > level.level
        if not through:
            continue
        if first_ts is None:
            first_ts = p.ts
        if extreme is None or (p.price > extreme if level.side == "buy" else p.price < extreme):
            extreme = p.price
    return first_ts, extreme


def judge(level: RestingLevel, prints: Iterable[Print], record: OrderRecord, *,
          now: float, grace_s: float, stale_s: float) -> Verdict:
    prints = list(prints)
    if record.order_id != level.order_id:
        return Verdict(UNDETERMINED, level.order_id, level.level,
                       f"record is for {record.order_id}, not {level.order_id}")
    if _satisfied(level, record):
        return Verdict(QUIET, level.order_id, level.level,
                       f"{level.side} {level.kind} {level.level:g} did its job (filled)")
    latest = max((p.ts for p in prints), default=None)
    if latest is None or latest < now - stale_s:
        return Verdict(UNDETERMINED, level.order_id, level.level,
                       "no print newer than %.0fs — cannot tell whether price crossed %g"
                       % (stale_s, level.level))
    crossed_at, extreme = _crossing(level, prints)
    if crossed_at is None:
        return Verdict(QUIET, level.order_id, level.level,
                       f"price has not traded through {level.level:g}")
    if record.read_at < crossed_at:
        return Verdict(UNDETERMINED, level.order_id, level.level,
                       f"price reached {extreme:g} through {level.level:g} but the order record "
                       f"predates the cross — re-read before judging", crossed_at)
    elapsed = now - crossed_at
    if elapsed < grace_s:
        return Verdict(PENDING, level.order_id, level.level,
                       f"price reached {extreme:g} through {level.level:g} {elapsed:.1f}s ago; "
                       f"venue record still {record.status} — inside the {grace_s:g}s grace",
                       crossed_at)
    state = (f"Activated, child {record.child_id} {record.child_status} filled={record.child_filled_qty:g}"
             if record.child_id and record.child_is_for_level
             else f"{record.status}, no fill for this level")
    return Verdict(TRADED_THROUGH_UNFILLED, level.order_id, level.level,
                   f"price reached {extreme:g} through {level.side} {level.kind} {level.level:g} "
                   f"{elapsed:.0f}s ago and the venue record ({state}) shows NO fill — "
                   f"slipped over; if this is protection the position is naked",
                   crossed_at)
