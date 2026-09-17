"""#132 W1 baseline — the SLIPPED-OVER scenario, pinned before the detector exists.

The operator's question (card #132): "if price went THROUGH a resting order's
level but the order did not fill, we have a slipped-over situation" — for a
stop-loss that is a naked position wearing a protection order's name.

Why every fixture here is REAL PRICES + a MOCKED ORDER RECORD: a correct venue
cannot be made to misbehave on command, so the alarm direction can only be
built from the venue's own prices with the order record held in the state the
venue WOULD show if it had slipped. The prices and the positive-control records
are copied verbatim from the 2026-09-17 live chain
(``fixtures/f132_20260917_l2b_chain.json``: stop entry ``dalp3kiv…`` trigger
1969.2 -> child 291756 FILLED 1969.4 at 07:08:05.7Z inside the 14:08 bar
[1968.8, 1970.4]; OCO umbrella ``dalp3liv…`` TP 1973.2 / SL 1965.2).

The bars are 1m OHLC, not per-print trades: a bar's high proves price reached
AT LEAST that high inside the bar, never the order of prints. The tests use the
bar extreme as one print stamped at the bar's END, which is the conservative
reading (the cross can only have happened earlier, never later).

RED-FIRST: this file must fail with ImportError until ``price_through.py``
exists, and the POSITIVE control (``__test_a_stop_that_activated_and_filled_is_quiet__``)
is the acceptance gate for every future implementation — the modal false alarm
is a stop that WORKED, and a detector that alarms on every cross passes every
other test here.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

# RED-FIRST RECORD (2026-09-17 16:0x, before this guard existed): running this
# file gave `1 error in 0.10s` — ImportError, the detector does not exist.
# The guard below turns that into a SKIP so the shared tree's full-suite runs
# (pytest.ini carries -x) are not stopped by a card that is deliberately ahead
# of its implementation; the pins go live the moment price_through.py exists.
_pt = pytest.importorskip("price_through",
                          reason="#132 W1 detector (plugins/dnse/tools/price_through.py) not built yet — baseline pins")
RestingLevel, Print, OrderRecord, judge = _pt.RestingLevel, _pt.Print, _pt.OrderRecord, _pt.judge
QUIET, PENDING, TRADED_THROUGH_UNFILLED, UNDETERMINED = (
    _pt.QUIET, _pt.PENDING, _pt.TRADED_THROUGH_UNFILLED, _pt.UNDETERMINED)

_FIXTURE = Path(__file__).parent / "fixtures" / "f132_20260917_l2b_chain.json"
_CHAIN = json.loads(_FIXTURE.read_text())

GRACE_S = 20.0        # how long a crossed level may sit un-activated before alarm
STALE_S = 60.0        # prints older than this cannot answer "did price cross since?"


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def _bar(ict_hhmm: str) -> dict:
    return next(b for b in _CHAIN["bars_1m"] if b["ict"] == ict_hhmm)


def _prints_from_bar(ict_hhmm: str) -> list[Print]:
    """The bar's extremes as prints stamped at the bar's END (conservative)."""
    b = _bar(ict_hhmm)
    end = b["t"] + 60
    return [Print(ts=end, price=b["l"]), Print(ts=end, price=b["h"])]


# --- the real chain, verbatim -------------------------------------------------

_ENTRY = _CHAIN["orders"]["dalp3kivfqkc7397meg0"]     # STOP, Activated, stopPrice 1969.2
_CHILD = _CHAIN["orders"]["291756"]                   # NORMAL, Filled 1969.4
_UMBRELLA = _CHAIN["orders"]["dalp3livfqkc7397megg"]  # OCO, TP 1973.2 / SL 1965.2
_TP_CHILD = _CHAIN["orders"]["291996"]                # NORMAL, Canceled (flatten)

ENTRY_LEVEL = RestingLevel(order_id="dalp3kivfqkc7397meg0", side="buy", kind="stop",
                           level=_ENTRY["stopPrice"], book="STOP",
                           resting_since=_ts("2026-09-17T07:07:30Z"))
FILL_TS = _ts(_CHILD["createdDate"])                  # 07:08:05.7Z — inside the 14:08 bar
CROSS_BAR = "14:08"                                   # [1968.8, 1970.4] contains 1969.2


def _record_activated_and_filled() -> OrderRecord:
    return OrderRecord(order_id=ENTRY_LEVEL.order_id, status="Activated",
                       child_id="291756", child_status="Filled", child_filled_qty=1,
                       read_at=FILL_TS + 1.0)


def _record_never_triggered(read_at: float) -> OrderRecord:
    """What the venue WOULD show if the stop slipped: still New, no child."""
    return OrderRecord(order_id=ENTRY_LEVEL.order_id, status="New",
                       child_id=None, child_status=None, child_filled_qty=0,
                       read_at=read_at)


# --- positive control: the acceptance gate -------------------------------------

def __test_a_stop_that_activated_and_filled_is_quiet__():
    """The modal false alarm is a stop that WORKED. Same prints as every negative
    below; the record says Activated + child Filled -> QUIET, never an alarm.
    A detector that fires on 'price crossed the level' alone passes every other
    test in this file and fails this one — that is the point of it."""
    verdict = judge(ENTRY_LEVEL, _prints_from_bar(CROSS_BAR),
                    _record_activated_and_filled(),
                    now=FILL_TS + GRACE_S + 30, grace_s=GRACE_S, stale_s=STALE_S)
    assert verdict.kind == QUIET, verdict


def __test_no_cross_is_quiet_whatever_the_record_says__():
    """The 14:06 bar tops at 1968.9 < 1969.2: price never reached the level, so a
    still-New record is the NORMAL resting state, not a slip. Guards against a
    detector that alarms on 'resting for a long time' instead of 'crossed'."""
    verdict = judge(ENTRY_LEVEL, _prints_from_bar("14:06"),
                    _record_never_triggered(read_at=_bar("14:06")["t"] + 60 + GRACE_S + 5),
                    now=_bar("14:06")["t"] + 60 + GRACE_S + 5, grace_s=GRACE_S, stale_s=STALE_S)
    assert verdict.kind == QUIET, verdict


# --- the scenario: price through the level, no fill -----------------------------

def __test_a_buy_stop_crossed_and_never_activated_is_TRADED_THROUGH_after_grace__():
    """S1 — the venue never activated the conditional although the 14:08 bar's
    high (1970.4) is above the trigger (1969.2). Record frozen at New, no child,
    read AFTER the grace window -> TRADED_THROUGH_UNFILLED, naming the level."""
    cross_ts = _bar(CROSS_BAR)["t"] + 60
    now = cross_ts + GRACE_S + 1
    verdict = judge(ENTRY_LEVEL, _prints_from_bar(CROSS_BAR),
                    _record_never_triggered(read_at=now), now=now,
                    grace_s=GRACE_S, stale_s=STALE_S)
    assert verdict.kind == TRADED_THROUGH_UNFILLED, verdict
    assert verdict.level == ENTRY_LEVEL.level and verdict.order_id == ENTRY_LEVEL.order_id
    assert "1969.2" in verdict.detail and "1970.4" in verdict.detail


def __test_a_crossed_stop_within_grace_is_PENDING_not_an_alarm__():
    """The engine measured ~1.1 s from child fill to bracket; venue activation
    itself takes a beat. Inside the grace window the honest verdict is PENDING —
    an alarm here would page on every healthy stop for a second or two."""
    cross_ts = _bar(CROSS_BAR)["t"] + 60
    now = cross_ts + GRACE_S - 1
    verdict = judge(ENTRY_LEVEL, _prints_from_bar(CROSS_BAR),
                    _record_never_triggered(read_at=now), now=now,
                    grace_s=GRACE_S, stale_s=STALE_S)
    assert verdict.kind == PENDING, verdict


def __test_activated_but_child_unfilled_past_grace_is_TRADED_THROUGH__():
    """S2 — the umbrella DID activate, but its normal-book child rests unfilled
    (the 'triggered, unfilled, still exposed' state broker._stop_fill_price's
    2x-slippage offset exists to prevent). Activation is not protection; only
    the child's fill is. Past grace -> TRADED_THROUGH_UNFILLED."""
    cross_ts = _bar(CROSS_BAR)["t"] + 60
    now = cross_ts + GRACE_S + 1
    record = OrderRecord(order_id=ENTRY_LEVEL.order_id, status="Activated",
                         child_id="291756", child_status="New", child_filled_qty=0,
                         read_at=now)
    verdict = judge(ENTRY_LEVEL, _prints_from_bar(CROSS_BAR), record, now=now,
                    grace_s=GRACE_S, stale_s=STALE_S)
    assert verdict.kind == TRADED_THROUGH_UNFILLED, verdict
    assert "291756" in verdict.detail


def __test_the_OCO_stoploss_leg_lives_on_the_umbrella_and_is_judged_there__():
    """The SL of a bracket is the umbrella's stopPrice (1965.2) — venue.py never
    lists it (#152) and a TP-only reading calls it UNSTOPPED. W1 must judge the
    SL leg from the umbrella record: prints fall to 1964.0, the umbrella still
    shows only the TP child and the position is still open -> the stop-loss was
    traded through without protecting -> TRADED_THROUGH_UNFILLED."""
    sl = RestingLevel(order_id="dalp3livfqkc7397megg", side="sell", kind="stop",
                      level=_UMBRELLA["stopPrice"], book="OCO",
                      resting_since=_ts(_UMBRELLA["createdDate"]))
    t = _ts(_UMBRELLA["createdDate"]) + 120
    prints = [Print(ts=t, price=1966.0), Print(ts=t + 5, price=1964.0)]
    now = t + 5 + GRACE_S + 1
    record = OrderRecord(order_id=sl.order_id, status="Activated",       # OCO is Activated from birth
                         child_id="291996", child_status="New", child_filled_qty=0,
                         child_is_for_level=False,                        # 291996 is the TP child, not the SL
                         position_open=True, read_at=now)
    verdict = judge(sl, prints, record, now=now, grace_s=GRACE_S, stale_s=STALE_S)
    assert verdict.kind == TRADED_THROUGH_UNFILLED, verdict
    assert "1965.2" in verdict.detail


def __test_the_OCO_stoploss_leg_that_fired_and_closed_the_position_is_quiet__():
    """Control for the previous pin: same prints, but the umbrella now names an
    SL child that Filled and the position is flat -> QUIET. Without this, the
    umbrella pin above is satisfied by 'always alarm on an OCO'."""
    sl = RestingLevel(order_id="dalp3livfqkc7397megg", side="sell", kind="stop",
                      level=_UMBRELLA["stopPrice"], book="OCO",
                      resting_since=_ts(_UMBRELLA["createdDate"]))
    t = _ts(_UMBRELLA["createdDate"]) + 120
    prints = [Print(ts=t, price=1966.0), Print(ts=t + 5, price=1964.0)]
    now = t + 5 + GRACE_S + 1
    record = OrderRecord(order_id=sl.order_id, status="Activated",
                         child_id="299999", child_status="Filled", child_filled_qty=1,
                         child_is_for_level=True, position_open=False, read_at=now)
    verdict = judge(sl, prints, record, now=now, grace_s=GRACE_S, stale_s=STALE_S)
    assert verdict.kind == QUIET, verdict


# --- limit orders: THROUGH means strictly beyond, a touch is not a slip ----------

def __test_a_sell_limit_traded_strictly_through_without_a_fill_is_TRADED_THROUGH__():
    """A resting sell LO at 1973.2 while a print at 1974.0 exists is impossible on
    a working book (card comment 3: a trade printing ABOVE your resting ask
    cannot happen if your order were live). Record New, fillQuantity 0, past
    grace -> TRADED_THROUGH_UNFILLED."""
    tp = RestingLevel(order_id="291996", side="sell", kind="limit",
                      level=_TP_CHILD["price"], book="NORMAL",
                      resting_since=_ts(_TP_CHILD["createdDate"]))
    t = tp.resting_since + 60
    prints = [Print(ts=t, price=1973.0), Print(ts=t + 3, price=1974.0)]
    now = t + 3 + GRACE_S + 1
    record = OrderRecord(order_id=tp.order_id, status="New", child_id=None,
                         child_status=None, child_filled_qty=0, own_filled_qty=0,
                         read_at=now)
    verdict = judge(tp, prints, record, now=now, grace_s=GRACE_S, stale_s=STALE_S)
    assert verdict.kind == TRADED_THROUGH_UNFILLED, verdict


def __test_a_sell_limit_merely_TOUCHED_is_quiet__():
    """Queue priority: a print AT 1973.2 does not owe our order a fill (others
    ahead in the queue). Touch != through. Guards against the over-alarm that
    would page on every limit that sits at the best ask."""
    tp = RestingLevel(order_id="291996", side="sell", kind="limit",
                      level=_TP_CHILD["price"], book="NORMAL",
                      resting_since=_ts(_TP_CHILD["createdDate"]))
    t = tp.resting_since + 60
    prints = [Print(ts=t, price=1973.0), Print(ts=t + 3, price=1973.2)]
    now = t + 3 + GRACE_S + 1
    record = OrderRecord(order_id=tp.order_id, status="New", child_id=None,
                         child_status=None, child_filled_qty=0, own_filled_qty=0,
                         read_at=now)
    verdict = judge(tp, prints, record, now=now, grace_s=GRACE_S, stale_s=STALE_S)
    assert verdict.kind == QUIET, verdict


# --- evidence that cannot answer ------------------------------------------------

def __test_stale_prints_are_UNDETERMINED_never_quiet__():
    """No print newer than ``stale_s``: the question 'did price cross since?'
    cannot be answered. A detector that reads 'no recent cross' as QUIET goes
    silent exactly when the price feed dies (#84's lesson, one layer out)."""
    cross_ts = _bar(CROSS_BAR)["t"] + 60
    now = cross_ts + STALE_S + 30
    verdict = judge(ENTRY_LEVEL, _prints_from_bar(CROSS_BAR),
                    _record_never_triggered(read_at=now), now=now,
                    grace_s=GRACE_S, stale_s=STALE_S)
    assert verdict.kind == UNDETERMINED, verdict


def __test_a_record_older_than_the_cross_cannot_clear_the_order__():
    """The record was read BEFORE the cross (read_at < cross_ts): 'still New'
    then says nothing about now. The detector must not treat a pre-cross read
    as evidence either way -> UNDETERMINED until a post-cross read exists."""
    cross_ts = _bar(CROSS_BAR)["t"] + 60
    now = cross_ts + GRACE_S + 1
    verdict = judge(ENTRY_LEVEL, _prints_from_bar(CROSS_BAR),
                    _record_never_triggered(read_at=cross_ts - 30), now=now,
                    grace_s=GRACE_S, stale_s=STALE_S)
    assert verdict.kind == UNDETERMINED, verdict


def __test_fixture_is_the_real_chain_not_an_invention__():
    """Pins the fixture to the venue record so nobody 'tidies' it into a
    plausible shape: the child's fill price, the umbrella's two levels and the
    14:08 bar's range are the values the venue served on 2026-09-17."""
    assert _CHILD["averagePrice"] == 1969.4 and _CHILD["orderStatus"] == "Filled"
    assert _UMBRELLA["orderCategory"] == "OCO" and _UMBRELLA["stopPrice"] == 1965.2
    assert _UMBRELLA["price"] == 1973.2 and _UMBRELLA["externalOrderId"] == "291996"
    b = _bar(CROSS_BAR)
    assert b["l"] <= ENTRY_LEVEL.level <= b["h"], b
    assert _bar("14:07")["t"] < FILL_TS < _bar("14:08")["t"] + 60
