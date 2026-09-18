"""The l2c enter-once latch must survive a round trip that completes inside ONE bar.

WHY THIS EXISTS. On 2026-09-18 the latch was written as::

    if strategy.opentrades > 0
        hasTraded := true

and it did not fire. Live-L2c run A opened at 1994.8 and its stop filled at 1993.7 INSIDE
bar 502. `strategy.opentrades` is only observable when the script runs, once per bar close,
and order commit is deferred — CLAUDE.md states it directly: the count does not change
until the NEXT bar after orders fill. So at bar 502's evaluation the entry had not filled,
at bar 503's the position was already closed, and `opentrades` read 0 at BOTH boundaries.
The latch never had a moment to set, the entry block re-issued, and a second position
opened. It then sat unprotected for 66 seconds because of a separate engine defect (#162).

`strategy.closedtrades` increments on the close and PERSISTS across bars, so it is the
condition that survives the same-bar case — which is precisely the fast stop this vehicle
is built to produce.

WHAT IS PINNED, AND WHY IT READS THE GENERATED .py. The `.py` is the build artifact that
actually runs and it can silently disagree with the `.pine` (measured the same day: the
transpiler dropped parentheses in the backstop expression and inverted its sign). So the
condition is extracted from the generated file and EVALUATED against the exact state the
live failure produced, rather than grepped for a substring — a grep for "closedtrades"
would pass on a condition that mentioned it without using it.

Run explicitly: ``pytest plugins/dnse/tests/test_l2c_latch.py -q``.
"""
from __future__ import annotations

import pathlib
import re
import types

import pytest

VEHICLE = (pathlib.Path(__file__).resolve().parents[1]
           / "testing" / "live_test" / "l2c_oco_execution.py")

#: The `if <condition>:` whose body sets the latch. Anchored on the assignment so a
#: different `if` cannot be picked up by accident.
_LATCH = re.compile(
    r"if\s+(?P<cond>[^\n:]+):\s*\n\s*hasTraded\s*=\s*True", re.MULTILINE)

#: The state the live failure produced: the round trip completed inside one bar, so at the
#: next bar-close evaluation there is no OPEN trade and exactly one CLOSED trade.
SAME_BAR_ROUND_TRIP = {"opentrades": 0, "closedtrades": 1}

#: An ordinary multi-bar position, still open at the bar boundary.
POSITION_STILL_OPEN = {"opentrades": 1, "closedtrades": 0}

#: Never traded at all — the latch must NOT fire, or the entry could never be placed.
NEVER_TRADED = {"opentrades": 0, "closedtrades": 0}


def _latch_condition() -> str:
    """The latch guard, from the GENERATED file.

    Raises rather than returning a default when it is missing or ambiguous: a pin that
    finds nothing and passes asserts nothing.
    """
    assert VEHICLE.exists(), f"generated vehicle not found: {VEHICLE}"
    matches = _LATCH.findall(VEHICLE.read_text())
    assert len(matches) == 1, (
        f"expected exactly one `hasTraded = True` guard in {VEHICLE.name}, "
        f"found {len(matches)}: {matches!r}"
    )
    return matches[0].strip()


def _evaluate(condition: str, state: dict) -> bool:
    strategy = types.SimpleNamespace(**state)
    return bool(eval(condition, {"__builtins__": {}}, {"strategy": strategy}))  # noqa: S307


def __test_latch_fires_on_a_same_bar_round_trip__():
    """THE REGRESSION. This is the exact state that let run A re-enter."""
    condition = _latch_condition()
    assert _evaluate(condition, SAME_BAR_ROUND_TRIP), (
        f"latch guard {condition!r} does NOT fire when a round trip completed inside one "
        f"bar (opentrades=0, closedtrades=1). That is the 2026-09-18 failure: the vehicle "
        f"re-enters after its stop, and the second position was naked for 66 s."
    )


def __test_latch_fires_while_a_position_is_open__():
    """The ordinary multi-bar case must still latch."""
    condition = _latch_condition()
    assert _evaluate(condition, POSITION_STILL_OPEN), (
        f"latch guard {condition!r} does not fire with a position open"
    )


def __test_latch_does_not_fire_before_any_trade__():
    """It must stay unset while flat and unfilled, or the entry can never be placed.

    Discriminating in the other direction: a guard hard-wired to True would pass both
    tests above and make the vehicle incapable of trading at all.
    """
    condition = _latch_condition()
    assert not _evaluate(condition, NEVER_TRADED), (
        f"latch guard {condition!r} fires before any trade — the entry would never be "
        f"dispatched and the vehicle could not run"
    )


@pytest.mark.parametrize("state,expected", [
    (SAME_BAR_ROUND_TRIP, False),
    (POSITION_STILL_OPEN, True),
    (NEVER_TRADED, False),
])
def __test_the_pin_catches_the_real_historical_bug__(state, expected):
    """RED-FIRST against the exact guard that shipped and failed on 2026-09-18.

    `strategy.opentrades > 0` is not a hypothetical mutant — it is what ran. It answers
    False for the same-bar round trip, which is why the vehicle re-entered, while
    answering correctly for the other two states. That mix is what made it look right.
    """
    assert _evaluate("strategy.opentrades > 0", state) is expected
