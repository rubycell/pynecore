"""
Characterization tests: when one bar crosses both a protective exit and an
opposite-direction entry stop, the position can end at TWO contracts even
though every order was written ``qty=1``.

WHY THIS IS PINNED (read before "fixing" it): this is **TradingView's own
behaviour**, not a PyneCore defect. Cross-checked 2026-09-10 by running the
same setup on TradingView (VN301! 15m, Pine v6): ``strategy.position_size``
went -1 -> +2 there exactly as it does here, and the probe's
``bgcolor(math.abs(strategy.position_size) > 1)`` guard lit on both engines.
Both probe files are kept at ``docs/probes/flip_exit/`` so the claim stays
reproducible.

THE MECHANISM IS THE ENTRY'S STALE FLIP QUANTITY -- NOT AN ORPHANED EXIT.
A price-based ``strategy.entry`` freezes its reversal augmentation when the
script places it (``lib/strategy/__init__.py:6136``, under
``elif limit is not None or stop is not None``). Placed while short 1, the
long entry stop therefore carries qty **2** -- one to close the short, one to
open the long. On the crossing bar the cover-stop fires first and closes the
short perfectly normally; the entry then fills against an ALREADY FLAT
position and opens all 2. Nothing re-derives the frozen flip component against
the position as it actually stands at fill time.

Consequence: the outcome is ORDER-DEPENDENT, which is what the second test
pins. Exit level crossed first -> 2 contracts. Entry level crossed first -> 1
contract, and the exit left orphaned opens nothing at all.

(An earlier revision of this docstring blamed "the now-parentless cover-stop
fires as an OPENING buy". That is REFUTED -- ``flip_exit_entry_first`` is the
control that kills it: under that theory it would also end at 2, and it ends
at 1. Measured 2026-09-11, see issue #105.)

Things this rules out, all measured on the way here:
  * ``pyramiding`` cannot prevent it -- the oversized fill is one entry order,
    not a second one. (PyneCore *does* enforce pyramiding, for market and stop
    entries alike.)
  * ``qty=1`` on every entry AND exit cannot prevent it -- the script-level
    quantity was never the mechanism; the flip augmentation is added on top.
  * Guarding entries on ``strategy.position_size`` makes it WORSE (measured
    2 -> 3 contracts): order commit is deferred and stop entries fill
    intrabar, so the guard reads a stale 0.
  * ``strategy.risk.max_position_size`` DOES bound it in backtest (the cap is
    applied at fill time, which re-reads the flat position and trims 2 -> 1),
    but NOT on the live path, whose cap runs at submit time only -- see #105.
    So a backtest proving "never exceeds 1" does not transfer to live.

If these tests ever fail after an upstream rebase, PyneCore has diverged from
TradingView here -- re-run the TradingView probe before assuming the new
behaviour is correct, and update the CLAUDE.md section "Pine behaviours that
LOOK like bugs and are NOT" either way.
"""
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'


def _make_syminfo():
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period='1', type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=0.0001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


def _make_ohlcv():
    """Bar 2 is the one that matters: its high crosses BOTH armed levels."""
    from pynecore.types.ohlcv import OHLCV
    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    rows = [
        # open,   high,   low,    close
        (100.00, 100.00, 100.00, 100.00),  # bar 0 - short signal
        (100.00, 100.05, 99.95, 100.00),   # bar 1 - short filled; both stops armed
        (100.10, 101.50, 100.00, 101.20),  # bar 2 - high crosses 100.20/100.50 AND 101.00
        (101.20, 101.30, 101.10, 101.20),  # bar 3 - tail; read the position here
    ]
    return [
        OHLCV(timestamp=base_ts + i * 60_000, open=o, high=h, low=lo, close=c,
              volume=100.0)
        for i, (o, h, lo, c) in enumerate(rows)
    ]


def _positions(script_name: str) -> list:
    """Run one @pyne script over the shared bars, return position_size per bar."""
    from pynecore.core.script_runner import ScriptRunner

    try:
        runner = ScriptRunner(
            DATA_DIR / f'{script_name}.py', iter(_make_ohlcv()), _make_syminfo(),
        )
        return [dict(plot_data).get('pos')
                for _candle, plot_data, _trades in runner.run_iter()]
    finally:
        sys.modules.pop(script_name, None)


def __test_a_flip_leaves_the_old_sides_exit_working_and_it_opens_a_position__():
    """
    Correct-looking arithmetic would be: cover-stop buys 1 -> flat; entry stop
    buys 1 -> long 1, i.e. net +1.

    What BOTH engines actually do: the cover-stop closes the short, then the
    entry fills carrying the qty-2 flip it froze while the position was still
    short, and opens 2 from flat.
    """
    positions = _positions('flip_exit_orphan')

    assert positions[1] == -1, (
        f"bar 1 should hold the 1-contract short, got {positions[1]}"
    )
    assert positions[-1] == 2, (
        "PINNED CROSS-ENGINE BEHAVIOUR: a bar crossing the protective exit "
        "BEFORE an opposite-direction entry stop must leave position_size == 2 "
        "(TradingView does the same -- see this module's docstring and "
        f"docs/probes/flip_exit/). Got {positions[-1]}. If this changed, verify "
        "against TradingView before treating the new value as correct."
    )


def __test_entry_level_crossed_first_leaves_one_contract_and_orphans_the_exit__():
    """
    DISCRIMINATING CONTROL for the mechanism, not a second flavour of the
    test above.

    Same two orders, same bars -- only the two stop LEVELS are swapped, so the
    entry fills first and the cover-stop is the one left parentless. Under the
    refuted "an orphaned exit fires as an opening order" theory this would also
    end at 2. It ends at 1: the orphan opens nothing, and the entry's flip
    quantity was spent correctly because the position was still short when it
    filled.

    Together with the test above this pins the ORDER DEPENDENCE, which is the
    part a single test cannot express: the same two orders give 2 or 1
    depending purely on which level the bar reaches first.
    """
    positions = _positions('flip_exit_entry_first')

    assert positions[1] == -1, (
        f"bar 1 should hold the 1-contract short, got {positions[1]}"
    )
    assert positions[-1] == 1, (
        "The entry stop is crossed BEFORE the protective exit, so the entry "
        "consumes its frozen flip against the still-open short (-1 -> +1) and "
        "the orphaned cover-stop opens nothing. Expected 1, got "
        f"{positions[-1]}. A 2 here would mean an orphaned exit DOES open a "
        "position -- re-open #105, the mechanism in this module's docstring "
        "would be wrong again."
    )
