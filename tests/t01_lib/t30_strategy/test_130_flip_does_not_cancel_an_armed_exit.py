"""
Characterization test: a REVERSAL does not cancel the old side's armed exit,
so one bar can leave the position at TWO contracts from qty=1 orders.

WHY THIS IS PINNED (read before "fixing" it): this is **TradingView's own
behaviour**, not a PyneCore defect. Cross-checked 2026-09-10 by running the
same setup on TradingView (VN301! 15m, Pine v6): ``strategy.position_size``
went -1 -> +2 there exactly as it does here, and the probe's
``bgcolor(math.abs(strategy.position_size) > 1)`` guard lit on both engines.
Both probe files are kept at ``docs/probes/flip_exit/`` so the claim stays
reproducible.

The mechanism: when a single bar crosses BOTH a protective exit and an
opposite-direction entry stop, the flip closes the short and opens the long
(+1) while the now-parentless cover-stop is still working and fires as an
OPENING buy (+1). The same buy is effectively consumed twice.

Things this rules out, all measured on the way here:
  * ``pyramiding`` cannot prevent it -- the second fill is not an entry order.
    (PyneCore *does* enforce pyramiding, for market and stop entries alike.)
  * ``qty=1`` on every entry AND exit cannot prevent it -- order size was
    never the mechanism.
  * Guarding entries on ``strategy.position_size`` makes it WORSE (measured
    2 -> 3 contracts): order commit is deferred and stop entries fill
    intrabar, so the guard reads a stale 0.
  * Therefore a hard position ceiling can only be enforced by the broker.

If this test ever fails after an upstream rebase, PyneCore has diverged from
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
        (100.10, 101.50, 100.00, 101.20),  # bar 2 - high crosses 100.50 AND 101.00
        (101.20, 101.30, 101.10, 101.20),  # bar 3 - tail; read the position here
    ]
    return [
        OHLCV(timestamp=base_ts + i * 60_000, open=o, high=h, low=lo, close=c,
              volume=100.0)
        for i, (o, h, lo, c) in enumerate(rows)
    ]


def __test_a_flip_leaves_the_old_sides_exit_working_and_it_opens_a_position__():
    """
    Correct-looking arithmetic would be: cover-stop buys 1 -> flat; entry stop
    buys 1 -> long 1, i.e. net +1.

    What BOTH engines actually do: the flip consumes the short and opens the
    long (+1), and the orphaned cover-stop opens a SECOND long (+1) -> +2.
    """
    from pynecore.core.script_runner import ScriptRunner

    try:
        runner = ScriptRunner(
            DATA_DIR / 'flip_exit_orphan.py', iter(_make_ohlcv()), _make_syminfo(),
        )
        positions = [dict(plot_data).get('pos')
                     for _candle, plot_data, _trades in runner.run_iter()]
    finally:
        sys.modules.pop('flip_exit_orphan', None)

    assert positions[1] == -1, (
        f"bar 1 should hold the 1-contract short, got {positions[1]}"
    )
    assert positions[-1] == 2, (
        "PINNED CROSS-ENGINE BEHAVIOUR: one bar crossing both a protective exit "
        "and an opposite-direction entry stop must leave position_size == 2 "
        "(TradingView does the same -- see this module's docstring and "
        f"docs/probes/flip_exit/). Got {positions[-1]}. If this changed, verify "
        "against TradingView before treating the new value as correct."
    )
