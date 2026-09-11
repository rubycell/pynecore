"""
Characterization pin for the closed-trade list limit -- upstream issue #84.

These assertions describe behaviour that is WRONG. They exist so the defect cannot
drift unnoticed and so the day it is fixed is loud rather than silent: each one
says both what PyneCore does today and what TradingView specifies, and fails when
either changes.

Once more trades close than the list retains, three things go wrong together:

1. ``strategy.closedtrades`` stops counting. The reference defines it as the number
   of trades "closed for the whole trading range", but it reports only what the
   list still holds. The win/loss/breakeven counters keep counting, so their sum is
   the control: it says how many really closed.
2. ``strategy.closedtrades.first_index`` stays 0. The reference is explicit that
   this is exactly when it stops being zero -- "If more trades than the allowed
   limit have been closed, the oldest trades are removed, and this number is the
   index of the oldest remaining trade."
3. The accessors silently address a different trade. ``entry_bar_index(0)`` no
   longer describes the run's first trade but whatever survived eviction, so every
   ``closedtrades.*(n)`` call is offset and the usual
   ``for i = 0 to strategy.closedtrades - 1`` walk reads the wrong trades while
   stopping short. Nothing raises.

The state is reachable from an ordinary backtest: the run below completes with no
order-cap error, and 20,000 bars is unremarkable intraday (~3.5 months of 15-minute
bars). This is the part that makes it worth pinning rather than dismissing.

The probe is authored in Pine (``data/closed_trade_list_overflow.pine``); the ``.py``
beside it is pine2pyne output and must not be hand-edited.
"""
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'
SCRIPT_NAME = 'closed_trade_list_overflow'

#: Bars to run. Each bar books one closed trade, so this must comfortably exceed
#: the retention limit for the overflow to happen at all.
BAR_COUNT = 20_000

#: What the run actually produces today. Kept as named constants so a change in
#: any of them reads as a deliberate update rather than a silent edit.
RETAINED = 9_000          # what strategy.closedtrades reports (the retention limit)
REALLY_CLOSED = 19_998    # what the win/loss/breakeven counters add up to
OLDEST_REMAINING = REALLY_CLOSED - RETAINED   # 10998 -- what first_index should be


def _make_syminfo():
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period='D', type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=0.001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


def _make_ohlcv():
    """A cheap oscillating series -- the prices do not matter, the churn does."""
    from pynecore.types.ohlcv import OHLCV

    base_ts = 1_735_689_600_000
    return [OHLCV(timestamp=base_ts + i * 86_400_000,
                  open=100.0 + (i % 7), high=108.0, low=99.0,
                  close=100.0 + ((i + 1) % 7), volume=10.0)
            for i in range(BAR_COUNT)]


def _run() -> dict:
    from pynecore.core.script_runner import ScriptRunner

    try:
        runner = ScriptRunner(DATA_DIR / f'{SCRIPT_NAME}.py', iter(_make_ohlcv()),
                              _make_syminfo())
        plots = [dict(p) for _candle, p, _trades in runner.run_iter()]
        assert plots, "the overflow probe produced no bars"
        return plots[-1]
    finally:
        sys.modules.pop(SCRIPT_NAME, None)


def __test_the_run_overflows_the_list_without_an_order_cap__():
    """The precondition: more trades close than the list keeps, and the run survives.

    If this ever stops holding -- an order cap starts firing, or the retention
    limit changes -- the three pins below are measuring nothing, so it is asserted
    first rather than assumed.
    """
    last = _run()
    assert last['real_total'] == REALLY_CLOSED, (
        f"expected {REALLY_CLOSED} trades to actually close, got {last['real_total']:.0f} "
        "-- the scenario no longer overflows the list, so the pins below are vacuous")
    assert last['real_total'] > RETAINED, "no overflow occurred"


def __test_closedtrades_stops_counting_at_the_retention_limit__():
    """BUG (#84): reports the retained count, not the whole trading range."""
    last = _run()
    assert last['closedtrades'] == RETAINED, (
        f"strategy.closedtrades reported {last['closedtrades']:.0f}, pinned at "
        f"{RETAINED}. If it now reports {REALLY_CLOSED} the bug is FIXED -- delete "
        "this pin.")
    assert last['closedtrades'] != last['real_total'], (
        "strategy.closedtrades now agrees with the real total -- #84 is fixed, "
        "remove this test")


def __test_first_index_stays_zero_after_eviction__():
    """BUG (#84): should become the index of the oldest remaining trade."""
    last = _run()
    assert last['first_index'] == 0, (
        f"first_index reported {last['first_index']:.0f}, pinned at 0. TradingView "
        f"specifies {OLDEST_REMAINING} here (trades closed minus trades retained); "
        "a non-zero reading means #84 is fixed -- delete this pin.")


def __test_accessors_silently_address_a_post_eviction_trade__():
    """BUG (#84), the damaging one: index 0 is no longer the run's first trade.

    The first trade entered at bar 1. After eviction, ``entry_bar_index(0)``
    describes a trade that entered roughly ``OLDEST_REMAINING`` bars later, so any
    loop over ``0 .. closedtrades - 1`` silently walks the wrong trades.
    """
    last = _run()
    trade0_bar = last['trade0_entry_bar']
    assert trade0_bar > OLDEST_REMAINING, (
        f"closedtrades.entry_bar_index(0) returned {trade0_bar:.0f}; pinned as "
        f"'well past bar {OLDEST_REMAINING}', i.e. NOT the run's first trade. A "
        "small value would mean the accessors are anchored correctly again and "
        "#84 is fixed -- delete this pin.")
    assert trade0_bar > 1, (
        "index 0 now looks like the run's first trade -- #84 is fixed, remove this test")
