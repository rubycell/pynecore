"""
Fix pins for the closed-trade list limit -- upstream issue #84, fixed in 6.9.3.

Until the 6.9.4 landing (card #151) this file was a characterization pin of the
DEFECT: three assertions described what PyneCore did wrong past the retention
limit, so the fix would be loud rather than silent. It was -- on the pure 6.9.4
replay exactly these three went red. They are now inverted into pins of the
behaviour TradingView specifies, so a regression of #84 is loud in its turn:

1. ``strategy.closedtrades`` counts the trades "closed for the whole trading
   range", not only what the list still holds. The win/loss/breakeven counters
   are the control: their sum says how many really closed.
2. ``strategy.closedtrades.first_index`` is "the index of the oldest remaining
   trade" once the oldest trades are evicted.
3. The accessors stay anchored to the whole range: index 0 names an evicted
   trade and reads ``na``; ``entry_bar_index(first_index)`` describes the oldest
   RETAINED trade. Each bar books one trade and trade ``k`` enters at bar
   ``k + 1`` (the entry placed on bar ``k`` fills at the next bar's open), so
   the oldest retained trade entered at bar ``OLDEST_REMAINING + 1``.

The state is reachable from an ordinary backtest: the run completes with no
order-cap error, and 20,000 bars is unremarkable intraday (~3.5 months of
15-minute bars). The precondition test is kept so the pins can never pass
vacuously below the limit.

The probe is authored in Pine (``data/closed_trade_list_overflow.pine``); the
transpiled module beside it is pine2pyne output and must not be hand-edited.
"""
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'
SCRIPT_NAME = 'closed_trade_list_overflow'

#: Bars to run. Each bar books one closed trade, so this must comfortably exceed
#: the retention limit for the overflow to happen at all.
BAR_COUNT = 20_000

#: Named constants so a change in any of them reads as a deliberate update
#: rather than a silent edit.
RETAINED = 9_000          # the retention limit (trades the list still holds)
REALLY_CLOSED = 19_998    # what the win/loss/breakeven counters add up to
OLDEST_REMAINING = REALLY_CLOSED - RETAINED   # 10998 -- first_index past the limit
OLDEST_ENTRY_BAR = OLDEST_REMAINING + 1       # 10999 -- trade k enters at bar k + 1


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


def __test_closedtrades_counts_the_whole_trading_range__():
    """#84 fixed: the count is the whole trading range, not the retained list."""
    last = _run()
    assert last['closedtrades'] == REALLY_CLOSED, (
        f"strategy.closedtrades reported {last['closedtrades']:.0f}, expected "
        f"{REALLY_CLOSED}. A reading of {RETAINED} is the #84 defect back again "
        "(the count stopped at the retention limit)")
    assert last['closedtrades'] == last['real_total'], (
        "strategy.closedtrades disagrees with the win+loss+even control")


def __test_first_index_is_the_oldest_remaining_trade__():
    """#84 fixed: first_index is the index of the oldest trade still retained."""
    last = _run()
    assert last['first_index'] == OLDEST_REMAINING, (
        f"first_index reported {last['first_index']:.0f}, expected {OLDEST_REMAINING} "
        "(trades closed minus trades retained). A constant 0 is the #84 defect")


def __test_accessors_stay_anchored_to_the_whole_range__():
    """#84 fixed: index 0 is evicted (na) and first_index names the oldest kept trade."""
    import math
    last = _run()
    assert math.isnan(last['trade0_entry_bar']), (
        f"closedtrades.entry_bar_index(0) returned {last['trade0_entry_bar']}; trade 0 "
        "was evicted, so it must read na -- a number here means the accessors are "
        "re-anchored to the retained list (the #84 defect)")
    assert last['oldest_entry_bar'] == OLDEST_ENTRY_BAR, (
        f"entry_bar_index(first_index) returned {last['oldest_entry_bar']:.0f}, expected "
        f"{OLDEST_ENTRY_BAR}: the oldest retained trade (index {OLDEST_REMAINING}) "
        "entered one bar after its index")
