"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    NA, barstate, close, display, high, input, log, na, plot, script, strategy
)
from pynecore.types import PersistentSeries, Series

@script.strategy('LIVE2c OCO leg execution', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, slippage=1, calc_on_every_tick=False, process_orders_on_close=False)
def main(
    tpPct=input.float(0.2, "TP % off trigger", minval=0.01, step=0.01),
    slPct=input.float(0.05, "SL % off trigger", minval=0.01, step=0.01)
):
    DOUBLE_CHECK_PCT = -0.25 - slPct

    tpLevel: PersistentSeries[float] = na(float)
    slLevel: PersistentSeries[float] = na(float)
    hasTraded: PersistentSeries[bool] = False

    higherHigh2 = (high if high > high[1] else high[1])

    if strategy.opentrades > 0 or strategy.closedtrades > 0:
        hasTraded = True

    if barstate.isrealtime and strategy.opentrades == 0 and (not hasTraded):
        tpLevel = higherHigh2 * (1.0 + tpPct / 100.0)
        slLevel = higherHigh2 * (1.0 - slPct / 100.0)
        strategy.entry("E", strategy.long, stop=higherHigh2, comment="STOP 1-lot")
        strategy.exit("X", from_entry="E", limit=tpLevel, stop=slLevel, oca_name="brk", comment_profit="TP", comment_loss="SL")
        log.info("L2C ENTRY dispatch trigger={0} tp={1} sl={2} tpPct={3} slPct={4}", higherHigh2, tpLevel, slLevel, tpPct, slPct)

    if strategy.opentrades > 0:
        strategy.exit("X", from_entry="E", limit=tpLevel, stop=slLevel, oca_name="brk", comment_profit="TP", comment_loss="SL")
        log.info("L2C EXIT reissue tp={0} sl={1} pos={2}", tpLevel, slLevel, strategy.position_size)

    if strategy.position_size == 0 and strategy.position_size[1] != 0 and strategy.closedtrades > 0:
        log.info("L2C CLOSED by={0} entry={1} exit={2} pnl={3}", strategy.closedtrades.exit_comment(strategy.closedtrades - 1), strategy.closedtrades.entry_price(strategy.closedtrades - 1), strategy.closedtrades.exit_price(strategy.closedtrades - 1), strategy.closedtrades.profit(strategy.closedtrades - 1))

    _bsEntry = (strategy.opentrades.entry_price(strategy.opentrades - 1) if strategy.opentrades > 0 else 0.0)
    _bsSize = (strategy.opentrades.size(strategy.opentrades - 1) if strategy.opentrades > 0 else 0.0)
    _bsDir = (-1.0 if _bsSize < 0 else 1.0)
    _bsLive = strategy.position_size != 0 and (not na(_bsEntry)) and _bsEntry != 0.0
    openPnlPct = ((close - _bsEntry) / _bsEntry * 100.0 * _bsDir if _bsLive else 0.0)
    if strategy.position_size != 0 and openPnlPct < DOUBLE_CHECK_PCT:
        strategy.cancel_all()
        strategy.close_all(comment="DOUBLECHECK")
        log.info("L2C BACKSTOP fired at {0} pct, threshold {1}", openPnlPct, DOUBLE_CHECK_PCT)

    plot(strategy.position_size, "pos", display=display.data_window)
    plot(tpLevel, "tp", display=display.data_window)
    plot(slLevel, "sl", display=display.data_window)