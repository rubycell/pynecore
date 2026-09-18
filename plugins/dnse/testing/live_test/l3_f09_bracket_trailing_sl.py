"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    NA, bar_index, barstate, close, display, high, input, log, na, plot, script, strategy
)
from pynecore.types import PersistentSeries, Series

@script.strategy('LIVE-L3-F09 bracket trailing SL', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, slippage=1, calc_on_every_tick=False, process_orders_on_close=False)
def main(
    tpPct=input.float(0.5, "TP % off trigger", minval=0.01, step=0.01),
    slPct=input.float(0.25, "SL % off trigger", minval=0.01, step=0.01),
    trailStep=input.float(0.8, "SL trail step, index points behind the high", minval=0.1, step=0.1),
    trailBars=input.int(2, "bars to trail after the fill", minval=1, maxval=10),
    holdBars=input.int(6, "flatten after this many bars held", minval=2, maxval=60)
):
    DOUBLE_CHECK_PCT = -0.25 - slPct

    tpLevel: PersistentSeries[float] = na(float)
    slBelieved: PersistentSeries[float] = na(float)
    hasTraded: PersistentSeries[bool] = False
    fillBar: PersistentSeries[int] = na(int)

    higherHigh2 = (high if high > high[1] else high[1])

    if strategy.opentrades > 0 or strategy.closedtrades > 0:
        hasTraded = True

    if barstate.isrealtime and strategy.opentrades == 0 and (not hasTraded):
        tpLevel = higherHigh2 * (1.0 + tpPct / 100.0)
        slBelieved = higherHigh2 * (1.0 - slPct / 100.0)
        strategy.entry("E", strategy.long, stop=higherHigh2, comment="STOP 1-lot")
        strategy.exit("X", from_entry="E", limit=tpLevel, stop=slBelieved, oca_name="f09", comment_profit="TP", comment_loss="SL")
        log.info("F09 ENTRY dispatch trigger={0} tp={1} sl={2} slPct={3} tpPct={4}", higherHigh2, tpLevel, slBelieved, slPct, tpPct)

    if strategy.opentrades > 0 and strategy.opentrades[1] == 0:
        fillBar = bar_index
        log.info("F09 FILLED at bar {0} entry={1} believedSL={2}", bar_index, strategy.opentrades.entry_price(strategy.opentrades - 1), slBelieved)

    barsHeld = (bar_index - fillBar if not na(fillBar) else -1)

    if strategy.opentrades > 0 and barsHeld >= 1 and barsHeld <= trailBars:
        trailCandidate = high - trailStep
        slBelieved = (trailCandidate if trailCandidate > slBelieved else slBelieved)
        strategy.exit("X", from_entry="E", limit=tpLevel, stop=slBelieved, oca_name="f09", comment_profit="TP", comment_loss="SL")
        log.info("F09 TRAIL barsHeld={0} high={1} step={2} believedSL={3} UNCONFIRMED at venue", barsHeld, high, trailStep, slBelieved)

    if strategy.position_size == 0 and strategy.position_size[1] != 0 and strategy.closedtrades > 0:
        log.info("F09 CLOSED by={0} entry={1} exit={2} pnl={3}", strategy.closedtrades.exit_comment(strategy.closedtrades - 1), strategy.closedtrades.entry_price(strategy.closedtrades - 1), strategy.closedtrades.exit_price(strategy.closedtrades - 1), strategy.closedtrades.profit(strategy.closedtrades - 1))

    if strategy.position_size != 0 and barsHeld >= holdBars:
        strategy.cancel_all()
        strategy.close("E", comment="HOLDCAP")
        log.info("F09 HOLDCAP flatten at barsHeld={0}", barsHeld)

    _bsEntry = (strategy.opentrades.entry_price(strategy.opentrades - 1) if strategy.opentrades > 0 else 0.0)
    _bsSize = (strategy.opentrades.size(strategy.opentrades - 1) if strategy.opentrades > 0 else 0.0)
    _bsDir = (-1.0 if _bsSize < 0 else 1.0)
    _bsLive = strategy.position_size != 0 and (not na(_bsEntry)) and _bsEntry != 0.0
    openPnlPct = ((close - _bsEntry) / _bsEntry * 100.0 * _bsDir if _bsLive else 0.0)
    if strategy.position_size != 0 and openPnlPct < DOUBLE_CHECK_PCT:
        strategy.cancel_all()
        strategy.close_all(comment="DOUBLECHECK")
        log.info("F09 BACKSTOP fired at {0} pct, threshold {1}", openPnlPct, DOUBLE_CHECK_PCT)

    plot(strategy.position_size, "pos", display=display.data_window)
    plot(tpLevel, "tp", display=display.data_window)
    plot(slBelieved, "slBelieved", display=display.data_window)