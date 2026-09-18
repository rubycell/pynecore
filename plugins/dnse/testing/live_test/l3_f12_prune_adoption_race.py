"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    NA, bar_index, barstate, close, display, high, input, log, na, plot, script, strategy
)
from pynecore.types import PersistentSeries, Series

@script.strategy('LIVE-L3-F12 prune adoption race', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, slippage=1, calc_on_every_tick=False, process_orders_on_close=False)
def main(
    crossOffset=input.float(0.2, "entry trigger, index points above the close", minval=0.1, step=0.1),
    attemptCap=input.int(6, "give up after this many race attempts", minval=1, maxval=30),
    tpPct=input.float(0.5, "TP % off trigger", minval=0.01, step=0.01),
    slPct=input.float(0.25, "SL % off trigger", minval=0.01, step=0.01)
):
    DOUBLE_CHECK_PCT = -0.25 - slPct

    tpLevel: PersistentSeries[float] = na(float)
    slLevel: PersistentSeries[float] = na(float)
    trigger: PersistentSeries[float] = na(float)
    hasTraded: PersistentSeries[bool] = False
    attempts: PersistentSeries[int] = 0
    races: PersistentSeries[int] = 0

    placedBar: PersistentSeries[int] = -1

    if strategy.opentrades > 0 or strategy.closedtrades > 0:
        hasTraded = True

    if barstate.isrealtime and strategy.opentrades == 0 and (not hasTraded) and placedBar < 0 and attempts < attemptCap:
        attempts += 1
        trigger = close + crossOffset
        tpLevel = trigger * (1.0 + tpPct / 100.0)
        slLevel = trigger * (1.0 - slPct / 100.0)
        placedBar = bar_index
        strategy.entry("E", strategy.long, stop=trigger, comment="STOP 1-lot")
        strategy.exit("X", from_entry="E", limit=tpLevel, stop=slLevel, oca_name="f12", comment_profit="TP", comment_loss="SL")
        log.info("F12 PLACE attempt={0} of {1} trigger={2} close={3} tp={4} sl={5}", attempts, attemptCap, trigger, close, tpLevel, slLevel)

    if placedBar >= 0 and bar_index == placedBar + 1:
        placedBar = -1
        strategy.cancel("E")
        log.info("F12 CANCEL attempt={0} trigger={1} barHigh={2} crossed={3} posNow={4}", attempts, trigger, high, high >= trigger, strategy.position_size)

    if strategy.opentrades > 0 and strategy.opentrades[1] == 0:
        races += 1
        log.info("F12 RACED — venue activated before the cancel landed. entry={0} pos={1} attempts={2}", strategy.opentrades.entry_price(strategy.opentrades - 1), strategy.position_size, attempts)

    if barstate.isrealtime and attempts >= attemptCap and races == 0 and strategy.position_size == 0:
        log.info("F12 NO RACE after {0} attempts — the condition under test was never reached", attempts)

    if strategy.position_size == 0 and strategy.position_size[1] != 0 and strategy.closedtrades > 0:
        log.info("F12 CLOSED by={0} entry={1} exit={2} pnl={3}", strategy.closedtrades.exit_comment(strategy.closedtrades - 1), strategy.closedtrades.entry_price(strategy.closedtrades - 1), strategy.closedtrades.exit_price(strategy.closedtrades - 1), strategy.closedtrades.profit(strategy.closedtrades - 1))

    if strategy.position_size != 0 and strategy.position_size[1] != 0:
        strategy.cancel_all()
        strategy.close("E", comment="FLATTEN")
        log.info("F12 FLATTEN at two candles held, pos={0}", strategy.position_size)

    _bsEntry = (strategy.opentrades.entry_price(strategy.opentrades - 1) if strategy.opentrades > 0 else 0.0)
    _bsSize = (strategy.opentrades.size(strategy.opentrades - 1) if strategy.opentrades > 0 else 0.0)
    _bsDir = (-1.0 if _bsSize < 0 else 1.0)
    _bsLive = strategy.position_size != 0 and (not na(_bsEntry)) and _bsEntry != 0.0
    openPnlPct = ((close - _bsEntry) / _bsEntry * 100.0 * _bsDir if _bsLive else 0.0)
    if strategy.position_size != 0 and openPnlPct < DOUBLE_CHECK_PCT:
        strategy.cancel_all()
        strategy.close_all(comment="DOUBLECHECK")
        log.info("F12 BACKSTOP fired at {0} pct, threshold {1}", openPnlPct, DOUBLE_CHECK_PCT)

    plot(strategy.position_size, "pos", display=display.data_window)
    plot(trigger, "trigger", display=display.data_window)
    plot(slLevel, "sl", display=display.data_window)