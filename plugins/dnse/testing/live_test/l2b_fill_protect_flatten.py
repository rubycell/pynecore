"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    NA, barstate, close, display, format, high, na, plot, script, strategy, string
)
from pynecore.types import PersistentSeries, Series

@script.strategy('LIVE2b fill+bracket+flatten', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, slippage=1, calc_on_every_tick=False, process_orders_on_close=False)
def main():
    traded: PersistentSeries[bool] = False
    tpLevel: PersistentSeries[float] = na(float)
    slLevel: PersistentSeries[float] = na(float)

    higherHigh2 = (high if high > high[1] else high[1])

    if barstate.isrealtime and (not traded):
        tpLevel = higherHigh2 * 1.002
        slLevel = higherHigh2 * 0.998
        strategy.entry("E", strategy.long, stop=higherHigh2, comment="STOP 1-lot")
        strategy.exit("X", from_entry="E", limit=tpLevel, stop=slLevel, oca_name="brk", comment_profit="TP@" + string.tostring(tpLevel, format.mintick), comment_loss="SL@" + string.tostring(slLevel, format.mintick))
        traded = True

    DOUBLE_CHECK_PCT = -0.3

    _bsEntry = (strategy.opentrades.entry_price(strategy.opentrades - 1) if strategy.opentrades > 0 else 0.0)
    _bsSize = (strategy.opentrades.size(strategy.opentrades - 1) if strategy.opentrades > 0 else 0.0)
    _bsDir = (-1.0 if _bsSize < 0 else 1.0)
    _bsLive = strategy.position_size != 0 and (not na(_bsEntry)) and _bsEntry != 0.0
    openPnlPct = ((close - _bsEntry) / _bsEntry * 100.0 * _bsDir if _bsLive else 0.0)
    if strategy.position_size != 0 and openPnlPct < DOUBLE_CHECK_PCT:
        strategy.cancel_all()
        strategy.close_all(comment="DOUBLECHECK")

    if strategy.position_size != 0 and strategy.position_size[1] != 0:
        strategy.cancel_all()
        strategy.close("E", comment="FLATTEN")

    plot(strategy.position_size, "pos", display=display.data_window)
    plot(tpLevel, "tp", display=display.data_window)
    plot(slLevel, "sl", display=display.data_window)