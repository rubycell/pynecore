"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    NA, barstate, display, format, high, input, na, plot, script, strategy, string
)
from pynecore.types import PersistentSeries, Series

@script.strategy('LIVE2b entry-update probe', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, slippage=1, calc_on_every_tick=False, process_orders_on_close=False)
def main(
    padPct=input.float(0.0, "No-fill padding % above breakout (0 = normal fill test)", minval=0.0, step=0.5)
):
    everFilled: PersistentSeries[bool] = False
    tpLevel: PersistentSeries[float] = na(float)
    slLevel: PersistentSeries[float] = na(float)

    higherHigh2 = (high if high > high[1] else high[1])
    entryStop = higherHigh2 * (1.0 + padPct / 100.0)
    if strategy.position_size != 0:
        everFilled = True

    if barstate.isrealtime and (not everFilled):
        strategy.entry("E", strategy.long, stop=entryStop, comment="STOP pad")

    if strategy.position_size > 0 and na(tpLevel):
        avg = strategy.position_avg_price
        tpLevel = avg * 1.002
        slLevel = avg * 0.998
        strategy.exit("X", from_entry="E", limit=tpLevel, stop=slLevel, oca_name="brk", comment_profit="TP@" + string.tostring(tpLevel, format.mintick), comment_loss="SL@" + string.tostring(slLevel, format.mintick))

    if (not na(tpLevel)) and strategy.position_size != 0 and strategy.position_size[1] != 0:
        strategy.cancel_all()
        strategy.close("E", comment="FLATTEN")

    plot(strategy.position_size, "pos", display=display.data_window)
    plot(entryStop, "entryStop", display=display.data_window)
    plot(tpLevel, "tp", display=display.data_window)
    plot(slLevel, "sl", display=display.data_window)