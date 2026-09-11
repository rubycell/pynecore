"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    NA, barstate, close, display, na, plot, script, strategy
)
from pynecore.types import PersistentSeries, Series

@script.strategy('LIVE2 fill+flatten', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, slippage=1, calc_on_every_tick=False, process_orders_on_close=False)
def main():
    traded: PersistentSeries[bool] = False
    closing: PersistentSeries[bool] = False

    if barstate.isrealtime and (not traded):
        strategy.entry("E", strategy.long, comment="MKT 1-lot")
        traded = True

    if traded and strategy.position_size != 0 and (not closing):
        strategy.cancel_all()
        strategy.close("E", comment="FLATTEN")
        closing = True

    DOUBLE_CHECK_PCT = -0.3

    _bsEntry = (strategy.opentrades.entry_price(strategy.opentrades - 1) if strategy.opentrades > 0 else 0.0)
    _bsSize = (strategy.opentrades.size(strategy.opentrades - 1) if strategy.opentrades > 0 else 0.0)
    _bsDir = (-1.0 if _bsSize < 0 else 1.0)
    _bsLive = strategy.position_size != 0 and (not na(_bsEntry)) and _bsEntry != 0.0
    openPnlPct = ((close - _bsEntry) / _bsEntry * 100.0 * _bsDir if _bsLive else 0.0)
    if strategy.position_size != 0 and openPnlPct < DOUBLE_CHECK_PCT:
        strategy.cancel_all()
        strategy.close_all(comment="DOUBLECHECK")

    plot(strategy.position_size, "pos", display=display.data_window)