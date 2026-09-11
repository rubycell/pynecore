"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    NA, close, display, format, na, open, plot, script, strategy, string
)

@script.strategy('LIVE3 OCA group place+cancel', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, calc_on_every_tick=False, process_orders_on_close=False)
def main():
    isGreen = close > open
    isRed = close < open
    upStop = close[1] * 1.05
    dnStop = close[1] * 0.95
    if isGreen:
        strategy.entry("Up", strategy.long, stop=upStop, oca_name="brk", oca_type=strategy.oca.cancel, comment="buy-stop@" + string.tostring(upStop, format.mintick))
        strategy.entry("Dn", strategy.short, stop=dnStop, oca_name="brk", oca_type=strategy.oca.cancel, comment="sell-stop@" + string.tostring(dnStop, format.mintick))

    if isRed:
        strategy.cancel("Up")
        strategy.cancel("Dn")

    PROTECT_PCT = -0.1

    _bsEntry = (strategy.opentrades.entry_price(strategy.opentrades - 1) if strategy.opentrades > 0 else 0.0)
    _bsSize = (strategy.opentrades.size(strategy.opentrades - 1) if strategy.opentrades > 0 else 0.0)
    _bsDir = (-1.0 if _bsSize < 0 else 1.0)
    _bsLive = strategy.position_size != 0 and (not na(_bsEntry)) and _bsEntry != 0.0
    openPnlPct = ((close - _bsEntry) / _bsEntry * 100.0 * _bsDir if _bsLive else 0.0)
    if strategy.position_size != 0 and openPnlPct < PROTECT_PCT:
        strategy.close_all(comment="PROTECT")

    plot(strategy.position_size, "pos", display=display.data_window)