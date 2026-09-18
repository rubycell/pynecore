"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    bar_index, close, input, open, plot, script, strategy
)

@script.strategy('R10 parity vehicle', overlay=True, default_qty_type=strategy.fixed, default_qty_value=1, pyramiding=0, initial_capital=10000000000, calc_on_every_tick=False)
def main(
    holdBars=input.int(3, "Bars to hold")
):
    if strategy.opentrades == 0 and close > open:
        strategy.entry("E", strategy.long, comment="parity entry")

    if strategy.opentrades == 1 and bar_index - strategy.opentrades.entry_bar_index(0) >= holdBars:
        strategy.close("E", comment="parity exit")

    plot(close, "close")