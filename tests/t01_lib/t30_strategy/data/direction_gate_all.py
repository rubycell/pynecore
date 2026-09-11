"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    bar_index, plot, script, strategy
)

@script.strategy('direction.all gate', initial_capital=100000, default_qty_type=strategy.fixed, default_qty_value=1)
def main():
    strategy.risk.allow_entry_in(strategy.direction.all)
    if bar_index == 2:
        strategy.entry("L", strategy.long)
    if bar_index == 5:
        strategy.close("L")
    if bar_index == 8:
        strategy.entry("S", strategy.short)
    if bar_index == 11:
        strategy.close("S")

    plot(strategy.position_size, "pos")
    plot(strategy.closedtrades, "closed")
    plot(strategy.wintrades + strategy.losstrades + strategy.eventrades, "booked")