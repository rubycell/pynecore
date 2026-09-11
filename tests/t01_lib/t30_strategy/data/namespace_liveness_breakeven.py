"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    bar_index, plot, script, strategy
)

@script.strategy('Breakeven trade', initial_capital=100000, default_qty_type=strategy.fixed, default_qty_value=1)
def main():
    if bar_index == 2:
        strategy.entry("BE", strategy.long, comment="be")
    if bar_index == 6:
        strategy.close("BE", comment="be_out")

    plot(strategy.eventrades, "eventrades")
    plot(strategy.wintrades, "wintrades")
    plot(strategy.losstrades, "losstrades")
    plot(strategy.closedtrades, "closedtrades")
    plot((strategy.closedtrades.profit(0) if strategy.closedtrades > 0 else 0.0), "profit")