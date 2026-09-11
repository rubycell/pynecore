"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    bar_index, plot, script, strategy
)

@script.strategy('closed trade list overflow', initial_capital=10000000, default_qty_type=strategy.fixed, default_qty_value=1)
def main():
    if bar_index % 2 == 0:
        strategy.entry("L", strategy.long)
    else:
        strategy.entry("S", strategy.short)

    plot(strategy.closedtrades, "closedtrades")
    plot(strategy.wintrades + strategy.losstrades + strategy.eventrades, "real_total")
    plot(strategy.closedtrades.first_index, "first_index")
    plot((strategy.closedtrades.entry_bar_index(0) if strategy.closedtrades > 0 else -1.0), "trade0_entry_bar")