"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    bar_index, plot, script, strategy
)

@script.strategy('oca.cancel', initial_capital=100000, default_qty_type=strategy.fixed, default_qty_value=2)
def main():
    if bar_index == 2:
        strategy.entry("L", strategy.long)
    if bar_index == 3:
        strategy.order("X1", strategy.short, qty=1, limit=110.0, oca_name="grp", oca_type=strategy.oca.cancel)
        strategy.order("X2", strategy.short, qty=2, stop=95.0, oca_name="grp", oca_type=strategy.oca.cancel)

    plot(strategy.position_size, "pos")
    plot(strategy.closedtrades, "closed")
    plot(strategy.opentrades, "open")