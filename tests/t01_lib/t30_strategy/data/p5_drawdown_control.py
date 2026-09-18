"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    bar_index, plot, script, strategy
)

@script.strategy('P5 drawdown control', initial_capital=100000, default_qty_type=strategy.fixed, default_qty_value=1, commission_type=strategy.commission.percent, commission_value=0)
def main():
    if bar_index == 2:
        strategy.entry("L", strategy.long)

    plot(strategy.position_size, "position_size")
    plot(strategy.openprofit, "openprofit")