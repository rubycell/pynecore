"""
@pyne

Script under test for ``test_130_flip_does_not_cancel_an_armed_exit``.

Goes short on bar 0; on bar 1 arms the short's protective cover-stop AND an
opposite-direction (long) entry stop just above it, so ONE later bar can cross
both levels. See the test module for the full argument and the TradingView
cross-check.
"""
from pynecore.lib import script, strategy, bar_index, plot


@script.strategy(
    "Flip Exit Orphan",
    overlay=False,
    initial_capital=1_000_000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=0,
    slippage=0,
)
def main():
    if bar_index == 0:
        strategy.entry('S', strategy.short, qty=1)

    if bar_index == 1 and strategy.position_size < 0:
        strategy.exit('Cover', 'S', stop=100.50, qty=1)
        strategy.entry('L', strategy.long, qty=1, stop=101.00)

    plot(strategy.position_size, "pos")
