"""
@pyne

Discriminating CONTROL for ``test_130``: identical to ``flip_exit_orphan``
except the entry stop sits BELOW the cover stop, so the rising bar crosses the
ENTRY level first and the protective exit is the order left orphaned.

If the doubling were caused by an orphaned exit firing as an opening order,
this script would also end at 2. It ends at 1 — see the test.
"""
from pynecore.lib import script, strategy, bar_index, plot


@script.strategy(
    "Flip Exit Entry First",
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
        # Cover ABOVE the entry stop: bar 2 rises through 100.20 before 100.50,
        # so the entry fills first and the cover is the orphan.
        strategy.exit('Cover', 'S', stop=100.50, qty=1)
        strategy.entry('L', strategy.long, qty=1, stop=100.20)

    plot(strategy.position_size, "pos")
