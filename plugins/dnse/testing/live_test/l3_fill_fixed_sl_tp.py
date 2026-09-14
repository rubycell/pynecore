"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    NA, barstate, close, display, format, input, na, plot, script, strategy, string
)
from pynecore.types import PersistentSeries, Series

@script.strategy('LIVE3 fill+fixed-SL+TP', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, slippage=1, calc_on_every_tick=False, process_orders_on_close=False)
def main(
    oracleMode=input.bool(False, "Oracle mode (OFFLINE backtest verification ONLY)"),
    slPct=input.float(0.2, "Stop-loss % below the fill", minval=0.01, step=0.01),
    tpPct=input.float(0.3, "Take-profit % above the fill", minval=0.01, step=0.01),
    maxHoldBars=input.int(3, "Live: flatten after N bars if SL/TP has not", minval=1),
    doubleCheckPct=input.float(-0.5, "Live: adverse-move double-check flatten %")
):
    traded: PersistentSeries[bool] = False
    slLevel: PersistentSeries[float] = na(float)
    tpLevel: PersistentSeries[float] = na(float)
    heldBars: PersistentSeries[int] = 0

    enterLive = barstate.isrealtime and (not traded)
    enterOracle = oracleMode and strategy.position_size == 0
    if enterLive or enterOracle:
        strategy.entry("E", strategy.long, comment="MKT 1-lot")
        traded = True
        slLevel = na
        tpLevel = na

    if strategy.position_size > 0 and na(slLevel):
        avg = strategy.position_avg_price
        slLevel = avg * (1.0 - slPct / 100.0)
        tpLevel = avg * (1.0 + tpPct / 100.0)
        strategy.exit("X", from_entry="E", stop=slLevel, limit=tpLevel, oca_name="brk", comment_profit="TP@" + string.tostring(tpLevel, format.mintick), comment_loss="SL@" + string.tostring(slLevel, format.mintick))

    heldBars = (heldBars + 1 if strategy.position_size != 0 else 0)
    if (not oracleMode) and strategy.position_size != 0 and heldBars >= maxHoldBars:
        strategy.cancel("X")
        strategy.close("E", comment="FLATTEN")

    _bsEntry = (strategy.opentrades.entry_price(strategy.opentrades - 1) if strategy.opentrades > 0 else 0.0)
    _bsLive = strategy.position_size != 0 and (not na(_bsEntry)) and _bsEntry != 0.0
    openPnlPct = ((close - _bsEntry) / _bsEntry * 100.0 if _bsLive else 0.0)
    if (not oracleMode) and strategy.position_size != 0 and openPnlPct < doubleCheckPct:
        strategy.cancel("X")
        strategy.close("E", comment="DOUBLECHECK")

    plot(strategy.position_size, "pos", display=display.data_window)
    plot(slLevel, "sl", display=display.data_window)
    plot(tpLevel, "tp", display=display.data_window)