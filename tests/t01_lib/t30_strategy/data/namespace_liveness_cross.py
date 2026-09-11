"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    NA, bar_index, close, na, plot, script, strategy, string, ta
)

try:
    from pynecore.core import _var_cache as _vcm
except ImportError:
    _vcm = None

__var_deps__ = {0: frozenset(), 1: frozenset()}
__num_cache_slots__ = 2

@script.strategy('Strategy namespace liveness (SMA cross)', initial_capital=100000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=50, margin_short=50)
def main():
    _vc = _vcm._data if _vcm else None
    _vb = _vcm._build if _vcm else None
    fast = _vc[0][int(bar_index)] if _vc is not None and _vc[0] is not None else ta.sma(close, 5)
    if _vb is not None and _vb[0] is not None: _vb[0].append(fast)
    slow = _vc[1][int(bar_index)] if _vc is not None and _vc[1] is not None else ta.sma(close, 20)
    if _vb is not None and _vb[1] is not None: _vb[1].append(slow)
    if ta.crossover(fast, slow):
        strategy.entry("L", strategy.long, comment="up")
    if ta.crossunder(fast, slow):
        strategy.entry("S", strategy.short, comment="dn")

    plot((na if na(strategy.account_currency) else string.length(strategy.account_currency)), "s__account_currency")
    plot(strategy.avg_losing_trade, "s__avg_losing_trade")
    plot(strategy.avg_losing_trade_percent, "s__avg_losing_trade_percent")
    plot(strategy.avg_trade, "s__avg_trade")
    plot(strategy.avg_trade_percent, "s__avg_trade_percent")
    plot(strategy.avg_winning_trade, "s__avg_winning_trade")
    plot(strategy.avg_winning_trade_percent, "s__avg_winning_trade_percent")
    plot(strategy.equity, "s__equity")
    plot(strategy.eventrades, "s__eventrades")
    plot(strategy.grossloss, "s__grossloss")
    plot(strategy.grossloss_percent, "s__grossloss_percent")
    plot(strategy.grossprofit, "s__grossprofit")
    plot(strategy.grossprofit_percent, "s__grossprofit_percent")
    plot(strategy.initial_capital, "s__initial_capital")
    plot(strategy.losstrades, "s__losstrades")
    plot(strategy.margin_liquidation_price, "s__margin_liquidation_price")
    plot(strategy.max_contracts_held_all, "s__max_contracts_held_all")
    plot(strategy.max_contracts_held_long, "s__max_contracts_held_long")
    plot(strategy.max_contracts_held_short, "s__max_contracts_held_short")
    plot(strategy.max_drawdown, "s__max_drawdown")
    plot(strategy.max_drawdown_percent, "s__max_drawdown_percent")
    plot(strategy.max_runup, "s__max_runup")
    plot(strategy.max_runup_percent, "s__max_runup_percent")
    plot(strategy.netprofit, "s__netprofit")
    plot(strategy.netprofit_percent, "s__netprofit_percent")
    plot(strategy.openprofit, "s__openprofit")
    plot(strategy.openprofit_percent, "s__openprofit_percent")
    plot(strategy.position_avg_price, "s__position_avg_price")
    plot((na if na(strategy.position_entry_name) else string.length(strategy.position_entry_name)), "s__position_entry_name")
    plot(strategy.position_size, "s__position_size")
    plot(strategy.wintrades, "s__wintrades")

    hasOpen = strategy.opentrades > 0
    plot(strategy.opentrades, "o__opentrades")
    plot((strategy.opentrades.capital_held if hasOpen else na), "o__capital_held")
    plot((strategy.opentrades.commission(0) if hasOpen else na), "o__commission")
    plot((strategy.opentrades.entry_bar_index(0) if hasOpen else na), "o__entry_bar_index")
    plot((string.length(strategy.opentrades.entry_comment(0)) if hasOpen and (not na(strategy.opentrades.entry_comment(0))) else na), "o__entry_comment")
    plot((string.length(strategy.opentrades.entry_id(0)) if hasOpen and (not na(strategy.opentrades.entry_id(0))) else na), "o__entry_id")
    plot((strategy.opentrades.entry_price(0) if hasOpen else na), "o__entry_price")
    plot((strategy.opentrades.entry_time(0) if hasOpen else na), "o__entry_time")
    plot((strategy.opentrades.max_drawdown(0) if hasOpen else na), "o__max_drawdown")
    plot((strategy.opentrades.max_drawdown_percent(0) if hasOpen else na), "o__max_drawdown_percent")
    plot((strategy.opentrades.max_runup(0) if hasOpen else na), "o__max_runup")
    plot((strategy.opentrades.max_runup_percent(0) if hasOpen else na), "o__max_runup_percent")
    plot((strategy.opentrades.profit(0) if hasOpen else na), "o__profit")
    plot((strategy.opentrades.profit_percent(0) if hasOpen else na), "o__profit_percent")
    plot((strategy.opentrades.size(0) if hasOpen else na), "o__size")

    hasClosed = strategy.closedtrades > 0
    plot(strategy.closedtrades, "c__closedtrades")
    plot(strategy.closedtrades.first_index, "c__first_index")
    plot((strategy.closedtrades.commission(0) if hasClosed else na), "c__commission")
    plot((strategy.closedtrades.entry_bar_index(0) if hasClosed else na), "c__entry_bar_index")
    plot((string.length(strategy.closedtrades.entry_comment(0)) if hasClosed and (not na(strategy.closedtrades.entry_comment(0))) else na), "c__entry_comment")
    plot((string.length(strategy.closedtrades.entry_id(0)) if hasClosed and (not na(strategy.closedtrades.entry_id(0))) else na), "c__entry_id")
    plot((strategy.closedtrades.entry_price(0) if hasClosed else na), "c__entry_price")
    plot((strategy.closedtrades.entry_time(0) if hasClosed else na), "c__entry_time")
    plot((strategy.closedtrades.exit_bar_index(0) if hasClosed else na), "c__exit_bar_index")
    plot((string.length(strategy.closedtrades.exit_comment(0)) if hasClosed and (not na(strategy.closedtrades.exit_comment(0))) else na), "c__exit_comment")
    plot((string.length(strategy.closedtrades.exit_id(0)) if hasClosed and (not na(strategy.closedtrades.exit_id(0))) else na), "c__exit_id")
    plot((strategy.closedtrades.exit_price(0) if hasClosed else na), "c__exit_price")
    plot((strategy.closedtrades.exit_time(0) if hasClosed else na), "c__exit_time")
    plot((strategy.closedtrades.max_drawdown(0) if hasClosed else na), "c__max_drawdown")
    plot((strategy.closedtrades.max_drawdown_percent(0) if hasClosed else na), "c__max_drawdown_percent")
    plot((strategy.closedtrades.max_runup(0) if hasClosed else na), "c__max_runup")
    plot((strategy.closedtrades.max_runup_percent(0) if hasClosed else na), "c__max_runup_percent")
    plot((strategy.closedtrades.profit(0) if hasClosed else na), "c__profit")
    plot((strategy.closedtrades.profit_percent(0) if hasClosed else na), "c__profit_percent")
    plot((strategy.closedtrades.size(0) if hasClosed else na), "c__size")