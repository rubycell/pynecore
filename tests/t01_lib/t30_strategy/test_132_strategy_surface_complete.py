"""
Surface tests: PyneCore implements the WHOLE ``strategy`` namespace TradingView
documents -- all 96 members, not just the data-bearing ones.

The liveness suite (``test_131_strategy_namespace_liveness``) proves the 66 data
members carry values. That leaves 30 members it cannot speak for: the constants,
the order-placing functions, and the four sub-namespaces. A member that vanished
or was renamed in any of those would break real scripts while every existing test
stayed green, because no test names them all in one place.

The member list below is TradingView's, not ours. It was taken from the v6
reference and verified against the LIVE page on 2026-09-09 by enumerating the
rendered anchors: 96 members, exactly matching this list.

One wrinkle recorded so the next reader does not re-chase it: the live page also
carries a link to ``strategy.risk_allow_entry_in()``, which is a typo in
TradingView's own "See also" -- the anchor it points at does not exist on the
page, and the real member is ``strategy.risk.allow_entry_in()``. It is NOT a 97th
member.
"""
import pytest

from pynecore.lib import strategy

#: Pine constants: plain values a script passes to ``strategy()`` or ``entry()``.
CONSTANTS = (
    'cash', 'fixed', 'long',
    'percent_of_equity', 'short',
)

#: Order-placing and helper functions on the namespace root.
FUNCTIONS = (
    'cancel', 'cancel_all', 'close',
    'close_all', 'convert_to_account', 'convert_to_symbol',
    'default_entry_qty', 'entry', 'exit',
    'order',
)

#: The four sub-namespaces: commission modes, direction and OCA constants, and
#: the ``risk.*`` rule setters.
SUBNAMESPACE = (
    'commission.cash_per_contract', 'commission.cash_per_order',
    'commission.percent', 'direction.all',
    'direction.long', 'direction.short',
    'oca.cancel', 'oca.none',
    'oca.reduce', 'risk.allow_entry_in',
    'risk.max_cons_loss_days', 'risk.max_drawdown',
    'risk.max_intraday_filled_orders', 'risk.max_intraday_loss',
    'risk.max_position_size',
)

#: The data-bearing members. Their VALUES are covered by the liveness suite; here
#: they only have to exist, so the surface count is complete in one place.
DATA = (
    'account_currency', 'avg_losing_trade', 'avg_losing_trade_percent',
    'avg_trade', 'avg_trade_percent', 'avg_winning_trade',
    'avg_winning_trade_percent', 'closedtrades', 'closedtrades.commission',
    'closedtrades.entry_bar_index', 'closedtrades.entry_comment', 'closedtrades.entry_id',
    'closedtrades.entry_price', 'closedtrades.entry_time', 'closedtrades.exit_bar_index',
    'closedtrades.exit_comment', 'closedtrades.exit_id', 'closedtrades.exit_price',
    'closedtrades.exit_time', 'closedtrades.first_index', 'closedtrades.max_drawdown',
    'closedtrades.max_drawdown_percent', 'closedtrades.max_runup', 'closedtrades.max_runup_percent',
    'closedtrades.profit', 'closedtrades.profit_percent', 'closedtrades.size',
    'equity', 'eventrades', 'grossloss',
    'grossloss_percent', 'grossprofit', 'grossprofit_percent',
    'initial_capital', 'losstrades', 'margin_liquidation_price',
    'max_contracts_held_all', 'max_contracts_held_long', 'max_contracts_held_short',
    'max_drawdown', 'max_drawdown_percent', 'max_runup',
    'max_runup_percent', 'netprofit', 'netprofit_percent',
    'openprofit', 'openprofit_percent', 'opentrades',
    'opentrades.capital_held', 'opentrades.commission', 'opentrades.entry_bar_index',
    'opentrades.entry_comment', 'opentrades.entry_id', 'opentrades.entry_price',
    'opentrades.entry_time', 'opentrades.max_drawdown', 'opentrades.max_drawdown_percent',
    'opentrades.max_runup', 'opentrades.max_runup_percent', 'opentrades.profit',
    'opentrades.profit_percent', 'opentrades.size', 'position_avg_price',
    'position_entry_name', 'position_size', 'wintrades',
)

ALL_MEMBERS = CONSTANTS + FUNCTIONS + SUBNAMESPACE + DATA


def _resolve(dotted: str):
    """Walk ``a.b.c`` from the strategy namespace, or raise AttributeError."""
    obj = strategy
    for part in dotted.split('.'):
        obj = getattr(obj, part)
    return obj


def __test_the_documented_surface_is_96_members__():
    """The list is TradingView's, and it has no duplicates or gaps."""
    assert len(ALL_MEMBERS) == 96, f"expected 96 documented members, listed {len(ALL_MEMBERS)}"
    assert len(set(ALL_MEMBERS)) == 96, "the member list contains duplicates"
    assert (len(CONSTANTS), len(FUNCTIONS), len(SUBNAMESPACE), len(DATA)) == (5, 10, 15, 66)


@pytest.mark.parametrize('member', ALL_MEMBERS)
def __test_every_documented_member_exists__(member):
    """Every member TradingView documents resolves in PyneCore."""
    try:
        _resolve(member)
    except AttributeError as e:
        pytest.fail(f"strategy.{member} is documented by TradingView but missing: {e}")


@pytest.mark.parametrize('member', FUNCTIONS + SUBNAMESPACE)
def __test_functions_and_subnamespace_members_are_usable__(member):
    """These are callables or constants -- never a bare stub.

    ``...`` in a stub file passes an existence check while handing a script
    something it cannot call, so existence alone is not enough here.
    """
    value = _resolve(member)
    assert value is not Ellipsis, f"strategy.{member} is a bare `...` stub"
    assert value is not None, f"strategy.{member} resolves to None"


def __test_pynecore_exposes_nothing_undocumented__():
    """The reverse direction: no stray public member TradingView does not document.

    A name PyneCore invents is a name scripts cannot rely on, and a name it keeps
    after TradingView removes one is a silent compatibility lie.
    """
    import inspect
    documented = {m.split('.')[0] for m in ALL_MEMBERS}
    # names the module merely imported, plus typing plumbing -- not Pine surface
    ignore = {
        'ABC', 'Context', 'Decimal', 'IntEnum', 'Literal', 'NA', 'OHLCV', 'Order',
        'PositionBase', 'PriceOrderBook', 'PyneFloat', 'PyneInt', 'PyneStr', 'QtyType',
        'SimPosition', 'Trade', 'abstractmethod', 'bisect_left', 'copy', 'datetime',
        'defaultdict', 'deque', 'insort', 'module_property', 'overload', 'na_str',
        'na_float', 'na_int', 'ADOPTED_STARTUP_ENTRY_ID', 'ROUND_FLOOR', 'ROUND_HALF_UP',
        'TYPE_CHECKING', 'UTC',
    }
    public = {
        n for n in dir(strategy)
        if not n.startswith('_') and n not in ignore
        and not inspect.ismodule(getattr(strategy, n))
    }
    # sub-namespaces are modules, so add them back by name
    public |= {n for n in dir(strategy) if not n.startswith('_')
               and inspect.ismodule(getattr(strategy, n))
               and n in {'commission', 'direction', 'oca', 'risk',
                         'opentrades', 'closedtrades'}}
    extra = sorted(public - documented)
    assert not extra, (
        f"PyneCore exposes strategy members TradingView does not document: {extra}. "
        "Either they are fork-only additions that need recording here, or upstream "
        "removed them and the shim should go."
    )
