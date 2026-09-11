"""
Coverage tests: every data-bearing member of ``strategy``,
``strategy.opentrades`` and ``strategy.closedtrades`` actually PRODUCES DATA.

A member can exist, be typed, be reachable from transpiled Pine and still be
dead -- reading a slot nothing ever writes. ``strategy.opentrades.profit_percent``
is exactly that: implemented in ``lib/strategy/opentrades.py``, declared in the
``.pyi``, emitted correctly by the transpiler, and returning ``0`` on every bar
of an open trade because the per-bar mark-to-market loop
(``lib/strategy/__init__.py``, the ``for trade in self.open_trades`` block that
refreshes ``profit``, ``max_drawdown*`` and ``max_runup*``) never assigns it --
all three ``profit_percent =`` sites live on CLOSE paths. Nothing caught it,
because nothing asserted that the member is alive.

These tests close that hole on two axes:

The probe scripts are **authored in Pine**: ``data/namespace_liveness*.pine`` are
the sources, and the ``.py`` beside each one is pine2pyne's output -- a build
artifact. Edit the ``.pine`` and re-transpile; a hand edit to the ``.py`` is lost
on the next transpile and the two silently disagree until then::

    cd /home/mike/workspace/github/pine2pyne
    .venv/bin/python -m pine2pyne <abs path>/namespace_liveness.pine \
        -o <abs path>/namespace_liveness.py

Authoring in Pine is what lets the same scenario be pasted into TradingView to
establish what the answer SHOULD be -- a hand-written ``@pyne`` file cannot be.
(These three exceed TradingView's 64-plot cap at 66 witnesses, so they need
splitting before they run there.)

1. **Liveness** -- ``data/namespace_liveness.py`` probes every covered member on
   every bar of a scenario built so each of them has something to report. A
   member that stays dead (``na`` or exactly ``0`` on every bar) fails unless it
   is listed in :data:`KNOWN_DEAD` with a reason.
2. **Completeness** -- the namespaces are introspected and compared against the
   probed set, so a member added upstream cannot silently go untested, and a
   probe left behind for a member that no longer exists is caught too.

:data:`KNOWN_DEAD` is strict in both directions, like pine2pyne's
``xfail(strict=True)`` bug pins: a newly dead member fails, and a member that
starts working also fails, so an entry has to be removed deliberately rather
than rotting into a permanent excuse.

The scenario (20 synthetic daily bars, market orders filling on the next open):

* bar 2 enters ``L1`` long, bar 3 arms a limit exit ``X1`` at 110, bar 4's high
  of 113 fills it -> a WINNING closed trade that exercises the ``strategy.exit``
  path, so ``exit_id`` / ``exit_comment`` carry real text.
* bar 6 enters ``L2`` long at 112, bar 8 closes it, bar 9's open of 101 books it
  -> a LOSING closed trade, exercising ``strategy.close``.
* bar 11 enters ``L3`` long at 101 and never closes it -> an OPEN trade on the
  final bar, in profit, so every ``opentrades`` member has a live value.

Both a win and a loss are required: ``wintrades`` / ``losstrades``,
``grossprofit`` / ``grossloss`` and the ``avg_winning_*`` / ``avg_losing_*``
families only leave zero once their own side has happened. A percent commission
makes the ``commission`` members non-zero, and every bar's high/low reaches past
its body so drawdown and runup accumulate.
"""
import sys
import math
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'

#: The probe script, also the module key to evict between runs.
SCRIPT_NAME = 'namespace_liveness'

#: Scenario B: the same 66 witnesses driven by an SMA(5/20) crossover over a
#: repeating triangle wave -- dozens of trades on both sides, fills landing on
#: exactly repeating integer prices, and NO commission, so a breakeven trade
#: (profit exactly 0, what ``strategy.eventrades`` counts) is reachable. A
#: member counts as alive if either scenario brings it to life.
CROSS_SCRIPT_NAME = 'namespace_liveness_cross'

#: Scenario C: a single trade constructed to close at exactly breakeven, which
#: is the only thing ``strategy.eventrades`` counts. It proves the counter is
#: alive, so its 0 in scenarios A and B is a correct reading rather than a dead
#: member -- the distinction :data:`EXPECTED_ZERO` exists to make.
BREAKEVEN_SCRIPT_NAME = 'namespace_liveness_breakeven'

#: Members that exist on a namespace but are NOT data: order-placing and helper
#: functions, Pine constants, and names the module merely imported. Subtracted
#: by the completeness check, which still fails on a genuinely new member.
NOT_DATA = {
    # order entry points / helpers
    'entry', 'exit', 'order', 'close', 'close_all', 'cancel', 'cancel_all',
    'default_entry_qty', 'convert_to_account', 'convert_to_symbol',
    # Pine constants
    'long', 'short', 'fixed', 'cash', 'percent_of_equity',
    'ADOPTED_STARTUP_ENTRY_ID', 'ROUND_FLOOR', 'ROUND_HALF_UP', 'TYPE_CHECKING', 'UTC',
    # imported symbols / typing plumbing
    'ABC', 'Context', 'Decimal', 'IntEnum', 'Literal', 'NA', 'OHLCV', 'Order',
    'PositionBase', 'PriceOrderBook', 'PyneFloat', 'PyneInt', 'PyneStr', 'QtyType',
    'SimPosition', 'Trade', 'abstractmethod', 'bisect_left', 'copy', 'datetime',
    'defaultdict', 'deque', 'insort', 'module_property', 'overload',
    'na_str', 'na_float', 'na_int',
}

#: Members measured to be DEAD -- reachable, typed, and constant for the whole
#: run even though the scenario gives them something to report. Strict in both
#: directions (see the module docstring).
KNOWN_DEAD = {
    'strategy.closedtrades.first_index':
        "BUG, upstream #84: constant 0. TradingView specifies the index of the "
        "oldest REMAINING trade once closed trades are evicted past the list "
        "limit -- measured, that state is reachable (20k bars, no order cap) and "
        "the correct answer there is 10998. Pinned by "
        "test_134_closed_trade_list_overflow. It stays in KNOWN_DEAD rather than "
        "EXPECTED_ZERO because 0 is only correct BELOW the limit; this was "
        "originally filed here as 'correct by design', which the overflow probe "
        "disproved",
    'strategy.opentrades.profit_percent':
        "returns 0 for an open trade: the per-bar open-trade loop refreshes "
        "profit / max_drawdown* / max_runup* but never assigns profit_percent, "
        "and all three assignment sites are on CLOSE paths",
}

#: Members whose value in THIS scenario is legitimately zero/na -- correct
#: behaviour, not a dead slot. Kept apart from :data:`KNOWN_DEAD` so a real
#: defect is never filed under "expected". Each entry says why, and what would
#: have to change in the scenario to exercise it for real.
EXPECTED_ZERO = {
    'strategy.eventrades':
        "counts trades closing at exactly 0 profit, which neither scenario "
        "produces (a percent commission makes it unreachable in A, and every "
        "crossover trade in B fills at different prices). The counter itself "
        "is proven alive by __test_eventrades_counts_a_breakeven_trade__",
}

#: Plot-name prefix -> the namespace it probes.
NAMESPACE_OF = {
    's': 'strategy',
    'o': 'strategy.opentrades',
    'c': 'strategy.closedtrades',
}


def _make_syminfo():
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period='D', type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=0.001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


def _make_ohlcv():
    """The scenario's bars: one trade wins, one loses, one is left open."""
    from pynecore.types.ohlcv import OHLCV

    # (open, high, low, close) -- every bar's high/low reaches past its body so
    # runup and drawdown have something to accumulate.
    rows = [
        (100, 101, 99, 100),    # 0
        (100, 101, 99, 100),    # 1
        (100, 101, 99, 100),    # 2  entry L1 issued
        (100, 104, 99, 103),    # 3  L1 fills at 100; exit X1 armed at 110
        (103, 113, 102, 112),   # 4  high 113 -> X1 limit fills at 110 (WIN)
        (112, 113, 111, 112),   # 5
        (112, 113, 111, 112),   # 6  entry L2 issued
        (112, 113, 105, 106),   # 7  L2 fills at 112
        (106, 107, 100, 101),   # 8  close L2 issued
        (101, 102, 100, 101),   # 9  L2 books at 101 (LOSS)
        (101, 102, 100, 101),   # 10
        (101, 102, 100, 101),   # 11 entry L3 issued
        (101, 106, 100, 105),   # 12 L3 fills at 101 and stays open
        (105, 107, 103, 104),   # 13
        (104, 108, 103, 107),   # 14
        (107, 109, 105, 106),   # 15
        (106, 110, 104, 109),   # 16
        (109, 111, 107, 110),   # 17
        (110, 112, 108, 111),   # 18
        (111, 113, 109, 112),   # 19 L3 still open
    ]
    base_ts = 1_735_689_600_000  # 2025-01-01 00:00:00 UTC, in ms
    return [
        OHLCV(timestamp=base_ts + i * 86_400_000,
              open=float(o), high=float(h), low=float(lo), close=float(c),
              volume=10.0)
        for i, (o, h, lo, c) in enumerate(rows)
    ]


def _make_ohlcv_cross():
    """A repeating triangle wave: SMA(5) crosses SMA(20) dozens of times.

    Prices are whole numbers that repeat exactly, so reversal fills land on the
    same price often enough for a breakeven trade to occur once the commission
    is off -- that is what makes ``strategy.eventrades`` reachable.
    """
    from pynecore.types.ohlcv import OHLCV

    # rise, PLATEAU, fall, PLATEAU -- the flat stretches are the point: the
    # fast average converges onto the slow one there, so a cross can open and
    # close a trade at the very same price, which is the only way profit lands
    # on exactly 0.
    closes = []
    for i in range(900):
        phase = i % 44
        if phase < 10:
            closes.append(100 + phase)          # rise 100 -> 109
        elif phase < 22:
            closes.append(110)                  # flat top
        elif phase < 32:
            closes.append(110 - (phase - 21))   # fall 109 -> 100
        else:
            closes.append(99)                   # flat bottom
    base_ts = 1_735_689_600_000
    bars = []
    for i, c in enumerate(closes):
        o = float(closes[i - 1]) if i else float(c)
        bars.append(OHLCV(timestamp=base_ts + i * 86_400_000,
                          open=o, high=float(max(o, c)) + 1.0,
                          low=float(min(o, c)) - 1.0, close=float(c), volume=10.0))
    return bars


def _make_ohlcv_flat():
    """A flat series: every fill lands on 100.0, so a trade can close at zero."""
    from pynecore.types.ohlcv import OHLCV

    base_ts = 1_735_689_600_000
    return [OHLCV(timestamp=base_ts + i * 86_400_000, open=100.0, high=101.0,
                  low=99.0, close=100.0, volume=10.0)
            for i in range(12)]


def _run_probe(script_name: str = SCRIPT_NAME) -> list[dict]:
    """Run one probe script, returning its plot dict per bar."""
    from pynecore.core.script_runner import ScriptRunner

    if script_name == CROSS_SCRIPT_NAME:
        bars = _make_ohlcv_cross()
    elif script_name == BREAKEVEN_SCRIPT_NAME:
        bars = _make_ohlcv_flat()
    else:
        bars = _make_ohlcv()
    try:
        runner = ScriptRunner(
            DATA_DIR / f'{script_name}.py', iter(bars), _make_syminfo(),
        )
        return [dict(plot_data) for _candle, plot_data, _trades in runner.run_iter()]
    finally:
        sys.modules.pop(script_name, None)


def _probed_members(plot_names) -> dict[str, set]:
    """Map the probe's plot-name prefixes back to namespace member names."""
    probed: dict[str, set] = {'strategy': set(), 'opentrades': set(), 'closedtrades': set()}
    key = {'s': 'strategy', 'o': 'opentrades', 'c': 'closedtrades'}
    for name in plot_names:
        prefix, sep, member = name.partition('__')
        if sep and prefix in key:
            probed[key[prefix]].add(member)
    # The counts (``o__opentrades`` / ``c__closedtrades``) land on their own
    # sub-namespace, which is where ``_declared_members`` finds them too: the
    # parent exposes them as MODULE objects, and modules are filtered out there.
    return probed


def _declared_members() -> dict[str, set]:
    """Introspect the three namespaces for data-bearing member names."""
    import inspect
    from pynecore.lib import strategy as _strategy
    from pynecore.lib.strategy import opentrades as _open, closedtrades as _closed

    def names(mod) -> set:
        out = set()
        for n in dir(mod):
            if n.startswith('_') or n in NOT_DATA:
                continue
            if inspect.ismodule(getattr(mod, n)):
                continue
            out.add(n)
        return out

    return {
        'strategy': names(_strategy),
        'opentrades': names(_open),
        'closedtrades': names(_closed),
    }


def _assert_scenario_traded(plots: list[dict]) -> None:
    """Without real trades every liveness verdict below would be vacuous.

    A member that is dead and one that was simply never exercised look
    identical, so the scenario's own outcome is asserted first.
    """
    last = plots[-1]
    assert last["c__closedtrades"] == 3, \
        f"scenario broken: expected 3 closed trades, got {last['c__closedtrades']}"
    assert last["s__wintrades"] == 1, \
        f"scenario broken: expected 1 winning trade, got {last['s__wintrades']}"
    assert last["s__losstrades"] == 2, \
        f"scenario broken: expected 2 losing trades, got {last['s__losstrades']}"
    assert last["s__max_contracts_held_short"] > 0, \
        "scenario broken: the short leg never opened, so short-side members are untested"
    assert last["o__opentrades"] == 1, \
        f"scenario broken: expected 1 open trade at the end, got {last['o__opentrades']}"


def __test_every_strategy_namespace_member_carries_data__():
    """Each probed member reports real data somewhere in the run.

    "Dead" means the member never produced a single usable value: every bar was
    ``na``, or every bar was exactly ``0``. Both are what a member backed by a
    slot nobody writes looks like.
    """
    plots = _run_probe()
    assert plots, "the probe script produced no bars"
    _assert_scenario_traded(plots)

    # Scenario B: an SMA(5/20) crossover over 900 bars, dozens of trades on both
    # sides and no commission. A member counts as alive if EITHER scenario
    # brings it to life, so one scenario's blind spot cannot condemn a member
    # that works.
    cross = _run_probe(CROSS_SCRIPT_NAME)
    assert cross[-1]["c__closedtrades"] >= 20, (
        "scenario B broken: the crossover should close dozens of trades, got "
        f"{cross[-1]['c__closedtrades']}")

    def alive(values) -> bool:
        real = [v for v in values
                if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if not real:
            return False                      # na on every bar
        return any(v != 0.0 for v in real)    # or pinned to exactly zero

    dead = set()
    for plot_name in plots[0]:
        prefix, sep, member = plot_name.partition('__')
        if not sep or prefix not in NAMESPACE_OF:
            continue                          # an OHLCV column, not a probe
        if not (alive([bar[plot_name] for bar in plots])
                or alive([bar[plot_name] for bar in cross])):
            dead.add(f"{NAMESPACE_OF[prefix]}.{member}")

    newly_dead = dead - set(KNOWN_DEAD) - set(EXPECTED_ZERO)
    assert not newly_dead, (
        "these members produced no data anywhere in the run (na or 0 on every "
        f"bar) and are not recorded in KNOWN_DEAD: {sorted(newly_dead)}"
    )

    # EXPECTED_ZERO is held just as strictly: if one starts reporting, the
    # justification no longer holds and the entry must go.
    now_reporting = set(EXPECTED_ZERO) - dead
    assert not now_reporting, (
        f"{sorted(now_reporting)} now reports data, so its EXPECTED_ZERO reason "
        "no longer holds. Remove the entry -- the member is testable for real now."
    )

    revived = set(KNOWN_DEAD) - dead
    assert not revived, (
        f"{sorted(revived)} now carries data -- the KNOWN_DEAD entry is stale. "
        "Remove it so the member is held to the liveness rule from now on."
    )


def __test_every_declared_member_is_probed__():
    """A member added upstream cannot slip past the liveness test untested."""
    plots = _run_probe()
    probed = _probed_members(plots[0])
    declared = _declared_members()

    missing = {space: sorted(declared[space] - probed[space])
               for space in declared if declared[space] - probed[space]}
    assert not missing, (
        "these namespace members are not probed by data/namespace_liveness.py "
        f"-- add a plot for each: {missing}"
    )

    # The other direction: a probe for a member that no longer exists would
    # quietly keep passing the liveness check on a stale plot.
    stale = {space: sorted(probed[space] - declared[space])
             for space in declared if probed[space] - declared[space]}
    assert not stale, f"these probes no longer match a declared member: {stale}"


def __test_eventrades_counts_a_breakeven_trade__():
    """``strategy.eventrades`` is alive: it counts a trade closing at zero.

    Without this, its 0 in the liveness scenarios would be ambiguous -- a
    correct reading and a dead member look identical. Here the condition is
    constructed (flat prices, no commission), so the entry and the close both
    fill at 100.0 and the trade's profit is exactly 0.0.
    """
    plots = _run_probe(BREAKEVEN_SCRIPT_NAME)
    last = plots[-1]
    assert last["closedtrades"] == 1, \
        f"expected 1 closed trade, got {last['closedtrades']}"
    assert last["profit"] == 0.0, \
        f"the trade must close at exactly breakeven, got profit {last['profit']}"
    assert last["wintrades"] == 0 and last["losstrades"] == 0, \
        "a breakeven trade must not be booked as a win or a loss"
    assert last["eventrades"] == 1, \
        f"eventrades did not count the breakeven trade (got {last['eventrades']})"
