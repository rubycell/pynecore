"""
Liveness pin for the fork's P5 drawdown / P&L-percent statistics (card #151).

The fork adds seven statistics to the sim position and its CSV: unrealized and
real max drawdown (absolute and percent) and the total / realized / unrealized
P&L percentages. Nothing in Pine reads them, so they have no consumer inside the
test-suite; the upstream 6.9.4 rebase rewrote the sim fill loop around their
accumulation in 48 hunks and the P5 commits merged silently. This file is the
consumer: it runs a scenario where every field has a hand-computable non-zero
value and reads them from BOTH objects that carry them -- ``runner.stats`` (what
the CSV is written from) and ``runner.script.position`` (what the per-bar
accumulation mutates) -- so a rename in one cannot silently read a default in
the other.

Red-first: ``__test_the_pin_sees_a_dead_accumulation__`` wraps the per-bar
finaliser so the four accumulated fields are zeroed after every bar, and shows
the same readers report 0 -- the failure mode the pin exists to catch. The
control probe (no adverse excursion) shows the accumulator is quiet when it
should be, so "non-zero" is not the whole assertion.

Probes are authored in Pine (``data/p5_drawdown_*.pine``); the transpiled modules
beside them are pine2pyne output and must not be hand-edited. SIM mode only: in
broker mode the drawdown fields are declared but not computed, by design.
"""
import math
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'

INITIAL_CAPITAL = 100_000.0

#: Hand-derived from the liveness probe's bars (pointvalue 1, qty 1, commission 0).
L1_REALIZED = 3.0                    # L1: 100 -> 103
L2_ENTRY = 100.0
L2_WORST_LOW = 90.0                  # bar 6 low
L2_BAR6_CLOSE = 95.0
L2_LAST_CLOSE = 96.0
REAL_MAX_DD = L2_ENTRY - L2_BAR6_CLOSE                        # 5.0, open loss at bar close
REAL_MAX_DD_PCT = REAL_MAX_DD / (L2_ENTRY * 1.0) * 100.0      # over the entry cost: 5.0
PEAK_REALIZED_EQUITY = INITIAL_CAPITAL + L1_REALIZED          # 100003.0
UNREALIZED_MAX_DD = L2_ENTRY - L2_WORST_LOW                   # 10.0, worst intrabar equity
UNREALIZED_MAX_DD_PCT = UNREALIZED_MAX_DD / PEAK_REALIZED_EQUITY * 100.0
UNREALIZED_PNL = L2_LAST_CLOSE - L2_ENTRY                     # -4.0, the run ends with L2 open
TOTAL_PNL = L1_REALIZED + UNREALIZED_PNL                      # -1.0


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


def _bars(rows):
    from pynecore.types.ohlcv import OHLCV
    base_ts = 1_735_689_600_000
    return [OHLCV(timestamp=base_ts + i * 86_400_000, open=float(o), high=float(h),
                  low=float(lo), close=float(c), volume=10.0)
            for i, (o, h, lo, c) in enumerate(rows)]


#: Liveness scenario -- see the probe header for the story each bar tells.
LIVENESS_BARS = [
    (100, 100, 100, 100),   # 0
    (100, 100, 100, 100),   # 1
    (100, 100, 100, 100),   # 2  L1 placed
    (100, 104, 100, 103),   # 3  L1 fills at 100; +3 at close, no dip
    (103, 104, 102, 103),   # 4  close L1 placed
    (103, 103, 103, 103),   # 5  L1 closes at 103 (+3); L2 placed
    (100, 101, 90, 95),     # 6  L2 fills at 100; low 90, close 95
    (95, 97, 92, 96),       # 7  smaller excursion; run ends with L2 open
]

#: Control scenario -- no bar low below the entry, the trade never loses.
CONTROL_BARS = [
    (100, 100, 100, 100),   # 0
    (100, 100, 100, 100),   # 1
    (100, 100, 100, 100),   # 2  L placed
    (100, 105, 100, 104),   # 3  L fills at 100
    (104, 108, 104, 107),   # 4
    (107, 110, 107, 109),   # 5  run ends with L open (+9)
]


def _run(script_name: str, rows):
    """Run a probe to the end; return the runner (its ``stats`` and position)."""
    from pynecore.core.script_runner import ScriptRunner

    try:
        runner = ScriptRunner(DATA_DIR / f'{script_name}.py', iter(_bars(rows)),
                              _make_syminfo())
        plots = [dict(p) for _candle, p, _trades in runner.run_iter()]
        assert plots, f"{script_name} produced no bars"
        return runner, plots[-1]
    finally:
        sys.modules.pop(script_name, None)


def _p5(runner) -> dict:
    """The seven P5 fields as the CSV writer sees them (``runner.stats``)."""
    s = runner.stats
    assert s is not None, "runner.stats was not populated -- is the probe a strategy?"
    return {
        'unrealized_max_drawdown': s.unrealized_max_drawdown,
        'unrealized_max_drawdown_percent': s.unrealized_max_drawdown_percent,
        'real_max_drawdown': s.real_max_drawdown,
        'real_max_drawdown_percent': s.real_max_drawdown_percent,
        'total_pnl_percent': s.total_pnl_percent,
        'realized_pnl_percent': s.realized_pnl_percent,
        'unrealized_pnl_percent': s.unrealized_pnl_percent,
    }


def __test_the_scenario_ends_with_the_losing_trade_open__():
    """Precondition: the liveness probe must end with L2 open at a loss.

    A run that ended flat would read ``unrealized_pnl == 0`` by construction and
    the percent pins below would be vacuous, so this is asserted first.
    """
    _runner, last = _run('p5_drawdown_liveness', LIVENESS_BARS)
    assert last['position_size'] == 1.0, f"expected L2 open, position {last['position_size']}"
    assert last['openprofit'] == UNREALIZED_PNL, f"open P&L {last['openprofit']}, expected {UNREALIZED_PNL}"
    assert last['netprofit'] == L1_REALIZED, f"net profit {last['netprofit']}, expected {L1_REALIZED}"


def __test_every_p5_field_carries_its_hand_computed_value__():
    """The seven fields read the derived values -- each percent over its OWN denominator.

    real DD % is over the entry cost, unrealized DD % over the peak realized
    equity, the P&L trio over the initial capital; asserting one shared
    denominator would pass a regression in any of them.
    """
    runner, _last = _run('p5_drawdown_liveness', LIVENESS_BARS)
    got = _p5(runner)
    expected = {
        'unrealized_max_drawdown': UNREALIZED_MAX_DD,
        'unrealized_max_drawdown_percent': UNREALIZED_MAX_DD_PCT,
        'real_max_drawdown': REAL_MAX_DD,
        'real_max_drawdown_percent': REAL_MAX_DD_PCT,
        'total_pnl_percent': TOTAL_PNL / INITIAL_CAPITAL * 100.0,
        'realized_pnl_percent': L1_REALIZED / INITIAL_CAPITAL * 100.0,
        'unrealized_pnl_percent': UNREALIZED_PNL / INITIAL_CAPITAL * 100.0,
    }
    for name, want in expected.items():
        assert math.isclose(got[name], want, rel_tol=0, abs_tol=1e-9), (
            f"{name}: got {got[name]!r}, expected {want!r}")
    assert got['unrealized_max_drawdown'] > got['real_max_drawdown'], (
        "on a bar with low < close < entry the intrabar drawdown must exceed the close drawdown")


def __test_stats_and_position_read_the_same_accumulation__():
    """Two readers of one object: the CSV-side stats equal the position's fields.

    A future rename in ``strategy_stats`` would otherwise read a dataclass
    default while the position still accumulates -- silently.
    """
    runner, _last = _run('p5_drawdown_liveness', LIVENESS_BARS)
    pos = runner.script.position
    got = _p5(runner)
    for name in ('unrealized_max_drawdown', 'unrealized_max_drawdown_percent',
                 'real_max_drawdown', 'real_max_drawdown_percent'):
        assert got[name] == getattr(pos, name), (
            f"{name}: stats {got[name]!r} != position {getattr(pos, name)!r}")


def __test_a_smaller_second_excursion_does_not_move_the_maxima__():
    """Bar 7 (low 92, close 96) is a smaller excursion than bar 6: both maxima stay.

    Catches an accumulator that overwrites instead of taking the running max.
    """
    runner, _last = _run('p5_drawdown_liveness', LIVENESS_BARS)
    got = _p5(runner)
    assert got['real_max_drawdown'] == REAL_MAX_DD, "real DD moved on the smaller bar"
    assert got['unrealized_max_drawdown'] == UNREALIZED_MAX_DD, "unrealized DD moved on the smaller bar"


def __test_the_control_without_adverse_excursion_reads_zero_drawdown__():
    """A trade that never loses and never dips below entry accumulates no drawdown."""
    runner, last = _run('p5_drawdown_control', CONTROL_BARS)
    assert last['position_size'] == 1.0 and last['openprofit'] == 9.0
    got = _p5(runner)
    assert got['real_max_drawdown'] == 0.0 and got['real_max_drawdown_percent'] == 0.0
    assert got['unrealized_max_drawdown'] == 0.0 and got['unrealized_max_drawdown_percent'] == 0.0
    assert got['unrealized_pnl_percent'] == 9.0 / INITIAL_CAPITAL * 100.0


def __test_the_pin_sees_a_dead_accumulation__(monkeypatch):
    """Red-first, kept in the file: zero the four accumulated fields after every
    bar and the same readers report 0 -- exactly what the pin above would fail on.

    The wrapper calls the real finaliser first, so net profit and the open P&L
    (upstream's own fields) stay intact: only the P5 accumulation is "dead".
    """
    from pynecore.lib.strategy import SimPosition

    original = SimPosition._finalize_bar_pnl

    def dead_accumulation(self):
        original(self)
        self.unrealized_max_drawdown = 0.0
        self.unrealized_max_drawdown_percent = 0.0
        self.real_max_drawdown = 0.0
        self.real_max_drawdown_percent = 0.0

    monkeypatch.setattr(SimPosition, '_finalize_bar_pnl', dead_accumulation)
    runner, last = _run('p5_drawdown_liveness', LIVENESS_BARS)
    assert last['netprofit'] == L1_REALIZED, "the wrapper must not disturb upstream's fields"
    got = _p5(runner)
    for name in ('unrealized_max_drawdown', 'unrealized_max_drawdown_percent',
                 'real_max_drawdown', 'real_max_drawdown_percent'):
        assert got[name] == 0.0, f"{name} still reports {got[name]!r} with the accumulation dead"
        assert getattr(runner.script.position, name) == 0.0
    # and the derived trio, which the accumulation never owned, is untouched
    assert math.isclose(got['unrealized_pnl_percent'], UNREALIZED_PNL / INITIAL_CAPITAL * 100.0, abs_tol=1e-9)
