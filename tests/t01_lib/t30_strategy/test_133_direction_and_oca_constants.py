"""
Behaviour tests for the two strategy constants nothing in the suite exercised.

A sweep of the 96-member ``strategy`` surface found exactly two members with no
test naming them anywhere: ``strategy.direction.all`` and ``strategy.oca.reduce``.
Both are constants that only mean something through the call they are passed to,
so "it exists" says nothing -- a constant that was silently equal to its sibling
would pass every existence check and change what real scripts do.

Each is therefore tested against its siblings, so the constants have to be
distinguishable rather than merely present:

* ``direction.all`` vs ``direction.long`` through ``strategy.risk.allow_entry_in``
* ``oca.reduce`` vs ``oca.cancel`` vs ``oca.none`` through an OCA group

The OCA trio only diverges on a PARTIAL fill, which is why those probes use a
2-contract position with a 1-contract first leg. With a full-size first leg
``reduce`` and ``cancel`` produce the identical outcome (the sibling's remaining
quantity is zero either way), and the test would pin nothing -- measured, not
assumed.

Probes are authored in Pine (``data/*.pine``); the ``.py`` beside each one is
pine2pyne output and must not be hand-edited.
"""
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'


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
    """Bars that reach the limit (110) and later the stop (95) after bar 3."""
    from pynecore.types.ohlcv import OHLCV

    rows = ([(100, 101, 99, 100)] * 3
            + [(100, 112, 99, 111),     # 3  high 112 -> the 110 limit leg fills
               (111, 113, 94, 96),      # 4  low 94   -> the 95 stop leg reachable
               (96, 97, 94, 95)]
            + [(95, 101, 94, 100)] * 10)
    base_ts = 1_735_689_600_000
    return [OHLCV(timestamp=base_ts + i * 86_400_000, open=float(o), high=float(h),
                  low=float(lo), close=float(c), volume=10.0)
            for i, (o, h, lo, c) in enumerate(rows)]


def _run(script_name: str) -> dict:
    """Run one probe, returning its final-bar plots."""
    from pynecore.core.script_runner import ScriptRunner

    try:
        runner = ScriptRunner(DATA_DIR / f'{script_name}.py', iter(_make_ohlcv()),
                              _make_syminfo())
        plots = [dict(p) for _candle, p, _trades in runner.run_iter()]
        assert plots, f"{script_name} produced no bars"
        return plots[-1]
    finally:
        sys.modules.pop(script_name, None)


def __test_direction_all_admits_both_sides__():
    """``direction.all`` lets a long AND a short entry through.

    The ``direction.long`` control is what makes this meaningful: a gate that
    ignored its argument would admit both sides in either script.
    """
    permissive = _run('direction_gate_all')
    long_only = _run('direction_gate_long')

    assert permissive['closed'] == 2, (
        "direction.all must admit both the long and the short entry, "
        f"but only {permissive['closed']:.0f} trade(s) closed")
    assert long_only['closed'] == 1, (
        "direction.long must refuse the short entry, but "
        f"{long_only['closed']:.0f} trades closed")
    assert permissive['closed'] != long_only['closed'], (
        "direction.all and direction.long produced the same outcome — the gate "
        "is not reading its argument, so neither constant is actually pinned")


def __test_oca_reduce_trims_its_sibling_rather_than_cancelling_it__():
    """The three OCA types must produce three DIFFERENT outcomes.

    Measured on a 2-contract position whose first leg fills 1:

    ===========  =========  ======  =====================================
    oca_type     final pos  closed  what happened to the sibling
    ===========  =========  ======  =====================================
    ``reduce``   0          2       trimmed 2 -> 1, stop closed the rest
    ``cancel``   +1         1       removed, 1 contract left open
    ``none``     -1         2       untouched at 2, overshot through flat
    ===========  =========  ======  =====================================
    """
    reduce_ = _run('oca_reduce_group')
    cancel = _run('oca_cancel_group')
    none = _run('oca_none_group')

    # reduce: the sibling keeps working at the reduced size, so the position ends flat
    assert reduce_['pos'] == 0.0, \
        f"oca.reduce should end flat, got position {reduce_['pos']}"
    assert reduce_['closed'] == 2, \
        f"oca.reduce should close both legs, got {reduce_['closed']:.0f}"

    # cancel: the sibling is gone, so the remainder is stranded
    assert cancel['pos'] == 1.0, \
        f"oca.cancel should strand 1 contract, got position {cancel['pos']}"

    # none: the sibling is untouched, so it oversells through flat
    assert none['pos'] == -1.0, \
        f"oca.none should overshoot into a short, got position {none['pos']}"

    outcomes = {'reduce': reduce_['pos'], 'cancel': cancel['pos'], 'none': none['pos']}
    assert len(set(outcomes.values())) == 3, (
        f"the three OCA types must be distinguishable, got {outcomes} — if any two "
        "match, the group is ignoring its type and none of them is pinned")
