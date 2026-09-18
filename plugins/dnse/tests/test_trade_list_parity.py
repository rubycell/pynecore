"""#157 R10 — the trade-list parity acceptance test, carried by the SUITE.

The comparison itself lives in ``testing/fixtures/offline/trade_list_parity.py``; this file is
what makes the suite carry it, so it cannot rot unnoticed as a script nobody runs.

It is OPT-IN, and deliberately so. The parity run drives two full ``pyne run`` subprocesses —
a file-mode backtest over ~700 bars and a paced live replay against the fake — and takes a
couple of minutes. Putting that in the default suite would push a 17-second run past two
minutes and it would be the first thing anyone disabled. Set ``FAKE_VENUE_PARITY=1`` to run it;
the gate runs it explicitly.

What is pinned when it runs: the same strategy over the same day produces the same number of
closed trades with the same sides and quantities in both engines, and prices within a stated
tolerance. The four known differences, including why trade TIMES are not compared at all, are
documented beside the comparison rather than absorbed into it.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "testing" / "fixtures" / "offline"))


@pytest.mark.skipif(not os.environ.get("FAKE_VENUE_PARITY"),
                    reason="parity drives two full pyne runs (~2 min); set FAKE_VENUE_PARITY=1")
def __test_the_backtest_and_the_fake_agree_on_the_closed_trade_list__():
    """The operator's acceptance test for #157.

    A mismatch is a FINDING about the fake or the plugin, never an accepted difference, so this
    asserts on an empty finding list rather than on a pass/fail flag — the findings themselves
    are the useful output when it fails.
    """
    import trade_list_parity as parity

    backtest_trades = parity.backtest()
    assert backtest_trades, "the backtest produced no trades: the comparison would be vacuous"

    fake_trades, offset_s = parity.fake_run(int(os.environ.get("FAKE_VENUE_LIVE_BARS", "25")))
    assert fake_trades, "the fake produced no trades"

    findings = parity.compare(backtest_trades, fake_trades, offset_s)

    assert not findings, "trade-list parity findings:\n  - " + "\n  - ".join(findings)


def __test_the_parity_comparison_reports_a_difference_rather_than_absorbing_it__():
    """The discriminating half, and it runs ALWAYS — no subprocesses, so it costs nothing.

    Without it, the skipped test above could sit green forever while ``compare`` quietly returned
    an empty list for every input. Feed it two lists that genuinely differ and require that each
    difference is reported.
    """
    import trade_list_parity as parity

    def _row(when, kind="Entry long", qty="1", price="1970.0"):
        return {"Date/Time": when, "Type": kind, "Contracts": qty, "Price VND": price}

    backtest = [_row("2026-09-17T13:00:00+0700"), _row("2026-09-17T13:10:00+0700")]
    fake = [_row("2026-09-17T13:00:00+0700", kind="Entry short", qty="2", price="1999.0")]

    findings = parity.compare(backtest, fake, offset_s=0)

    text = " ".join(findings)
    assert findings, "a real difference must be reported, not absorbed"
    assert "COUNT" in text, "a missing trade must be reported"
    assert "side" in text or "type" in text.lower(), "a side difference must be reported"
    assert "quantity" in text, "a quantity difference must be reported"
    assert "price" in text, "a price outside tolerance must be reported"
