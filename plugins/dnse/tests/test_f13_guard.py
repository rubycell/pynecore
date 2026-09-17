"""#146 F1/F2 — pins for the F13 runner guards. No venue: `run` is injected.

These cover the decisions that used to be inline bash on the ORDER PATH, where
nothing could reach them. The shipped version had three routes to a RESTING
STOP with a MARKET POSITION on top of it, and every one of them is a pin below.

MUTATION NOTE: `f13_guard.py` is path-loaded here, so run mutants with
`PYTHONDONTWRITEBYTECODE=1` and read colours from BEHAVIOUR — a stale .pyc
survives a source restore and `inspect.getsource()` cannot see it.
"""
import importlib.util
import pathlib
import sys
from datetime import datetime, timedelta, timezone

_TOOL = (pathlib.Path(__file__).resolve().parents[1]
         / "testing" / "live_test" / "f13_guard.py")

spec = importlib.util.spec_from_file_location("dnse_f13_guard", _TOOL)
guard = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = guard
spec.loader.exec_module(guard)

ICT = timezone(timedelta(hours=7))

#: Two entries dispatched by one l2b run — it re-issues strategy.entry("E") on
#: every flat bar, so a 6-bar window can produce several.
TWO_ENTRIES = """\
1789456800.100 [2026-09-16 13:41:00+0700] bar: 501 INFO [BROKER] dispatched ENTRY BUY id='E' qty=1.0 type=stop stop=1874.2 -> ['da2he286p09g1n1vkb7g']
1789456860.100 [2026-09-16 13:42:00+0700] bar: 502 INFO [BROKER] dispatched ENTRY BUY id='E' qty=1.0 type=stop stop=1875.0 -> ['da2he2g6p09g1n1vkb80']
"""


class _Runner:
    """Records venue.py calls; answers from a canned script."""

    def __init__(self, **codes):
        self.codes = codes          # "cancel:<id>" / "flat" -> exit code
        self.calls = []

    def __call__(self, args):
        self.calls.append(list(args))
        if args and args[0] == "cancel":
            return self.codes.get(f"cancel:{args[1]}",
                                  self.codes.get("cancel", 0)), ""
        if args and args[0] == "flat":
            return self.codes.get("flat", 0), ""
        return 0, ""


# === F1 — the fallback must never run over a resting entry ==================

def __test_every_dispatched_entry_is_cancelled_not_just_the_last__():
    """Catches: `tail -1` (the shipped parse).

    l2b re-issues `strategy.entry("E")` on every flat bar, so a 6-bar window
    dispatches up to six ids. Cancelling only the newest leaves the earlier
    stops RESTING, and the fallback then puts a market position under them.
    """
    assert guard.entry_ids(TWO_ENTRIES) == [
        "da2he286p09g1n1vkb7g", "da2he2g6p09g1n1vkb80"]

    runner = _Runner()
    decision = guard.cleanup_decision(TWO_ENTRIES, run=runner)
    cancelled = [c[1] for c in runner.calls if c and c[0] == "cancel"]
    assert cancelled == ["da2he286p09g1n1vkb7g", "da2he2g6p09g1n1vkb80"], (
        f"only {cancelled} was cancelled — every dispatched entry must be")
    assert decision.proceed is True


def __test_a_refused_cancel_stops_the_ladder_and_blocks_the_fallback__():
    """THE F1 pin. Catches: echoing the cancel's exit code without testing it.

    `venue.py cancel` refuses to write when the trading token is not GOOD
    (venue.py:283-289) — the 8-hour token expires mid-session — and DNSE is
    documented to answer INVALID_TRADING_TOKEN on conditional-book writes after
    the operator's first app trade (#46/#51). Either leaves the stop RESTING,
    and the shipped runner then launched a market entry on top of it.
    """
    runner = _Runner(**{"cancel:da2he2g6p09g1n1vkb80": 2})
    decision = guard.cleanup_decision(TWO_ENTRIES, run=runner)

    assert decision.proceed is False, "a fallback was allowed after a refused cancel"
    assert decision.exit_code != guard.EXIT_OK
    assert "FLATTEN NOW" in decision.reason, (
        "the operator must be told to flatten: this runner cannot, and the "
        "entry is still resting")
    assert not any(c[0] == "flat" for c in runner.calls[2:]), (
        "proceeded to the flat check after a refused cancel instead of stopping")


def __test_unreadable_entry_id_does_not_fall_through_to_the_fallback__():
    """Catches the shipped control flow exactly: the runner REFUSED to guess an
    unreadable id — printing a careful warning — and then launched the fallback
    two lines later anyway. A refusal that does not change what happens next is
    decoration."""
    runner = _Runner(**{"flat": 1})
    decision = guard.cleanup_decision("a log with no dispatch line at all",
                                      run=runner)
    assert decision.proceed is False
    assert decision.exit_code == guard.EXIT_UNKNOWN
    assert "FLATTEN NOW" in decision.reason


def __test_no_entry_and_a_flat_account_may_still_fall_back__():
    """The over-block control: if 'no id found' always stopped the ladder, a
    run that simply never entered would end the arm. With the venue confirming
    FLAT there is nothing resting and the fallback is safe."""
    runner = _Runner(**{"flat": 0})
    decision = guard.cleanup_decision("no dispatch here", run=runner)
    assert decision.proceed is True, decision.reason


def __test_unconfirmed_flat_blocks_the_fallback_even_when_cancels_passed__():
    """Catches: treating 'every cancel returned 0' as proof the account is
    clear. A cancel ACK is not a fill-state, and exit 2 from `flat` is
    could-not-determine — never 'no', and never a reason to place an order."""
    for flat_code in (1, 2):
        runner = _Runner(**{"flat": flat_code})
        decision = guard.cleanup_decision(TWO_ENTRIES, run=runner)
        assert decision.proceed is False, (
            f"fallback allowed with venue.py flat exit {flat_code}")
        assert "FLATTEN NOW" in decision.reason


# === F2 — the session/window gate ==========================================

def __test_run_that_would_cross_the_ATC_boundary_is_refused__():
    """F2(b). Catches: no session gate at all — which is what shipped.

    L0 cannot serve as this gate: it SKIPS conditional probes outside a session
    and still exits 0 (l0_order_semantics.py:271-278, :382), so `--arm poll`
    launched at 14:00 passed every check and ran 45 minutes straight through
    the 14:30 ATC, where DNSE refuses cancels and fills whatever rests.
    """
    now = datetime(2026, 9, 18, 14, 0, tzinfo=ICT)
    decision = guard.window_decision(now, "continuous", 2700)
    assert decision.proceed is False, decision.reason
    assert "14:25" in decision.reason


def __test_run_that_fits_before_the_boundary_is_allowed__():
    """Over-block control: a gate that refused everything would pass the pin
    above while making the runner unusable."""
    now = datetime(2026, 9, 18, 9, 30, tzinfo=ICT)
    decision = guard.window_decision(now, "continuous", 2700)
    assert decision.proceed is True, decision.reason


def __test_morning_run_crossing_the_lunch_close_is_refused__():
    """The morning boundary is 11:25, not 14:25 — catches a single hardcoded
    afternoon deadline, under which a 10:50 start would sail past 11:30."""
    now = datetime(2026, 9, 18, 10, 50, tzinfo=ICT)
    decision = guard.window_decision(now, "continuous", 2700)
    assert decision.proceed is False, decision.reason
    assert "11:25" in decision.reason


def __test_non_continuous_phase_refuses_to_start__():
    """Catches starting an arm in lunch/atc/closed — and the failed-read form
    too, since an unrecognised phase is not a licence to trade."""
    for phase in ("lunch", "atc", "closed", "UNKNOWN (ImportError)"):
        decision = guard.window_decision(
            datetime(2026, 9, 18, 9, 30, tzinfo=ICT), phase, 600)
        assert decision.proceed is False, (
            f"phase {phase!r} was allowed to start an order-placing run")
