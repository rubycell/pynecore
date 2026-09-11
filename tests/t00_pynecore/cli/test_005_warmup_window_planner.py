"""
Baseline + regression for the ``-N bars`` warmup window planner (#106).

THE BUG. ``pyne run`` asks the venue for bars by TIME RANGE but the user asks by
bar COUNT, so ``run.py`` guesses a window, downloads, counts, and widens-and-
redownloads while short (``max_retries = 4``). Two defects make deep warmup
unreachable:

1. THE PLANNER (``run.py`` ~888) anchors the next window on the oldest bar the
   venue ACTUALLY SERVED. When the widened slice lands inside a market closure no
   older bar comes back, the anchor does not move, the candidate window is
   byte-identical and ``min()`` is a no-op -- a FIXED POINT that repeats one
   attempt until the retries run out.

2. THE HORIZON CHECK (``run.py:848``) is the PRIMARY defect and fires first::

       horizon_reached = (prev_oldest_ts is not None and oldest_ts is not None
                          and oldest_ts >= prev_oldest_ts)

   It never consults the REQUESTED window, so two passes that both land inside a
   closure "prove" the venue has no more history. Measured: even a perfect
   doubling planner ABORTS at attempt 1 after a Tet-length closure.

MEASURED LIVE 2026-09-11 (DNSE VN30F1M @1m, target 2584):
  391 -> 1114 -> 1596 -> 1596 -> 1596 ... (identical through attempt 20)
Trigger: 2026-08-29..09-02, five consecutive non-trading days (Vietnam National
Day), against a flat ``timedelta(days=3)`` buffer documented as clearing "any
single weekend / session gap in one jump". The venue was never the limit -- the
same API served 4,579 bars for 30 days and 20,003 for 120.

WHY THESE TESTS LOOK PARANOID. An earlier version of this file was NOT
discriminating: the panel on #106 showed it passed under two implementations that
leave the defect architecture untouched --
  * the shipped formula with the magic constant bumped 3 -> 10 days, and
  * a constant 10-year window (no planning at all).
So the closure LENGTH is parametrized (a bumped constant fails once the closure
outruns it) and OVERSHOOT is bounded (a giant constant window fails that). A
pass-count bound alone excludes neither.
"""
from datetime import datetime, timedelta, timezone

import pytest

UTC = timezone.utc

TF_SECONDS = 60
BARS_PER_SESSION = 241
NOW = datetime(2026, 9, 11, 5, 0, tzinfo=UTC)

#: Closure lengths in CALENDAR days, ending the day before NOW's week starts.
#: 5 = the measured National Day cluster; 21 ~ Tet.
CLOSURE_LENGTHS = [5, 9, 14, 21]


def _closed_dates(closure_days: int) -> set:
    """A closure of ``closure_days`` ending 2026-09-02, plus ordinary weekends."""
    end = datetime(2026, 9, 2, tzinfo=UTC)
    closed = {(end - timedelta(days=i)).strftime("%Y-%m-%d")
              for i in range(closure_days)}
    day = datetime(2026, 6, 1, tzinfo=UTC)
    while day < NOW:
        if day.weekday() >= 5:
            closed.add(day.strftime("%Y-%m-%d"))
        day += timedelta(days=1)
    return closed


def make_venue(closure_days: int):
    """A synthetic venue: BARS_PER_SESSION per trading day, with a closure."""
    closed = _closed_dates(closure_days)

    def venue(window_from: datetime, window_to: datetime = NOW):
        bars, oldest = 0, None
        day = window_from.replace(hour=2, minute=0, second=0, microsecond=0)
        while day <= window_to:
            if (day >= window_from
                    and day.strftime("%Y-%m-%d") not in closed
                    and day.weekday() < 5):
                bars += BARS_PER_SESSION
                if oldest is None:
                    oldest = day
            day += timedelta(days=1)
        return bars, oldest

    return venue


def _drive(planner, venue, target, max_passes=8):
    """Run planner+venue to convergence. Returns (bars, passes, final_span_days)."""
    frm = NOW - timedelta(seconds=TF_SECONDS * (target + 1))
    bars = 0
    passes = 0
    for passes in range(1, max_passes + 1):
        bars, oldest = venue(frm)
        if bars >= target:
            break
        frm = planner(time_from=frm, window_to=NOW, oldest=oldest,
                      bar_count=target, real_bars=bars, tf_seconds=TF_SECONDS)
    return bars, passes, (NOW - frm).total_seconds() / 86400


# === the legacy formula, transcribed from run.py ~888 (documents the bug) =====


def _legacy(time_from, window_to, oldest, bar_count, real_bars, tf_seconds):
    anchor = oldest if oldest is not None else time_from
    missing = bar_count - real_bars
    return min(time_from,
               anchor - timedelta(seconds=tf_seconds * (missing + 10))
               - timedelta(days=3))


def __test_legacy_window_arithmetic_reaches_a_fixed_point__():
    """Characterizes the shipped bug offline. Keeps passing after the fix."""
    venue = make_venue(5)
    target = 2584
    frm = NOW - timedelta(seconds=TF_SECONDS * (target + 1))
    seen = []
    for _ in range(25):
        bars, oldest = venue(frm)
        seen.append((frm, bars))
        if bars >= target:
            break
        frm = _legacy(frm, NOW, oldest, target, bars, TF_SECONDS)

    assert max(b for _, b in seen) < target, "legacy formula unexpectedly converged"
    windows = [f for f, _ in seen]
    assert len(set(windows)) < len(windows), (
        "expected the legacy formula to revisit a window it had already tried"
    )


# === the planner contract (#106) =============================================


@pytest.mark.parametrize("closure_days", CLOSURE_LENGTHS)
@pytest.mark.parametrize("target", [500, 1024, 2000, 2584])
def __test_planner_converges_across_any_closure_length__(target, closure_days):
    """Must reach the requested count regardless of how long the market was shut.

    Parametrizing the CLOSURE LENGTH is what makes this discriminating: an
    implementation that merely enlarges the shipped magic constant (3 -> 10 days)
    converges for short closures and fails once the closure outruns the constant.
    """
    from pynecore.cli.commands.run import plan_next_window  # noqa: PLC0415

    bars, passes, _ = _drive(plan_next_window, make_venue(closure_days), target)
    assert bars >= target, (
        f"#106: planner fell short at target={target} across a {closure_days}-day "
        f"closure: {bars} bars after {passes} passes"
    )


@pytest.mark.parametrize("closure_days", CLOSURE_LENGTHS)
def __test_planner_does_not_overshoot_wildly__(closure_days):
    """Bounds the request. Kills the 'just ask for ten years' non-fix.

    The honest span for 2584 bars at 241/session is ~11 sessions ~ 15 calendar
    days; a closure adds its own length. Anything beyond that plus generous slack
    means the planner is not planning.
    """
    from pynecore.cli.commands.run import plan_next_window  # noqa: PLC0415

    target = 2584
    bars, _, span_days = _drive(plan_next_window, make_venue(closure_days), target)
    assert bars >= target
    budget = 3 * (15 + closure_days)
    assert span_days <= budget, (
        f"#106: planner requested {span_days:.1f} days for {target} bars across a "
        f"{closure_days}-day closure; budget is {budget} days. A planner that just "
        f"asks for an enormous window is not a fix."
    )


def __test_planner_makes_progress_when_a_pass_returns_nothing__():
    """Density is undefined at zero bars — must fall through to the floor, not divide."""
    from pynecore.cli.commands.run import plan_next_window  # noqa: PLC0415

    frm = NOW - timedelta(hours=1)
    nxt = plan_next_window(time_from=frm, window_to=NOW, oldest=None,
                           bar_count=2584, real_bars=0, tf_seconds=TF_SECONDS)
    assert nxt < frm, "a zero-bar pass must still widen the window"


def __test_planner_never_moves_the_window_later__():
    """The window may only ever grow backwards."""
    from pynecore.cli.commands.run import plan_next_window  # noqa: PLC0415

    frm = NOW - timedelta(days=30)
    nxt = plan_next_window(time_from=frm, window_to=NOW, oldest=NOW - timedelta(days=29),
                           bar_count=100, real_bars=99, tf_seconds=TF_SECONDS)
    assert nxt <= frm, "planner must never shrink the window"
