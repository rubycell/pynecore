"""#146 — pins for the F13 grader. No venue, no sockets, canned logs only.

Each pin names the wrong implementation it catches. The headline one is the
CLOCK refusal: `[BROKER]` lines carry the PINE BAR time (`lib/log.py:66-81`
resolves `epoch = lib._time / 1000`), identical for every event inside a bar,
so a grader that computes latency from them reports bar-grid noise with three
decimal places. The sample below is copied verbatim from a real capture
(`logs/f13_ws_fill1_134039.log`) — one order's PendingNew/New/Filled frames,
all stamped `13:41:00 bar: 501` — and is inlined rather than read from that
file so the pin does not depend on an untracked log.

MUTATION NOTE: `f13_grade.py` is path-loaded here, so a mutant leaves a
`__pycache__` entry that can survive a source restore, and `inspect.getsource`
cannot see it. Prefer runtime patches, run with `PYTHONDONTWRITEBYTECODE=1`,
and read every colour from BEHAVIOUR.
"""
import importlib.util
import pathlib
import sys

_TOOL = (pathlib.Path(__file__).resolve().parents[1]
         / "testing" / "live_test" / "f13_grade.py")

spec = importlib.util.spec_from_file_location("dnse_f13_grade", _TOOL)
grader = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = grader
spec.loader.exec_module(grader)


# Verbatim shape from a real run: bar-stamped, no wall-clock prefix.
BAR_STAMPED = """\
[2026-09-16 13:41:00+0700] bar:    501 INFO     [BROKER] [BROKER] order frame via WS: id=**2376 status=PendingNew fillQty=0 avgPx=0.0
[2026-09-16 13:41:00+0700] bar:    501 INFO     [BROKER] [BROKER] order frame via WS: id=**2376 status=New fillQty=0 avgPx=0.0
[2026-09-16 13:41:00+0700] bar:    501 INFO     [BROKER] [BROKER] order frame via WS: id=**2376 status=Filled fillQty=1 avgPx=1964.6
[2026-09-16 13:41:00+0700] bar:    501 INFO     [BROKER] [BROKER] WS ORDER SOURCE FIRST LIVE FRAME id=**2376
[2026-09-16 13:41:00+0700] bar:    501 INFO     [BROKER] event FILLED id=215286 side=buy qty=1.0 filled=1.0 price=1964.6 pine='E' leg=entry
[2026-09-16 13:41:00+0700] bar:    501 INFO     [BROKER] dispatched EXIT id='X' from='E' qty=1.0 tp=1948.4 sl=1940.6 -> ['159736']
"""

# The same run as the runner will capture it from now on: wall-clock prefixed.
PREFIXED = """\
1789456800.100 [2026-09-16 13:41:00+0700] bar:    501 INFO     [BROKER] [BROKER] order frame via WS: id=**2376 status=PendingNew fillQty=0
1789456800.400 [2026-09-16 13:41:00+0700] bar:    501 INFO     [BROKER] [BROKER] order frame via WS: id=**2376 status=New fillQty=0
1789456800.850 [2026-09-16 13:41:00+0700] bar:    501 INFO     [BROKER] [BROKER] WS ORDER SOURCE FIRST LIVE FRAME id=**2376
1789456800.900 [2026-09-16 13:41:00+0700] bar:    501 INFO     [BROKER] [BROKER] order frame via WS: id=159736 status=Filled fillQty=1
1789456801.250 [2026-09-16 13:41:00+0700] bar:    501 INFO     [BROKER] event FILLED id=215286 side=buy qty=1.0 filled=1.0 price=1964.6 pine='E' leg=entry
1789456802.500 [2026-09-16 13:41:00+0700] bar:    502 INFO     [BROKER] dispatched EXIT id='X' from='E' qty=1.0 tp=1948.4 sl=1940.6 -> ['159736']
"""

#: 215286 is an ACTIVATED conditional whose child 159736 did the filling (#41).
#: The child's fill time is deliberately chosen so the two latencies come out
#: DIFFERENT (+0.750s venue->print, +1.250s print->arm): with one shared value a
#: grader that swapped the two computations would pass the pin unnoticed.
VENUE = {
    "215286": {"orderStatus": "Activated", "externalOrderId": "159736",
               "createdDate": 1789456799000},
    "159736": {"orderStatus": "Filled", "modifiedDate": 1789456800500},
}


def _log(tmp_path, text, name="f13_ws_fill1_120000.log"):
    path = tmp_path / name
    path.write_text(text)
    return path


def __test_bar_stamped_log_refuses_to_report_a_latency__(tmp_path, capsys):
    """THE headline pin (Fable's ruling, #146).

    Catches: a grader that parses the `[BROKER]` bracket as an event time. On
    such a log every event in a bar shares one stamp, so `T(exit) - T(fill)` is
    exactly 0.000s whenever the exit dispatches in the fill's own bar — which
    #121 arm-on-fill makes the EXPECTED case. The wrong implementation is not
    noisy; it is confidently precise and wrong, and it would be most convincing
    on the ws arm the decision rests on.

    Mutation: make `clock_verdict` return `(True, ...)` unconditionally ->
    a numeric latency appears and this reddens.
    """
    rc = grader.main([str(_log(tmp_path, BAR_STAMPED)), "--arm", "ws"])
    out = capsys.readouterr().out
    assert rc == grader.EXIT_UNKNOWN, (
        f"a bar-stamped log graded {rc}; it must be could-not-determine (2)")
    assert "COULD-NOT-DETERMINE" in out
    assert "bar-stamped only" in out, (
        "the refusal must say WHY, or whoever re-grades an old capture cannot "
        "tell a broken log from a broken run")
    assert "0.000s" not in out, (
        "a latency number was produced from bar-stamped lines — this is the "
        "precise-looking garbage the tool exists to refuse")


def __test_prefixed_log_produces_real_latencies__(tmp_path, capsys):
    """The over-block control: a grader that refuses EVERYTHING would pass the
    pin above while being useless. Catches a refusal that never lifts.

    Also pins the arithmetic, with two DIFFERENT expected values so a grader
    that swapped the computations cannot pass: the venue's child fill at
    ...800.500 against our FILL print at ...801.250 is +0.750s, and our FILL
    print to our EXIT print at ...802.500 is +1.250s.
    """
    venue_json = tmp_path / "venue.json"
    venue_json.write_text(__import__("json").dumps(VENUE))
    rc = grader.main([str(_log(tmp_path, PREFIXED)), "--arm", "ws",
                      "--venue-json", str(venue_json)])
    out = capsys.readouterr().out
    assert "+0.750s" in out, f"fill latency wrong or missing:\n{out}"
    assert "+1.250s" in out, f"arm latency wrong or missing:\n{out}"
    assert rc == grader.EXIT_OK, f"a clean ws run graded {rc}:\n{out}"


def __test_venue_fill_follows_the_activated_conditional_to_its_child__():
    """#41/#39: a stop entry's fill NEVER appears on the id we placed — the
    conditional goes `Activated` (closed, not filled) and a NORMAL-book child
    executes. F13 now runs exactly that vehicle.

    Catches: reading the umbrella's own date as the fill time, which would
    silently grade the CONDITIONAL'S CREATION as the fill and make every
    latency ~1s too large here (and arbitrarily wrong in general).
    """
    epoch, note = grader.venue_fill_epoch(VENUE, "215286")
    assert epoch == 1789456800.5, f"took the umbrella's own date ({epoch})"
    assert "child 159736" in note


def __test_activated_umbrella_without_its_child_record_is_undetermined__():
    """Catches: falling back to the umbrella's date when the child record is
    absent. Absence of the child is could-not-determine, never a substitute."""
    partial = {"215286": VENUE["215286"]}
    epoch, note = grader.venue_fill_epoch(partial, "215286")
    assert epoch is None, "substituted the umbrella's date for a missing child"
    assert "child" in note


def __test_missing_first_live_frame_grades_ws_as_delivering_nothing__(
        tmp_path, capsys):
    """#134: the subscribe line is NOT delivery evidence. Catches a grader that
    accepts 'subscribe REQUESTED' (or any frame at all) as proof the WS arm
    worked."""
    text = PREFIXED.replace(
        "[BROKER] WS ORDER SOURCE FIRST LIVE FRAME id=**2376",
        "[BROKER] WS order feed subscribe REQUESTED channel=order.DERIVATIVE.json")
    rc = grader.main([str(_log(tmp_path, text)), "--arm", "ws"])
    out = capsys.readouterr().out
    assert "WS delivered nothing" in out, f"ws failure not reported:\n{out}"
    assert rc == grader.EXIT_NEGATIVE, f"a failed delivery gate graded {rc}"


def __test_ws_frame_naming_the_child_id_answers_130_for_stop_entries__(
        tmp_path, capsys):
    """The open #130 question: does the WS arm attribute the conditional's
    normal-book CHILD? Catches a grader that only ever looks at the placed id
    and so can never answer it either way."""
    rc = grader.main([str(_log(tmp_path, PREFIXED)), "--arm", "ws"])
    out = capsys.readouterr().out
    assert "#130 child frame" in out
    assert "PASS" in out.split("#130 child frame")[0].split("[")[-1] or \
           "a WS frame names the child id" in out, f"child frame not credited:\n{out}"

    without = PREFIXED.replace("id=159736 status=Filled", "id=**9999 status=Filled")
    grader.main([str(_log(tmp_path, without, "f13_ws_fill2_120000.log")),
                 "--arm", "ws"])
    out2 = capsys.readouterr().out
    assert "no WS frame names any of" in out2, (
        f"a run where NO frame names the child still passed the #130 gate:\n{out2}")


def __test_poll_arm_with_ws_frames_is_not_poll_only__(tmp_path, capsys):
    """Catches: grading the poll arm without checking it was actually poll-only.
    A poll arm that quietly had WS frames is not a poll measurement at all, and
    the whole ws-vs-poll comparison silently compares ws to ws."""
    rc = grader.main([str(_log(tmp_path, PREFIXED)), "--arm", "poll"])
    out = capsys.readouterr().out
    assert "not poll-only" in out, f"poll purity not checked:\n{out}"
    assert rc == grader.EXIT_NEGATIVE

    clean = ("1789456800.100 [2026-09-16 13:41:00+0700] bar: 501 INFO [BROKER] "
             "WS order feed disabled by config\n"
             "1789456801.250 [2026-09-16 13:41:00+0700] bar: 501 INFO [BROKER] "
             "event FILLED id=215286 side=buy qty=1.0 filled=1.0 price=1964.6 "
             "pine='E' leg=entry\n"
             "1789456802.500 [2026-09-16 13:41:00+0700] bar: 502 INFO [BROKER] "
             "dispatched EXIT id='X' from='E' qty=1.0 sl=1940.6 -> ['159736']\n")
    grader.main([str(_log(tmp_path, clean, "f13_poll_fill1_120000.log")),
                 "--arm", "poll"])
    assert "WS disabled and zero frames" in capsys.readouterr().out, (
        "a genuinely poll-only run failed the purity gate — the check would "
        "reject every valid poll arm")


def __test_no_sample_row_is_reported_as_such__(tmp_path, capsys):
    """The `--window-bars` outcome: an unfilled stop entry is a NO-SAMPLE row
    counted 0, not a missing run and not a zero-latency run. Catches a grader
    that silently omits the slot, which would make an arm look complete when
    half its runs never filled."""
    text = ("1789456800.100 [2026-09-16 13:41:00+0700] bar: 501 INFO [BROKER] "
            "F13 NO-SAMPLE: stop entry unfilled after 6 bars; cancelled\n")
    rc = grader.main([str(_log(tmp_path, text)), "--arm", "ws"])
    out = capsys.readouterr().out
    assert "NO-SAMPLE" in out, f"the NO-SAMPLE slot vanished from the table:\n{out}"
    assert rc != grader.EXIT_OK, "a run that produced no sample graded as a pass"


def __test_missing_venue_record_is_undetermined_not_zero__(tmp_path, capsys):
    """exit-2-never-no, on the venue side. Catches defaulting the venue fill
    time to 0 (or to our own print), which yields a latency equal to the epoch
    or to 0.000s — both plausible-looking."""
    rc = grader.main([str(_log(tmp_path, PREFIXED)), "--arm", "ws"])
    out = capsys.readouterr().out
    assert "COULD-NOT-DETERMINE" in out and "no venue record supplied" in out
    assert rc == grader.EXIT_UNKNOWN, f"graded {rc} with no venue times at all"
