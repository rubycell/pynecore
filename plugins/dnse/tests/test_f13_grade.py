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


# FIXTURES REBUILT FROM THREE REAL l2b RUNS (l2b_orig_141527, l2b_fill_133954,
# f11_retry), ids masked. The first cut invented a plausible shape instead —
# `event FILLED id=<the umbrella>` — and the engine does not produce it. That
# made the #130 gate look verified while it answered COULD-NOT-DETERMINE on
# every real run: a fixture that pins a shape the engine never emits is worse
# than no fixture, because it converts an untested path into a green one.
#
# The real sequence for a chased STOP entry, which every pin below now uses:
#   dispatched ENTRY … -> ['dakf1rav…']            the first conditional
#   event CREATED id=dakf1rav… leg=entry
#   cancel -> wire | order=dakf1rav… book=STOP     the chase cancels it
#   event CREATED id=dakf2aav… leg=entry           the REPLACEMENT — no dispatch line
#   conditional ACTIVATED … parent=dakf2aav… child=214806
#   event FILLED id=214806 … leg=entry             the CHILD fills, not the umbrella

# Bar-stamped (no wall-clock prefix): what the runner produced before #146.
BAR_STAMPED = """\
[2026-09-15 14:16:00+0700] bar:    501 INFO     [BROKER] dispatched ENTRY BUY id='E' qty=1.0 type=stop stop=1953.0 -> ['dakf1ravfqkc7397iko0']
[2026-09-15 14:17:00+0700] bar:    502 INFO     [BROKER] conditional ACTIVATED -> tracking child | parent=dakf2aavfqkc7397ikqg child=214806 pine=E polls=1
[2026-09-15 14:17:00+0700] bar:    502 INFO     [BROKER] [BROKER] order frame via WS: id=**4806 status=Filled fillQty=1 avgPx=1952.9
[2026-09-15 14:17:00+0700] bar:    502 INFO     [BROKER] [BROKER] WS ORDER SOURCE FIRST LIVE FRAME id=**4806
[2026-09-15 14:17:00+0700] bar:    502 INFO     [BROKER] event FILLED id=214806 side=buy qty=1.0 filled=1.0 price=1952.9 pine='E' leg=entry
[2026-09-15 14:17:00+0700] bar:    502 INFO     [BROKER] dispatched EXIT id='X' from='E' qty=1.0 tp=1956.8 sl=1948.4 -> ['215286']
"""

# The same run as the runner captures it now: wall-clock prefixed.
PREFIXED = """\
1789456800.100 [2026-09-15 14:16:00+0700] bar:    501 INFO     [BROKER] dispatched ENTRY BUY id='E' qty=1.0 type=stop stop=1953.0 -> ['dakf1ravfqkc7397iko0']
1789456800.150 [2026-09-15 14:16:00+0700] bar:    501 INFO     [BROKER] event CREATED id=dakf1ravfqkc7397iko0 side=buy qty=1.0 filled=0.0 pine='E' leg=entry
1789456800.400 [2026-09-15 14:17:00+0700] bar:    502 INFO     [BROKER] cancel -> wire | order=dakf1ravfqkc7397iko0 book=STOP pine=E from_entry=None leg=ENTRY
1789456800.450 [2026-09-15 14:17:00+0700] bar:    502 INFO     [BROKER] event CREATED id=dakf2aavfqkc7397ikqg side=buy qty=1.0 filled=0.0 pine='E' leg=entry
1789456800.800 [2026-09-15 14:17:00+0700] bar:    502 INFO     [BROKER] conditional ACTIVATED -> tracking child | parent=dakf2aavfqkc7397ikqg child=214806 pine=E polls=1
1789456800.850 [2026-09-15 14:17:00+0700] bar:    502 INFO     [BROKER] [BROKER] WS ORDER SOURCE FIRST LIVE FRAME id=**4806
1789456800.900 [2026-09-15 14:17:00+0700] bar:    502 INFO     [BROKER] [BROKER] order frame via WS: id=**4806 status=Filled fillQty=1 avgPx=1952.9
1789456801.250 [2026-09-15 14:17:00+0700] bar:    502 INFO     [BROKER] event FILLED id=214806 side=buy qty=1.0 filled=1.0 price=1952.9 pine='E' leg=entry
1789456802.500 [2026-09-15 14:18:00+0700] bar:    503 INFO     [BROKER] dispatched EXIT id='X' from='E' qty=1.0 tp=1956.8 sl=1948.4 -> ['215286']
"""

#: The venue record of the CHILD — the id that actually filled. `venue.py order
#: --json` captures it because the runner now collects `child=` ids too.
VENUE = {
    "214806": {"orderStatus": "Filled", "modifiedDate": 1789456800500,
               "createdDate": 1789456800200},
}


#: The other real venue shape: the CONDITIONAL's record, which is Activated and
#: names its child. Used by the venue-hop pins — both shapes occur, depending on
#: which id `venue.py order --json` was given.
UMBRELLA_VENUE = {
    "dakf2aavfqkc7397ikqg": {"orderStatus": "Activated",
                             "externalOrderId": "214806",
                             "createdDate": 1789456800200},
    "214806": {"orderStatus": "Filled", "modifiedDate": 1789456800500},
}


def _edit(text, needle, replacement):
    """`str.replace` that REFUSES to be a no-op.

    A fixture mutated with a needle that no longer occurs changes nothing,
    raises nothing, and leaves the pin quietly testing the unmutated case —
    which is exactly what happened when the fixtures were rebuilt from real
    logs and a pin still reached for an id from the invented shape. The pin
    stayed green while asserting nothing. Same family as every other blind
    check today: the operation reported success about the wrong object.
    """
    assert needle in text, (
        f"fixture mutation is a NO-OP: {needle!r} does not occur. The pin "
        f"would test the UNMUTATED fixture and pass for no reason.")
    return text.replace(needle, replacement)


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
    epoch, note = grader.venue_fill_epoch(UMBRELLA_VENUE,
                                          "dakf2aavfqkc7397ikqg")
    assert epoch == 1789456800.5, f"took the umbrella's own date ({epoch})"
    assert "child 214806" in note


def __test_activated_umbrella_without_its_child_record_is_undetermined__():
    """Catches: falling back to the umbrella's date when the child record is
    absent. Absence of the child is could-not-determine, never a substitute."""
    partial = {"dakf2aavfqkc7397ikqg": UMBRELLA_VENUE["dakf2aavfqkc7397ikqg"]}
    epoch, note = grader.venue_fill_epoch(partial, "dakf2aavfqkc7397ikqg")
    assert epoch is None, "substituted the umbrella's date for a missing child"
    assert "child" in note


def __test_missing_first_live_frame_grades_ws_as_delivering_nothing__(
        tmp_path, capsys):
    """#134: the subscribe line is NOT delivery evidence. Catches a grader that
    accepts 'subscribe REQUESTED' (or any frame at all) as proof the WS arm
    worked."""
    text = _edit(PREFIXED,
                 "[BROKER] WS ORDER SOURCE FIRST LIVE FRAME id=**4806",
                 "[BROKER] WS order feed subscribe REQUESTED "
                 "channel=order.DERIVATIVE.json")
    rc = grader.main([str(_log(tmp_path, text)), "--arm", "ws",
                      "--venue-json", _venue_json(tmp_path)])
    out = capsys.readouterr().out
    assert "WS delivered nothing" in out, f"ws failure not reported:\n{out}"
    assert rc == grader.EXIT_NEGATIVE, f"a failed delivery gate graded {rc}"


def _venue_json(tmp_path, records=None):
    path = tmp_path / "venue.json"
    path.write_text(__import__("json").dumps(records if records else VENUE))
    return str(path)


def __test_child_frame_gate_asks_about_the_ENTRY_child_not_the_bracket__(
        tmp_path, capsys):
    """F3. Catches the shipped gate, which built its child ids from the
    `dispatched EXIT -> [...]` line — the BRACKET's orders.

    #130 asks whether the ENTRY conditional's normal-book child is attributed
    over WS. Grading the bracket's ids answers a different question and calls
    it a PASS. Here the entry's child (dakf2aav… -> 214806, stated by the
    engine's own ACTIVATED line) IS named by a frame, so the gate passes for
    the RIGHT reason.
    """
    rc = grader.main([str(_log(tmp_path, PREFIXED)), "--arm", "ws",
                      "--venue-json", _venue_json(tmp_path)])
    out = capsys.readouterr().out
    assert "names the entry's normal-book child 214806" in out, (
        f"the #130 gate did not resolve the ENTRY's child:\n{out}")
    assert rc == grader.EXIT_OK


def __test_child_frame_gate_uses_the_engines_own_ACTIVATED_line__(
        tmp_path, capsys):
    """The child comes from OUR log, not from a venue call.

    The engine states the mapping itself — `conditional ACTIVATED -> tracking
    child | parent=… child=214806` — and that is BETTER evidence for #130 than
    the venue record, because #130 asks about OUR attribution of the child.

    This pin replaced one asserting the opposite (that the gate is
    COULD-NOT-DETERMINE without a venue record). That premise came from a
    fixture using the umbrella's id as the FILLED id, which the engine never
    emits. Catches a regression to a venue-only lookup, which would answer
    could-not-determine on every run where the conditional record was not
    captured — i.e. most of them, since the runner asks about the FILLED id.
    """
    grader.main([str(_log(tmp_path, PREFIXED)), "--arm", "ws"])
    out = capsys.readouterr().out
    assert "child from engine ACTIVATED line" in out, (
        f"the #130 gate did not use the engine's own mapping:\n{out}")


def __test_a_masked_frame_id_must_not_substring_match_a_foreign_child__(
        tmp_path, capsys):
    """F3's demonstrated collision, and the sharpest pin in this file.

    `ws_order_source` logs MASKED ids (`id=**9736`), and the shipped test was
    `f["id"].lstrip("*") in cid or cid in f["id"]`. Against child `1597312`
    the unrelated frame `**5973` PASSES, because "5973" occurs inside it — a
    four-character needle finds itself almost anywhere in a longer haystack.
    That is a false PASS on the one measurement the ws-vs-poll decision rests
    on.

    Contract: a masked id matches only if the child ENDS WITH its visible
    suffix, and the frame is at/after the entry fill.
    """
    collide = _edit(_edit(PREFIXED, "child=214806", "child=1597312"),
                    "id=**4806", "id=**5973")
    grader.main([str(_log(tmp_path, collide, "f13_ws_fill3_120000.log")),
                 "--arm", "ws"])
    out = capsys.readouterr().out
    assert "no WS frame names the entry's child 1597312" in out, (
        f"an unrelated masked id substring-matched a foreign child:\n{out}")


def __test_a_masked_suffix_that_really_is_the_child_still_matches__(
        tmp_path, capsys):
    """The over-block control: masked ids are the NORMAL case on the ws arm, so
    a rule that rejected all of them would fail every real run and still pass
    the collision pin above. A genuine suffix match is labelled as such so the
    table never presents it as an exact id match."""
    grader.main([str(_log(tmp_path, PREFIXED, "f13_ws_fill4_120000.log")),
                 "--arm", "ws", "--venue-json", _venue_json(tmp_path)])
    out = capsys.readouterr().out
    assert "suffix-matched" in out, f"a genuine masked child was rejected:\n{out}"


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
             "event FILLED id=214806 side=buy qty=1.0 filled=1.0 price=1952.9 "
             "pine='E' leg=entry\n"
             "1789456802.500 [2026-09-15 14:18:00+0700] bar: 503 INFO [BROKER] "
             "dispatched EXIT id='X' from='E' qty=1.0 sl=1948.4 -> ['215286']\n")
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
