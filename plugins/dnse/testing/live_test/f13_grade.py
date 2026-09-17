#!/usr/bin/env python3
"""Live-L3-F13 grader — READ-ONLY, offline, no venue contact whatsoever.

Grades one arm of an F13 run from its LOG FILES plus an optional venue-record
JSON capture. It opens no sockets, constructs no broker, and imports nothing
from the plugin: everything it needs arrives as a file path. That is the point
of splitting it out of ``run_f13_latency.sh`` — the runner places live orders,
so no test affordance may live there, and a grader that cannot be exercised
offline is a grader nobody has ever checked.

    f13_grade.py --arm ws  logs/f13_ws_fill1_*.log [--venue-json venue.json]

THE CLOCK PROBLEM, and why this tool refuses things (measured 2026-09-17):

``[BROKER]`` log lines are stamped with the PINE BAR TIME, not the moment the
event arrived — ``lib/log.py:66-81`` resolves ``epoch = lib._time / 1000`` and
only falls back to the record's wall clock before the first bar. Measured on a
real F13 log, one order's ``PendingNew`` / ``New`` / ``Filled`` frames all carry
the identical stamp ``[2026-09-16 13:41:00+0700] bar: 501``. So a latency
computed from those stamps is quantised to the bar grid: at 5m it reports where
in the bar the fill landed, and for an exit dispatched in the fill's own bar it
reports exactly 0.0 — precise-looking, and meaningless.

The runner therefore prefixes each line with a real wall clock as it tees, and
this grader reads THAT. A log without the prefix is graded
``COULD-NOT-DETERMINE`` (exit 2) rather than silently producing bar-grid
numbers — exit-2-never-no, applied to the grader itself.

CAVEAT that must appear in every report: the prefix records WHEN OUR PROCESS
PRINTED THE LINE. It is an upper bound on arrival, including the logging path.
Venue-side times come from the venue record (its own clock, on the venue's
host) — the two are DIFFERENT CLOCKS and their difference carries both offsets.

EXIT CODES (venue.py convention): 0 graded and every gate passed; 1 graded and
a gate FAILED; 2 could not determine (no usable clock, no parsable events).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

EXIT_OK, EXIT_NEGATIVE, EXIT_UNKNOWN = 0, 1, 2

#: The runner's wall-clock prefix: epoch seconds with millisecond resolution,
#: written BEFORE the engine's own ``[bar-time]`` bracket.
_PREFIX = re.compile(r"^(?P<epoch>\d{10}\.\d{1,6})\s")
#: The engine's bar-time bracket — detected only so a log carrying ONLY this
#: can be refused with an accurate reason.
_BAR_STAMP = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4}\]")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")

_ENTRY_FILL = re.compile(r"event FILLED id=(?P<id>\S+).*?leg=entry")
_ENTRY_DISPATCH = re.compile(r"dispatched ENTRY [A-Z]+ ")
#: The engine states the conditional -> normal-book child mapping IN OUR LOG:
#:   conditional ACTIVATED -> tracking child | parent=dakf2aav… child=214806
#: Measured on three real l2b runs. This is better evidence than the venue
#: record for the #130 question, because it is OUR attribution — the thing the
#: question is actually about — and it needs no extra call.
_ACTIVATED_CHILD = re.compile(
    r"conditional ACTIVATED -> tracking child \| parent=(?P<parent>\S+) "
    r"child=(?P<child>\S+)")
_DISPATCH_EXIT = re.compile(r"dispatched EXIT id='(?P<pine>[^']*)'.*?->\s*\[(?P<ids>[^\]]*)\]")
_WS_FRAME = re.compile(r"order frame via WS: id=(?P<id>\S+) status=(?P<status>\S+)")
_FIRST_FRAME = re.compile(r"WS ORDER SOURCE FIRST LIVE FRAME")
_WS_DISABLED = re.compile(r"WS order feed disabled by config")
#: Written by the runner when a stop entry did not fill inside its window.
_NO_SAMPLE = re.compile(r"F13 NO-SAMPLE")
_FALLBACK = re.compile(r"F13 FALLBACK transport-only")


class ClockUnavailable(Exception):
    """The log carries no wall-clock prefix — latency is not computable."""


def _read(path: Path) -> list[tuple[float | None, str]]:
    """-> [(epoch|None, text)]. Strips ANSI; never raises on a weird byte."""
    out: list[tuple[float | None, str]] = []
    for raw in path.read_text(errors="replace").splitlines():
        line = _ANSI.sub("", raw)
        match = _PREFIX.match(line)
        if match:
            out.append((float(match.group("epoch")), line[match.end():]))
        else:
            out.append((None, line))
    return out


def clock_verdict(lines: list[tuple[float | None, str]]) -> tuple[bool, str]:
    """Is there a usable EVENT clock in this log? -> (usable, why).

    Usable means the runner's wall-clock prefix is present. A log carrying only
    the engine's bar stamp is explicitly named as such, because that is the
    failure this tool exists to refuse — and the reason has to be legible to
    whoever re-runs the grade on an old capture.
    """
    prefixed = sum(1 for epoch, _ in lines if epoch is not None)
    if prefixed:
        return True, f"{prefixed} line(s) carry the runner's wall-clock prefix"
    bar_stamped = sum(1 for _, text in lines if _BAR_STAMP.match(text))
    if bar_stamped:
        return False, (
            f"no usable event clock in this log (bar-stamped only: "
            f"{bar_stamped} line(s)). The [BROKER] bracket is the PINE BAR "
            f"time (lib/log.py:66-81), identical for every event in a bar, so "
            f"a latency from it is bar-grid noise. Re-run with a runner that "
            f"prefixes the wall clock.")
    return False, "no recognisable timestamps at all"


def parse_run(path: Path) -> dict:
    """Everything one run's log can tell us. No judgement here, just facts."""
    lines = _read(path)
    usable, why = clock_verdict(lines)
    run: dict = {
        "log": path.name, "clock_usable": usable, "clock_why": why,
        "entry_fill": None, "entry_id": None, "entry_dispatch": None,
        "activated_parent": None, "activated_child": None,
        "exit_dispatch": None,
        "exit_child_ids": [], "ws_frames": [], "first_live_frame": False,
        "ws_disabled": False, "no_sample": False, "fallback": False,
    }
    for epoch, text in lines:
        if _FIRST_FRAME.search(text):
            run["first_live_frame"] = True
        if _WS_DISABLED.search(text):
            run["ws_disabled"] = True
        if _NO_SAMPLE.search(text):
            run["no_sample"] = True
        if _FALLBACK.search(text):
            run["fallback"] = True
        if _ENTRY_DISPATCH.search(text) and run["entry_dispatch"] is None:
            run["entry_dispatch"] = epoch
        activated = _ACTIVATED_CHILD.search(text)
        if activated:
            run["activated_parent"] = activated.group("parent")
            run["activated_child"] = activated.group("child")
        fill = _ENTRY_FILL.search(text)
        if fill and run["entry_fill"] is None:
            run["entry_fill"] = epoch
            run["entry_id"] = fill.group("id")
        exit_match = _DISPATCH_EXIT.search(text)
        if exit_match and run["exit_dispatch"] is None:
            run["exit_dispatch"] = epoch
            run["exit_child_ids"] = [
                piece.strip().strip("'\"")
                for piece in exit_match.group("ids").split(",") if piece.strip()]
        frame = _WS_FRAME.search(text)
        if frame:
            run["ws_frames"].append(
                {"t": epoch, "id": frame.group("id"),
                 "status": frame.group("status")})
    return run


def venue_fill_epoch(venue: dict, entry_id: str | None) -> tuple[float | None, str]:
    """-> (epoch, note). Follows an Activated conditional to its CHILD.

    A stop entry's fill NEVER appears on the id we placed: the conditional goes
    ``Activated`` (closed, not filled) and the venue creates a NORMAL-book child
    named by ``externalOrderId`` — that child is what executes (#41/#39). So the
    fill time comes from the child's record, and reading the umbrella's own
    ``modifiedDate`` as a fill time would be wrong for exactly the vehicle F13
    now runs.
    """
    if not venue or not entry_id:
        return None, "no venue record supplied"
    record = venue.get(str(entry_id))
    if record is None:
        return None, f"no venue record for {entry_id}"
    note = ""
    status = str(record.get("orderStatus", "")).upper()
    child_id = record.get("externalOrderId")
    if status == "ACTIVATED" and child_id:
        child = venue.get(str(child_id))
        if child is None:
            return None, (f"{entry_id} is Activated -> child {child_id}, but no "
                          f"record for the child was supplied (#41: the "
                          f"umbrella never fills; the child does)")
        note = f"via child {child_id}"
        record = child
    stamp = record.get("modifiedDate") or record.get("createdDate")
    if stamp is None:
        return None, f"venue record for {entry_id} carries no date field"
    try:
        value = float(stamp)
    except (TypeError, ValueError):
        return None, f"unparsable venue date {stamp!r}"
    # DNSE serves epoch MILLISECONDS; a seconds value would be ~1e9.
    return (value / 1000.0 if value > 1e11 else value), note


def entry_child_id(run: dict, venue: dict) -> tuple[str | None, str]:
    """-> (child id, where it came from). The normal-book order that FILLED.

    Measured on three real l2b runs (l2b_orig_141527, l2b_fill_133954,
    f11_retry): for a STOP entry the engine's ``event FILLED id=`` carries the
    CHILD, not the conditional —

        conditional ACTIVATED -> tracking child | parent=dakf2aav… child=214806
        event FILLED id=214806 … leg=entry

    The first cut took ``entry_id`` from the FILLED line and then asked the
    VENUE for *that* record's ``externalOrderId`` — but a child has none, so
    the #130 gate answered COULD-NOT-DETERMINE on every real run while the
    fixture (which used the umbrella's id) said it worked. The fixture pinned a
    shape the engine does not produce, which is worse than no fixture: it made
    a broken gate look verified.

    Three sources, best first: the engine's own ACTIVATED line (our attribution,
    which is what #130 asks about), the venue's ``externalOrderId`` if a record
    for the conditional was captured, and finally the FILLED id itself — which
    for a stop entry already IS the child.
    """
    if run.get("activated_child"):
        return str(run["activated_child"]), "engine ACTIVATED line"
    entry_id = run.get("entry_id")
    if venue and entry_id:
        record = venue.get(str(entry_id)) or {}
        child = record.get("externalOrderId")
        if child:
            return str(child), "venue externalOrderId"
    if entry_id:
        return str(entry_id), "the FILLED id (already the normal-book child)"
    return None, ""


def match_child_frame(frames: list, child_id: str, entry_dispatch_t: float | None):
    """-> (matched, how). Does any WS frame name this child?

    ``ws_order_source`` logs MASKED ids (``id=**9736``), so an exact compare is
    impossible for them — but the first cut's loose substring test was far
    worse than imprecise, it was WRONG: ``f["id"].lstrip("*") in cid or cid in
    f["id"]`` matches an unrelated frame ``**5973`` against child ``1597312``,
    because "5973" appears inside it. A four-character needle in a
    six-or-more-character haystack finds itself almost anywhere.

    So: a full id matches exactly. A masked id must have its visible suffix
    EQUAL the child's last characters, and its frame must arrive at or after
    the entry DISPATCH — a masked suffix alone is weak evidence and the
    timestamp is the cheap corroboration we already hold. The caller labels the
    result so the table never presents a suffix match as an exact one.

    The corroborating anchor is the DISPATCH, not the fill, and the difference
    is not cosmetic: the child's frame is what CAUSES our `event FILLED` line,
    so it necessarily PRECEDES it. Anchoring on the fill (the first cut here)
    rejected every genuine frame and would have reported "WS did not attribute
    the child" on runs where it plainly did — a false FAIL on the exact
    measurement the ws-vs-poll decision rests on. Caught by this module's own
    over-block control.
    """
    for frame in frames:
        ident = str(frame.get("id") or "")
        if not ident:
            continue
        if "*" not in ident:
            if ident == child_id:
                return True, "exact id"
            continue
        suffix = ident.lstrip("*")
        if not suffix or len(suffix) < 4:
            continue                       # too little to claim anything
        if not child_id.endswith(suffix):
            continue
        stamp = frame.get("t")
        if (entry_dispatch_t is not None and stamp is not None
                and stamp < entry_dispatch_t):
            continue                  # predates our entry: cannot be its child
        return True, f"suffix-matched on {suffix!r}, frame at/after the entry dispatch"
    return False, ""


def grade(run: dict, arm: str, venue: dict) -> dict:
    """Judgement. Every gate answers PASS / FAIL / COULD-NOT-DETERMINE."""
    gates: list[tuple[str, str, str]] = []          # (name, verdict, detail)

    if run["no_sample"]:
        gates.append(("sample", "NO-SAMPLE",
                      "the stop entry did not fill inside its window; counted 0"))
    if run["fallback"]:
        gates.append(("vehicle", "FALLBACK",
                      "transport-only (l2 market entry) — no bracket chain in "
                      "this row"))

    if arm == "ws":
        gates.append(("ws delivery (#134)",
                      "PASS" if run["first_live_frame"] else "FAIL",
                      "FIRST LIVE FRAME present" if run["first_live_frame"] else
                      "WS delivered nothing: no FIRST LIVE FRAME line. The "
                      "subscribe line is NOT evidence of delivery (#134)."))
        # F3: the question #130 asks is whether the ENTRY conditional's
        # normal-book CHILD is attributed over WS. The first cut compared
        # against the BRACKET's ids from the `dispatched EXIT -> [...]` line,
        # which is a different order entirely — a PASS there answered a
        # question nobody asked, and an OCO umbrella (Activated from birth with
        # its own child) would likely have produced a false FAIL as well.
        entry_child, source = entry_child_id(run, venue)
        if entry_child:
            matched, how = match_child_frame(
                run["ws_frames"], entry_child, run["entry_dispatch"])
            gates.append(("#130 child frame",
                          "PASS" if matched else "FAIL",
                          f"a WS frame names the entry's normal-book child "
                          f"{entry_child} ({how}; child from {source})"
                          if matched else
                          f"no WS frame names the entry's child {entry_child} "
                          f"(child from {source}) — the conditional's "
                          f"normal-book child was not attributed over WS"))
        else:
            gates.append(("#130 child frame", "COULD-NOT-DETERMINE",
                          "no entry fill and no ACTIVATED line in this run — "
                          "there is no child to ask the #130 question about"))
    elif arm == "poll":
        pure = run["ws_disabled"] and not run["ws_frames"]
        gates.append(("poll purity",
                      "PASS" if pure else "FAIL",
                      "WS disabled and zero frames" if pure else
                      f"not poll-only: ws_disabled={run['ws_disabled']}, "
                      f"{len(run['ws_frames'])} frame(s) present"))

    latencies: dict[str, str] = {}
    if not run["clock_usable"]:
        latencies["fill latency"] = f"COULD-NOT-DETERMINE — {run['clock_why']}"
        latencies["arm latency"] = "COULD-NOT-DETERMINE — same reason"
        gates.append(("event clock", "COULD-NOT-DETERMINE", run["clock_why"]))
    else:
        venue_t, note = venue_fill_epoch(venue, run["entry_id"])
        if venue_t is None:
            latencies["fill latency"] = f"COULD-NOT-DETERMINE — {note}"
        elif run["entry_fill"] is None:
            latencies["fill latency"] = "COULD-NOT-DETERMINE — no entry FILL event in the log"
        else:
            latencies["fill latency"] = (
                f"{run['entry_fill'] - venue_t:+.3f}s  (venue->our print{', ' + note if note else ''}; "
                f"DIFFERENT CLOCKS: venue host vs ours)")
        if run["entry_fill"] is not None and run["exit_dispatch"] is not None:
            latencies["arm latency"] = (
                f"{run['exit_dispatch'] - run['entry_fill']:+.3f}s  "
                f"(our print of FILL -> our print of dispatched EXIT; one clock)")
        else:
            latencies["arm latency"] = "COULD-NOT-DETERMINE — missing FILL or EXIT line"
    return {"gates": gates, "latencies": latencies}


def render(results: list[tuple[dict, dict]], arm: str) -> int:
    """Print the table; return the exit code.

    PRECEDENCE, and it is not the obvious one: an unusable EVENT CLOCK
    DOMINATES every other outcome. The presence gates (WS delivery, #130 child
    attribution) do not need a clock and stay meaningful, but a log we have
    declared ungradeable must not leave through the same door as a graded run
    that merely failed a gate — the operator reading `exit=1` would believe the
    measurement happened and one check failed, when in fact the measurement the
    whole test exists for never ran. Exit-2-never-no, pointed at the grader.
    """
    worst = EXIT_OK
    if any(not run["clock_usable"] for run, _ in results):
        worst = EXIT_UNKNOWN
    print(f"=== F13 grade — arm {arm} — {len(results)} run(s) ===")
    print("CAVEAT: our timestamps are WHEN OUR PROCESS PRINTED the line — an "
          "upper bound on\n        arrival, including the logging path. Venue "
          "times are the VENUE's clock.\n")
    for run, verdict in results:
        print(f"-- {run['log']}")
        print(f"   entry id       : {run['entry_id'] or '(none seen)'}")
        print(f"   exit children  : {run['exit_child_ids'] or '(none)'}")
        print(f"   ws frames      : {len(run['ws_frames'])}")
        for name, state, detail in verdict["gates"]:
            print(f"   [{state:<19}] {name}: {detail}")
            if state == "FAIL" and worst != EXIT_UNKNOWN:
                worst = EXIT_NEGATIVE
            elif state == "COULD-NOT-DETERMINE":
                worst = EXIT_UNKNOWN
        for name, value in verdict["latencies"].items():
            print(f"   {name:<15}: {value}")
            if value.startswith("COULD-NOT-DETERMINE") and worst == EXIT_OK:
                worst = EXIT_UNKNOWN
        print()
    if not results:
        print("no logs graded — COULD NOT DETERMINE")
        return EXIT_UNKNOWN
    return worst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", help="run log file(s)")
    parser.add_argument("--arm", choices=("ws", "poll"), required=True)
    parser.add_argument("--venue-json", default=None,
                        help="JSON object mapping venue order id -> its record "
                             "(orderStatus, createdDate/modifiedDate, "
                             "externalOrderId). Offline-capturable.")
    args = parser.parse_args(argv)

    venue: dict = {}
    if args.venue_json:
        path = Path(args.venue_json)
        if not path.is_file():
            print(f"--venue-json {path} does not exist — venue-side times "
                  f"unavailable (grading continues without them)")
        else:
            try:
                venue = json.loads(path.read_text())
            except (OSError, ValueError) as exc:
                print(f"--venue-json unreadable ({exc}) — venue-side times "
                      f"unavailable")
    results = []
    for name in args.logs:
        path = Path(name)
        if not path.is_file():
            print(f"missing log: {path}")
            continue
        run = parse_run(path)
        results.append((run, grade(run, args.arm, venue)))
    return render(results, args.arm)


if __name__ == "__main__":
    raise SystemExit(main())
