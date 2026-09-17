#!/usr/bin/env python3
"""F13 runner guards — the decisions that must NOT live in bash.

Two jobs, both on the ORDER PATH, both previously inline shell that nothing
could pin (#146 review, findings F1 and F2):

  window   may this arm START at all? (session phase + enough time before the
           next hard boundary for the whole run to finish)
  cleanup  after an l2b run that produced NO SAMPLE: cancel EVERY entry the run
           dispatched, verify the account is flat, and only then say whether a
           fallback l2 may run.

WHY IT MOVED HERE. The shipped version parsed one id with ``tail -1``, echoed
the cancel's exit code without testing it, and launched the fallback regardless
— including down the branch that had just refused to guess an unreadable id. It
had three reachable paths to a RESTING STOP plus a MARKET POSITION on top of
it, and none of them could be exercised offline. Bash on the order path that
nobody can test is how that shipped.

Every venue command goes through an injected ``run`` callable, so the pins in
``plugins/dnse/tests/test_f13_guard.py`` drive the refusal paths — cancel
refused, id unreadable, not flat — with no venue anywhere near them.

EXIT CODES (venue.py convention, and the runner obeys them):
  0  proceed (window open / cleanup complete and a fallback may run)
  1  do NOT proceed (window shut, or cleanup says stop the ladder)
  2  could not determine — treat exactly as 1, and say so loudly
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path

EXIT_OK, EXIT_NEGATIVE, EXIT_UNKNOWN = 0, 1, 2

ICT = timezone(timedelta(hours=7))
REPO = Path(__file__).resolve().parents[4]
VENUE = REPO / "plugins" / "dnse" / "tools" / "venue.py"

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
#: EVERY id the run dispatched as an entry — l2b re-issues ``strategy.entry``
#: on every flat bar, so a 6-bar window can dispatch six of them. The shipped
#: ``tail -1`` cancelled the newest and left the rest resting.
_ENTRY_DISPATCH = re.compile(r"dispatched ENTRY [A-Z]+ .*?->\s*\[(?P<ids>[^\]]*)\]")

#: Hard boundaries a run must finish BEFORE (ICT). 11:25 leaves five minutes to
#: flatten before the 11:30 lunch close; 14:25 leaves five before the 14:30 ATC,
#: where DNSE refuses cancels and fills whatever rests.
MORNING_DEADLINE = dtime(11, 25)
AFTERNOON_DEADLINE = dtime(14, 25)


@dataclass
class Decision:
    proceed: bool
    exit_code: int
    reason: str
    cancelled: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)


def _default_run(args: list[str]) -> tuple[int, str]:
    completed = subprocess.run(
        [sys.executable, str(VENUE), *args], capture_output=True, text=True,
        stdin=subprocess.DEVNULL, timeout=180)
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


# ------------------------------------------------------------------ window

def window_decision(now: datetime, phase: str, run_timeout_s: float) -> Decision:
    """May an arm START now and still finish inside its session block?

    F2(b): there was no session gate at all, and L0 does not supply one — it
    SKIPS conditional probes outside a session and still exits 0
    (l0_order_semantics.py:271-278, :382). So `--arm poll` launched at 14:00
    passed every gate and ran 45 minutes straight through the 14:30 ATC, the
    one phase where DNSE refuses cancels and fills whatever is resting.
    """
    token = (phase or "").split()[0] if (phase or "").strip() else ""
    if token != "continuous":
        return Decision(False, EXIT_NEGATIVE,
                        f"session phase is {phase!r}, not continuous — refusing "
                        f"to start (L0 exits 0 off-session and cannot be used "
                        f"as this gate)")
    finish = now + timedelta(seconds=float(run_timeout_s))
    deadline_t = MORNING_DEADLINE if now.time() < dtime(12, 0) else AFTERNOON_DEADLINE
    deadline = now.replace(hour=deadline_t.hour, minute=deadline_t.minute,
                           second=0, microsecond=0)
    if finish > deadline:
        return Decision(False, EXIT_NEGATIVE,
                        f"a run started now would finish at "
                        f"{finish:%H:%M:%S} ICT, past the {deadline:%H:%M} "
                        f"boundary — refusing. Shorten --window-bars or use the "
                        f"other session block.")
    return Decision(True, EXIT_OK,
                    f"window open: finishes by {finish:%H:%M:%S} ICT, before "
                    f"{deadline:%H:%M}")


# ----------------------------------------------------------------- cleanup

def entry_ids(log_text: str) -> list[str]:
    """EVERY venue id this run dispatched as an entry, in order, deduped."""
    found: list[str] = []
    for match in _ENTRY_DISPATCH.finditer(_ANSI.sub("", log_text)):
        for piece in match.group("ids").split(","):
            ident = piece.strip().strip("'\"")
            if ident and ident not in found:
                found.append(ident)
    return found


def cleanup_decision(log_text: str, *, run=_default_run,
                     symbol: str | None = None) -> Decision:
    """Cancel every dispatched entry, prove flat, then allow a fallback.

    F1: the fallback l2 places a MARKET order. If ANY entry stop is still
    resting when it runs, the stop can trigger afterwards and the account holds
    two contracts, one of them unbracketed. So the fallback is gated on
    POSITIVE evidence — every cancel returned 0 AND the account reads flat
    immediately before — and every failure path stops the ladder instead.

    An id we could not read is NOT a reason to proceed: the shipped version
    refused to guess and then launched the fallback anyway two lines later.
    """
    ids = entry_ids(log_text)
    symbol_args = ["--symbol", symbol] if symbol else []
    if not ids:
        # No dispatch line at all. Either the run never entered (harmless) or
        # our log is incomplete (not harmless). Only the venue can tell us.
        code, output = run([*symbol_args, "flat"])
        if code == EXIT_OK:
            return Decision(True, EXIT_OK,
                            "no entry was dispatched and the account is flat — "
                            "a fallback may run")
        return Decision(False, EXIT_UNKNOWN,
                        f"no entry id could be read from our log AND the "
                        f"account is not provably flat (venue.py flat exit "
                        f"{code}). FLATTEN NOW, operator — a resting entry may "
                        f"exist that this runner cannot name.\n{output.strip()}")

    cancelled: list[str] = []
    unresolved: list[str] = []
    for ident in ids:
        code, output = run([*symbol_args, "cancel", ident])
        if code == EXIT_OK:
            cancelled.append(ident)
        else:
            unresolved.append(f"{ident} (exit {code})")
    if unresolved:
        return Decision(
            False, EXIT_NEGATIVE,
            f"cancel did NOT confirm for {', '.join(unresolved)} — an expired "
            f"trading token or a #46/#51 conditional-write refusal leaves the "
            f"entry RESTING. FLATTEN NOW, operator; the ladder is stopped. "
            f"(cancelled: {cancelled or 'none'})",
            cancelled, unresolved)

    code, output = run([*symbol_args, "flat"])
    if code != EXIT_OK:
        return Decision(
            False, EXIT_UNKNOWN if code == EXIT_UNKNOWN else EXIT_NEGATIVE,
            f"every cancel returned 0 but the account is not provably flat "
            f"(venue.py flat exit {code}; 2 = COULD NOT DETERMINE, never "
            f"'no'). FLATTEN NOW, operator — refusing to place a fallback "
            f"market order on top of it.\n{output.strip()}",
            cancelled, unresolved)
    return Decision(True, EXIT_OK,
                    f"cleanup complete: {len(cancelled)} entry order(s) "
                    f"cancelled and the account reads FLAT — a fallback may run",
                    cancelled, unresolved)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    win = sub.add_parser("window", help="may this arm start?")
    win.add_argument("--phase", required=True)
    win.add_argument("--run-timeout", type=float, required=True)
    clean = sub.add_parser("cleanup", help="post-NO-SAMPLE cleanup + fallback gate")
    clean.add_argument("--log", required=True)
    clean.add_argument("--symbol", default=None)
    args = parser.parse_args(argv)

    if args.cmd == "window":
        decision = window_decision(datetime.now(ICT), args.phase, args.run_timeout)
    else:
        path = Path(args.log)
        if not path.is_file():
            print(f"log {path} does not exist — COULD NOT DETERMINE; the "
                  f"ladder stops rather than guessing.")
            return EXIT_UNKNOWN
        decision = cleanup_decision(path.read_text(errors="replace"),
                                    symbol=args.symbol)
    print(decision.reason)
    return decision.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
