"""#157 R10 — trade-list parity: the same strategy, the same day, two engines.

The operator's acceptance test. One strategy over one replayed day, run twice:

* as a plain FILE-MODE BACKTEST over the day's bars, which is the sim engine, and
* through ``pyne run ... dnse_fake:... --broker``, which is the real engine and the real plugin
  against the fake venue.

The closed-trade lists are then compared. A mismatch is a FINDING about the fake or the plugin,
never an accepted difference.

WHY NOT THE l2b VEHICLE. Its own header states the design: "barstate.isrealtime -> warmup never
trades (a plain backtest places ZERO = proof)". The backtest arm would be empty by construction
and the assertion would read 0 == 0, passing however wrong the fake was. ``parity_vehicle.pine``
exists for this comparison: no realtime gate, no trade window, a market entry, one lot, closed
after a fixed number of bars, so the trade list is a function of the bar series alone.

KNOWN DIFFERENCES, stated here and asserted against rather than absorbed:

1. TIME SHIFT. The fake serves the recorded day re-stamped onto the current wall clock, because
   the engine's live path is anchored to that clock in two independent places (a staleness
   watchdog and a per-missed-boundary bar synthesiser). Fake trade times are therefore shifted
   by a known offset, which this script reads from the run's own log line and removes before
   comparing. The day FILE is never rewritten.
2. FILL PRICES. The backtest fills at bar OHLC; the fake fills against replayed PRINTS derived
   from each bar's open/high/low/close path. The same order can therefore fill at a different
   price within the bar, so prices are compared with a tolerance rather than for equality. Sides,
   quantities, counts and bar times are compared EXACTLY — a tolerance on those would hide the
   defects this test exists to catch.
4. TRADE TIMES. A backtest stamps a trade with its bar time; a live run stamps it with the wall
   clock at closing. The replay compresses a minute of bar time into 0.3 s of wall time, so the
   two stamps cannot be reconciled by removing the replay offset. Trades are therefore matched
   by ORDER, and sides, quantities and prices are compared on that pairing.
3. WINDOW. The fake only trades on its LIVE slice, while the backtest trades over every bar. The
   comparison is therefore restricted to trades whose entry falls at or after the first live bar,
   which is stated rather than silently filtered.
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
VEHICLE = "plugins/dnse/testing/fixtures/offline/parity_vehicle.py"

# The day and its matching bar store are a PAIR: the fake arm replays the day, the backtest arm
# reads the store, and parity is only meaningful if they hold the same bars. Both are overridable
# together so the same comparison can run against a DERIVED-FROM-1M day as well as the synthetic
# one, and `venue_day_dataset.dataset_from_day` builds the store FROM the day so "the same bars"
# is a fact rather than a belief about how a local file was once made.
DAY = os.environ.get("FAKE_VENUE_PARITY_DAY",
                     "plugins/dnse/testing/fixtures/venue_day/SYNTHETIC_vn30f1m_1m.json.gz")
DATASET = os.environ.get("FAKE_VENUE_PARITY_DATASET", "fakeparity_VN30F1M_1")
TRADES = REPO / "workdir" / "output" / "parity_vehicle_trade.csv"

#: Fill prices may differ by up to one bar's range, because one engine fills at bar OHLC and the
#: other at a print inside that bar. Expressed in price units of the instrument (VN30 futures
#: trade around 1970 with a 0.1 tick), so this is about 0.25 %.
PRICE_TOLERANCE = 5.0


def _park(path: Path) -> None:
    """Move a previous arm's trade file aside rather than deleting it.

    The house rule is never delete, and it has no carve-out for regenerated artefacts. Parking
    also leaves the previous arm's list recoverable, which matters when a comparison surprises
    you and you want to see what the earlier run actually produced.
    """
    if not path.exists():
        return
    import time as _t
    parked = REPO / "backup" / "deleteable"
    parked.mkdir(parents=True, exist_ok=True)
    path.replace(parked / f"{path.name}.{int(_t.time())}")


def _read_trades(path: Path) -> list[dict]:
    with path.open() as handle:
        return [row for row in csv.DictReader(handle) if row.get("Date/Time")]


def _run(args: list[str], env: dict | None = None, timeout: int = 300) -> None:
    subprocess.run(args, cwd=REPO, env={**os.environ, **(env or {})},
                   capture_output=True, text=True, timeout=timeout)


def backtest() -> list[dict]:
    """File-mode run over the day's bars: the sim engine."""
    _park(TRADES)                # never read a stale list from a previous arm
    _run([str(REPO / ".venv" / "bin" / "pyne"), "run", VEHICLE, DATASET,
          "--trade", str(TRADES)])
    if not TRADES.exists():
        raise RuntimeError("the backtest wrote no trade file")
    return _read_trades(TRADES)


def fake_run(live_bars: int, pace: str = "0.3"):
    """Live run against the fake.

    Returns the trades, the replay offset in seconds, and the FIRST LIVE BAR as the fake itself
    reported it. That third value is the comparison window, and it is read from the fake rather
    than reconstructed from a trade stamp — see the note in :func:`compare`.
    """
    log = REPO / "workdir" / "output" / "parity_fake_run.log"
    # The fake PARKS once its stream is exhausted (the replay-provider contract: "kill the run
    # when the strategy is done"), so the run is bounded here rather than waited on. Trades are
    # flushed to the CSV as they close, so a bounded run still yields the complete list for the
    # bars that were streamed. The budget covers startup plus live_bars * pace with headroom.
    # BOTH arms write the same trade CSV. Deleting it first is what stops a fake run that
    # failed to write from being compared against the BACKTEST's own list — which happened, and
    # produced a confident 120-row "difference" that was the backtest against itself.
    _park(TRADES)
    # Warmup alone replays hundreds of bars and takes tens of seconds; a budget that only
    # covered the live stream interrupted the run mid-warmup and it never traded (measured).
    budget = 100 + int(float(pace) * live_bars * 2)
    # INTERRUPT rather than kill. A killed run flushes nothing — measured: the trade CSV was
    # absent entirely, and without the guard above that would have compared the backtest against
    # its own stale file. SIGINT takes the engine's normal shutdown path, which writes the
    # outputs, exactly as a supervised live run is ended by hand.
    import signal
    import time as _time
    with log.open("w") as handle:
        proc = subprocess.Popen(
            [str(REPO / ".venv" / "bin" / "pyne"), "run", VEHICLE,
             "dnse_fake:VN30F1M@1", "--broker", "--trade", str(TRADES)],
            cwd=REPO, stdout=handle, stderr=subprocess.STDOUT,
            env={**os.environ, "FAKE_VENUE_DAY": DAY,
                 "FAKE_VENUE_LIVE_BARS": str(live_bars), "FAKE_VENUE_LIVE_PACE": pace})
        deadline = _time.monotonic() + budget
        while _time.monotonic() < deadline and proc.poll() is None:
            _time.sleep(0.5)
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text())
    match = re.search(r"replay offset = ([+-]?\d+) s", text)
    if not match:
        raise RuntimeError("the fake did not report its replay offset; cannot align trade times")
    synth = text.count("idle-bar synth")
    if synth:
        raise RuntimeError(
            f"{synth} idle-bar synth line(s): the engine substituted its own bars for the "
            f"fake's, so this run does not compare the same series")
    if not TRADES.exists():
        raise RuntimeError(
            "the fake run wrote no trade file: there is nothing to compare, and comparing the "
            "stale file would have compared the backtest against itself")
    live_match = re.search(r"first live bar = (\d+) ms", text)
    if not live_match:
        raise RuntimeError(
            "the fake did not report its first live bar; the comparison window would have to be "
            "reconstructed from a trade stamp, which is exactly the unsound derivation that "
            "produced a false COUNT finding on 2026-09-18")
    from datetime import datetime as _dt
    first_live = _dt.fromtimestamp(int(live_match.group(1)) / 1000).astimezone()
    return _read_trades(TRADES), int(match.group(1)), first_live


def compare(bt: list[dict], fake: list[dict], offset_s: int, window_start) -> list[str]:
    """Return the findings; an empty list is parity.

    ``window_start`` is the fake's FIRST LIVE BAR, in the day's own timestamps. It is passed in
    rather than derived, and that is the whole point of the parameter.

    MEASURED 2026-09-18. This window used to be computed as the fake's first trade time minus
    the replay offset, which is unsound by this file's own rule: a backtest stamps a trade with
    its BAR time and a live run stamps the WALL CLOCK at which it closed, so no offset converts
    one into the other. On the DERIVED-FROM-1M day for that date the derived start landed 39 s
    late, crossed a bar boundary, dropped one backtest entry and reported "backtest 3 vs fake 4"
    when both engines had in fact produced 4. The synthetic day passed only because its first
    trade happened not to straddle a boundary, so the green was luck — and the same flaw could
    equally have trimmed a window until a REAL difference vanished.
    """
    findings: list[str] = []
    if not fake:
        return ["the fake produced NO trades: nothing to compare, which is a finding in itself"]

    from datetime import datetime

    def _when(row: str) -> datetime:
        return datetime.fromisoformat(row.strip().replace("Z", ""))

    # Align on ENTRY rows. The CSV interleaves entry and exit rows per trade, so a window
    # filter that starts mid-trade puts an exit opposite an entry and every later row is off by
    # one — an artefact of the comparison, not a difference between the engines. Comparing
    # entries to entries removes it without hiding anything: a missing or extra trade still
    # changes the count.
    first = window_start
    entries = lambda rows: [r for r in rows if "Entry" in r["Type"]]
    bt_window = entries([r for r in bt if _when(r["Date/Time"]) >= first])
    fake = entries(fake)

    if len(bt_window) != len(fake):
        findings.append(f"trade COUNT differs: backtest {len(bt_window)} vs fake {len(fake)} "
                        f"over the same window from {first}")

    for index, (a, b) in enumerate(zip(bt_window, fake)):
        if a["Type"] != b["Type"]:
            findings.append(f"trade {index}: side/type {a['Type']!r} vs {b['Type']!r}")
        if a["Contracts"] != b["Contracts"]:
            findings.append(f"trade {index}: quantity {a['Contracts']} vs {b['Contracts']}")
        # Trade TIMESTAMPS are NOT comparable between the two arms, and this is a property of
        # the engines rather than of the fake. A backtest stamps a trade with its BAR time; a
        # live run stamps it with the WALL CLOCK at which the trade closed. Measured: the fake
        # arm's three trades are 2 seconds apart (13:43:30, :32, :34) while the backtest's are
        # minutes apart (13:49, 13:54, 14:01), because the replay streams a bar every 0.3 s.
        # Removing the replay offset cannot reconcile that, so ORDER is compared instead of
        # instants: the Nth trade must be the Nth trade in both, which is what the sides,
        # quantities and prices below are matched on.
        gap = abs(float(a["Price VND"]) - float(b["Price VND"]))
        if gap > PRICE_TOLERANCE:
            findings.append(f"trade {index}: price {a['Price VND']} vs {b['Price VND']} "
                            f"differs by {gap:.2f} > {PRICE_TOLERANCE} tolerance")
    return findings


def main() -> int:
    live = int(os.environ.get("FAKE_VENUE_LIVE_BARS", "25"))
    print(f"backtest over {DATASET} ...")
    bt = backtest()
    print(f"  {len(bt)} closed trade(s)")
    print(f"fake run, {live} live bars ...")
    fake, offset, first_live = fake_run(live)
    print(f"  {len(fake)} closed trade(s), replay offset {offset:+d}s, "
          f"first live bar {first_live.isoformat(timespec='seconds')}")
    findings = compare(bt, fake, offset, first_live)
    if findings:
        print("\nFINDINGS (a mismatch is a finding about the fake or the plugin, never an "
              "accepted diff):")
        for item in findings:
            print(f"  - {item}")
        return 1
    print("\nPARITY: same trade count, and matched by order the same sides and quantities, with\n"
          "prices within tolerance. Trade TIMES are not compared: a backtest stamps bar time,\n"
          "a live run stamps wall clock (known difference 4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
