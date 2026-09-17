#!/usr/bin/env python3
"""#132 W0 — naked-position sidecar. READ-ONLY, ALARM-ONLY.

Watches ONE question, out of process, while a live run trades:

    an OPEN, BOT-OWNED position must have a RESTING protective order at the
    venue that can reduce it.

It sends NOTHING. No close, no cancel, no re-arm, no engine hook. The action
arm is deferred by the card's adjudication behind #122, a run_controls seam, a
measured leg-fire and a scored shadow period — so this process cannot, by
construction, become the second writer that turns a detection into an incident.

WHY OUT OF PROCESS. An in-engine check shares its subject's fate and its
subject's beliefs. #120's aftermath is an engine that DIED holding a position;
#135 was an engine whose own position belief went to 2.0 against a venue
holding 1. A watchdog reading ``self._position.size`` would have concurred with
both. This one reads the VENUE for position and the durable JOURNAL for
ownership, so it is independent of the belief it audits.

WHAT IT CANNOT DO, stated plainly: it cannot see a protective exit the engine
never journaled, it cannot price-check a level (that is W1), and its session
ladder is a CLOCK, so an unlisted exchange holiday reads as a trading day. The
alarm ladder bounds what that costs.

    naked_watch.py                    # loop until stopped
    naked_watch.py --once             # one cycle, exit 0/1/2
    naked_watch.py --interval 5       # poll cadence (default 5s)

EXIT CODES — ``venue.py``'s meanings exactly (0 affirmative / 1 negative /
2 could-not-determine). In loop mode the precedence differs deliberately:
ALARM outranks could-not-determine, so a run that ever alarmed can never
report as merely inconclusive because a later read failed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import naked_position as core                                     # noqa: E402
import venue                                                      # noqa: E402
from flatten import _read_position_size_confirmed                 # noqa: E402
from naked_position import Observation, RestingOrder, Verdict     # noqa: E402
# IMPORTED, never retyped: the engine owns the wording graders grep for.
from pynecore.core.broker.sync_engine import (                    # noqa: E402
    PROTECTION_UNPROTECTED_MARKER,
)

ICT = timezone(timedelta(hours=7))
STORE = REPO / "workdir" / "output" / "logs" / "broker.sqlite"
ALARM_LOG = REPO / "workdir" / "output" / "logs" / "naked_watch_alarms.log"

#: Journal ``extras`` keys mirrored from the engine's own exclusions
#: (sync_engine.py:4531-4536): a startup-adopted row is ANOTHER run's slice,
#: and a retirement counter is exposure the venue no longer holds.
_ADOPTED_KEYS = ("adopted_startup_extra", "adopted_startup")
_RETIRED_KEYS = ("journal_exposure_retired", "exposure_retired")


# ----------------------------------------------------------------- sight

def prove_sight(broker) -> tuple[bool, str]:
    """Can this watcher actually SEE the account? -> (proven, detail).

    #145: ``resolve_contract`` caches per instance FOREVER and, on a FAILED
    instruments read, caches the UNRESOLVED ALIAS — which is also the
    legitimate passthrough for stocks, so the error path and success path are
    byte-identical. ``get_position`` then filters venue rows on that alias,
    matches nothing, and returns ``None`` — FLAT — with no exception and no
    log line, for the life of the process. A watchdog on that read answers
    OK forever over a real position.

    Two agreeing reads CANNOT catch it (both agree, both blind), so sight is
    proven separately: the instruments read must succeed AND, for a
    derivative alias, the resolved code must differ from the alias.
    """
    symbol = broker.symbol or ""
    try:
        resolved = broker.resolve_contract()
    except Exception as exc:                                      # noqa: BLE001
        return False, f"resolve_contract RAISED {type(exc).__name__}: {exc}"
    try:
        status, body = broker.client.get_instruments(limit=200)
    except Exception as exc:                                      # noqa: BLE001
        return False, f"instruments read RAISED {type(exc).__name__}: {exc}"
    if status != 200 or not isinstance(body, dict):
        return False, (f"instruments read answered HTTP {status} — a cached "
                       f"alias cannot be ruled out (#145)")
    codes = {str(r.get("symbol")) for r in (body.get("data") or [])}
    if symbol.upper().startswith("VN30F") and resolved == symbol:
        return False, (f"{symbol!r} resolved to ITSELF — that is the #145 "
                       f"signature: a derivative alias is never a tradable "
                       f"code, so this is a cached failed read")
    if codes and resolved not in codes:
        return False, (f"resolved contract {resolved!r} is absent from the "
                       f"instruments catalogue — stale cache across a roll "
                       f"(#113)?")
    return True, f"{symbol} -> {resolved} (confirmed against the catalogue)"


# --------------------------------------------------------------- journal

def journal_attribution(store_path, account_id: str):
    """-> (owned_ids, exposure, per_id) or ``(None, None, None)``.

    READ-ONLY over a direct sqlite connection, concurrently with a live run —
    the ``flatten.py::owned_live_ids`` pattern, and for its reasons: the
    normal API's ``open_run`` raises on the live-row collision for a whole
    5-minute stale window and would steal the run identity from the #77/T16
    restart adoption. ``BrokerStore`` is never instantiated: its constructor
    CREATES the file, which would make absence read as a clean account.

    ``None`` = attribution UNAVAILABLE (missing/unreadable store, or no runs
    for this account) and the caller must degrade to could-not-determine —
    never to "nothing owned" (the #91 vacuous-pass guard).

    TWO DIFFERENT SCOPES, deliberately:

    * **ownership of an ORDER id** is day-scoped for digit-shaped (NORMAL-
      class) ids, because DNSE REUSES those across days and a stale row could
      claim the operator's order on a shared netting account (#96).
    * **exposure** is NOT day-scoped. Day-scoping it would drop the entry row
      of a position held overnight and answer "we own nothing" — silence over
      a real naked position, the one direction this tool must not have. The
      cost is the opposite error: stale rows from finished runs inflate
      exposure (measured: a finished run sums to -2 with every contributing
      row Filled-but-unclosed, #89). That inflation is bounded by the venue
      clamp and surfaces as an alarm or a sign disagreement — both LOUD.
    """
    path = Path(store_path)
    if not path.is_file():
        print(f"attribution UNAVAILABLE: no store at {path}")
        return None, None, None
    day_start_ms = int(datetime.now(ICT).replace(
        hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            known = conn.execute("SELECT COUNT(*) FROM runs WHERE account_id = ?",
                                 (account_id,)).fetchone()[0]
            if not known:
                print(f"attribution UNAVAILABLE: no journalled runs for this "
                      f"account in {path.name}")
                return None, None, None
            rows = conn.execute(
                "SELECT o.exchange_order_id, o.side, o.filled_qty, o.from_entry,"
                "       o.extras,"
                "       MAX(COALESCE(o.created_ts_ms,0), COALESCE(o.updated_ts_ms,0))"
                " FROM orders o JOIN runs r"
                "   ON r.run_instance_id = o.run_instance_id"
                " WHERE o.closed_ts_ms IS NULL AND r.account_id = ?",
                (account_id,)).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        print(f"attribution UNAVAILABLE: store read failed: {exc}")
        return None, None, None

    owned: set[str] = set()
    per_id: dict[str, dict] = {}
    exposure = 0.0
    for venue_id, side, filled_qty, from_entry, extras_json, last_ms in rows:
        extras = {}
        if extras_json:
            try:
                extras = json.loads(extras_json)
            except (TypeError, ValueError):
                extras = {}
        filled = float(filled_qty or 0.0)
        if filled and not any(extras.get(k) for k in _ADOPTED_KEYS):
            for key in _RETIRED_KEYS:
                retired = extras.get(key)
                if isinstance(retired, (int, float)):
                    filled = max(0.0, filled - min(filled, float(retired)))
            if filled:
                exposure += filled if side == "buy" else -filled
        vid = str(venue_id or "")
        if not vid:
            continue
        if vid.isdigit() and (last_ms or 0) < day_start_ms:
            continue                     # the #96 cross-day id-reuse trap
        owned.add(vid)
        per_id[vid] = {"from_entry": from_entry,
                       "leg_kind": extras.get("leg_kind") or None}
    return owned, exposure, per_id


# ----------------------------------------------------------------- venue

class _PhantomCache:
    """Memoized #41 classification. POSITIVE results only.

    A triggered conditional stays ``Activated`` forever — that is terminal,
    so caching it is sound and stops the per-cycle detail cost from growing
    monotonically all day as shells accumulate (an unmemoized watcher walks
    itself into a self-inflicted 429). A NEGATIVE result is NOT cached: a
    ``New`` conditional can still become ``Activated`` later.
    """

    def __init__(self) -> None:
        self._phantom: dict[str, str] = {}

    def classify(self, broker, order) -> tuple[bool, bool, dict]:
        """-> (is_phantom, classifiable, row)."""
        oid = str(order.id)
        if oid in self._phantom:
            return True, True, {}
        try:
            _book, row = venue._detail_today(broker, oid)
        except Exception as exc:                                  # noqa: BLE001
            print(f"   detail read FAILED for {oid}: {type(exc).__name__}: {exc}")
            return False, False, {}
        if row is None:
            return False, False, {}
        if (str(row.get("orderStatus", "")).upper() == "ACTIVATED"
                and row.get("externalOrderId")):
            self._phantom[oid] = str(row.get("externalOrderId"))
            return True, True, row
        return False, True, row


def read_resting(broker, symbol, owned, per_id, phantoms):
    """-> tuple of :class:`RestingOrder`, or ``None`` when a book failed.

    ``get_open_orders`` refuses a partial book union (broker.py:3148-3155),
    so a failed read raises rather than looking empty — exactly the shape
    this tool needs. Phantom shells are dropped: they are consumed, and
    counting one as cover would be silence over a naked position.
    """
    try:
        working = asyncio.run(broker.get_open_orders(symbol))
    except Exception as exc:                                      # noqa: BLE001
        print(f"   working-order read FAILED: {type(exc).__name__}: {exc}")
        return None
    resting = []
    for order in working:
        is_phantom, classifiable, row = phantoms.classify(broker, order)
        if is_phantom:
            continue
        oid = str(order.id)
        attribution = per_id.get(oid) or {}
        resting.append(RestingOrder(
            venue_id=oid,
            side=str(getattr(order, "side", "") or ""),
            qty=float(getattr(order, "qty", 0.0) or 0.0),
            owned=oid in owned,
            from_entry=attribution.get("from_entry"),
            leg_kind=attribution.get("leg_kind"),
            classifiable=classifiable,
            book=str(row.get("orderCategory")) if row else None,
            stop_price=getattr(order, "stop_price", None)))
    return tuple(resting)


# ------------------------------------------------------------- heartbeat

class Heartbeat:
    """A daemon ticker carrying the AGE OF THE LAST COMPLETED EVALUATION.

    Not "I am alive" — that is what a loop-emitted heartbeat would say right
    up until the loop hung. There is no asyncio deadline on the venue detail
    reads and urllib3 defaults are connect=30/read=60, so a worst-case cycle
    runs for minutes; a heartbeat emitted BY the cycle would simply stop, and
    silence is ambiguous between "hung" and "not started". Reporting the age
    from a separate thread makes a hang VISIBLE and rising.

    Copied in shape from ``_LiveHeartbeat`` (cli/commands/run.py:185-215):
    ``threading.Event().wait`` rather than a busy-poll, so ``stop()`` returns
    promptly.
    """

    def __init__(self, interval_s: float, emit) -> None:
        self._interval = interval_s
        self._emit = emit
        self._stop = threading.Event()
        self._last_done = time.monotonic()
        self._seq = 0
        self._thread = threading.Thread(
            target=self._run, name="naked-watch-heartbeat", daemon=True)

    def mark_evaluated(self) -> None:
        self._last_done = time.monotonic()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._seq += 1
            self._emit(self._seq, time.monotonic() - self._last_done)

    def start(self) -> None:
        if self._interval > 0.0:
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()


# ----------------------------------------------------------------- cycle

def _stamp() -> str:
    return f"{datetime.now(ICT):%Y-%m-%d %H:%M:%S} ICT"


def evaluate_once(broker, symbol, store_path, account_id, *,
                  bar_period_s=0.0, sight=None):
    """One full cycle -> :class:`Assessment`. Every read failure degrades."""
    proven, detail = sight if sight is not None else prove_sight(broker)
    if not proven:
        print(f"   sight: {detail}")
    owned, exposure, per_id = (None, None, None)
    resting = None
    position = None
    if proven:
        position = _read_position_size_confirmed(broker, symbol)
        owned, exposure, per_id = journal_attribution(store_path, account_id)
        if position is not None and owned is not None:
            resting = read_resting(broker, symbol, owned, per_id, phantoms=_PHANTOMS)
    return core.evaluate(Observation(
        phase=venue.session_phase(),
        contract_proven=proven,
        position_signed=position,
        owned_exposure=exposure,
        resting=resting,
        bar_period_s=bar_period_s))


_PHANTOMS = _PhantomCache()


def _alarm(line: str, alarm_log: Path) -> None:
    """Loud on stdout AND appended to a dedicated file, flushed immediately.

    The dedicated file IS the out-of-band channel for this stage: the prior
    panel established that no pager, webhook or push path exists in this repo
    and that a live run's stdout is buried in ~500K of ANSI spinner noise, so
    an EMERGENCY line emitted only there is lost on arrival. A real push
    channel is its own card.
    """
    print(line, flush=True)
    try:
        alarm_log.parent.mkdir(parents=True, exist_ok=True)
        with alarm_log.open("a") as handle:
            handle.write(line + "\n")
            handle.flush()
    except OSError as exc:
        print(f"   ALARM LOG WRITE FAILED ({exc}) — this line exists only on "
              f"stdout", flush=True)


def run(args) -> int:
    broker = venue.broker(args.symbol)
    symbol = broker.symbol or args.symbol
    account_id = getattr(broker, "account_id", None) or ""
    store_path = Path(args.store)
    alarm_log = Path(args.alarm_log)
    ladder = core.AlarmLadder(window_s=core.confirm_window_s(args.bar_period))
    seen: list[Verdict] = []

    print(f"naked_watch: {symbol} every {args.interval:g}s, confirm window "
          f"{ladder.window_s:.0f}s, store {store_path.name}, alarms -> "
          f"{alarm_log}")
    sight = prove_sight(broker)
    print(f"sight: {'PROVEN' if sight[0] else 'NOT PROVEN'} — {sight[1]}")

    heartbeat = Heartbeat(args.heartbeat, lambda seq, age: print(
        f"[HEARTBEAT {seq}] {_stamp()} last evaluation {age:.0f}s ago",
        flush=True))
    if not args.once:
        heartbeat.start()
    try:
        while True:
            assessment = evaluate_once(
                broker, symbol, store_path, account_id,
                bar_period_s=args.bar_period,
                sight=sight if sight[0] else None)
            seen.append(assessment.verdict)
            heartbeat.mark_evaluated()
            print(f"[{assessment.verdict.value}] {_stamp()} "
                  f"{assessment.reason}", flush=True)
            alarm_line, note = ladder.observe(assessment.verdict, time.monotonic())
            if note:
                print(f"   {note}", flush=True)
            if alarm_line:
                _alarm(f"{PROTECTION_UNPROTECTED_MARKER} [{core.SIDECAR_TAG}] "
                       f"{_stamp()}: {alarm_line} — {assessment.reason}. "
                       f"Operator: check `venue.py status` NOW. This watchdog "
                       f"only ALARMS; it has sent nothing.", alarm_log)
            if args.once:
                return assessment.verdict.exit_code
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    finally:
        heartbeat.stop()
    return core.worst_exit_code(seen)


def main(argv: "list[str] | None" = None) -> int:
    # STRICT parsing BEFORE anything touches the venue, and --help never
    # contacts it (the flatten_api.py lesson: a loose `in sys.argv` check let
    # --help fall through into a live account).
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default=venue.SYMBOL)
    parser.add_argument("--once", action="store_true",
                        help="one cycle, then exit with its verdict")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="poll cadence in seconds (default 5)")
    parser.add_argument("--heartbeat", type=float, default=30.0,
                        help="heartbeat cadence in seconds (0 disables)")
    parser.add_argument("--bar-period", type=float, default=0.0,
                        help="the run's bar period in seconds; the naked-confirm "
                             "window is max(30s, this)")
    parser.add_argument("--store", default=str(STORE))
    parser.add_argument("--alarm-log", default=str(ALARM_LOG))
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
