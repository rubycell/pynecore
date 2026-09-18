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
both. This one reads the VENUE for position, so its view of WHETHER A
POSITION EXISTS is independent of the belief it audits.

Its view of WHOSE the position is, is NOT — and an earlier draft of this
docstring wrongly claimed otherwise (review finding F1). Ownership comes
entirely from the engine's journalled ``filled_qty``, so a fill the engine
never journalled reads as foreign. That is why a venue position with no
journalled exposure is UNATTRIBUTED rather than OK: the honest answer is that
this tool cannot tell the operator's position from our own unjournalled one.

WHAT IT CANNOT DO, stated plainly: it cannot see a protective exit the engine
never journaled, it cannot price-check a level (that is W1), and its session
ladder is a CLOCK, so an unlisted exchange holiday reads as a trading day. The
alarm ladder bounds what that costs.

    naked_watch.py                    # loop until stopped
    naked_watch.py --once             # one cycle, exit 0/1/2
    naked_watch.py --interval 15      # poll cadence (default 15s)

EXPECT A CONTINUOUS EXIT 2 BESIDE A MANUAL POSITION. If the operator holds a
position this bot never journalled, the verdict is UNATTRIBUTED (exit 2) every
cycle: on a shared netting account "the venue holds a position our journal
never claimed" genuinely IS could-not-determine — it may be the operator's, or
it may be our own fill with the engine down between a stop trigger and the
child's adoption (#39/#120). That is the honest answer, not noise, and it is
deliberately not OK.

ONE STORE PER SYMBOL, or expect UNDETERMINED. Journal attribution is scoped by
ACCOUNT, not by symbol, so a stock run's open rows in the same store are summed
into the owned exposure of a futures run — a protected futures short beside an
open HPG position grades UNDETERMINED every cycle. A symbol filter is the fix
and is a follow-up on #132; until then, point ``--store`` at a store whose runs
share this symbol.

EXIT CODES — ``venue.py``'s meanings exactly (0 affirmative / 1 negative /
2 could-not-determine). NOTE on loop mode: exit 1 means SOME CYCLE GRADED
NAKED, which is not the same as "an alarm fired" — a single sub-window naked
cycle that recovered (the healthy post-entry gap) still sets it, while the
alarm ladder deliberately stayed quiet. Read the ALARM LINE for whether the
operator was paged; read the exit code for whether any cycle was ever naked. In loop mode the precedence differs deliberately:
ALARM outranks could-not-determine, so a run that ever alarmed can never
report as merely inconclusive because a later read failed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import os
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
        # require_contract (#145) RAISES rather than falling back to the alias,
        # and it discriminates on PROVENANCE — a 200 that simply lacks the row
        # (the catalogue is paged) is unresolved too, which a status check
        # cannot see. W0 is its first consumer, which is fitting: a watchdog
        # that cannot establish the contract must say BLIND, not read FLAT.
        # This deliberately does NOT re-derive "is it resolved?" — that policy
        # now lives in one place, and the second copy is what produced the
        # worst_exit_code miss two rounds ago.
        resolved = broker.require_contract()
    except Exception as exc:                                      # noqa: BLE001
        return False, (f"contract UNRESOLVED ({type(exc).__name__}: {exc}) — "
                       f"every position read would filter on an alias and "
                       f"answer FLAT (#145)")
    try:
        status, body = broker.client.get_instruments(limit=200)
    except Exception as exc:                                      # noqa: BLE001
        return False, f"instruments read RAISED {type(exc).__name__}: {exc}"
    if status != 200 or not isinstance(body, dict):
        return False, (f"instruments read answered HTTP {status} — a cached "
                       f"alias cannot be ruled out (#145)")
    rows = body.get("data") or []
    if not rows:
        # A 200 with an EMPTY catalogue proved sight under the first cut
        # (`if codes and ...` skipped the check entirely). Empty is not an
        # answer: it is a read that told us nothing.
        return False, ("the instruments catalogue came back EMPTY — that is "
                       "not evidence the cached contract is current")
    # The alias-resolves-to-itself check that stood here is GONE: that is
    # exactly what require_contract now refuses, above, and keeping a local
    # copy would mean two definitions of "unresolved" drifting apart.
    #
    # THE ROLL (#113), and membership alone does not survive it. The provider
    # caches a RESOLVED code permanently, and for a while after the repoint the
    # catalogue lists BOTH the expired and the new contract — so a stale code
    # is still "in codes", passes, and then get_position filters on a contract
    # the account no longer holds, answers None, and the sidecar reports
    # "nothing bot-owned is open" over a position held in the NEW code. The
    # binding question is not "does this code exist?" but "is this code the one
    # our ALIAS points at TODAY?".
    current = [str(r.get("symbol")) for r in rows
               if str(r.get("symbolType") or "") == symbol.upper()]
    if current:
        if resolved not in current:
            return False, (f"{symbol} now maps to {current[0]!r} but this "
                           f"process has {resolved!r} cached — the alias has "
                           f"REPOINTED (#113 roll). Restart the sidecar; its "
                           f"position reads are filtering on an expired "
                           f"contract and will answer FLAT.")
        return True, (f"{symbol} -> {resolved} (confirmed as the CURRENT "
                      f"mapping, not merely present in the catalogue)")
    if resolved not in {str(r.get("symbol")) for r in rows}:
        return False, (f"resolved contract {resolved!r} is absent from the "
                       f"instruments catalogue — stale cache across a roll "
                       f"(#113)?")
    return True, (f"{symbol} -> {resolved} (present in the catalogue; no "
                  f"symbolType row to confirm it is the current mapping)")


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
                "SELECT o.exchange_order_id, o.side, o.qty, o.filled_qty,"
                "       o.from_entry, o.extras,"
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
    for venue_id, side, qty, filled_qty, from_entry, extras_json, last_ms in rows:
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
        # N1: a PRIOR-DAY numeric id is not dropped outright any more. Dropping
        # it made a real OVERNIGHT bracket stop counting as cover, so a
        # protected position graded NAKED and re-warned all morning; keeping it
        # unconditionally would let an id the venue REISSUED today (the #96
        # trap) pass as our cover. It is kept but FLAGGED, and `read_resting`
        # corroborates it against the venue record before believing it.
        stale_numeric = bool(vid.isdigit() and (last_ms or 0) < day_start_ms)
        per_id[vid] = {"from_entry": from_entry,
                       "leg_kind": extras.get("leg_kind") or None,
                       "side": side, "qty": qty,
                       "stale_numeric": stale_numeric}
        if not stale_numeric:
            owned.add(vid)
    return owned, exposure, per_id


def _today_start_ms() -> int:
    """Midnight ICT today, in epoch ms — the same boundary the journal uses."""
    return int(datetime.now(ICT).replace(
        hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)


def _epoch_ms(value) -> "float | None":
    """DNSE serves epoch MILLISECONDS; tolerate seconds and ISO, else None.

    A date we cannot parse must not become a number — it would silently answer
    the reissue question in whichever direction the default happened to fall.
    """
    if value is None:
        return None
    try:
        number = float(value)
        return number if number > 1e11 else number * 1000.0
    except (TypeError, ValueError):
        pass
    try:
        text = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(text).timestamp() * 1000.0
    except (TypeError, ValueError):
        return None


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
        venue_side = str(getattr(order, "side", "") or "")
        venue_qty = float(getattr(order, "qty", 0.0) or 0.0)
        is_owned = oid in owned
        if attribution.get("stale_numeric"):
            # N1: our journal claims this id from a PRIOR day. Believe it only
            # if the resting order still looks like the one we journalled AND
            # the venue did not create it today (a today-created record under a
            # prior-day journal id is a reissue by definition).
            if core.stale_numeric_id_verdict(
                    attribution.get("side"), attribution.get("qty"),
                    venue_side, venue_qty,
                    venue_created_ms=_epoch_ms(row.get("createdDate")) if row else None,
                    day_start_ms=_today_start_ms()) == "owned":
                is_owned = True
            else:
                print(f"   id {oid}: prior-day journal row does not match the "
                      f"resting order (ours: {attribution.get('side')} "
                      f"{attribution.get('qty')}, venue: {venue_side} "
                      f"{venue_qty}) — the venue may have REISSUED this id "
                      f"(#96). Unclassifiable, not counted either way.")
                # BOTH flags matter. `owned` stays TRUE because our journal DOES
                # claim this id; `classifiable` goes False because we cannot
                # corroborate it. The core's poison rule requires owned AND
                # not-classifiable, so leaving owned False would drop it to a
                # plain foreign order and the cycle would grade NAKED — a
                # confident alarm built on evidence we just admitted we cannot
                # read.
                is_owned = True
                classifiable = False
        resting.append(RestingOrder(
            venue_id=oid,
            side=venue_side,
            qty=venue_qty,
            owned=is_owned,
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

    def __init__(self, interval_s: float, emit, stall_after_s: float = 0.0,
                 on_stall=None) -> None:
        self._interval = interval_s
        self._emit = emit
        self._stall_after = stall_after_s
        #: What a STALL does. The default IS the production behaviour — a hard
        #: ``os._exit`` from this thread, because the main thread is blocked in
        #: a venue read and will never observe a flag or an exception. It is a
        #: parameter only so the behaviour can be OBSERVED: with the hard exit
        #: wired in unconditionally, the stall pin killed the pytest process
        #: itself (exit 2, no summary, the whole suite gone), which is both
        #: untestable and a fair warning about what this does to anything
        #: sharing the interpreter. Injecting the ACTION keeps the shipped
        #: default honest while making it pinnable.
        self._on_stall = on_stall if on_stall is not None else os._exit
        self._stop = threading.Event()
        self._last_done = time.monotonic()
        self._seq = 0
        #: Set once the age passes ``stall_after_s``. The LOOP's exit code
        #: consults it: a heartbeat that only PRINTS a stall is a warning
        #: nobody's wrapper can act on, and a hung cycle is precisely when the
        #: operator needs a non-zero exit rather than a line in a log.
        self.stalled = False
        self._thread = threading.Thread(
            target=self._run, name="naked-watch-heartbeat", daemon=True)

    def mark_evaluated(self) -> None:
        self._last_done = time.monotonic()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._seq += 1
            age = time.monotonic() - self._last_done
            if self._stall_after and age >= self._stall_after and not self.stalled:
                self.stalled = True
                self._emit(self._seq, age)
                # EXIT FROM THE THREAD. The flag was consulted only at loop END,
                # so a genuinely hung cycle — a venue read with no asyncio
                # deadline, urllib3 defaults connect=30/read=60 — never reached
                # it: the process printed STALLED forever and exited 2 only if
                # someone pressed Ctrl-C. A watchdog that has stopped watching
                # must STOP, loudly and by itself, or its supervisor cannot
                # restart it. os._exit because the main thread is blocked in a
                # read and will not observe a flag or an exception.
                print("!! HEARTBEAT STALL — exiting could-not-determine so a "
                      "supervisor can restart this watchdog", flush=True)
                sys.stdout.flush()
                self._on_stall(core.EXIT_UNKNOWN)
                return          # reached only when a caller injected a no-op
            self._emit(self._seq, age)

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
    """One full cycle -> :class:`Assessment`. Every read failure degrades.

    ``sight`` is normally left None so sight is RE-PROVED every cycle (review
    finding F3). Proving it once before the loop caught the #145 cached-alias
    signature at startup but could never fire mid-run — and the catalogue-
    membership arm exists precisely for the ROLL case (#113), which happens
    while the process is already running: across the boundary the cached dated
    code expires, ``get_position`` matches nothing, answers None, and the
    sidecar would print OK forever. One cached-catalogue GET per cycle is the
    price of the check being able to fire at all.
    """
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
    # #152: the stop leg of an OCO bracket lives on a book `get_open_orders`
    # never scans, so `resting` alone cannot answer "is this exposure stopped".
    # Read it here and map it onto the core's own type — the core stays pure and
    # never imports the toolkit. A FAILED read stays None, which the core treats
    # as NOT READ (could-not-see), never as proven absence.
    try:
        found = venue.oco_umbrellas(broker)
    except Exception:                                             # noqa: BLE001
        found = None
    umbrellas = None if found is None else tuple(
        core.CoverUmbrella(id=u.id, state=u.state.value, stop_price=u.stop_price,
                           side=u.side, qty=u.quantity)
        for u in found)
    return core.evaluate(Observation(
        phase=venue.session_phase(),
        contract_proven=proven,
        position_signed=position,
        owned_exposure=exposure,
        resting=resting,
        bar_period_s=bar_period_s,
        umbrellas=umbrellas))


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
    ladder = core.AlarmLadder(window_s=core.confirm_window_s(
        args.bar_period, arm_grace_s=args.arm_grace))
    seen: list[Verdict] = []

    print(f"naked_watch: {symbol} every {args.interval:g}s, confirm window "
          f"{ladder.window_s:.0f}s, store {store_path.name}, alarms -> "
          f"{alarm_log}")
    sight = prove_sight(broker)
    print(f"sight: {'PROVEN' if sight[0] else 'NOT PROVEN'} — {sight[1]}")

    def _beat(seq, age):
        stalled = args.stall_after and age >= args.stall_after
        print(f"[HEARTBEAT {seq}] {_stamp()} last evaluation {age:.0f}s ago"
              + (f"  !! STALLED (>{args.stall_after:.0f}s): a cycle is hung; "
                 f"this watchdog is NOT watching" if stalled else ""),
              flush=True)

    heartbeat = Heartbeat(args.heartbeat, _beat, stall_after_s=args.stall_after)
    if not args.once:
        heartbeat.start()
    try:
        while True:
            # sight=None: re-proved every cycle (F3). The startup proof above is
            # for the operator's launch line, not a value carried forward.
            assessment = evaluate_once(
                broker, symbol, store_path, account_id,
                bar_period_s=args.bar_period)
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
    code = core.worst_exit_code(seen)
    if heartbeat.stalled and code == core.EXIT_OK:
        # A run whose cycles hung has not observed anything, whatever its
        # verdicts said before the hang. Exit 0 would tell a wrapper the
        # invariant held throughout.
        print(f"!! a heartbeat STALL was observed (>{args.stall_after:.0f}s "
              f"without a completed evaluation) — reporting could-not-determine",
              flush=True)
        return core.EXIT_UNKNOWN
    return code


def main(argv: "list[str] | None" = None) -> int:
    # STRICT parsing BEFORE anything touches the venue, and --help never
    # contacts it (the flatten_api.py lesson: a loose `in sys.argv` check let
    # --help fall through into a live account).
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default=venue.SYMBOL)
    parser.add_argument("--once", action="store_true",
                        help="one cycle, then exit with its verdict")
    parser.add_argument("--interval", type=float, default=15.0,
                        help="poll cadence in seconds (default 15 — 5s is ~50 "
                             "prod REST calls/min alongside a live run)")
    parser.add_argument("--heartbeat", type=float, default=30.0,
                        help="heartbeat cadence in seconds (0 disables)")
    parser.add_argument("--bar-period", type=float, default=0.0,
                        help="the run's bar period in seconds — PASS THIS. The "
                             "naked-confirm window is max(30s, this), and a "
                             "reactively placed exit arms a bar late by design, "
                             "so leaving it 0 at 5m/15m pages on every entry "
                             "(300 / 900 are the values for those timeframes)")
    parser.add_argument("--arm-grace", type=float, default=None, metavar="SEC",
                        help="how long this VEHICLE legitimately takes to arm its "
                             "protection after a fill. Unset (default) derives the "
                             "window from --bar-period, which assumes a REACTIVE "
                             "exit arming a bar late. A PRE-PLACED bracket (l2b, "
                             "l2c) arms on the fill — measured 0.874s — so pass "
                             "30 for those. Measured 2026-09-18: the derived 300s "
                             "at 5m silently covered a 66s naked exposure end to "
                             "end, four NAKED verdicts, zero alarms. Floored at "
                             f"{core.NAKED_CONFIRM_FLOOR_S:g}s: a zero grace pages "
                             "on every entry and is muted within a day")
    parser.add_argument("--stall-after", type=float, default=120.0,
                        help="seconds without a completed evaluation before the "
                             "heartbeat reports a STALL and the run exits "
                             "could-not-determine (0 disables)")
    parser.add_argument("--store", default=str(STORE))
    parser.add_argument("--alarm-log", default=str(ALARM_LOG))
    args = parser.parse_args(argv)
    # (1) A stall threshold with no heartbeat thread is a GATE THAT CANNOT
    # FIRE: `--heartbeat 0` never starts the ticker, so `--stall-after` would
    # be silently inert — the operator would believe a hung cycle exits and it
    # would not. REFUSED rather than warned, because a warning in a long
    # transcript is how a disarmed guard survives; disabling the stall must be
    # said out loud.
    if args.heartbeat <= 0.0 and args.stall_after > 0.0:
        parser.error("--heartbeat 0 disables the heartbeat thread, which is "
                     "what enforces --stall-after; pass --stall-after 0 too if "
                     "you really want no stall detection")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
