#!/usr/bin/env python3
"""#128 thread 2 — does the VENUE cancel a BARE, untriggered STOP entry?

THE QUESTION
------------
On 2026-09-15 a plain conditional STOP **entry** — no OCO, no bracket, no
position, in-TTL and never triggered — went ``Canceled`` at ~200 s with nothing
of ours touching it. #128 thread 1 explains the OCO-bracket cancels through the
umbrella's leg transition; a bare stop has NO OCO machinery at all, so whatever
ended it is a SECOND phenomenon. This probe isolates it exactly the way
``probe_124_oco_lifecycle_isolation.py`` isolated thread 1: NO position, NO
strategy, NO engine, NO sync loop, NO wake — one standalone conditional STOP,
placed through the plugin's own write funnel (``DNSEBroker._place``), left to
rest, watched.

    * it gets cancelled again  -> REPRODUCIBLE. The final metadata
      (``cancel_ip`` — who sent it, ``originCategory``, ``eventNo``) and the
      timing are then the discriminator between a venue housekeeping job and
      anything account-side.
    * it rests for the full watch -> the 09-15 cancel was a one-off FOR A
      WINDOW THIS LONG. Absence over N seconds is evidence of absence for N
      seconds and nothing more.

WHY ``--duration`` IS AN A/B WORTH RUNNING
------------------------------------------
Yesterday's CANCELLED bare stop entry was GTD; the OCO that SURVIVED the same
session was DAY. Duration is therefore a live hypothesis for what the venue
retires, and it is one flag apart. Two things to know before reading a DAY run:

* ``durationType=DAY`` on a bare STOP was measured on prod (2026-09-14) to be
  REFUSED with ``400 CO-ORD-004`` — see ``DNSEBroker._gtd``'s docstring. So a
  DAY run here may well measure the REJECTION rather than a lifecycle. That is
  a result (it re-measures the claim), not a probe failure — it is reported as
  INDETERMINATE for the cancel question, with the venue's own error printed.
* GTD goes through the plugin's normal #118-clamped path (``_gtd()``), which in
  expiry week clamps to the contract's final trade date. The resolved value is
  printed in the pre-flight so the two runs are comparable.

HOW TO RUN (dry run FIRST — it touches nothing)
-----------------------------------------------
    .venv/bin/python plugins/dnse/testing/live_test/probe_bare_stop_lifecycle.py
    .venv/bin/python plugins/dnse/testing/live_test/probe_bare_stop_lifecycle.py --yes
    ... --yes --duration DAY          # the A/B arm

Run the mandatory L0 gate (``level0_venue_semantics``) before the ``--yes`` run,
like every other live case, and schedule it BEFORE the operator's first EntradeX
app trade of the day — conditional-book writes start answering
``INVALID_TRADING_TOKEN`` after that boundary (see CLAUDE.md / README).

SAFETY GUARANTEES (this runs on a REAL money account)
-----------------------------------------------------
    * qty = 1 contract, ONE conditional order, nothing else.
    * the trigger is priced from a LIVE reference read and must sit at least
      ``--away`` (default 5%, house floor 4.5%) on the side that cannot fill
      (a BUY stop triggers only ``>=`` its price, so it is put ABOVE market),
      and the LO the stop would emit is the engine's own ``_stop_fill_price``
      — further through the trigger, so also unreachable. Both levels are
      proven inside the venue's ceiling/floor band before anything is sent.
    * WITHOUT ``--yes`` nothing is placed at all — the pre-flight summary prints
      and the probe exits 0.
    * it REFUSES to run (exit 2) when: the trading token is not GOOD, the
      reference-price read fails, the band read fails, the computed levels are
      not provably far enough / inside the band, or the session phase cannot
      host a conditional. A FAILED READ IS NEVER REPORTED AS "SAFE".
    * cleanup always runs — normal exit, Ctrl-C, or any exception — and VERIFIES
      the order reached a terminal state. If it cannot, it prints a loud,
      unmissable warning naming the order id so the operator can cancel it in
      the app.
    * if anything ever shows a FILL, the watch aborts immediately and says so.
    * secrets are never printed: the trading token is read only inside the
      plugin's own write path, and the account number is masked.

EXIT CODES
----------
    0  a verdict was measured (VENUE-CANCELLED or RESTED-UNTOUCHED)
    1  measured, but cleanup could NOT prove the order is terminal (operator
       action required)
    2  INDETERMINATE / refused to run — never confuse this with "rested fine"
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "plugins" / "dnse" / "tools"))
sys.path.insert(0, str(Path(__file__).parent / "level0_venue_semantics"))

from pynecore.core.config import ensure_config                       # noqa: E402
from pynecore_dnse.broker import DNSEBroker, DNSEBrokerConfig        # noqa: E402
# The SAME metadata parse the plugin's #128 instrument uses, so probe evidence
# and live broker logs can never disagree about what a field held (``metadata``
# arrives as a JSON STRING, not a dict).
from pynecore_dnse.broker import _order_metadata as order_metadata   # noqa: E402
from l0_order_semantics import reference_close, session_phase        # noqa: E402
# venue.py owns the conventions for reading order state: it scans BOTH books
# (NORMAL / STOP / OCO) and knows an ``Activated`` conditional is a phantom
# shell whose NORMAL-book child did the work (#41). Reusing it is the point.
from venue import _detail_today as detail_on_any_book                # noqa: E402
from venue import token_verdict                                      # noqa: E402

ICT = timezone(timedelta(hours=7))

SYMBOL = "VN30F1M"
QUANTITY = 1
#: House no-fill convention for live probes (L1 cases use >= 4.5%).
MINIMUM_AWAY_FRACTION = 0.045
DEFAULT_AWAY_FRACTION = 0.05
#: The 09-15 cancel landed at ~200 s, so the old 120 s watch would have missed
#: it entirely. 420 s is >2x that.
DEFAULT_WATCH_SECONDS = 420

EXIT_MEASURED, EXIT_DIRTY_CLEANUP, EXIT_INDETERMINATE = 0, 1, 2

TERMINAL_STATUSES = {"CANCELED", "CANCELLED", "FILLED", "EXPIRED", "REJECTED"}
CANCELLED_STATUSES = {"CANCELED", "CANCELLED"}
#: A conditional that has fired is ``Activated`` — CLOSED, not filled — and its
#: NORMAL-book child is what actually executes (#41). Never read as terminal.
ACTIVATED_STATUS = "ACTIVATED"


def now_text() -> str:
    return datetime.now(ICT).strftime("%H:%M:%S")


def log_line(message: str) -> None:
    print(f"[128B] {now_text()} {message}", flush=True)


def build_broker(symbol: str) -> DNSEBroker:
    config = ensure_config(
        DNSEBrokerConfig,
        REPO / "workdir" / "config" / "plugins" / "dnse_broker.toml")
    return DNSEBroker(symbol=symbol, timeframe="1", config=config)


def probe_envelope(tag: str) -> SimpleNamespace:
    """Minimal stand-in for the engine's order envelope — enough for ``_place``.

    Deliberately NOT an engine envelope: no sync loop, no wake, no add-a-leg
    path is attached to it, which is the whole isolation this probe buys.
    """
    return SimpleNamespace(
        intent=SimpleNamespace(intent_key=f"p128-{tag}", pine_id=f"P128-{tag}",
                               from_entry=None, limit=None, stop=None))


# ------------------------------------------------------------------ safety math

class UnsafeLevels(RuntimeError):
    """The computed order levels could not be PROVEN safe — refuse to place."""


def distance_fraction(reference_price: float, level: float) -> float:
    return abs(level - reference_price) / reference_price


def compute_stop_trigger(reference_price: float, side: str,
                         away_fraction: float) -> float:
    """The trigger for an UNREACHABLE stop ENTRY.

    A buy stop fires ``>=`` its trigger, so putting it ABOVE market makes it
    unreachable without a rise of ``away_fraction``; a sell stop mirrors.
    """
    if side == "buy":
        return round(reference_price * (1 + away_fraction), 1)
    return round(reference_price * (1 - away_fraction), 1)


def assert_levels_are_safe(reference_price: float, levels: dict,
                           away_fraction: float, band_ceiling: float,
                           band_floor: float) -> None:
    """Raise :class:`UnsafeLevels` unless EVERY level is provably unreachable.

    Two independent proofs, because either alone can be wrong:
      * distance — each level is at least the house no-fill margin from the
        live reference price;
      * band — each level is inside the venue's own ceiling/floor, so the order
        cannot be refused (and so a +5% level has not silently landed outside
        the +/-7% daily band after a big move).
    """
    if reference_price <= 0:
        raise UnsafeLevels(f"reference price is not positive: {reference_price!r}")
    if away_fraction < MINIMUM_AWAY_FRACTION:
        raise UnsafeLevels(
            f"--away {away_fraction:.4f} is below the house no-fill floor "
            f"{MINIMUM_AWAY_FRACTION:.4f}")
    if not band_ceiling or not band_floor or band_ceiling <= band_floor:
        raise UnsafeLevels(
            f"unusable price band ceiling={band_ceiling!r} floor={band_floor!r}")
    for name, level in levels.items():
        gap = distance_fraction(reference_price, level)
        if gap < MINIMUM_AWAY_FRACTION:
            raise UnsafeLevels(
                f"{name} {level} is only {gap * 100:.2f}% from the reference "
                f"{reference_price} — below the {MINIMUM_AWAY_FRACTION * 100:.1f}% "
                f"no-fill floor")
        if not (band_floor <= level <= band_ceiling):
            raise UnsafeLevels(
                f"{name} {level} is OUTSIDE the venue band "
                f"[{band_floor}, {band_ceiling}] — the venue would refuse it")


# --------------------------------------------------- the DAY arm of the A/B

def install_day_duration_override(broker: DNSEBroker) -> None:
    """Send ``durationType=DAY`` instead of the plugin's GTD, PROBE-LOCALLY.

    ``DNSEBroker._place`` hard-codes GTD (+ the #118-clamped
    ``durationDateTime``) for every conditional STOP, and that is the right
    production behaviour — DAY was measured to be refused. So the A/B arm is
    implemented as a one-call interceptor on THIS broker instance's client:
    nothing in the plugin changes, and the intercept is visible in the log.
    """
    original_post_order = broker.client.post_order

    def post_order_with_day_duration(account_no, market_type, payload,
                                     trading_token, **kwargs):
        # New dict, never a mutation of the plugin's payload.
        day_payload = {key: value for key, value in payload.items()
                       if key != "durationDateTime"}
        day_payload["durationType"] = "DAY"
        log_line("duration override ACTIVE: durationType=DAY "
                 "(durationDateTime dropped) — measured 2026-09-14 to answer "
                 "400 CO-ORD-004 on a bare STOP")
        return original_post_order(account_no, market_type, day_payload,
                                   trading_token, **kwargs)

    broker.client.post_order = post_order_with_day_duration


# ------------------------------------------------------------------ venue reads

def read_order_row(broker: DNSEBroker, order_id: str) -> "tuple[str | None, dict | None]":
    """(book, today's venue record) for ``order_id`` on whichever book holds it."""
    try:
        return detail_on_any_book(broker, str(order_id))
    except Exception as exc:                                          # noqa: BLE001
        log_line(f"detail read for {order_id} RAISED {type(exc).__name__}: {exc}")
        return None, None


def summarize_row(row: dict) -> tuple[str, str, float, str]:
    """(status_upper, child_id, filled_quantity, printable) from a venue row."""
    status = str(row.get("orderStatus") or "").upper()
    child_id = str(row.get("externalOrderId") or "") or ""
    try:
        filled = float(row.get("fillQuantity") or 0)
    except (TypeError, ValueError):
        filled = 0.0
    printable = (f"status={row.get('orderStatus')} price={row.get('price')} "
                 f"stop={row.get('stopPrice')} qty={row.get('quantity')} "
                 f"duration={row.get('durationType')} "
                 f"filled={row.get('fillQuantity')} child={child_id or '-'}")
    return status, child_id, filled, printable


def field_or_absent(row: dict, *names: str) -> str:
    """The first of ``names`` the row actually carries — else ``absent``.

    "absent" is printed rather than blank so a reader can tell a field the venue
    did not serve from one it served empty. A STOP detail may carry no metadata
    at all; that itself is the measurement.
    """
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return "absent"


def print_cancel_evidence(broker: DNSEBroker, order_id: str) -> None:
    """Everything the venue exposes about a cancel, fetched the moment we see it.

    ``cancel_ip`` / ``originCategory`` / ``eventNo`` live inside ``metadata``,
    which the venue serves as a JSON STRING; ``modifiedDate`` times the cancel
    and ``error``/``errorMessage`` is the only place a reason could appear.
    """
    book, row = read_order_row(broker, order_id)
    if row is None:
        print(f"cancel-evidence: id={order_id} UNREADABLE — the detail read "
              f"failed, so NOTHING is known about this cancel (not 'no "
              f"metadata')", flush=True)
        return
    metadata = order_metadata(row)
    print(
        f"cancel-evidence: id={order_id} book={book} "
        f"status={row.get('orderStatus')} "
        f"createdDate={field_or_absent(row, 'createdDate')} "
        f"modifiedDate={field_or_absent(row, 'modifiedDate')} "
        f"errorMessage={field_or_absent(row, 'errorMessage', 'error')} "
        f"cancel_ip={field_or_absent(metadata, 'cancel_ip')} "
        f"originCategory={field_or_absent(metadata, 'originCategory')} "
        f"eventNo={field_or_absent(metadata, 'eventNo')} "
        f"(metadata: {f'{len(metadata)} fields' if metadata else 'ABSENT'})",
        flush=True)
    condition = metadata.get("condition")
    if condition:
        print(f"cancel-evidence: id={order_id} condition={condition}", flush=True)


def open_stop_ids(broker: DNSEBroker) -> "set[str] | None":
    """Ids currently on the STOP book, or ``None`` when the read is unprovable."""
    rows, _classified = broker._read_book_rows_sync("STOP")
    if rows is None:
        return None
    return {str(row.get("id")) for row in rows if row.get("id") is not None}


# ------------------------------------------------------------------ measurement

class Watch:
    """The measurement state: which ids we follow and what the venue said."""

    def __init__(self, broker: DNSEBroker):
        self.broker = broker
        self.tracked_ids: list[str] = []
        self.last_status: dict[str, str] = {}
        self.transitions: list[str] = []
        self.cancel_events: list[str] = []
        self.fill_events: list[str] = []
        self.unreadable_polls: dict[str, int] = {}
        self.evidence_taken: set[str] = set()
        self.confirmed_resting = False

    def track(self, order_id: str, role: str) -> None:
        order_id = str(order_id)
        if order_id and order_id not in self.tracked_ids:
            self.tracked_ids.append(order_id)
            self.unreadable_polls[order_id] = 0
            log_line(f"tracking {role} id={order_id}")

    def poll_once(self) -> None:
        for order_id in list(self.tracked_ids):
            _book, row = read_order_row(self.broker, order_id)
            if row is None:
                self.unreadable_polls[order_id] += 1
                self._note(order_id, "UNREADABLE",
                           "venue record could not be read (NOT proof of anything)")
                continue
            status, child_id, filled, printable = summarize_row(row)
            if child_id:
                # #41: an Activated conditional is a phantom shell; the child on
                # the NORMAL book is the order that actually acts. Watch it too.
                self.track(child_id, "child (externalOrderId)")
            if status and status not in TERMINAL_STATUSES and status != ACTIVATED_STATUS:
                self.confirmed_resting = True
            self._note(order_id, status, printable)
            if filled > 0:
                self.fill_events.append(
                    f"{now_text()} id={order_id} FILLED quantity={filled}")
            if status in CANCELLED_STATUSES:
                self.cancel_events.append(f"{now_text()} id={order_id} -> {status}")
                if order_id not in self.evidence_taken:
                    # Immediately, while the record is still today's: the whole
                    # point of the probe is the cancel's own metadata.
                    self.evidence_taken.add(order_id)
                    print_cancel_evidence(self.broker, order_id)

    def _note(self, order_id: str, status: str, printable: str) -> None:
        if self.last_status.get(order_id) == status:
            return
        previous = self.last_status.get(order_id, "(first read)")
        self.last_status[order_id] = status
        line = f"{now_text()} id={order_id} {previous} -> {status} | {printable}"
        self.transitions.append(line)
        log_line(f"TRANSITION {line}")

    @property
    def saw_a_cancel(self) -> bool:
        return bool(self.cancel_events)

    @property
    def saw_a_fill(self) -> bool:
        return bool(self.fill_events)

    def every_id_unreadable(self) -> bool:
        return bool(self.tracked_ids) and all(
            self.last_status.get(order_id) in (None, "UNREADABLE")
            for order_id in self.tracked_ids)


# ------------------------------------------------------------------ cleanup

def cleanup(broker: DNSEBroker, watch: Watch) -> bool:
    """Cancel everything this probe placed and PROVE each id is terminal.

    Returns True only when every tracked id reads back terminal. Anything else
    prints a loud warning naming the ids — a silent failure here would leave a
    live conditional on a real-money account.
    """
    print("\n--- cleanup ---", flush=True)
    for order_id in watch.tracked_ids:
        try:
            outcome = asyncio.run(broker._cancel_one_disposition(order_id))
            log_line(f"cancel {order_id}: {outcome}")
        except Exception as exc:                                      # noqa: BLE001
            log_line(f"cancel {order_id} RAISED {type(exc).__name__}: {exc}")

    time.sleep(3)   # DNSE detail reads are eventually consistent
    not_terminal: list[str] = []
    for order_id in watch.tracked_ids:
        _book, row = read_order_row(broker, order_id)
        if row is None:
            not_terminal.append(f"{order_id} (UNREADABLE)")
            log_line(f"final record {order_id}: COULD NOT READ")
            continue
        status, _child, _filled, printable = summarize_row(row)
        log_line(f"final record {order_id}: {printable}")
        if status not in TERMINAL_STATUSES:
            not_terminal.append(f"{order_id} ({status or 'unknown'})")
    if not_terminal:
        print("\n" + "!" * 78)
        print("!! CLEANUP FAILED — these orders are NOT proven terminal:")
        for item in not_terminal:
            print(f"!!   {item}")
        print("!! CANCEL THEM IN THE ENTRADEX APP, or:")
        print("!!   .venv/bin/python plugins/dnse/tools/venue.py status")
        print("!!   .venv/bin/python plugins/dnse/tools/venue.py cancel <id>")
        print("!" * 78 + "\n", flush=True)
        return False
    log_line("cleanup clean — every placed order is terminal")
    return True


# ------------------------------------------------------------------ verdict

def print_verdict(watch: Watch, watch_seconds: int, duration: str,
                  indeterminate_reason: "str | None") -> int:
    print("\n" + "=" * 78)
    print(f"BARE STOP ENTRY, durationType={duration}, watch={watch_seconds}s")
    if indeterminate_reason:
        print("VERDICT: INDETERMINATE")
        print(f"  reason: {indeterminate_reason}")
        print("  A failed, refused or empty read is NOT evidence that the order "
              "rested fine. Nothing is concluded about the 09-15 bare-stop cancel.")
        print("=" * 78, flush=True)
        return EXIT_INDETERMINATE
    if watch.saw_a_cancel:
        print("VERDICT: VENUE-CANCELLED")
        print("  A resting, untriggered, in-TTL conditional STOP ENTRY was "
              "cancelled with NO position, NO engine, NO sync loop, NO OCO "
              "umbrella and NO bracket machinery of any kind running.")
        print("  -> the 09-15 bare-stop cancel is REPRODUCIBLE and is a SECOND "
              "phenomenon, distinct from the #128 thread-1 OCO leg transition. "
              "The cancel-evidence lines above (cancel_ip / originCategory / "
              "eventNo / modifiedDate) are the discriminator.")
        print("  cancel events:")
        for event in watch.cancel_events:
            print(f"    {event}")
        print("  full transition log:")
        for line in watch.transitions:
            print(f"    {line}")
        print("=" * 78, flush=True)
        return EXIT_MEASURED
    print("VERDICT: RESTED-UNTOUCHED")
    print(f"  No cancel was observed for the FULL watch of {watch_seconds} s "
          f"({watch_seconds / 60:.1f} min).")
    print("  -> NOT reproduced in THIS window with durationType="
          f"{duration}. Absence over {watch_seconds} s is evidence of absence "
          f"for {watch_seconds} s and nothing longer: it does not rule out a "
          "cancel triggered later in the session, by a position existing, by "
          "the other duration, or by a longer rest.")
    print("  transition log:")
    for line in watch.transitions:
        print(f"    {line}")
    print("=" * 78, flush=True)
    return EXIT_MEASURED


# ------------------------------------------------------------------ main flow

def run(args) -> int:
    broker = build_broker(args.symbol)

    phase = session_phase()
    try:
        reference_price, reference_source = reference_close(broker)
    except Exception as exc:                                          # noqa: BLE001
        print(f"REFUSING — the reference price read FAILED "
              f"({type(exc).__name__}: {exc}). This is NOT 'safe to place'.")
        return EXIT_INDETERMINATE
    try:
        band_ceiling, band_floor = broker._band()
    except Exception as exc:                                          # noqa: BLE001
        print(f"REFUSING — the price-band read FAILED "
              f"({type(exc).__name__}: {exc}). Safety cannot be proven.")
        return EXIT_INDETERMINATE
    try:
        stop_trigger_price = compute_stop_trigger(reference_price, args.side,
                                                  args.away)
        # The engine's own LO pricing for a triggered stop (trigger +/- 2x
        # slippage ticks): identical to what a live strategy would place.
        stop_limit_price = broker._stop_fill_price(args.side, stop_trigger_price)
        assert_levels_are_safe(
            reference_price,
            {"stop-trigger": stop_trigger_price, "stop LO": stop_limit_price},
            args.away, band_ceiling, band_floor)
    except UnsafeLevels as exc:
        print(f"REFUSING — unsafe order levels: {exc}")
        return EXIT_INDETERMINATE
    except Exception as exc:                                          # noqa: BLE001
        print(f"REFUSING — could not compute safe levels "
              f"({type(exc).__name__}: {exc}).")
        return EXIT_INDETERMINATE

    try:
        contract = broker.resolve_contract()
    except Exception as exc:                                          # noqa: BLE001
        print(f"REFUSING — cannot resolve the tradable contract "
              f"({type(exc).__name__}: {exc}).")
        return EXIT_INDETERMINATE

    if args.duration == "GTD":
        try:
            duration_detail = f"GTD until {broker._gtd()} (#118-clamped)"
        except Exception as exc:                                      # noqa: BLE001
            duration_detail = (f"GTD — the clamp could not be previewed "
                               f"({type(exc).__name__}: {exc})")
    else:
        duration_detail = ("DAY (probe-local override; measured 2026-09-14 to "
                           "answer 400 CO-ORD-004 on a bare STOP)")

    print("=" * 78)
    print("PRE-FLIGHT — exactly what this probe would place (ONE order):")
    print(f"  account          : {broker._mask_id(broker.account_id)} (masked)")
    print(f"  symbol / contract: {args.symbol} -> {contract} "
          f"({broker.market_type})")
    print(f"  session phase    : {phase}   ({datetime.now(ICT):%Y-%m-%d %H:%M:%S} ICT)")
    print(f"  reference price  : {reference_price} ({reference_source})")
    print(f"  venue band       : floor={band_floor} ceiling={band_ceiling}")
    print("  order category   : STOP (conditional book, NO OCO, NO bracket)")
    print(f"  side / quantity  : {args.side} / {QUANTITY}")
    print(f"  stop trigger     : {stop_trigger_price}  "
          f"({distance_fraction(reference_price, stop_trigger_price) * 100:.2f}% "
          f"from market, unreachable)")
    # Display only: the computed price carries a binary float-repr tail
    # (1834.1000000000001). The WIRE value stays ``stop_limit_price`` untouched.
    print(f"  stop LO price    : {stop_limit_price:.1f} (emitted only if the "
          f"trigger ever fires)")
    print(f"  duration         : {duration_detail}")
    print(f"  watch            : {args.watch_seconds} s, polling every "
          f"{args.poll_seconds} s")
    print("  cleanup          : cancel + verify terminal, always (incl. Ctrl-C)")
    print("=" * 78, flush=True)

    if not args.yes:
        print("\nDRY RUN — nothing was sent to the venue (no order, no cancel).")
        print("NOTE: the gates below run ONLY under --yes, so a clean dry run is "
              "NOT evidence they pass: session phase (needs continuous/lunch), "
              "trading-token GOOD, and the STOP-book provability read. The "
              "pre-flight above only proves the LEVELS are safe.")
        print("Re-run with --yes to place it, after the L0 gate.")
        return EXIT_MEASURED

    if phase not in ("continuous", "lunch"):
        print(f"REFUSING — the conditional book needs phase continuous/lunch, "
              f"got {phase!r}. (ATC refuses cancels; closed refuses placement.)")
        return EXIT_INDETERMINATE
    verdict = token_verdict()
    if not verdict.startswith("GOOD"):
        print(f"REFUSING — trading token is not GOOD: {verdict}\n"
              f"mint one first: .venv/bin/python plugins/dnse/tools/refresh_token.py "
              f"--send   (then --otp <code>)")
        return EXIT_INDETERMINATE

    before_ids = open_stop_ids(broker)
    if before_ids is None:
        print("REFUSING — the STOP book read is unprovable (pagination), so a "
              "new order could not be identified afterwards. INDETERMINATE.")
        return EXIT_INDETERMINATE

    if args.duration == "DAY":
        install_day_duration_override(broker)

    watch = Watch(broker)
    indeterminate_reason: "str | None" = None
    try:
        placed = broker._place(
            probe_envelope("stop"), args.side, QUANTITY,
            price=stop_limit_price, category="STOP",
            stop_price=stop_trigger_price)
        tracked = placed[0]
        log_line(f"PLACED STOP tracked_id={tracked.id} "
                 f"trigger={stop_trigger_price} lo={stop_limit_price:.1f} "
                 f"duration={args.duration} "
                 f"filled={getattr(tracked, 'filled_qty', 0)}")
        watch.track(str(tracked.id), "tracked conditional STOP entry")

        after_ids = open_stop_ids(broker)
        if after_ids is not None:
            extra = sorted(after_ids - before_ids - set(watch.tracked_ids))
            for candidate in extra:
                log_line(f"WARNING: an unexpected new STOP row appeared "
                         f"({candidate}) — tracking it too")
                watch.track(candidate, "unexpected new STOP row")
        else:
            log_line("WARNING: could not re-read the STOP book after placing — "
                     "watching the tracked order only")

        deadline = time.monotonic() + args.watch_seconds
        while time.monotonic() < deadline:
            watch.poll_once()
            if watch.saw_a_fill:
                print("\n" + "!" * 78)
                print("!! THE STOP REPORTED A FILL — aborting the watch and "
                      "cleaning up.")
                for event in watch.fill_events:
                    print(f"!!   {event}")
                print("!! Check the position immediately: "
                      ".venv/bin/python plugins/dnse/tools/venue.py status")
                print("!" * 78, flush=True)
                indeterminate_reason = (
                    "the stop FILLED during the watch, so it never rested "
                    "untouched and no lifecycle conclusion is possible")
                break
            if watch.saw_a_cancel:
                log_line("cancel observed — watching a few more polls for the "
                         "follow-on transitions, then stopping")
                for _ in range(3):
                    time.sleep(args.poll_seconds)
                    watch.poll_once()
                break
            time.sleep(args.poll_seconds)

        if indeterminate_reason is None:
            if not watch.confirmed_resting and not watch.saw_a_cancel:
                indeterminate_reason = (
                    "the order was never observed RESTING on the venue (every "
                    "read was unreadable or non-working) — nothing can be "
                    "concluded about who cancels what")
            elif watch.every_id_unreadable():
                indeterminate_reason = (
                    "every tracked id became unreadable — an absent read is not "
                    "an absent cancel")
    except KeyboardInterrupt:
        indeterminate_reason = "interrupted by the operator (Ctrl-C)"
        log_line("interrupted — going straight to cleanup")
    except Exception as exc:                                          # noqa: BLE001
        indeterminate_reason = f"the probe raised {type(exc).__name__}: {exc}"
        if args.duration == "DAY" and not watch.tracked_ids:
            indeterminate_reason += (
                " — this is the DAY arm and NOTHING was placed, so what was "
                "measured is the PLACEMENT's answer to durationType=DAY on a "
                "bare STOP (see _gtd's 400 CO-ORD-004 note), not a lifecycle")
        log_line(f"RAISED {type(exc).__name__}: {exc} — going to cleanup")
    finally:
        cleanup_ok = cleanup(broker, watch)

    exit_code = print_verdict(watch, args.watch_seconds, args.duration,
                              indeterminate_reason)
    if exit_code == EXIT_MEASURED and not cleanup_ok:
        return EXIT_DIRTY_CLEANUP
    return exit_code


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true",
                        help="actually place the order (without this the probe "
                             "is a DRY RUN and touches nothing)")
    parser.add_argument("--watch-seconds", type=int,
                        default=DEFAULT_WATCH_SECONDS,
                        help=f"how long to watch the resting stop (default "
                             f"{DEFAULT_WATCH_SECONDS}; the 09-15 cancel came "
                             f"at ~200 s, so 120 was too short)")
    parser.add_argument("--poll-seconds", type=float, default=2.0,
                        help="venue poll interval in seconds (default 2.0)")
    parser.add_argument("--duration", choices=("GTD", "DAY"), default="GTD",
                        help="order TTL: GTD is the plugin's normal "
                             "#118-clamped path and what the CANCELLED 09-15 "
                             "entry used; DAY is the probe-local override and "
                             "what the SURVIVING OCO used (default GTD)")
    parser.add_argument("--away", type=float, default=DEFAULT_AWAY_FRACTION,
                        help=f"fraction away from market for the trigger "
                             f"(default {DEFAULT_AWAY_FRACTION}; values below "
                             f"{MINIMUM_AWAY_FRACTION} are refused)")
    parser.add_argument("--side", choices=("buy", "sell"), default="buy",
                        help="entry side: 'buy' puts the trigger ABOVE market "
                             "(the 09-15 shape), 'sell' below")
    parser.add_argument("--symbol", default=SYMBOL)
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
