#!/usr/bin/env python3
"""#124 part-B — does the VENUE cancel a resting conditional OCO all by itself?

THE QUESTION
------------
Issue #124 (CRITICAL, blocks live): a protective OCO exit that the engine places
right after an entry fill is CANCELLED moments after it was CREATED, the engine
then quarantines, and a REAL position is left naked. Reproduced live twice on
2026-09-14. What is still unknown is WHO cancels it. Three suspects:

    (i)   the VENUE's own OCO/conditional lifecycle — e.g. the OCO umbrella's
          spawned NORMAL LO being auto-amended into the stop leg and the old
          working order being retired (#93), or any other server-side
          housekeeping on the conditional book;
    (ii)  the #121 arm-on-fill WAKE re-dispatching the exit while the bar-close
          sync is still in flight (our cancel, racing ourselves);
    (iii) the #123 add-a-leg path cancelling/replacing a leg while growing the
          protection for a later partial-entry slice.

This probe DISCRIMINATES (i) from (ii) and (iii) by removing everything that is
ours: NO position, NO strategy, NO engine, NO sync loop, NO wake. It places ONE
standalone conditional OCO order FAR FROM MARKET through the plugin's own write
funnel (``DNSEBroker._place``), lets it REST, and simply watches the venue record.

    * the venue cancels it (or its child) with nothing else running
              -> suspect (i) CONFIRMED. The fix for #124 belongs in how we
                 place/track conditionals (or in not using native OCO at all),
                 NOT in the wake or the add-a-leg path.
    * it rests untouched for the whole watch
              -> suspect (i) REFUTED for that window. The cancel in #124 is
                 OURS: look at the #121 wake re-dispatch and the #123 add-a-leg
                 path (both of which this probe deliberately does not run).

HOW TO RUN (dry run FIRST — it touches nothing)
-----------------------------------------------
    .venv/bin/python plugins/dnse/testing/live_test/probe_124_oco_lifecycle_isolation.py
    .venv/bin/python plugins/dnse/testing/live_test/probe_124_oco_lifecycle_isolation.py --yes

Run the mandatory L0 gate (``level0_venue_semantics``) before the ``--yes`` run,
like every other live case, and schedule it BEFORE the operator's first EntradeX
app trade of the day — conditional-book writes start answering
``INVALID_TRADING_TOKEN`` after that boundary (see CLAUDE.md / README).

SAFETY GUARANTEES (this runs on a REAL money account)
-----------------------------------------------------
    * qty = 1 contract, ONE order, nothing else.
    * every leg is priced from a LIVE reference read and must sit at least
      ``--away`` (default 5%, house floor 4.5%) away from it, in the
      direction that cannot fill: the TP limit is unmarketable, the stop trigger
      is 5% on the far side. Both levels are additionally proven to be inside
      the venue's own ceiling/floor band before anything is sent.
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
from l0_order_semantics import reference_close, session_phase        # noqa: E402
# venue.py owns the conventions for reading order state: it scans BOTH books
# (NORMAL / STOP / OCO) and it is the tool that knows an ``Activated``
# conditional is a phantom shell whose NORMAL-book child did the work (#41).
# Reusing it is the point — a hand-rolled lookup here is exactly the habit that
# produced the false "account is FLAT" report on 2026-08-19.
from venue import _detail_today as detail_on_any_book                # noqa: E402
from venue import token_verdict                                      # noqa: E402

ICT = timezone(timedelta(hours=7))

SYMBOL = "VN30F1M"
QUANTITY = 1
#: House no-fill convention for live probes (L1 cases use >= 4.5%).
MINIMUM_AWAY_FRACTION = 0.045
DEFAULT_AWAY_FRACTION = 0.05

EXIT_MEASURED, EXIT_DIRTY_CLEANUP, EXIT_INDETERMINATE = 0, 1, 2

TERMINAL_STATUSES = {"CANCELED", "CANCELLED", "FILLED", "EXPIRED", "REJECTED"}
CANCELLED_STATUSES = {"CANCELED", "CANCELLED"}
#: A conditional that has fired is ``Activated`` — CLOSED, not filled — and its
#: NORMAL-book child is what actually executes (#41). Never read as terminal.
ACTIVATED_STATUS = "ACTIVATED"


def now_text() -> str:
    return datetime.now(ICT).strftime("%H:%M:%S")


def log_line(message: str) -> None:
    print(f"[124B] {now_text()} {message}", flush=True)


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
        intent=SimpleNamespace(intent_key=f"p124-{tag}", pine_id=f"P124-{tag}",
                               from_entry=None, limit=None, stop=None))


# ------------------------------------------------------------------ safety math

class UnsafeLevels(RuntimeError):
    """The computed order levels could not be PROVEN safe — refuse to place."""


def compute_levels(reference_price: float, side: str, away_fraction: float,
                   band_ceiling: float, band_floor: float
                   ) -> tuple[float, float]:
    """(take_profit_price, stop_trigger_price) for a protective exit ``side``.

    A protective exit for a LONG is a ``sell``: its take-profit sits ABOVE the
    market (an unmarketable sell limit) and its stop trigger BELOW. For a SHORT
    (``buy``) both mirror. Either way neither leg is reachable without a move of
    ``away_fraction``.
    """
    if side == "sell":
        take_profit_price = round(reference_price * (1 + away_fraction), 1)
        stop_trigger_price = round(reference_price * (1 - away_fraction), 1)
    else:
        take_profit_price = round(reference_price * (1 - away_fraction), 1)
        stop_trigger_price = round(reference_price * (1 + away_fraction), 1)
    assert_levels_are_safe(reference_price, take_profit_price, stop_trigger_price,
                           away_fraction, band_ceiling, band_floor)
    return take_profit_price, stop_trigger_price


def distance_fraction(reference_price: float, level: float) -> float:
    return abs(level - reference_price) / reference_price


def assert_levels_are_safe(reference_price: float, take_profit_price: float,
                           stop_trigger_price: float, away_fraction: float,
                           band_ceiling: float, band_floor: float) -> None:
    """Raise :class:`UnsafeLevels` unless BOTH legs are provably unreachable.

    Two independent proofs, because either alone can be wrong:
      * distance — each leg is at least the house no-fill margin from the live
        reference price;
      * band — each leg is inside the venue's own ceiling/floor, so the order
        cannot be refused (and so a +/-5% level has not silently landed outside
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
    for name, level in (("take-profit", take_profit_price),
                        ("stop-trigger", stop_trigger_price)):
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


# ------------------------------------------------------------------ venue reads

def read_order_row(broker: DNSEBroker, order_id: str) -> "dict | None":
    """Today's venue record for ``order_id`` on whichever book holds it."""
    try:
        _book, row = detail_on_any_book(broker, str(order_id))
    except Exception as exc:                                          # noqa: BLE001
        log_line(f"detail read for {order_id} RAISED {type(exc).__name__}: {exc}")
        return None
    return row


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
                 f"filled={row.get('fillQuantity')} child={child_id or '-'}")
    return status, child_id, filled, printable


def open_oco_umbrella_ids(broker: DNSEBroker) -> "set[str] | None":
    """Ids currently on the OCO book, or ``None`` when the read is unprovable."""
    rows, _classified = broker._read_book_rows_sync("OCO")
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
        self.confirmed_resting = False

    def track(self, order_id: str, role: str) -> None:
        order_id = str(order_id)
        if order_id and order_id not in self.tracked_ids:
            self.tracked_ids.append(order_id)
            self.unreadable_polls[order_id] = 0
            log_line(f"tracking {role} id={order_id}")

    def poll_once(self) -> None:
        for order_id in list(self.tracked_ids):
            row = read_order_row(self.broker, order_id)
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

def cleanup(broker: DNSEBroker, watch: Watch, umbrella_id: "str | None") -> bool:
    """Cancel everything this probe placed and PROVE each id is terminal.

    Returns True only when every tracked id reads back terminal. Anything else
    prints a loud warning naming the ids — a silent failure here would leave a
    live order on a real-money account.
    """
    print("\n--- cleanup ---", flush=True)
    if umbrella_id:
        # The umbrella lives on the OCO book; cancelling it should retire the
        # child LO with it. Cancel it FIRST, then mop up anything still working.
        try:
            status, body = broker._write(lambda token: broker.client.cancel_order(
                broker.account_id, str(umbrella_id), broker.market_type, token,
                order_category="OCO"))
            log_line(f"cancel umbrella {umbrella_id}: http={status} "
                     f"body={str(body)[:120]}")
        except Exception as exc:                                      # noqa: BLE001
            log_line(f"cancel umbrella {umbrella_id} RAISED "
                     f"{type(exc).__name__}: {exc}")
    for order_id in watch.tracked_ids:
        if order_id == str(umbrella_id or ""):
            continue
        try:
            outcome = asyncio.run(broker._cancel_one_disposition(order_id))
            log_line(f"cancel {order_id}: {outcome}")
        except Exception as exc:                                      # noqa: BLE001
            log_line(f"cancel {order_id} RAISED {type(exc).__name__}: {exc}")

    time.sleep(3)   # DNSE detail reads are eventually consistent
    not_terminal: list[str] = []
    for order_id in watch.tracked_ids:
        row = read_order_row(broker, order_id)
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

def print_verdict(watch: Watch, watch_seconds: int, indeterminate_reason: str | None
                  ) -> int:
    print("\n" + "=" * 78)
    if indeterminate_reason:
        print("VERDICT: INDETERMINATE")
        print(f"  reason: {indeterminate_reason}")
        print("  A failed or empty read is NOT evidence that the order rested "
              "fine. Nothing is concluded about suspect (i).")
        print("=" * 78, flush=True)
        return EXIT_INDETERMINATE
    if watch.saw_a_cancel:
        print("VERDICT: VENUE-CANCELLED")
        print("  A resting conditional OCO (or its NORMAL-book child) was "
              "cancelled with NO position, NO engine, NO sync loop, NO #121 wake "
              "and NO #123 add-a-leg running.")
        print("  -> #124 suspect (i) CONFIRMED: the venue's own OCO/conditional "
              "lifecycle retires the order. Fix direction is the placement/"
              "tracking of native OCO, not the wake or add-a-leg paths.")
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
    print("  -> #124 suspect (i) REFUTED FOR THIS WINDOW ONLY. Absence over "
          f"{watch_seconds} s is evidence of absence for {watch_seconds} s and "
          "nothing longer: it does not rule out a venue cancel triggered by a "
          "later session event, by a position existing, or by a longer rest.")
    print("  -> the #124 cancel is therefore OURS: look next at the #121 "
          "arm-on-fill wake re-dispatch (ii) and the #123 add-a-leg path (iii).")
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
        take_profit_price, stop_trigger_price = compute_levels(
            reference_price, args.side, args.away, band_ceiling, band_floor)
        stop_limit_price = broker._stop_fill_price(args.side, stop_trigger_price)
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

    print("=" * 78)
    print("PRE-FLIGHT — exactly what this probe would place (ONE order):")
    print(f"  account          : {broker._mask_id(broker.account_id)} (masked)")
    print(f"  symbol / contract: {args.symbol} -> {contract} "
          f"({broker.market_type})")
    print(f"  session phase    : {phase}   ({datetime.now(ICT):%Y-%m-%d %H:%M:%S} ICT)")
    print(f"  reference price  : {reference_price} ({reference_source})")
    print(f"  venue band       : floor={band_floor} ceiling={band_ceiling}")
    print("  order category   : OCO (conditional book)")
    print(f"  side / quantity  : {args.side} / {QUANTITY}")
    print(f"  take-profit LO   : {take_profit_price}  "
          f"({distance_fraction(reference_price, take_profit_price) * 100:.2f}% "
          f"from market, unmarketable)")
    print(f"  stop trigger     : {stop_trigger_price}  "
          f"({distance_fraction(reference_price, stop_trigger_price) * 100:.2f}% "
          f"from market)")
    print(f"  stop LO price    : {stop_limit_price} (emitted only if the trigger "
          f"ever fires)")
    print(f"  watch            : {args.watch_seconds} s, polling every "
          f"{args.poll_seconds} s")
    print("  cleanup          : cancel + verify terminal, always (incl. Ctrl-C)")
    print("=" * 78, flush=True)

    if not args.yes:
        print("\nDRY RUN — nothing was sent to the venue (no order, no cancel).")
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

    before_ids = open_oco_umbrella_ids(broker)
    if before_ids is None:
        print("REFUSING — the OCO book read is unprovable (pagination), so the "
              "umbrella id could not be identified afterwards. INDETERMINATE.")
        return EXIT_INDETERMINATE

    watch = Watch(broker)
    umbrella_id: str | None = None
    indeterminate_reason: str | None = None
    try:
        placed = broker._place(
            probe_envelope("oco"), args.side, QUANTITY,
            price=take_profit_price, category="OCO",
            stop_price=stop_trigger_price, stop_order_price=stop_limit_price)
        tracked = placed[0]
        log_line(f"PLACED OCO tracked_id={tracked.id} tp={take_profit_price} "
                 f"stop={stop_trigger_price} filled="
                 f"{getattr(tracked, 'filled_qty', 0)}")
        watch.track(str(tracked.id), "tracked working order")

        after_ids = open_oco_umbrella_ids(broker)
        if after_ids is not None:
            new_umbrella = sorted(after_ids - before_ids)
            if len(new_umbrella) == 1:
                umbrella_id = new_umbrella[0]
                watch.track(umbrella_id, "OCO umbrella")
            elif new_umbrella:
                log_line(f"WARNING: {len(new_umbrella)} new OCO rows appeared "
                         f"({new_umbrella}) — cannot attribute the umbrella; "
                         f"tracking all of them")
                for candidate in new_umbrella:
                    watch.track(candidate, "OCO umbrella candidate")
        else:
            log_line("WARNING: could not re-read the OCO book — the umbrella id "
                     "is unknown; watching the tracked working order only")

        deadline = time.monotonic() + args.watch_seconds
        while time.monotonic() < deadline:
            watch.poll_once()
            if watch.saw_a_fill:
                print("\n" + "!" * 78)
                print("!! A LEG REPORTED A FILL — aborting the watch and cleaning up.")
                for event in watch.fill_events:
                    print(f"!!   {event}")
                print("!! Check the position immediately: "
                      ".venv/bin/python plugins/dnse/tools/venue.py status")
                print("!" * 78, flush=True)
                indeterminate_reason = (
                    "a leg FILLED during the watch, so the order never rested "
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
        indeterminate_reason = (f"the probe raised {type(exc).__name__}: {exc}")
        log_line(f"RAISED {type(exc).__name__}: {exc} — going to cleanup")
    finally:
        cleanup_ok = cleanup(broker, watch, umbrella_id)

    exit_code = print_verdict(watch, args.watch_seconds, indeterminate_reason)
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
    parser.add_argument("--watch-seconds", type=int, default=120,
                        help="how long to watch the resting order (default 120)")
    parser.add_argument("--poll-seconds", type=float, default=2.0,
                        help="venue poll interval in seconds (default 2.0)")
    parser.add_argument("--away", type=float, default=DEFAULT_AWAY_FRACTION,
                        help=f"fraction away from market for both legs "
                             f"(default {DEFAULT_AWAY_FRACTION}; values below "
                             f"{MINIMUM_AWAY_FRACTION} are refused)")
    parser.add_argument("--side", choices=("sell", "buy"), default="sell",
                        help="protective-exit side: 'sell' is a long's exit "
                             "(the #124 shape), 'buy' a short's")
    parser.add_argument("--symbol", default=SYMBOL)
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
