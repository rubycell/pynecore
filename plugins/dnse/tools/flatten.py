"""#91 — fill-tier flatten: close FIRST, then sweep OUR protection.

Measured live 2026-09-08 (operator-caught): the old close-only flatten left
the closed position's protective stop ARMED on a flat account — with the
engine down, a naked entry-stop until the 14:45 DAY expiry (#82 class).

Design (card #91, panel-adjudicated, unanimous):

* **Close-first** — a rejected close under protection-first is one failure
  from an UNBOUNDED naked open (near-market closes CAN reject, measured
  F6), and re-arming protection after a refused close is a conditional
  write of the same class that just refused. Close-first's failure state
  leaves the account unchanged and retryable.
* **The sweep runs only after the close is CONFIRMED flat** — never on an
  unconfirmed close.
* **Attribution is a READ-ONLY store query** (:func:`owned_live_ids`),
  never ``connect()``/``open_run``: ``open_run`` raises on the live-row
  collision for the whole 5-minute stale window (storage.py:886) — exactly
  the post-crash emergency this tool exists for — and would steal the run
  identity from the #77/T16 restart adoption. A missing/unreadable store
  is "attribution UNAVAILABLE" (``None``) and the caller must exit 2 —
  ``BrokerStore`` is never instantiated here because its constructor
  CREATES the file, which would make absence read as "clean".
* **The netting account is SHARED** — a venue order outside the owned set
  is the operator's: reported, never cancelled (hard rule).
* Sweep cancels go through the plugin's disposition core
  (``_cancel_one_disposition`` — the same path ``venue.py cancel`` uses):
  a #41 phantom shell resolves ``ALREADY_FILLED`` via its child, a
  double-cancel racing a live engine answers the measured-terminal
  ``ORDER_CANCEL_STATUS_REJECTED``, and a #51 refusal classifies
  ``UNKNOWN`` — reported and exit 1, never retried (#58 write policy).

Exit codes (venue.py convention): 0 flat AND owned orders swept/resolved;
1 not flat, or an owned order's disposition stayed UNKNOWN; 2 could not
determine (position unreadable, or attribution unavailable).
"""
import asyncio
import sqlite3
import time
from pathlib import Path

from pynecore.core.broker.models import (
    CancelDispositionOutcome, CloseIntent, DispatchEnvelope,
)

#: Sweep outcomes that mean "this order is provably done" (fill outranks —
#: an ALREADY_FILLED here is a consumed entry/shell, not a working order).
_RESOLVED = frozenset({
    CancelDispositionOutcome.CANCEL_CONFIRMED,
    CancelDispositionOutcome.ALREADY_FILLED,
    CancelDispositionOutcome.TOO_LATE_TO_CANCEL,
    CancelDispositionOutcome.STILL_OPEN,
})


def owned_live_ids(store_path, account_id: str) -> "set[str] | None":
    """Venue ids the BOT owns on THIS account, from the journal — read-only.

    Scoping (#96, panel-adjudicated): rows join to their run's
    ``account_id`` (the real custody boundary — the shared store holds
    other venues' runs; plugin_name is NOT used because probe subclasses
    journal under their own display names and their leftovers must stay
    sweepable). Digit-shaped (NORMAL-class) ids additionally require
    activity TODAY — ``MAX(created_ts_ms, updated_ts_ms)`` on the current
    ICT calendar date — because DNSE REUSES NORMAL ids across days
    (measured: 09-08 issued lower ids than 09-07) and a stale row could
    claim a foreign order the venue reissued. MAX, never created alone: a
    reopened deterministic-coid row (#77/T16) carries TODAY's venue id
    under YESTERDAY's created date, and dropping it would leave OUR OWN
    protection out of the sweep. String (conditional hash) ids have no
    reuse class and stay day-unscoped (multi-day GTD conditionals).

    Deliberately NO symbol filter (the store speaks WIRE symbols — the
    #77 lesson); the sweep's venue-working intersection scopes symbols.

    ``None`` = attribution UNAVAILABLE: store missing/unreadable, OR no
    ``runs`` rows exist for this account (an unmatched key must exit 2,
    never read as a clean empty set — the #91 vacuous-pass guard). The
    file is NEVER created here.
    """
    from datetime import datetime, timezone, timedelta
    path = Path(store_path)
    if not path.is_file():
        return None
    tz = timezone(timedelta(hours=7))
    day_start_ms = int(datetime.now(tz).replace(
        hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            known = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE account_id = ?",
                (account_id,)).fetchone()[0]
            if not known:
                print(f"attribution UNAVAILABLE: no journalled runs for "
                      f"this account in {path.name}")
                return None
            rows = conn.execute(
                "SELECT DISTINCT o.exchange_order_id, "
                "       MAX(COALESCE(o.created_ts_ms,0), COALESCE(o.updated_ts_ms,0)) "
                "FROM orders o JOIN runs r "
                "  ON r.run_instance_id = o.run_instance_id "
                "WHERE o.closed_ts_ms IS NULL "
                "  AND o.exchange_order_id IS NOT NULL "
                "  AND o.exchange_order_id != '' "
                "  AND r.account_id = ?", (account_id,)).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        print(f"attribution UNAVAILABLE: store read failed: {exc}")
        return None
    owned: set = set()
    for venue_id, last_ms in rows:
        vid = str(venue_id)
        if vid.isdigit() and (last_ms or 0) < day_start_ms:
            continue        # stale NORMAL-class id: the cross-day reuse trap
        owned.add(vid)
    return owned


def _read_position_size(broker, symbol: str) -> "float | None":
    """Signed net size, or ``None`` when the read failed (never 'flat')."""
    try:
        pos = asyncio.run(broker.get_position(symbol))
    except Exception as exc:                                  # noqa: BLE001
        print(f"COULD NOT READ position: {type(exc).__name__}: {exc}")
        return None
    if pos is None:
        return 0.0
    # ``ExchangePosition.size`` is a MAGNITUDE — the sign lives in ``.side``
    # (broker.py builds it as ``size=abs(net)`` with
    # ``side="long" if net > 0 else "short"``). Reading ``.size`` alone drops
    # the sign, and the sign is the whole decision here: the caller branches
    # on it to choose BUY or SELL. Unsigned, ``size > 0`` is true for every
    # non-flat position, so the "buy" arm was unreachable and flattening a
    # SHORT sold into it — short 1 -> short 2 (the 2026-09-16 incident).
    # The engine re-derives the sign from ``.side`` at three separate places
    # (sync_engine.py ~3849, ~4763, ~5098); this mirrors them, including the
    # refusal to guess an unrecognised label.
    magnitude = abs(float(pos.size or 0.0))
    side = str(pos.side or "").lower()
    if magnitude == 0.0 or side == "flat":
        return 0.0
    if side == "long":
        return magnitude
    if side == "short":
        return -magnitude
    print(f"COULD NOT READ position: unrecognised side {pos.side!r} with "
          f"size {pos.size!r} — refusing to guess the sign")
    return None


def _read_position_size_confirmed(broker, symbol: str,
                                  *, gap_s: float = 1.5) -> "float | None":
    """Signed net size CONFIRMED by two agreeing reads, else ``None``.

    WHY TWO READS (live incident 2026-09-16): a single read decides the SIGN,
    and the sign decides whether we BUY or SELL. Get it wrong and the "flatten"
    DOUBLES the position instead of closing it — which is exactly what happened:
    with the account really SHORT 1, one read answered ``long 1.0``, so the tool
    sold 1 and took it to SHORT 2. The read was not cached — ``get_position``
    hits the venue every time — the VENUE served a stale view, which CLAUDE.md
    already documents for order detail reads ("a Canceled order served as New by
    a lagging replica ~10 s later", 08-17).

    So a directional action now requires agreement. Disagreement means we cannot
    prove the sign, and the fail-closed answer to "which way do I trade?" is DO
    NOTHING — never split the difference, never prefer the newer read (we have no
    evidence which of the two is the stale one).
    """
    first = _read_position_size(broker, symbol)
    if first is None:
        return None
    time.sleep(gap_s)
    second = _read_position_size(broker, symbol)
    if second is None:
        return None
    if first != second:
        print(f"position reads DISAGREE ({first} then {second}) — refusing to "
              f"act. A wrong SIGN doubles the position instead of closing it "
              f"(live 2026-09-16). Re-run once the venue settles, or flatten in "
              f"the app.")
        return None
    return first


def flatten(broker, symbol: str, owned_ids: "set[str] | None",
            *, close_wait_s: int = 25) -> int:
    """Close-first flatten + owned-order sweep. See module docstring."""
    size = _read_position_size_confirmed(broker, symbol)
    if size is None:
        return 2

    if size != 0.0:
        close_side = "sell" if size > 0 else "buy"
        qty = abs(size)
        print(f"position: {'long' if size > 0 else 'short'} {qty} -> "
              f"closing via execute_close ({close_side} {int(qty)} @ band edge)")
        envelope = DispatchEnvelope(
            intent=CloseIntent(pine_id="F-FLATTEN", symbol=symbol,
                               side=close_side, qty=qty, immediately=True),
            run_tag="flat", bar_ts_ms=int(time.time() * 1000),
            retry_seq=0, coid_max_len=30)
        order = asyncio.run(broker.execute_close(envelope))
        print(f"close order placed: id={order.id} {order.side} {order.qty}")
        deadline = time.time() + close_wait_s
        flat = False
        while time.time() < deadline:
            time.sleep(2)
            # CONFIRMED read, not a bare one. This verdict authorises the
            # protection SWEEP below, so it decides an IRREVERSIBLE action on
            # the position's EMPTINESS — the other half of the rule that
            # covers its sign. A bare read here is worse than elsewhere, not
            # better: the loop polls ~12 times and breaks on the FIRST flat,
            # so it actively samples FOR a stale empty page. One lagging
            # response mid-poll declared "FLAT confirmed", cancelled the
            # protective conditionals over a still-open position and returned
            # 0. Disagreement resolves to None (could-not-determine), which is
            # not 0.0 and therefore keeps polling rather than sweeping.
            size_now = _read_position_size_confirmed(broker, symbol)
            if size_now == 0.0:
                flat = True
                break
        if not flat:
            # Unconfirmed close: NEVER sweep (cancelling protection over a
            # possibly-open position is the naked-open the panel rejected).
            print(f"NOT FLAT after {close_wait_s}s — sweep WITHHELD; "
                  f"check venue.py status NOW")
            return 1
        print("FLAT confirmed")
    else:
        print("FLAT already")

    if owned_ids is None:
        print("attribution UNAVAILABLE — cannot prove which resting orders "
              "are ours; position is flat but the protection sweep could "
              "not run. Inspect venue.py status and the store path.")
        return 2

    try:
        working = asyncio.run(broker.get_open_orders(symbol))
    except Exception as exc:                                  # noqa: BLE001
        print(f"COULD NOT READ open orders: {type(exc).__name__}: {exc}")
        return 2
    ours = [order for order in working if str(order.id) in owned_ids]
    foreign = [order for order in working if str(order.id) not in owned_ids]
    for order in foreign:
        # SHARED account: report, never cancel (hard rule).
        print(f"foreign live order (operator's — NOT touched): "
              f"id={order.id} {order.side} qty={order.qty}")
    if not ours:
        print("sweep: no owned working orders — clean")
        return 0

    unresolved = []
    for order in ours:
        outcome = asyncio.run(broker._cancel_one_disposition(str(order.id)))
        print(f"sweep: cancel {order.id} -> {outcome.value}")
        if outcome not in _RESOLVED:
            unresolved.append(str(order.id))
    if unresolved:
        print(f"sweep INCOMPLETE — disposition UNKNOWN for {unresolved}; "
              f"the venue may still hold them (e.g. a #51 refusal window). "
              f"Cancel via the app or retry later. NOT retrying (#58).")
        return 1
    print(f"sweep complete: {len(ours)} owned order(s) resolved; "
          f"{len(foreign)} foreign order(s) reported untouched")
    return 0
