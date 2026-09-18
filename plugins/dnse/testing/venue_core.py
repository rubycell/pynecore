"""TESTING ONLY (#157): the fake DNSE venue STATE MACHINE.

One state machine, two adapters. This module is the machine: pure Python, no socket, no clock,
no network. The in-process adapter drives it directly (pytest, the conformance suite); the
socket adapter (``venue_http.py``) serves the same machine over HTTP so ``pyne run --broker``
can reach it unchanged. Keeping ONE machine is the point — two fakes answering the same question
is how the answers drift apart.

Named ``venue_core`` rather than ``fake_venue`` because ``fake_venue.py`` already exists in this
directory (card #10, ``FakeDNSEVenue``, alive in ``test_fixes_end_to_end.py``); its measured
quirks are absorbed here rather than re-derived.

**Rows leave this machine in the VENUE's vocabulary, not in ours.** Sides are ``NB``/``NS``, not
``buy``/``sell``. That is not cosmetic: the plugin's read funnel is
``side=_DNSE_TO_SIDE.get(raw.get("side",""), "buy")`` (``broker.py:1134``), a lookup with a
``buy`` DEFAULT, so a row carrying ``"sell"`` is silently booked as a BUY and every position sign
derived from it is wrong while nothing raises. Internal logic still reasons in buy/sell through
:func:`_is_buy`, which is a readability choice that must never leak to the wire.

EVERY behaviour below is a MEASURED venue fact with its source, EXCEPT the two marked DESIGN
CHOICE, which are labelled because a fake that presents a design choice as a measurement is the
most expensive kind of wrong this project produces.

Deliberately NOT modelled (card #157 non-goals): a matching engine beyond the recorded prints,
market impact, multiple accounts.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from typing import Any

# Status vocabulary: the venue's own, never the uppercase variant that testing/fake_dnse.py
# synthesised. The plugin's _STATUS_MAP (broker.py:162-171) is keyed on uppercased strings and
# maps both spellings to one value, so it cannot tell the difference — which is exactly why the
# fake must get it right and pin it, rather than rely on the plugin to absorb the error.
NEW = "New"
PENDING_REPLACE = "PendingReplace"
PARTIALLY_FILLED = "PartiallyFilled"
FILLED = "Filled"
CANCELED = "Canceled"
ACTIVATED = "Activated"

_TERMINAL = {FILLED, CANCELED}

#: Wire side codes. The venue speaks these; so does this machine, on every row it emits.
BUY, SELL = "NB", "NS"
_TO_WIRE = {"buy": BUY, "sell": SELL, BUY: BUY, SELL: SELL}

#: Phases in which the venue refuses order placement.
#:
#: DESIGN CHOICE, partly unmeasured: ``CO-ORD-006`` is measured for post-close writes, but
#: ``CANNOT_PLACE_ORDER_IN_THE_CLOSED_SESSION`` appears nowhere in ``live_test/README.md`` — only
#: in ``errors.py`` and in ``t33_closed_hours.py``, a probe that has never been run. It is served
#: here because the plugin can classify it, and it is labelled because it is not evidence.
#: Settled by: running the L1-T33 post-close probe once and recording which code the venue sends.
_CLOSED_PHASES = {"post_close", "closed"}


def _is_wire_buy(order: dict) -> bool:
    """Internal side test. Rows carry the WIRE code, so nothing may compare against 'buy'."""
    return order["side"] == BUY


class VenueReject(Exception):
    """A venue refusal carrying its STRUCTURED code.

    DNSE's write rejects are coded, not free text (CLAUDE.md), and the engine branches on the
    code, so the code is part of the contract rather than a human-readable message.
    """

    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message


class VenueServerError(Exception):
    """A 5xx from the venue, which is NOT a coded rejection.

    Kept separate from :class:`VenueReject` because the engine treats them differently: a coded
    400 is a decision it can branch on, a 500 is the venue failing to answer. The measured
    derivative amend is the latter.
    """

    def __init__(self, status: int = 500, message: str = ""):
        super().__init__(message or f"HTTP {status}")
        self.status = status
        self.message = message



#: VN30 derivatives session, ICT. The values are the measured venue behaviour recorded in
#: CLAUDE.md and in the L0 gate's own phase table, kept in ONE place so the fake and the gate
#: cannot drift: continuous 09:00-11:30 and 13:00-14:30, lunch 11:30-13:00, ATC 14:30-14:45.
_ICT = timezone(timedelta(hours=7))


def phase_at(timestamp_ms: int) -> str:
    """The trading phase at an epoch-millisecond instant, read in ICT.

    Used with the ORIGINAL timestamp of the bar being replayed, never the shifted one and never
    the wall clock. The day is re-stamped onto the current minute so the engine's live path stays
    anchored to a clock it believes, but a replay of a 09:00-to-14:45 session is still a replay of
    that session: the venue must walk its real phases whatever hour someone runs it at. Deriving
    the phase from the shifted stamp would merely reproduce the wall clock, which is the thing
    that made a closed-hours question impossible to pose.
    """
    when = datetime.fromtimestamp(timestamp_ms / 1000, _ICT)
    minutes = when.hour * 60 + when.minute
    if when.weekday() > 4:
        return "closed"
    if 9 * 60 <= minutes < 11 * 60 + 30:
        return "continuous"
    if 11 * 60 + 30 <= minutes < 13 * 60:
        return "lunch"
    if 13 * 60 <= minutes < 14 * 60 + 30:
        return "continuous"
    if 14 * 60 + 30 <= minutes < 14 * 60 + 45:
        return "atc"
    return "closed"


class FakeVenue:
    """An offline DNSE venue driven by market prints rather than by a clock.

    Nothing happens here on a timer. A conditional triggers, a limit fills and a partial
    completes only when :meth:`feed_print` delivers a print that warrants it — which is what
    makes a replay deterministic and is the opposite of the real DNSE sandbox, whose fills are
    a fixed server-side timer (CLAUDE.md, measured 2026-09-12).
    """

    #: Fields that exist for this machine's own bookkeeping and must NEVER reach a listing.
    _INTERNAL = ("parent_id", "book", "_category")

    def __init__(self, *, phase: str = "continuous", symbol: str = "41I1G9000",
                 market_type: str = "DERIVATIVE", last_price: float = 0.0,
                 seed: int | None = None, record_file=None):
        self.phase = phase
        self.symbol = symbol
        self.market_type = market_type
        self.last_price = last_price
        self._seed = seed
        # Where to keep the record so a run can be GRADED from the venue rather than from its
        # log, exactly as a live result is. Off unless asked for; see _record for why the write
        # happens after every transition instead of at exit.
        self._record_file = Path(record_file) if record_file is not None else None
        # Seeded, monotonic id allocation: ids must never come from a clock or a random source,
        # or two replays of one day diverge and every pin becomes unreliable evidence.
        self._next_normal = 100000 + (seed or 0) % 100000
        self._next_cond = 0
        self._orders: dict[Any, dict] = {}
        self._records: list[dict] = []
        self._seq = 0

    def advance_to(self, timestamp_ms: int) -> str:
        """Move the venue's clock to the instant of the bar being replayed.

        The phase is DERIVED here rather than passed in, so a caller cannot hand the venue a
        phase that disagrees with the bar it is replaying. A venue that is never advanced keeps
        whatever phase it was constructed with, which is what every existing unit pin relies on.
        """
        self.phase = phase_at(int(timestamp_ms))
        return self.phase

    # ----------------------------------------------------------------- ids

    def _new_normal_id(self) -> str:
        """NORMAL book: integer-shaped ids (CLAUDE.md, e.g. 437346)."""
        self._next_normal += 1
        return str(self._next_normal)

    def _new_conditional_id(self) -> str:
        """Conditional book: long string ids (CLAUDE.md, e.g. da203hg6p09g1n1vipog)."""
        self._next_cond += 1
        base = (self._seed or 0) * 1000 + self._next_cond
        return f"d{base:012x}pg"

    # ----------------------------------------------------------------- record

    def _record(self, order: dict) -> None:
        """Append an ordered snapshot — the venue's own history, and what a grader reads."""
        self._seq += 1
        self._records.append({
            "seq": self._seq,
            "id": order["id"],
            "orderStatus": order["orderStatus"],
            "fillQuantity": order.get("fillQuantity", 0.0),
        })
        if self._record_file is not None:
            # Rewritten in full after EVERY transition, not appended at exit. A supervised run
            # is stopped by the operator and a run killed by a signal reaches no exit hook, so
            # an end-of-run flush would lose precisely the runs a grader cares about. The
            # record is tens of rows, so rewriting it costs nothing worth optimising.
            self._record_file.parent.mkdir(parents=True, exist_ok=True)
            self._record_file.write_text(json.dumps(self._records, indent=2), encoding="utf-8")

    def all_orders(self) -> list[dict]:
        """Every order this venue has ever held, in creation order.

        The order BOOKS only show what is still working; history shows what happened, including
        rows that are terminal. Keeping them separate is the point: a cancelled order leaves the
        book but stays in history, which is what makes a previous-day lookup possible at all.
        """
        return [dict(order) for order in self._orders.values()]

    def records(self) -> list[dict]:
        return [dict(r) for r in self._records]

    # ----------------------------------------------------------------- views

    def _detail(self, order: dict) -> dict:
        """The DETAIL view: everything the venue publishes, internals stripped.

        ``externalOrderId`` lives HERE and only here (measured; ``broker.py:1594`` reads the
        child from the detail). A listing that volunteered it would let an engine path find the
        child without the detail read production forces, so a #39-class regression could pass
        offline and fail live.
        """
        return {k: v for k, v in order.items() if k not in self._INTERNAL}

    def _listing(self, order: dict) -> dict:
        """The LISTING view: the detail minus ``externalOrderId``."""
        row = self._detail(order)
        row.pop("externalOrderId", None)
        return row

    # ----------------------------------------------------------------- orders

    def place(self, *, category: str, side: str, qty: float,
              price: float | None = None, stop_price: float | None = None,
              stop_order_price: float | None = None) -> dict:
        """Place on the NORMAL book or the conditional book.

        An OCO is Activated FROM BIRTH and spawns exactly ONE normal-book child at placement
        (measured 2026-09-15, re-measured in #159). There is no second leg: the stop side is not
        an order, it is a trigger that will later rewrite this same child.

        ``stop_order_price`` is the LIMIT the triggered stop will post, which the plugin computes
        THROUGH the trigger (``broker.py:1496`` sends it as ``stopOrderPrice``). Discarding it —
        as an earlier version of this fake did — makes the gap-through-unfilled case unreachable,
        and that case is the entire reason ``_stop_fill_price`` exists.
        """
        if self.phase in _CLOSED_PHASES:
            raise VenueReject("CANNOT_PLACE_ORDER_IN_THE_CLOSED_SESSION", f"phase={self.phase}")

        wire_side = _TO_WIRE.get(side, side)
        if category == "NORMAL":
            return dict(self._detail(
                self._make(self._new_normal_id(), "NORMAL", wire_side, qty, price, None,
                           None, NEW)))

        if category not in ("STOP", "OCO"):
            raise VenueReject("UNSUPPORTED_ORDER_CATEGORY", category)

        # The venue's order-category matrix (changelog 2026-08-06):
        #
        #   orderCategory   NORMAL   STOP   OCO
        #   STOCK             yes     yes    NO
        #   DERIVATIVE        yes     yes    yes
        #   BOND              yes     NO     NO
        #
        # This accepted OCO on a stock until the matrix was checked. That is the same error as
        # the amend model corrected in round 1: a fake MORE PERMISSIVE than the venue teaches the
        # engine a capability that will fail live, and no pin of ours can catch it, because the
        # pin and the fake were written from the same belief.
        # Two independent pages state the same matrix: changelog.md:24-28 as a table, and
        # dnse-place-order.md:11-14 in prose.
        if category == "OCO" and self.market_type != "DERIVATIVE":
            raise VenueReject("UNSUPPORTED_ORDER_CATEGORY",
                              f"OCO is a DERIVATIVE-only category; {self.market_type} cannot")
        if category == "STOP" and self.market_type == "BOND":
            raise VenueReject("UNSUPPORTED_ORDER_CATEGORY",
                              "BOND supports NORMAL only — no STOP and no OCO")

        cond = self._make(self._new_conditional_id(), "STOP_BOOK", wire_side, qty,
                          price, stop_price, stop_order_price, NEW, category=category)

        if category == "OCO":
            child = self._make(self._new_normal_id(), "NORMAL", wire_side, qty, price, None,
                               None, NEW)
            child["parent_id"] = cond["id"]
            cond["externalOrderId"] = child["id"]
            cond["orderStatus"] = ACTIVATED
            self._record(cond)
        return dict(self._detail(cond))

    def _make(self, order_id: str, book: str, wire_side: str, qty: float,
              price: float | None, stop_price: float | None,
              stop_order_price: float | None, status: str,
              category: str | None = None) -> dict:
        order = {
            "id": order_id,
            "symbol": self.symbol,
            "side": wire_side,                       # WIRE code, never buy/sell
            "orderType": "LO",
            "orderCategory": category or "NORMAL",
            "quantity": float(qty),
            "price": price,
            "stopPrice": stop_price,
            "stopOrderPrice": stop_order_price,
            "orderStatus": status,
            "fillQuantity": 0.0,
            "averagePrice": None,
            "canceledQuantity": 0.0,
            "externalOrderId": None,
            # internals, stripped from every view
            "parent_id": None,
            "book": book,
            "_category": category or "NORMAL",
        }
        self._orders[order_id] = order
        self._record(order)
        return order

    def order(self, order_id, *, category: str | None = None) -> dict | None:
        """DETAIL for one id.

        ``category`` selects the book. The venue answers ``RESOURCE_NOT_FOUND`` for an id looked
        up on the WRONG book, which is how a caller learns the books are separate; a fake that
        ignored the parameter would let book-confused code pass.
        """
        found = self._orders.get(order_id)
        if found is None:
            return None
        if category is not None and not self._on_book(found, category):
            return None
        return self._detail(found)

    @staticmethod
    def _on_book(order: dict, category: str) -> bool:
        wanted_conditional = category in ("STOP", "OCO")
        return (order["book"] == "STOP_BOOK") is wanted_conditional

    def orders(self, *, book: str) -> list[dict]:
        """LISTING for one book — internals and ``externalOrderId`` stripped.

        The STOP book keeps returning Activated shells for the rest of the day (#41).
        """
        want = "STOP_BOOK" if book == "STOP" else "NORMAL"
        return [self._listing(o) for o in self._orders.values() if o["book"] == want]

    def cancel(self, order_id, *, category: str | None = None) -> dict:
        """Cancel by id, with the venue's own refusal codes."""
        if self.phase == "atc":
            raise VenueReject("CANNOT_CANCEL_THE_ORDER_IN_THE_ATC_SESSION", str(order_id))

        order = self._orders.get(order_id)
        if order is None or (category is not None and not self._on_book(order, category)):
            raise VenueReject("RESOURCE_NOT_FOUND", str(order_id))
        if order["orderStatus"] == ACTIVATED:
            raise VenueReject("CO-ORD-013", "order status is not new")
        if order["orderStatus"] in _TERMINAL:
            raise VenueReject("ORDER_CANCEL_STATUS_REJECTED", "order is done")

        order["orderStatus"] = CANCELED
        order["canceledQuantity"] = order["quantity"] - order["fillQuantity"]
        self._record(order)
        return self._detail(order)

    def amend(self, order_id, *, price: float | None = None, qty: float | None = None,
              category: str | None = None) -> dict:
        """Amend an order. Three outcomes, and the discriminator is the BOOK, not the asset.

        * CONDITIONAL book (STOP/OCO), any asset: HTTP 500. Measured as
          Live-L1-T07-AmendConditional500 (#18), and it is why the plugin routes a conditional
          modify away from a PUT entirely at broker.py:2278 — a conditional ENTRY becomes its
          own outcome-gated cancel+replace, a conditional EXIT becomes a park.
        * NORMAL book, DERIVATIVE: 200, amended IN PLACE, same id with the new price and
          quantity. Measured as Live-L1-T06-AmendNormal, PASS 2026-08-14, re-verified 08-17.
        * NORMAL book, STOCK: 200 with a NEW order id, the venue having cancelled the old one
          itself so it reads Canceled untouched (#117, prod 2026-09-15). Both price and quantity
          land in one PUT, and anything still tracking the OLD id goes blind.

        An earlier version of this method gated on ASSET TYPE and answered 500 for every
        derivative amend. That was wrong and it made the fake teach a venue that does not exist:
        the staged probe's T6 parked on a 500 the real venue would never have sent for a
        normal-book order.
        """
        order = self._orders.get(order_id)
        if order is None or (category is not None and not self._on_book(order, category)):
            raise VenueReject("RESOURCE_NOT_FOUND", str(order_id))
        if order["orderStatus"] in _TERMINAL:
            raise VenueReject("ORDER_IS_DONE", "order is done")

        if order["book"] == "STOP_BOOK":
            raise VenueServerError(500, "conditional amend is not supported")

        if self.market_type != "STOCK":
            # DOCUMENTED, NOT MEASURED (FAQ #33, "Sua lenh co so va phai sinh khac nhau nhu the
            # nao"): a derivative amend changes price OR quantity, never both, and the amended
            # quantity must exceed what has already filled. A stock amend may change both,
            # which IS measured (#117, prod 2026-09-15) and agrees with the same FAQ row.
            #
            # Flagged as documentation-sourced because the project rule is that a measurement
            # beats a document. Nothing has been measured against this restriction, so a live
            # amend sending both fields on a derivative would overturn it — and whoever measures
            # that should delete this branch rather than argue with it.
            # The rule is about what CHANGES, not about which fields are PRESENT. The plugin
            # sends the full order on a PUT, so an unchanged quantity travels alongside a changed
            # price on every amend — and Live-L1-T06-AmendNormal, which PASSED on the venue
            # 2026-08-14, is exactly that shape. A first draft of this guard refused on presence
            # and broke that measured pin: stricter than the venue, from a rule that never said
            # so. Compare against the order's current values instead.
            changes_price = price is not None and price != order["price"]
            changes_qty = qty is not None and qty != order["quantity"]
            if changes_price and changes_qty:
                raise VenueReject(
                    "INPUT_INVALID",
                    "a DERIVATIVE amend changes price OR quantity, not both (FAQ #33)")
            if qty is not None and qty <= float(order.get("fillQuantity") or 0):
                raise VenueReject(
                    "INVALID_QUANTITY",
                    f"amended quantity {qty} must exceed the filled quantity "
                    f"{order.get('fillQuantity')}")
            # Normal-book derivative: amended in place, same id.
            if price is not None:
                order["price"] = price
            if qty is not None:
                order["quantity"] = qty
            order["orderStatus"] = NEW
            self._record(order)
            return self._detail(order)

        order["orderStatus"] = CANCELED
        order["canceledQuantity"] = order["quantity"] - order["fillQuantity"]
        self._record(order)
        replacement = self._make(
            self._new_normal_id(), order["book"], order["side"],
            qty if qty is not None else order["quantity"],
            price if price is not None else order["price"],
            order["stopPrice"], order["stopOrderPrice"], NEW,
            category=order["_category"])
        return self._detail(replacement)

    # ----------------------------------------------------------------- market

    def feed_print(self, *, price: float, volume: float) -> None:
        """Deliver one market print — the only thing that moves this venue."""
        self.last_price = price
        self._trigger_conditionals(price)
        self._match_resting(price, volume)

    def _trigger_conditionals(self, price: float) -> None:
        for order in list(self._orders.values()):
            if order["book"] != "STOP_BOOK" or order["stopPrice"] is None:
                continue
            if not self._crossed(order, price):
                continue

            if order["_category"] == "OCO":
                # #159: the stop leg AMENDS the existing child IN PLACE. New -> PendingReplace
                # -> New on ONE id; no order is created, so there is no far leg to cancel.
                child = self._orders.get(order["externalOrderId"])
                if child is None or child["orderStatus"] in _TERMINAL:
                    continue
                child["orderStatus"] = PENDING_REPLACE
                self._record(child)
                # Amended to the LIMIT the plugin computed through the trigger, NOT to the print.
                # If the market has already gapped past that limit the child RESTS, unfilled and
                # unprotecting — which is the failure _stop_fill_price exists to make unlikely and
                # which a fake that filled at the print could never reproduce.
                child["price"] = (order["stopOrderPrice"]
                                  if order["stopOrderPrice"] is not None else order["stopPrice"])
                child["orderStatus"] = NEW
                self._record(child)
            elif order["orderStatus"] == NEW:
                order["orderStatus"] = ACTIVATED
                child = self._make(self._new_normal_id(), "NORMAL", order["side"],
                                   order["quantity"],
                                   order["stopOrderPrice"] if order["stopOrderPrice"] is not None
                                   else order["price"],
                                   None, None, NEW)
                child["parent_id"] = order["id"]
                order["externalOrderId"] = child["id"]
                self._record(order)

    @staticmethod
    def _crossed(order: dict, price: float) -> bool:
        """A buy stop triggers at or above its trigger, a sell stop at or below it."""
        return price >= order["stopPrice"] if _is_wire_buy(order) else price <= order["stopPrice"]

    def _match_resting(self, price: float, volume: float) -> None:
        """Fill resting NORMAL orders against this print.

        Quantity is matched against the print's VOLUME, not a timer.

        DESIGN CHOICE, unmeasured: matching by traded volume is a model, not a recorded venue
        behaviour. The only PartiallyFilled evidence in the project is the SANDBOX, which
        CLAUDE.md describes as a fixed server-side timer emitting one variable chunk — a
        different mechanism. Settled by: one prod stock fill of qty>1 with the executions or WS
        frames captured, showing whether chunk size tracks traded volume.
        """
        remaining_volume = float(volume)
        for order in list(self._orders.values()):
            if order["book"] != "NORMAL" or order["orderStatus"] in _TERMINAL:
                continue
            if order["price"] is None or not self._marketable(order, price):
                continue

            outstanding = order["quantity"] - order["fillQuantity"]
            if self.market_type == "DERIVATIVE" and order["quantity"] == 1:
                take = outstanding          # a qty-1 derivative never partial-fills
            else:
                take = min(outstanding, remaining_volume)
                remaining_volume -= take
            if take <= 0:
                continue

            filled_before = order["fillQuantity"]
            order["fillQuantity"] = filled_before + take
            # Cumulative VWAP, because that is what the plugin books against (the executions
            # endpoint 404s on this account) and fill_price is None without it (broker.py:3086).
            previous = (order["averagePrice"] or 0.0) * filled_before
            order["averagePrice"] = round((previous + price * take) / order["fillQuantity"], 4)
            order["orderStatus"] = (FILLED if order["fillQuantity"] >= order["quantity"]
                                    else PARTIALLY_FILLED)
            self._record(order)

    @staticmethod
    def _marketable(order: dict, price: float) -> bool:
        return price <= order["price"] if _is_wire_buy(order) else price >= order["price"]
