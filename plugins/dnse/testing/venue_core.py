"""TESTING ONLY (#157): the fake DNSE venue STATE MACHINE.

One state machine, two adapters. This module is the machine: pure Python, no socket, no clock,
no network. The in-process adapter drives it directly (pytest, the conformance suite); the
socket adapter (``fake_dnse.py``, revived) serves the same machine over HTTP and WS so
``pyne run --broker`` can reach it unchanged. Keeping ONE machine is the point — two fakes
answering the same question is how the answers drift apart.

Named ``venue_core`` rather than ``fake_venue`` because ``fake_venue.py`` already exists in this
directory (card #10, ``FakeDNSEVenue``, alive in ``test_fixes_end_to_end.py``); its measured
quirks are absorbed here rather than re-derived.

EVERY behaviour below is a MEASURED venue fact with its source. Where the venue's documented
behaviour and its measured behaviour disagree, the measured one wins — the documentation fills
in only what was never measured. A fake that is wrong in a direction the plugin happens to
absorb teaches nothing and still passes, which is the failure mode this file exists to avoid.

Deliberately NOT modelled (card #157 non-goals): a matching engine beyond the recorded prints,
market impact, multiple accounts.
"""
from __future__ import annotations

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

#: Phases in which the venue refuses order placement (live_test/README.md).
_CLOSED_PHASES = {"post_close", "closed"}


class VenueReject(Exception):
    """A venue refusal carrying its STRUCTURED code.

    DNSE's write rejects are coded, not free text (CLAUDE.md), and the engine branches on the
    code, so the code is part of the contract rather than a human-readable message.
    """

    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message


class FakeVenue:
    """An offline DNSE venue driven by market prints rather than by a clock.

    Nothing happens here on a timer. A conditional triggers, a limit fills and a partial
    completes only when :meth:`feed_print` delivers a print that warrants it — which is what
    makes a replay deterministic and is the opposite of the real DNSE sandbox, whose fills are
    a fixed server-side timer (CLAUDE.md, measured 2026-09-12).
    """

    def __init__(self, *, phase: str = "continuous", symbol: str = "41I1G9000",
                 market_type: str = "DERIVATIVE", last_price: float = 0.0,
                 seed: int | None = None):
        self.phase = phase
        self.symbol = symbol
        self.market_type = market_type
        self.last_price = last_price
        self._seed = seed
        # Seeded, monotonic id allocation: ids must never come from a clock or a random source,
        # or two replays of one day diverge and every pin becomes unreliable evidence.
        self._next_normal = 100000 + (seed or 0) % 100000
        self._next_cond = 0
        self._orders: dict[Any, dict] = {}
        self._records: list[dict] = []
        self._seq = 0

    # ----------------------------------------------------------------- ids

    def _new_normal_id(self) -> str:
        """NORMAL book: integer-shaped ids (CLAUDE.md, e.g. 437346)."""
        self._next_normal += 1
        return str(self._next_normal)

    def _new_conditional_id(self) -> str:
        """Conditional book: long string ids (CLAUDE.md, e.g. da203hg6p09g1n1vipog).

        Derived from a counter, not from time, so it is reproducible across replays.
        """
        self._next_cond += 1
        base = (self._seed or 0) * 1000 + self._next_cond
        return f"d{base:012x}pg"

    # ----------------------------------------------------------------- record

    def _record(self, order: dict) -> None:
        """Append an ordered snapshot. The record is the venue's own history, and it is what a
        grader reads — never the run log (the standing rule for live grading applies offline)."""
        self._seq += 1
        self._records.append({
            "seq": self._seq,
            "id": order["id"],
            "orderStatus": order["orderStatus"],
            "fillQuantity": order.get("fillQuantity", 0.0),
        })

    def records(self) -> list[dict]:
        """The ordered record of every state the venue passed through."""
        return [dict(r) for r in self._records]

    # ----------------------------------------------------------------- orders

    def place(self, *, category: str, side: str, qty: float,
              price: float | None = None, stop_price: float | None = None) -> dict:
        """Place on the NORMAL book or the conditional book.

        An OCO is Activated FROM BIRTH and spawns exactly ONE normal-book child at placement
        (measured 2026-09-15, re-measured in #159, umbrella-to-child gap 9-13 ms). There is no
        second leg: the stop side of an OCO is not an order, it is a trigger that will later
        rewrite this same child (see :meth:`feed_print`).
        """
        if self.phase in _CLOSED_PHASES:
            raise VenueReject("CANNOT_PLACE_ORDER_IN_THE_CLOSED_SESSION",
                              f"phase={self.phase}")

        if category == "NORMAL":
            order = self._make(self._new_normal_id(), "NORMAL", side, qty, price, None, NEW)
            return dict(order)

        if category not in ("STOP", "OCO"):
            raise VenueReject("UNSUPPORTED_ORDER_CATEGORY", category)

        cond = self._make(self._new_conditional_id(), "STOP_BOOK", side, qty,
                          price, stop_price, NEW)

        if category == "OCO":
            # Activated from birth, with its child already on the normal book.
            child = self._make(self._new_normal_id(), "NORMAL", side, qty, price, None, NEW)
            child["parent_id"] = cond["id"]
            cond["externalOrderId"] = child["id"]
            cond["orderStatus"] = ACTIVATED
            cond["category"] = "OCO"
            self._record(cond)
        return dict(cond)

    def _make(self, order_id: str, book: str, side: str, qty: float,
              price: float | None, stop_price: float | None, status: str) -> dict:
        order = {
            "id": order_id, "book": book, "side": side,
            "quantity": float(qty), "price": price, "stopPrice": stop_price,
            "orderStatus": status, "fillQuantity": 0.0,
            "externalOrderId": None, "parent_id": None, "category": book,
        }
        self._orders[order_id] = order
        self._record(order)
        return order

    def order(self, order_id) -> dict | None:
        """One order record, or None when the venue does not resolve the id."""
        found = self._orders.get(order_id)
        return dict(found) if found else None

    def orders(self, *, book: str) -> list[dict]:
        """Every order on one book.

        The STOP book keeps returning Activated shells for the rest of the day (#41): a
        triggered conditional never disappears, and `venue.py status` depends on still seeing it.
        """
        want = "STOP_BOOK" if book == "STOP" else "NORMAL"
        return [dict(o) for o in self._orders.values() if o["book"] == want]

    def cancel(self, order_id) -> dict:
        """Cancel by id, with the venue's own refusal codes.

        Three refusals, each measured:
        * ATC refuses cancels outright (live_test/README.md);
        * an Activated conditional is DONE and cannot be cancelled (CO-ORD-013, CLAUDE.md);
        * a terminal order refuses permanently with ORDER_CANCEL_STATUS_REJECTED — and when it
          became terminal by venue AMENDMENT the refusal never becomes transient, which is the
          assumption #162's park was built on and the reason it could never clear.
        """
        if self.phase == "atc":
            raise VenueReject("CANNOT_CANCEL_THE_ORDER_IN_THE_ATC_SESSION", str(order_id))

        order = self._orders.get(order_id)
        if order is None:
            raise VenueReject("RESOURCE_NOT_FOUND", str(order_id))
        if order["orderStatus"] == ACTIVATED:
            raise VenueReject("CO-ORD-013", "order status is not new")
        if order["orderStatus"] in _TERMINAL:
            raise VenueReject("ORDER_CANCEL_STATUS_REJECTED", "order is done")

        order["orderStatus"] = CANCELED
        self._record(order)
        return dict(order)

    # ----------------------------------------------------------------- market

    def feed_print(self, *, price: float, volume: float) -> None:
        """Deliver one market print — the only thing that moves this venue.

        Order matters: conditionals are evaluated before resting orders, because a conditional
        that triggers on this print creates or rewrites a normal-book order that the same print
        may then fill. That is the sequence the WS frames showed in #159.
        """
        self.last_price = price
        self._trigger_conditionals(price)
        self._match_resting(price, volume)

    def _trigger_conditionals(self, price: float) -> None:
        for order in list(self._orders.values()):
            if order["book"] != "STOP_BOOK" or order["stopPrice"] is None:
                continue
            if not self._crossed(order, price):
                continue

            if order["category"] == "OCO":
                # #159, measured 2026-09-18: the stop leg AMENDS the existing child IN PLACE.
                # New -> PendingReplace -> New on ONE id. No order is created, so there is no
                # far leg to cancel and one-cancels-other never appears as a sibling being
                # cancelled. The umbrella is a FOLLOWER here: its record was written 0.473 s
                # after the child's, so the child is the actor.
                child = self._orders.get(order["externalOrderId"])
                if child is None or child["orderStatus"] in _TERMINAL:
                    continue
                child["orderStatus"] = PENDING_REPLACE
                self._record(child)
                # Rewritten to execute against THIS print. The venue prices a triggered stop
                # THROUGH its trigger so it crosses the spread rather than resting (the reason
                # broker.py:_stop_fill_price offsets by 2x slippage: an order left AT the
                # trigger becomes a stop-LIMIT that never fills on a gap). Modelling the
                # rewritten price as the triggering print is that behaviour's outcome without
                # inventing a tick offset the venue never published: #159 saw
                # PendingReplace -> New -> Filled in one burst, so the rewrite executes here.
                child["price"] = price
                child["orderStatus"] = NEW
                self._record(child)
                # The umbrella stays Activated with its stopPrice populated, exactly as a spent
                # one does on a flat account (#159) — armed and spent are indistinguishable from
                # this row, which is why #152 resolves through the child.
            elif order["orderStatus"] == NEW:
                # A STOP ENTRY spawns a NEW normal-book child, named by externalOrderId (#39).
                order["orderStatus"] = ACTIVATED
                child = self._make(self._new_normal_id(), "NORMAL", order["side"],
                                   order["quantity"], order["price"], None, NEW)
                child["parent_id"] = order["id"]
                order["externalOrderId"] = child["id"]
                self._record(order)

    @staticmethod
    def _crossed(order: dict, price: float) -> bool:
        """A buy stop triggers at or above its trigger, a sell stop at or below it."""
        if order["side"] == "buy":
            return price >= order["stopPrice"]
        return price <= order["stopPrice"]

    def _match_resting(self, price: float, volume: float) -> None:
        """Fill resting NORMAL orders against this print.

        A buy limit fills when the print trades at or below its price, a sell limit at or above.
        Quantity is matched against the print's VOLUME, not a timer, so a large order takes
        several prints — the real partial-fill shape. A qty-1 DERIVATIVE never partial-fills, so
        it completes on any matching print regardless of the print's size; emitting a partial
        tick there would invent a state the venue cannot produce.
        """
        remaining_volume = float(volume)
        for order in list(self._orders.values()):
            if order["book"] != "NORMAL" or order["orderStatus"] in _TERMINAL:
                continue
            if order["price"] is None or not self._marketable(order, price):
                continue

            outstanding = order["quantity"] - order["fillQuantity"]
            if self.market_type == "DERIVATIVE" and order["quantity"] == 1:
                take = outstanding
            else:
                take = min(outstanding, remaining_volume)
                remaining_volume -= take
            if take <= 0:
                continue

            order["fillQuantity"] += take
            order["orderStatus"] = (FILLED if order["fillQuantity"] >= order["quantity"]
                                    else PARTIALLY_FILLED)
            self._record(order)

    @staticmethod
    def _marketable(order: dict, price: float) -> bool:
        if order["side"] == "buy":
            return price <= order["price"]
        return price >= order["price"]
