"""TESTING ONLY (#157): the offline fake DNSE venue — STUB, no behaviour implemented yet.

This module exists so the conformance suite written test-first
(``plugins/dnse/tests/test_fake_venue_conformance.py``) can FAIL for the right reason. Every
method below raises :class:`NotImplementedError`; nothing here talks to a socket, a clock or a
fixture. It is deliberately NOT stage C: the transport decision (a standalone server on a real
socket, an in-process transport fake, or cassettes) is still with the adversarial panel on card
#157, and the stage A endpoint inventory goes to the leader for review before any server code.

What this file DOES fix is the API the conformance suite pins, because the tests were written
first and the tests define the contract. Whichever transport the panel picks must satisfy it.

Structural precedent: ``replay_sandbox.py`` (#114) — a TESTING ONLY module living beside the
production plugin, reached only by name, never imported by a production order path.
"""
from __future__ import annotations

_NOT_YET = "#157: the fake venue is not implemented yet — the conformance suite is written first"


class VenueReject(Exception):
    """A venue refusal carrying its STRUCTURED code.

    DNSE's write rejects are coded, not free text (CLAUDE.md: ``ORDER_IS_DONE``,
    ``ORDER_CANCEL_STATUS_REJECTED``, ``CO-ORD-013``, ``CANNOT_CANCEL_THE_ORDER_IN_THE_ATC_SESSION``),
    and the engine branches on the code, so the code is part of the contract rather than a message.
    """

    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message


class FakeVenue:
    """Offline DNSE venue. Every method is a stub until the panel settles the transport."""

    def __init__(self, *, phase: str = "continuous", symbol: str = "41I1G9000",
                 market_type: str = "DERIVATIVE", last_price: float = 0.0,
                 seed: int | None = None):
        self.phase = phase
        self.symbol = symbol
        self.market_type = market_type
        self.last_price = last_price
        self.seed = seed

    # ----------------------------------------------------------------- orders

    def place(self, *, category: str, side: str, qty: float,
              price: float | None = None, stop_price: float | None = None) -> dict:
        """Place on the NORMAL book (integer id) or the conditional book (string id)."""
        raise NotImplementedError(_NOT_YET)

    def cancel(self, order_id) -> dict:
        """Cancel by id, raising :class:`VenueReject` with the venue's own code when refused."""
        raise NotImplementedError(_NOT_YET)

    def order(self, order_id) -> dict | None:
        """One order record, or ``None`` when the venue does not resolve the id."""
        raise NotImplementedError(_NOT_YET)

    def orders(self, *, book: str) -> list[dict]:
        """Every order on one book, including Activated shells (#41)."""
        raise NotImplementedError(_NOT_YET)

    # ----------------------------------------------------------------- market

    def feed_print(self, *, price: float, volume: float) -> None:
        """Deliver one market print: the only thing that triggers conditionals and fills orders."""
        raise NotImplementedError(_NOT_YET)

    # ----------------------------------------------------------------- record

    def records(self) -> list[dict]:
        """The venue's own ordered record of everything that happened, for determinism checks."""
        raise NotImplementedError(_NOT_YET)
