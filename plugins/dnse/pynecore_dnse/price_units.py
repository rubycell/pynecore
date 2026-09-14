"""DNSE price-unit codec (#119) — the FEED unit vs the ORDER-BOOK unit.

DNSE's unit split is by DOMAIN, not by endpoint family:

===================  ==========================  ==========================
domain               STOCK                       DERIVATIVE / INDEX
===================  ==========================  ==========================
market data + secdef THOUSANDS of VND (21.3)     index points (2058.6)
ORDER BOOK           ĐỒNG (21300)                index points (2058.6)
===================  ==========================  ==========================

Measured live on the prod order book 2026-09-14 (HPG, book bid 21.3):
``price=21.0`` -> ``400 PRICE_MUST_GREATER_THAN_OR_EQUAL_TO_FLOOR_PRICE``
(the venue floor-checked it against 19,850 đ); ``price=21000`` -> ``200``.

Everything the engine/strategy sees stays in the **feed** unit — bars, quotes,
secdef bands and therefore every Pine price literal are thousands for stocks.
Only the wire is đồng, so the conversion lives at the venue boundary:

    feed -> wire :  :func:`to_wire` then :func:`quantize_wire`   (writes)
    wire -> feed :  :func:`from_wire`                            (readbacks)

Pure functions only — no I/O, no broker state. The unit SELECTOR (which scale a
symbol gets, and the two hard guards around it) lives on the broker, because it
needs the venue's ``securityGroupId``.
"""
from __future__ import annotations

import math

from pynecore.core.broker.exceptions import ExchangeOrderRejectedError

#: Wire scale for a STOCK: the order book counts đồng, the feed counts thousands.
STOCK_WIRE_SCALE = 1000.0
#: Derivatives (and indices) quote index points on BOTH sides of the boundary.
DERIVATIVE_WIRE_SCALE = 1.0

#: Tick size of a DERIVATIVE, in index points (VN30F* = 0.1).
DERIVATIVE_TICK = 0.1

#: HOSE price ladder, in ĐỒNG: ``(exclusive upper bound, tick)``. Below 10,000 đ
#: the tick is 10 đ, 10,000-49,950 đ it is 50 đ, from 50,000 đ up it is 100 đ.
#: Expressed in đồng because that is the unit the ladder is DEFINED in — the
#: same ladder in thousands (0.01/0.05/0.10) is unrepresentable in the old
#: ``round(price, 1)`` quantizer, which snapped every stock price to 100 đ.
HOSE_LADDER_DONG = ((10_000.0, 10.0), (50_000.0, 50.0), (math.inf, 100.0))


class PriceUnitError(ExchangeOrderRejectedError):
    """Base for the #119 hard guards.

    Subclasses :class:`ExchangeOrderRejectedError` so the engine handles a
    refused write through its ordinary reject path instead of dying on an
    unknown exception type — the write provably never left the process.
    """


class UnverifiedClassificationError(PriceUnitError):
    """G1: the stock x1000 scale was about to be applied to a GUESSED
    classification (see ``DNSEBroker._wire_scale``). Fail loud instead."""


class StockOrdersDisabledError(PriceUnitError):
    """G2: live STOCK order placement is opt-in until a hand-verified real
    stock fill confirms the readback units (see ``enable_stock_orders``)."""


def hose_tick_dong(price_dong: float) -> float:
    """HOSE tick for ``price_dong``, in đồng (10 / 50 / 100 by band)."""
    price = abs(float(price_dong))
    for upper, tick in HOSE_LADDER_DONG:
        if price < upper:
            return tick
    return HOSE_LADDER_DONG[-1][1]


def snap_to_step(value: float, step: float) -> float:
    """``value`` snapped to the nearest multiple of ``step`` (half away from 0).

    ``round()`` is banker's rounding and would send 21,325 đ down to 21,300
    while 21,375 goes up — the venue's ladder is a plain half-up grid. The
    epsilon absorbs the binary-float wobble of ``x / step`` (21310 / 50 is not
    exactly 426.2 in IEEE754).
    """
    if step <= 0:
        return float(value)
    sign = -1.0 if value < 0 else 1.0
    return sign * math.floor(abs(value) / step + 0.5 + 1e-9) * step


def to_wire(price: float, scale: float) -> float:
    """FEED price -> venue WIRE price (unquantized)."""
    return float(price) * scale


def from_wire(price: float, scale: float) -> float:
    """Venue WIRE price -> FEED price (the unit the engine/strategy speaks)."""
    return float(price) / scale


def quantize_wire(price_wire: float, scale: float) -> float:
    """Snap a WIRE price onto the venue's tick grid.

    STOCK: the HOSE ladder, in đồng. Anything else: the pre-#119 behaviour,
    ``round(price, 1)`` — byte-identical for derivatives (the control the #119
    baseline test pins), whose 0.1-point tick that expression already encodes.
    """
    if scale == STOCK_WIRE_SCALE:
        return snap_to_step(float(price_wire), hose_tick_dong(price_wire))
    return round(float(price_wire), 1)
