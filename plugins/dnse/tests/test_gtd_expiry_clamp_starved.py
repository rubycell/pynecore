"""RED baseline for #118 — the GTD clamp starves when the venue drops ``finalTradeDate``.

Measured live on prod 2026-09-14: ``get_security_definition("41I1G9000")`` (the dated
VN30F1M contract) answers with **no** ``finalTradeDate`` at all — the field was
populated (``2026-08-20``) on 2026-08-14 and is now ``None``; ``/market/instruments``
carries no expiry either, so the plugin has no venue-served expiry left.

``broker._gtd`` (``broker.py:870``) then takes its documented FAIL-OPEN branch and
returns the plain ``now + 7 days``. On 2026-09-14 that is **2026-09-21**, which is past
the contract's real final trade date (**Thu 2026-09-17**, the 3rd Thursday — 2026-08-20
was also a 3rd Thursday).  DNSE refuses every conditional whose GTD reaches past the
final trade date with ``CO-ORD-006 "Validate Order Failed"``, so for the whole of expiry
week NO native STOP/OCO can be placed — protective stop-losses included.

These tests pin the behaviour the plugin MUST have: with no venue-served expiry, the GTD
it puts on the wire must still not reach past the contract's real final trade date.
They FAIL today (the fail-open +7d wins) and are the RED baseline for #118.

Same fake-client seam as ``test_broker_orders.py`` — no live network, no real files.
Time is frozen by swapping ``broker.datetime`` for a subclass whose ``now()`` is fixed,
so the verdict cannot drift with the wall clock (a real-``now`` test would go green on
its own once the contract rolls, silently un-pinning the bug).
"""
import asyncio
from datetime import datetime, timezone

import pytest

import pynecore.lib as lib

lib.bar_index = 0  # let the [BROKER] log formatter render during broker._emit()

from pynecore_dnse import broker
from pynecore.core.broker.models import ExitIntent, DispatchEnvelope, OrderType, EntryIntent

#: The day the regression was measured on prod. A Monday inside expiry week.
FROZEN_NOW = datetime(2026, 9, 14, 3, 0, 0, tzinfo=timezone.utc)

#: The contract's REAL final trade date: 3rd Thursday of September 2026.
#: (2026-08-20, the value the venue served on 2026-08-14, was also a 3rd Thursday.)
#: The usable CEILING is the end of that date's CONTINUOUS session —
#: 14:30 ICT = 07:30Z, the ATC start — not the 14:45 close and not midnight
#: UTC. Measured 2026-09-17: 07:45Z was REFUSED four times on the expiring
#: contract while the operator's app order with a 14:30 expiry was accepted
#: and rested. Midnight UTC is 07:00 ICT *on* the final date, before the
#: session opens, so it is already past for any in-session order (#118).
FINAL_TRADE_DATE = datetime(2026, 9, 17, 7, 30, 0, tzinfo=timezone.utc)

#: The dated VN30F1M contract for September 2026 (measured live 2026-09-14).
FRONT_CONTRACT = "41I1G9000"

#: The secdef row the venue actually returns today: bands + group, and NO
#: ``finalTradeDate`` key — this absence IS the regression under test.
_SECDEF_NO_EXPIRY = [{"ceilingPrice": "1550", "floorPrice": "1450",
                      "securityGroupId": "FU", "symbol": FRONT_CONTRACT,
                      "listingDate": "2026-03-19"}]

#: ``/market/instruments`` as measured — alias -> dated code, and no expiry field.
_INSTRUMENTS = (200, {"data": [
    {"symbol": FRONT_CONTRACT, "symbolType": "VN30F1M", "marketId": "DVX",
     "securityGroupId": "FU", "listedDate": ""},
    {"symbol": "41I1GA000", "symbolType": "VN30F2M", "marketId": "DVX",
     "securityGroupId": "FU", "listedDate": ""},
]})

_LOAN_OK = (200, {"loanPackages": [{"id": 42}]})


class _FrozenDatetime(datetime):
    """``datetime`` with a fixed ``now()`` — everything else (``strptime``) intact."""

    @classmethod
    def now(cls, tz=None):                                   # noqa: D102
        return FROZEN_NOW if tz is None else FROZEN_NOW.astimezone(tz)


@pytest.fixture
def frozen_clock(monkeypatch):
    """Freeze ``broker.datetime.now`` at :data:`FROZEN_NOW`."""
    monkeypatch.setattr(broker, "datetime", _FrozenDatetime)
    return FROZEN_NOW


def _broker_without_venue_expiry(fake_client, tmp_path, **client_responses):
    """A ``DNSEBroker`` whose venue serves NO ``finalTradeDate`` — today's prod shape."""
    responses = {
        "get_security_definition": (200, _SECDEF_NO_EXPIRY),
        "get_instruments": _INSTRUMENTS,
        "get_loan_packages": _LOAN_OK,
    }
    responses.update(client_responses)
    config = broker.DNSEBrokerConfig(
        api_key="k", api_secret="s", account_no="ACC001", trading_token="tok-A",
        token_file=str(tmp_path / "missing_token.json"))
    b = broker.DNSEBroker(symbol="VN30F1M", timeframe="15", config=config)
    b._client = fake_client(**responses)
    return b


def _envelope(intent, *, run_tag="abcd", bar_ts_ms=1_700_000_000_000):
    return DispatchEnvelope(intent=intent, run_tag=run_tag, bar_ts_ms=bar_ts_ms)


def _posted_payload(client):
    matches = [c for c in client.calls if c[0] == "post_order"]
    assert matches, "post_order was never called"
    return matches[-1][1][2]


def _parse_gtd(raw: str) -> datetime:
    return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


# --- the wire-level pin (what DNSE actually judges) ---------------------------

def __test_protective_stop_gtd_does_not_reach_past_the_final_trade_date__(
        fake_client, tmp_path, frozen_clock):
    """A protective STOP placed in expiry week must carry a placeable GTD.

    This is the exact live failure: the entry fills, the engine arms the protective
    stop, and DNSE refuses the conditional with ``CO-ORD-006`` because the GTD
    (``now + 7d`` = 2026-09-21) reaches past the 2026-09-17 final trade date.
    """
    b = _broker_without_venue_expiry(fake_client, tmp_path, post_order=(
        201, {"id": "5", "symbol": FRONT_CONTRACT, "side": "NB", "quantity": 2,
              "orderStatus": "New"}))
    envelope = _envelope(ExitIntent(pine_id="X", from_entry="L", symbol="VN30F1M",
                                    side="sell", qty=2, sl_price=1480.0))

    asyncio.run(b.execute_exit(envelope))

    payload = _posted_payload(b._client)
    assert payload["durationType"] == "GTD", "a native STOP is placed GTD"
    gtd = _parse_gtd(payload["durationDateTime"])
    assert gtd <= FINAL_TRADE_DATE, (
        f"GTD {gtd:%Y-%m-%d} reaches past the contract's final trade date "
        f"{FINAL_TRADE_DATE:%Y-%m-%d} -> DNSE refuses the protective stop with "
        f"CO-ORD-006 and the position is left unprotected (#118)")


def __test_stop_entry_gtd_does_not_reach_past_the_final_trade_date__(
        fake_client, tmp_path, frozen_clock):
    """The same starved GTD rides on conditional ENTRIES, so nothing conditional places."""
    b = _broker_without_venue_expiry(fake_client, tmp_path, post_order=(
        201, {"id": "1", "symbol": FRONT_CONTRACT, "side": "NB", "quantity": 1,
              "orderStatus": "New"}))
    envelope = _envelope(EntryIntent(pine_id="L", symbol="VN30F1M", side="buy", qty=1,
                                     order_type=OrderType.STOP, stop=1500.0))

    asyncio.run(b.execute_entry(envelope))

    gtd = _parse_gtd(_posted_payload(b._client)["durationDateTime"])
    assert gtd <= FINAL_TRADE_DATE, (
        f"GTD {gtd:%Y-%m-%d} past {FINAL_TRADE_DATE:%Y-%m-%d} -> CO-ORD-006 (#118)")


# --- the helper-level pin (same bug, one call deep) ---------------------------

def __test_gtd_helper_is_bounded_when_the_venue_serves_no_expiry__(
        fake_client, tmp_path, frozen_clock):
    """``_gtd`` itself must not fail OPEN into an unplaceable date.

    ``broker.py:884`` documents the fallback as "a missing field cannot make the plugin
    unable to place anything at all" — but in expiry week the plain +7d window produces
    exactly that, silently. Any acceptable fix (computed 3rd Thursday, cached
    last-known-good, or refusing loudly) satisfies this assertion; only the current
    fail-open does not.
    """
    b = _broker_without_venue_expiry(fake_client, tmp_path)

    gtd = _parse_gtd(b._gtd(days=7))

    assert gtd <= FINAL_TRADE_DATE, (
        f"_gtd fell open to {gtd:%Y-%m-%d}, past the real final trade date "
        f"{FINAL_TRADE_DATE:%Y-%m-%d} (#118)")
    assert gtd > FROZEN_NOW, "a GTD in the past would be refused too"
