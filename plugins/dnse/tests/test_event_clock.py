"""#101 — the event clock's offline harness gate (mandatory pre-live).

Pins the three panel guards against the fake-venue seam: consumer
handshake (no free-run into the unbounded producer queue), positive-
evidence settlement with a LOUD timeout (a park never drifts into
idle-synth), and the wall-clock-seeded strictly-monotone bar grid (a
repeated grid would replay deterministic coids and REOPEN a prior run's
journal rows — store_helpers reopen-on-coid semantics).
"""
import asyncio

import pytest
import pynecore.lib as lib

lib.bar_index = 0

from pynecore_dnse.event_clock import EventClockBroker
from pynecore_dnse import broker as broker_mod
from pynecore.core.broker.models import EntryIntent, DispatchEnvelope, OrderType

_SECDEF_ROW = [{"ceilingPrice": "2100", "floorPrice": "1800", "securityGroupId": "FU"}]
_LOAN_OK = (200, {"loanPackages": [{"id": 42}]})
_OHLC_1M = (200, {"t": [1788921780], "o": [1970.0], "h": [1971.0],
                  "l": [1969.0], "c": [1970.5], "v": [100]})


def _ec(fake_client, tmp_path, **client_responses):
    responses = {"get_security_definition": (200, _SECDEF_ROW),
                 "get_loan_packages": _LOAN_OK, "get_ohlc": _OHLC_1M}
    responses.update(client_responses)
    config = broker_mod.DNSEBrokerConfig(
        api_key="k", api_secret="s", account_no="ACC001", trading_token="tok-A",
        token_file=str(tmp_path / "missing_token.json"))
    instance = EventClockBroker(symbol="VN30F1M", timeframe="15S",
                                config=config)
    instance._client = fake_client(**responses)
    instance._EC_IDLE_GRACE_S = 0.05
    instance._EC_SETTLE_TIMEOUT_S = 0.5
    instance._EC_POLL_S = 0.01
    return instance


def _envelope(pine_id="E", price=1900.0):
    return DispatchEnvelope(
        intent=EntryIntent(pine_id=pine_id, symbol="VN30F1M", side="buy",
                           qty=1, order_type=OrderType.LIMIT, limit=price),
        run_tag="abcd", bar_ts_ms=1_700_000_000_000, retry_seq=0,
        coid_max_len=30)


def __test_grid_is_wall_seeded_monotone_and_flat_priced__(fake_client,
                                                          tmp_path):
    """Guard 3: the first bar seeds near wall-clock ms; each emission moves
    +tf; prices are flat at the last real venue close. Catches a constant
    seed (coid replay -> journal-row reopen) and a made-up price."""
    import time as _time
    b = _ec(fake_client, tmp_path)
    before = int(_time.time() * 1000)
    bar1 = asyncio.run(b.watch_ohlcv("VN30F1M", "15S"))
    assert abs(bar1.timestamp - before) < 5_000, "grid must be wall-seeded"
    assert bar1.open == bar1.close == 1970.5, "flat at the last real close"
    bar2 = asyncio.run(b.watch_ohlcv("VN30F1M", "15S"))  # idle-grace advance
    assert bar2.timestamp == bar1.timestamp + 15_000, "strictly +tf"


def __test_no_free_run_while_a_dispatch_is_unsettled__(fake_client,
                                                       tmp_path):
    """Guard 1+2: after a dispatch BEGAN following an emission, the next
    bar is withheld until the new venue id is POSITIVELY observed
    (_last_seen) — and a never-settling id ABORTS loudly instead of
    hanging or advancing. Catches the venue-only vacuous predicate (which
    would free-run the ladder into the unbounded queue) and any silent
    drift."""
    b = _ec(fake_client, tmp_path,
            post_order=(201, {"id": "900001", "symbol": "VN30F1M",
                              "side": "NB", "quantity": 1,
                              "orderStatus": "New"}))
    asyncio.run(b.watch_ohlcv("VN30F1M", "15S"))          # bar 1 out
    asyncio.run(b.execute_entry(_envelope()))              # dispatch begins+ends
    # the new id 900001 is tracked but NOT yet observed via _last_seen:
    with pytest.raises(RuntimeError, match="aborting LOUDLY"):
        asyncio.run(b.watch_ohlcv("VN30F1M", "15S"))
    # once the poll ladder observes it, the bar releases immediately:
    b._last_seen["900001"] = (0.0, "New")
    b._ec_emit_wall = 0.0                                  # reset the window
    bar = asyncio.run(b.watch_ohlcv("VN30F1M", "15S"))
    assert bar.is_closed


def __test_observation_only_state_advances_after_idle_grace__(fake_client,
                                                              tmp_path):
    """A bar consumed WITHOUT any dispatch (observe-only states, DONE)
    advances after the idle grace — the ladder never wedges on a quiet
    state. Catches a handshake that requires a dispatch unconditionally."""
    b = _ec(fake_client, tmp_path)
    bar1 = asyncio.run(b.watch_ohlcv("VN30F1M", "15S"))
    bar2 = asyncio.run(b.watch_ohlcv("VN30F1M", "15S"))
    assert bar2.timestamp == bar1.timestamp + 15_000


def __test_production_broker_is_structurally_untouched__():
    """Default-OFF condition: the event clock exists only under its own
    entry point; DNSEBroker carries none of the clock state. Catches a
    config flag leaking into the live class."""
    from pynecore_dnse.broker import DNSEBroker
    assert not hasattr(DNSEBroker, "_EC_SETTLE_TIMEOUT_S")
    assert "event" not in (getattr(DNSEBroker, "plugin_name", "") or "").lower()
    from importlib.metadata import entry_points
    eps = {e.name: e.value for e in entry_points(group="pyne.plugin")}
    assert eps.get("dnse_event") == "pynecore_dnse.event_clock:EventClockBroker"
    assert eps.get("dnse_broker") == "pynecore_dnse.broker:DNSEBroker"
