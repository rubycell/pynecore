"""#100 phase 2a — the sub-minute LTF feed path, offline.

Pins the panel-adjudicated contract: WS-only synthesis (closed bars via
the shipped TickAggregator), outage RAISES instead of hanging (a hanging
feed makes the runner fabricate frozen filler bars), quiet-phase silence
is normal, the LTF store is disjoint from every provider truncation path,
and sub-minute never enters the venue timeframe map.
"""
import asyncio

import pytest
import pynecore.lib as lib

lib.bar_index = 0

from pynecore_dnse import broker
from pynecore.core.broker.exceptions import ExchangeConnectionError
from pynecore.core.tick_aggregator import TickAggregator

_SECDEF_ROW = [{"ceilingPrice": "2100", "floorPrice": "1800", "securityGroupId": "FU"}]
_LOAN_OK = (200, {"loanPackages": [{"id": 42}]})


def _broker(fake_client, tmp_path, **client_responses):
    responses = {"get_security_definition": (200, _SECDEF_ROW),
                 "get_loan_packages": _LOAN_OK}
    responses.update(client_responses)
    config = broker.DNSEBrokerConfig(
        api_key="k", api_secret="s", account_no="ACC001", trading_token="tok-A",
        token_file=str(tmp_path / "missing_token.json"))
    instance = broker.DNSEBroker(symbol="VN30F1M", timeframe="15S", config=config)
    instance._client = fake_client(**responses)
    return instance


class _FakeSource:
    """Scripted tick source: a list of ('tick', (ts,p,q,cum)) or
    ('silence', None) events; silence raises TimeoutError like the real
    ``next_tick`` on a quiet wire."""
    def __init__(self, events):
        self.events = list(events)
        self.overflowed = False

    async def next_tick(self, timeout):
        if not self.events:
            raise asyncio.TimeoutError
        kind, payload = self.events.pop(0)
        if kind == "silence":
            raise asyncio.TimeoutError
        return payload


def _seed_ltf(b, events, tf_seconds=15):
    b._ltf_state = {"source": _FakeSource(events),
                    "agg": TickAggregator(tf_seconds), "pending": []}


def __test_ltf_feed_yields_closed_bars_from_ticks__(fake_client, tmp_path,
                                                    monkeypatch):
    """Ticks in → the engine's watch_ohlcv contract out (closed 15S bar,
    epoch-aligned, ms timestamp). Catches a feed that returns forming
    bars or mis-scales the timestamp."""
    monkeypatch.chdir(tmp_path)
    b = _broker(fake_client, tmp_path)
    _seed_ltf(b, [("tick", (1000.0, 100.0, 1, None)),
                  ("tick", (1002.0, 101.0, 2, None)),
                  ("tick", (1006.0, 100.5, 1, None))])   # crosses 1005

    bar = asyncio.run(b.watch_ohlcv("VN30F1M", "15S"))

    assert bar.is_closed and bar.timestamp == 990 * 1000
    assert (bar.open, bar.high, bar.low, bar.close, bar.volume) == \
        (100.0, 101.0, 100.0, 101.0, 3.0)


def __test_ltf_outage_raises_instead_of_hanging__(fake_client, tmp_path,
                                                  monkeypatch):
    """The panel's idle-synth guard: silence past grace OUTSIDE a quiet
    phase must raise ExchangeConnectionError (the runner's retry path) —
    a hanging feed lets the core fabricate frozen zero-volume bars.
    Catches a wait-forever loop."""
    monkeypatch.chdir(tmp_path)
    b = _broker(fake_client, tmp_path)
    monkeypatch.setattr(b, "_in_feed_quiet_phase", lambda: False)
    _seed_ltf(b, [("silence", None)])

    with pytest.raises(ExchangeConnectionError, match="refusing to fabricate"):
        asyncio.run(b.watch_ohlcv("VN30F1M", "15S"))


def __test_ltf_quiet_phase_silence_is_not_an_outage__(fake_client, tmp_path,
                                                      monkeypatch):
    """ATC-class silence: the feed keeps waiting through a quiet phase and
    delivers the next real bar. Catches treating the ATC as an outage
    (a daily false ExchangeConnectionError at 14:30)."""
    monkeypatch.chdir(tmp_path)
    b = _broker(fake_client, tmp_path)
    quiet = {"on": True}
    monkeypatch.setattr(b, "_in_feed_quiet_phase", lambda: quiet["on"])
    _seed_ltf(b, [("silence", None), ("silence", None),
                  ("tick", (2000.0, 100.0, 1, None)),
                  ("tick", (2011.0, 100.5, 1, None))])    # crosses 2010

    bar = asyncio.run(b.watch_ohlcv("VN30F1M", "15S"))
    assert bar.timestamp == 1995 * 1000


def __test_ltf_bars_persist_to_the_separate_store__(fake_client, tmp_path,
                                                    monkeypatch):
    """The truncation trap (seats 1+3): LTF history lives under
    workdir/data/ltf/ — a path no provider download target ever touches
    (their targets live flat in workdir/data/). Catches persisting into
    the shared provider cache."""
    monkeypatch.chdir(tmp_path)
    b = _broker(fake_client, tmp_path)
    _seed_ltf(b, [("tick", (1000.0, 100.0, 1, None)),
                  ("tick", (1006.0, 100.5, 1, None))])

    asyncio.run(b.watch_ohlcv("VN30F1M", "15S"))

    store = tmp_path / "workdir" / "output" / "ltf" / "dnsebroker_VN30F1M_15S.ohlcv"
    alt = tmp_path / "workdir" / "data" / "ltf" / "dnsebroker_VN30F1M_15S.ohlcv"
    assert store.exists() or alt.exists(), "LTF bar was not persisted"
    shared = [p for p in (tmp_path / "workdir").rglob("*.ohlcv")
              if p.parent.name != "ltf"]
    assert not shared, (
        f"LTF history leaked into a provider-truncatable path: {shared}")


def __test_sub_minute_never_enters_the_venue_map__():
    """Seat 3's trap: the download path consumes the timeframe map
    verbatim — '15S' there would fetch 15-MINUTE bars into a file named
    15S. The predicate routes; the map keeps raising."""
    from pynecore_dnse.provider import DNSEProvider
    assert DNSEProvider.is_sub_minute("15S")
    assert DNSEProvider.is_sub_minute("5S")
    assert not DNSEProvider.is_sub_minute("1")
    assert not DNSEProvider.is_sub_minute("1D")
    with pytest.raises(ValueError, match="no resolution"):
        DNSEProvider.to_exchange_timeframe("15S")


def __test_quiet_phase_clock_both_directions__(fake_client, tmp_path,
                                               monkeypatch):
    """The declared phase covers the measured ATC and nothing else
    (discriminating both ways: 14:35 in, 10:00 and 14:45 out)."""
    import pynecore_dnse.broker as broker_mod

    class _FakeDT:
        @staticmethod
        def now(tz=None):
            return _now

    from datetime import datetime as real_dt, timezone, timedelta
    tz7 = timezone(timedelta(hours=7))
    b = _broker(fake_client, tmp_path)
    monkeypatch.setattr(broker_mod, "datetime", _FakeDT)
    global _now
    _now = real_dt(2026, 9, 9, 14, 35, tzinfo=tz7)
    assert b._in_feed_quiet_phase() is True
    _now = real_dt(2026, 9, 9, 10, 0, tzinfo=tz7)
    assert b._in_feed_quiet_phase() is False
    _now = real_dt(2026, 9, 9, 14, 45, tzinfo=tz7)
    assert b._in_feed_quiet_phase() is False, "14:45 = settlement, phase over"


def __test_minute_and_above_keep_the_venue_candle_paths__(fake_client,
                                                          tmp_path,
                                                          monkeypatch):
    """Coexistence pin (operator requirement 2026-09-09): BOTH bar methods
    stay available — venue-fetched candles for >=1m (the parity-proven
    paths, byte-identical), synthesis ONLY for sub-minute. Catches a
    dispatch that routes minute timeframes into the LTF path (which would
    replace venue truth with synthesis) or vice versa."""
    calls = []

    async def _fake_closed(self, symbol, timeframe):
        calls.append(("closed", timeframe)); return "closed-bar"

    async def _fake_tick(self, symbol, timeframe):
        calls.append(("tick", timeframe)); return "tick-bar"

    async def _fake_ltf(self, symbol, timeframe):
        calls.append(("ltf", timeframe)); return "ltf-bar"

    monkeypatch.setattr(broker.DNSEBroker, "_watch_ohlcv_closed", _fake_closed)
    monkeypatch.setattr(broker.DNSEBroker, "_watch_ohlcv_tick", _fake_tick)
    monkeypatch.setattr(broker.DNSEBroker, "_watch_ohlcv_ltf", _fake_ltf)

    b = _broker(fake_client, tmp_path)
    b._feed_mode = "ohlc"
    assert asyncio.run(b.watch_ohlcv("VN30F1M", "1")) == "closed-bar"
    assert asyncio.run(b.watch_ohlcv("VN30F1M", "5")) == "closed-bar"
    b._feed_mode = "tick"
    assert asyncio.run(b.watch_ohlcv("VN30F1M", "1")) == "tick-bar"
    # sub-minute routes to synthesis REGARDLESS of feed_mode — the venue
    # has no candles there; feed_mode only selects among venue-bar paths.
    assert asyncio.run(b.watch_ohlcv("VN30F1M", "15S")) == "ltf-bar"
    b._feed_mode = "ohlc"
    assert asyncio.run(b.watch_ohlcv("VN30F1M", "15S")) == "ltf-bar"
    assert [c[0] for c in calls] == ["closed", "closed", "tick", "ltf", "ltf"]
