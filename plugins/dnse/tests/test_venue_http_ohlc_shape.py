"""#157 second pass — the fake's ``/price/ohlc`` must answer in the VENUE's shape. Tests FIRST.

MEASURED 2026-09-18, twice, from opposite directions. The real endpoint answers TradingView-UDF
parallel arrays — the keys are exactly ``c, h, l, nextTime, o, t, v``, confirmed against
production while building the 1m downloader. The fake answered ``{"data": [ {timestamp, open,
...} ]}``. Nothing in the normal fake path noticed, because ``FakeVenueBroker`` overrides both
``download_ohlcv`` and ``watch_ohlcv``, so the array-parsing readers were never reached through
that door. They were reached through another door — the direct-client scripts — and there the
shape broke two readers in two different ways.

**These tests drive the READERS, not the payload.** An assertion on the payload's keys is exactly
what failed to catch this: the payload was internally consistent and self-explanatory, and every
test that looked at it agreed with it. What nobody did was hand it to the code that consumes it.
So each test below calls a real consumer and asserts on the ANSWER it produces:

* ``reference_close`` (the loud reader) — it raised, and took the L0 gate and T33 down with it;
* ``_stop_already_crossed`` (the silent reader) — it FAILED OPEN and answered False, which turns
  crossed-stop detection off with no error anywhere.

The silent one is the reason this file exists. A fake that makes a safety check answer "no
problem" is worse than a fake that crashes.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "testing"))
from venue_core import FakeVenue                                       # noqa: E402
from venue_http import VenueHTTP                                       # noqa: E402

MINUTE_MS = 60_000
SESSION_OPEN_MS = 1_789_524_000_000


def _bars(count=5, start=SESSION_OPEN_MS):
    return [{"timestamp": start + index * MINUTE_MS,
             "open": 1980.0 + index, "high": 1985.0 + index, "low": 1979.0 + index,
             "close": 1984.0 + index, "volume": 100.0 + index}
            for index in range(count)]


@pytest.fixture
def served():
    """A running fake venue, torn down after the test."""
    bars = _bars()
    venue = FakeVenue(symbol="41I1GA000", market_type="DERIVATIVE", last_price=1980.0, seed=7)
    server = VenueHTTP(venue, contract="VN30F1M", bars=bars,
                       final_trade_date="2026-12-17", band=(2118.6, 1841.4)).start()
    try:
        yield server, bars
    finally:
        server.stop()


def _client(server):
    from pynecore_dnse.client import DNSEClient
    return DNSEClient("fake-venue-key", "fake-venue-secret", base_url=server.base_url)


# --------------------------------------------------------------------------- the loud reader

def __test_the_l0_gates_reference_close_reads_a_close_from_the_fake__(served):
    """The reader that broke loudly. MEASURED against the fake on 2026-09-18:

        RuntimeError: cannot read a reference close from /price/ohlc (symbol=VN30F1M)
        at l0_order_semantics.py:193

    It took the L0 gate and the T33 closed-hours test down with it, before either placed
    anything. This drives that exact function rather than re-implementing its logic, because a
    re-implementation would agree with whatever the fake happened to serve.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                           / "testing" / "live_test" / "level0_venue_semantics"))
    from l0_order_semantics import reference_close

    server, bars = served

    class _Broker:
        symbol = "VN30F1M"
        market_type = "DERIVATIVE"

        def __init__(self, client):
            self.client = client

    close, label = reference_close(_Broker(_client(server)))

    assert close == pytest.approx(bars[-1]["close"]), "the reference must be the LAST close served"
    assert label


# --------------------------------------------------------------------------- the silent reader

def __test_a_stop_below_the_market_is_seen_as_already_crossed__(served):
    """The reader that broke SILENTLY, and the reason this file exists.

    ``_stop_already_crossed`` reads ``body.get("c")``. Against the dict-shaped payload it found
    nothing and returned False — fail-open — so a stop the market had already passed was reported
    as not crossed. No exception, no log line, a plausible record, and the branch under test
    never runs. This asserts on the ANSWER, so a payload the reader cannot parse fails here.
    """
    from pynecore_dnse.broker import DNSEBroker

    server, bars = served
    last_close = bars[-1]["close"]

    # ``market_type`` is a read-only PROPERTY derived from the venue's own secdef, so it is left
    # alone and the symbol is the alias, which classifies DERIVATIVE either way. Assigning it
    # raised at fixture setup on the first draft of this test, and a test that goes red before
    # it reaches its assertion has proven nothing about the thing it names.
    broker = DNSEBroker.__new__(DNSEBroker)
    broker.symbol = "VN30F1M"
    broker._client = _client(server)

    # A BUY stop BELOW the last traded price has already been passed by the market.
    assert broker._stop_already_crossed("buy", last_close - 10.0) is True


def __test_a_stop_the_market_has_not_reached_is_not_reported_as_crossed__(served):
    """The discriminating half. Without it, a reader that answered True unconditionally would
    satisfy the test above and turn every resting stop into an immediate market order."""
    from pynecore_dnse.broker import DNSEBroker

    server, bars = served
    last_close = bars[-1]["close"]

    broker = DNSEBroker.__new__(DNSEBroker)
    broker.symbol = "VN30F1M"
    broker._client = _client(server)

    assert broker._stop_already_crossed("buy", last_close + 10.0) is False


# --------------------------------------------------------------------------- the shape itself

def __test_the_endpoint_answers_the_venues_parallel_arrays__(served):
    """Kept LAST and deliberately narrow. It pins the shape against the production keys measured
    on 2026-09-18, but it is the weakest test here: the previous shape was internally consistent
    too. The reader tests above are what actually catch a regression.
    """
    server, bars = served

    status, body = _client(server).get_ohlc("DERIVATIVE", {
        "symbol": "VN30F1M", "resolution": "1",
        "from": bars[0]["timestamp"] // 1000, "to": bars[-1]["timestamp"] // 1000})

    assert status == 200
    for key in ("t", "o", "h", "l", "c", "v"):
        assert key in body, f"production serves {key!r}; the fake must too"
        assert len(body[key]) == len(bars)
    assert body["t"][0] == bars[0]["timestamp"] // 1000, "the venue answers SECONDS, not ms"
    assert body["c"][-1] == bars[-1]["close"]


def __test_the_history_read_is_still_clamped_to_the_replay_cursor__(served):
    """The clamp must survive the reshape. A history endpoint that serves bars the replay has not
    reached is serving the FUTURE as history: warmup then eats the whole day and the live stream
    has nothing left, which is the failure that cost a day before the re-stamping work."""
    server, bars = served
    server.catalogue["replay_cursor"] = bars[1]["timestamp"]

    _, body = _client(server).get_ohlc("DERIVATIVE", {
        "symbol": "VN30F1M", "resolution": "1", "from": 0, "to": 9_999_999_999})

    assert len(body["t"]) == 2, "bars past the cursor must not be served as history"
    assert body["t"][-1] == bars[1]["timestamp"] // 1000
