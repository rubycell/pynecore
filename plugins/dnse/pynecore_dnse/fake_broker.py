"""TESTING ONLY (#157): the offline fake-venue broker — opt-in by NAME.

DATA comes from a recorded or synthetic venue day; ORDERS go to a local fake venue served over
loopback HTTP by :mod:`venue_http`, which is one adapter over the single state machine in
:mod:`venue_core`. Production ``dnse_broker`` runs are structurally untouched: this is a separate
class reached only via ``dnse_fake:...``, exactly like ``dnse_event`` and ``dnse_replay_sandbox``,
and it reads its OWN config file so the live one is never involved.

What this buys over the Sandbox Replay E2E (#114): conditional orders. The sandbox rejects the
STOP and OCO categories outright and has no price simulation, so activation, the normal-book
child, partial fills by traded volume and the measured refusal codes could only ever be exercised
against production. Here they run offline and deterministically.

The bars and the order book advance TOGETHER. Each bar's prints are fed into the venue before the
bar is handed to the engine, so a conditional triggers on the same trade the market printed —
which is the whole reason the venue is driven by prints rather than by a clock.
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

from pynecore.core.plugin import override
from pynecore.types.ohlcv import OHLCV

from .broker import DNSEBroker
from .config import DNSEBrokerConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "testing"))
from venue_core import FakeVenue                                      # noqa: E402
from venue_day import load_day                                        # noqa: E402
from venue_http import VenueHTTP                                      # noqa: E402

#: Environment pin naming the day to replay. No default: a run must say which day it replayed,
#: and a silently chosen fixture is how a result gets attributed to the wrong data.
_DAY_ENV = "FAKE_VENUE_DAY"


class FakeVenueConfig(DNSEBrokerConfig):
    """The fake's OWN config class (R6).

    Not cosmetic. ``ensure_config`` caches on ``config_cls._ensured`` and tests for it with
    ``hasattr``, which follows inheritance (#165), so whichever class in a hierarchy is ensured
    FIRST answers for every relative afterwards. With the fake sharing ``DNSEBrokerConfig``, a
    machine whose first ``pyne`` command touched the fake could hand a later LIVE ``dnse_broker``
    run the fake's defaults — loopback endpoints and a nonsense token. A distinct leaf class
    cannot be reached from ``DNSEBrokerConfig``, so the poisoning has no path to production.
    """


class FakeVenueBroker(DNSEBroker):
    """Replays a venue day while routing orders to a local fake venue."""

    Config = FakeVenueConfig

    #: Disable the engine's feed-staleness watchdog FOR THE FAKE ONLY.
    #:
    #: live_runner.py:794-798 computes feed_stale_after from this attribute and leaves it None
    #: when the attribute is falsy; the idle-bar synthesiser at :1813 is gated on
    #: ``feed_stale_after is not None and time.time() - last_real_update >= feed_stale_after``.
    #: That clock is the WALL clock, so a replayed past day can never arrive "on time": measured
    #: 2026-09-18, the first synthetic bar appeared on the SAME bar as the first live bar,
    #: because warmup itself burns wall seconds while the staleness clock runs. From that moment
    #: the engine preferred its own flat synthetic bars and stopped consuming the fake's, so no
    #: print ever reached a resting order and nothing could fill.
    #:
    #: The watchdog exists to catch a half-open socket on a LIVE feed. A replay cannot go
    #: half-open — the stream is a list — so disabling it here removes a guard that has nothing
    #: to guard, and it changes nothing for production: DNSEProvider keeps its 17 (provider.py:179).
    feed_timeout_bars = None

    _bars: list[OHLCV]
    _idx: int

    # ----------------------------------------------------------------- wiring

    def _repair_config_if_degraded(self) -> None:
        """Rebuild this run's config when the framework handed us the PARENT class's instance.

        Framework defect, measured 2026-09-18 and filed separately. ``core/config.py``'s
        ``ensure_config`` caches on ``config_cls._ensured`` and tests for it with ``hasattr``,
        which follows inheritance. So once ``DNSEConfig`` has been ensured, every SUBCLASS —
        including ``DNSEBrokerConfig`` — silently receives the parent's instance, missing every
        field the subclass adds. The symptom is an ``AttributeError`` on ``config.account_no``
        deep inside the broker contract check, which reads as a broker bug and is not one.

        This repairs only THIS run's object, from THIS broker's own toml. It does not touch the
        framework, which is out of scope here; the one-line fix (read ``__dict__`` rather than
        ``hasattr``) is proposed on the card instead.
        """
        # Two degradations, not one. The class can be wrong (the inherited-cache defect), and
        # the CONTENT can be wrong: measured 2026-09-18, a run arrived with a correctly typed
        # DNSEBrokerConfig carrying empty credentials, meaning the framework had resolved a
        # different file than this broker's own. Either way the run cannot proceed, and either
        # way the fix is the same: rebuild from the tracked example, which nothing rewrites.
        if isinstance(self.config, FakeVenueConfig) and getattr(self.config, "api_key", ""):
            return

        import dataclasses
        import tomllib

        # Read the TRACKED example, not the workdir copy. Measured 2026-09-18: the same
        # framework defect does not merely return the parent's instance, it REWRITES the config
        # file to the parent's schema, deleting every key the subclass adds — account_no,
        # trading_token, token_file and the poll intervals all vanished from the workdir file
        # after one run. So the workdir copy cannot be trusted as the source of truth here.
        source = Path(__file__).resolve().parents[1] / "testing" / "dnse_fake.toml.example"
        if not source.exists():
            # R7: fail LOUD. Proceeding with raw={} would leave the PRODUCTION defaults in
            # place — the real token_file path and the real endpoints — on a run that believes
            # it is driving a fake. Silence here is the one outcome that could route a test to
            # the live venue.
            raise RuntimeError(
                f"{source} is missing: refusing to build a fake-venue config from defaults, "
                f"because those defaults are the PRODUCTION token path and endpoints.")
        with source.open("rb") as handle:
            raw = tomllib.load(handle)

        known = {f.name for f in dataclasses.fields(FakeVenueConfig)}
        carried = {}
        if self.config is not None:                     # keep whatever the parent did load
            for field in dataclasses.fields(type(self.config)):
                if field.name in known:
                    carried[field.name] = getattr(self.config, field.name)
        carried.update({k: v for k, v in raw.items() if k in known})

        self.config = FakeVenueConfig(**carried)

        import logging
        logging.getLogger(__name__).warning(
            "[FAKE VENUE] rebuilt a degraded config: the framework returned %s because "
            "ensure_config's cache is inherited by subclasses; see the card.",
            type(self.config).__name__)

    def _reset_own_bar_store(self) -> None:
        """Park this fake's OWN accumulated bar file so warmup replays only this run's slice.

        Measured 2026-09-18, and it invalidated a whole afternoon of probe results. The bar store
        ACCUMULATES across runs: a run that saved 399 warmup bars then streamed 300 live ones
        left all 699 in the file, so the NEXT run warmed up on 704 bars — including every bar
        this run intended to stream live. The staged probe's window therefore opened during
        WARMUP (measured: its first gated bar was 2026-09-16 13:28, while live trading only began
        at 2026-09-17 14:11) and every stage was consumed against the backtest engine, which
        routes no orders. The symptom was "the probe places nothing", and the cause was nothing
        to do with the probe or the venue.

        Only THIS broker's own file is touched: ``fakevenuebroker_*`` is written by nothing else,
        and no real ``dnse_*`` data is involved. It is MOVED, never deleted, per the house rule —
        it is a regenerated artefact, but the rule does not carve out exceptions.
        """
        path = getattr(self, "ohlcv_path", None)
        if path is None:
            return
        store, stem = Path(path).parent, Path(path).stem
        # Anchored to the repo, not the cwd: a run started elsewhere would otherwise
        # scatter parked files into whatever directory it happened to be in.
        parked = Path(__file__).resolve().parents[3] / "backup" / "deleteable"
        stamp = int(datetime.now().timestamp())
        for existing in sorted(store.glob(f"{stem}.*")) + [Path(path)]:
            if existing.exists() and existing.name.startswith(stem):
                parked.mkdir(parents=True, exist_ok=True)
                existing.replace(parked / f"{existing.name}.{stamp}")
        # The writer was built in __init__ against the file just moved away, so it is REBUILT
        # rather than cleared: the provider's context manager asserts it is not None, and
        # nulling it turns this reset into an AssertionError at teardown (measured).
        from pynecore.core.ohlcv import OHLCVWriter
        from pynecore.lib.timeframe import _process_tf
        modifier, multiplier = _process_tf(self.timeframe or "1")
        period = f"{multiplier}{modifier}" if modifier else str(multiplier)
        self.ohlcv_file = OHLCVWriter(Path(path), period)
        # The provider opens its writer in __enter__, which already ran against the old file, so
        # the replacement must be opened here or every save raises "writer is not open".
        self.ohlcv_file.open()

    def _ensure_venue(self) -> None:
        """Start the fake venue and repoint this run's endpoints at it, once."""
        if getattr(self, "_server", None) is not None:
            return

        self._repair_config_if_degraded()

        path = os.environ.get(_DAY_ENV)
        if not path:
            raise RuntimeError(
                f"{_DAY_ENV} is unset — point it at a venue day (RECORDED or SYNTHETIC). "
                f"There is no default on purpose: a run must state which day it replayed.")

        # Refuse a production endpoint from the CONFIG side, not only the server side. The
        # broker overwrites base_url with its own loopback port moments later, so a production
        # host in the toml would usually be harmless by accident — and "harmless by accident"
        # is not a safety property. Checked before anything is started, so a mis-edited toml
        # stops the run instead of being silently overwritten.
        for label, url in (("base_url", self.config.base_url), ("ws_url", self.config.ws_url)):
            VenueHTTP.assert_not_production(url)
            _ = label

        self._reset_own_bar_store()

        day = load_day(path)
        self._day = day

        # RE-STAMP the replayed day onto the current wall clock, ONE offset for every surface.
        #
        # The engine anchors its live path to the wall clock in TWO independent places
        # (live_runner.py:1813 staleness, :763-775 one synthetic bar per MISSED timeframe
        # boundary at :787), so a day stamped in the past misses a boundary every real minute
        # and the synthesiser wins forever. Measured: 65 synthetic bars even with the staleness
        # watchdog disabled. Shifting the FIRST LIVE bar onto the current boundary makes the
        # stream arrive at or ahead of the clock, so no boundary is ever missed.
        #
        # The day FILE keeps its recorded timestamps; the shift is applied at SERVE time only,
        # so the fixture stays a faithful record and the offset is reported in the log.
        live_count_preview = max(1, int(os.environ.get("FAKE_VENUE_LIVE_BARS", "120")))
        if len(day.bars) <= live_count_preview:
            live_count_preview = max(1, len(day.bars) // 2)
        first_live_ts = int(day.bars[-live_count_preview]["timestamp"])
        tf_ms = 60_000
        now_boundary_ms = (int(datetime.now().timestamp() * 1000) // tf_ms) * tf_ms
        self._offset_ms = now_boundary_ms - first_live_ts

        self._orig_ts = [int(b["timestamp"]) for b in day.bars]
        self._bars = [OHLCV(timestamp=int(b["timestamp"]) + self._offset_ms,
                            open=float(b["open"]),
                            high=float(b["high"]), low=float(b["low"]),
                            close=float(b["close"]), volume=float(b["volume"]))
                      for b in day.bars]
        # SPLIT the day into warmup and live. The first attempt replayed every bar as warmup,
        # which left watch_ohlcv with nothing to yield: the engine then filled the silence with
        # synthesised idle bars forever (1280 of them before the run was killed) and the
        # strategy never saw a live tick. The replay fixture carries separate warmup and live
        # lists for exactly this reason; a day is one stream, so the split is made here.
        #: Seconds between live bars. Small, but non-zero: see watch_ohlcv.
        self._live_pace = float(os.environ.get("FAKE_VENUE_LIVE_PACE", "0.25"))
        live_count = max(1, int(os.environ.get("FAKE_VENUE_LIVE_BARS", "120")))
        if len(self._bars) <= live_count:
            live_count = max(1, len(self._bars) // 2)
        self._warmup_bars = self._bars[:-live_count]
        self._live_bars = self._bars[-live_count:]
        self._live_orig_ts = self._orig_ts[-live_count:]
        self._idx = 0

        contract = day.symbol
        reference = self._bars[0].close if self._bars else 2000.0
        venue = FakeVenue(symbol=contract, market_type="DERIVATIVE",
                          last_price=reference, seed=1157)
        self._venue = venue
        shifted_bars = [dict(b, timestamp=int(b["timestamp"]) + self._offset_ms)
                        for b in day.bars]
        from datetime import timedelta as _td
        shifted_ftd = (datetime.now() + _td(days=90) + _td(milliseconds=self._offset_ms)
                       ).strftime("%Y-%m-%d")
        self._server = VenueHTTP(
            venue, contract=contract, bars=shifted_bars, final_trade_date=shifted_ftd,
            band=(round(reference * 1.07, 1), round(reference * 0.93, 1)),
        ).start()

        # Repoint THIS run only. The live config file is never touched: this broker reads its
        # own, and the endpoints are overwritten in memory after it is loaded.
        self.config.base_url = self._server.base_url
        self.config.ws_url = f"ws://127.0.0.1:{self._server.port}"

        import logging
        if os.environ.get("FAKE_VENUE_DEBUG"):
            # The poll's own view is logged at DEBUG (lib/log.py:broker_debug), which the
            # operator log level hides. This is the only way to see what the poll READ versus
            # what the adapter SERVED without instrumenting the engine, which is out of scope.
            logging.getLogger().setLevel(logging.DEBUG)
            logging.getLogger("pynecore").setLevel(logging.DEBUG)
        logging.getLogger(__name__).warning(
            "[FAKE VENUE] day=%s label=%s partial=%s bars=%d prints=%d rest=%s "
            "finalTradeDate=%s (served in the REAL future so the GTD clamp admits conditionals)",
            Path(path).name, day.label.value, day.partial, len(day.bars), len(day.prints),
            self._server.base_url, self._server.served_final_trade_date)
        logging.getLogger(__name__).warning(
            "[FAKE VENUE] split: %d warmup bar(s), %d live bar(s) (FAKE_VENUE_LIVE_BARS)",
            len(self._warmup_bars), len(self._live_bars))
        logging.getLogger(__name__).warning(
            "[FAKE VENUE] replay offset = %+d s (day %s served as today); the day file keeps "
            "its recorded timestamps, the shift is applied at serve time on bars, prints, "
            "/price/ohlc and finalTradeDate alike",
            self._offset_ms // 1000,
            datetime.fromtimestamp(self._orig_ts[0] / 1000).date())

    @property
    def client(self):
        """Repair the config before the first client build.

        The degraded-config defect bites at whichever boundary touches the config FIRST, and
        that is not always a method this class overrides — the broker contract check reaches
        ``account_id`` and then the client on its own. Repairing here covers every path, because
        nothing can reach the venue without going through this property.
        """
        self._repair_config_if_degraded()
        return super().client

    # ----------------------------------------------------------------- data

    @override
    def download_ohlcv(self, time_from: datetime, time_to: datetime,
                       on_progress: Callable[[datetime], None] | None = None,
                       limit: int | None = None, with_extra: bool = False):
        """Warm up from the day's bars. The window is ignored: the day IS the history."""
        self._ensure_venue()
        # The history cursor starts at the END OF WARMUP: everything at or before this is
        # history, everything after it is the live stream that has not happened yet.
        if self._warmup_bars:
            self._server.catalogue["replay_cursor"] = self._warmup_bars[-1].timestamp
        for bar in self._warmup_bars:
            self.save_ohlcv_data(bar)
        if on_progress is not None:
            on_progress(time_to)

    @override
    async def connect(self) -> None:
        self._ensure_venue()
        self._idx = 0
        self._exhausted = asyncio.Event()
        self._connected = True

    @override
    async def disconnect(self) -> None:
        self._connected = False
        server = getattr(self, "_server", None)
        if server is not None:
            server.stop()
            self._server = None

    @property
    @override
    def is_connected(self) -> bool:
        return bool(getattr(self, "_connected", False))

    @override
    async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
        """Feed this bar's prints into the venue, THEN hand the bar to the engine.

        Order matters. Feeding first means a conditional triggers on the trade that actually
        printed inside the bar, so intrabar activation and fills are reproduced rather than
        deferred to a bar boundary — the behaviour a bar-close-only fake would silently lose.
        """
        self._ensure_venue()
        if self._idx < len(self._live_bars):
            # PACE the live stream. Measured 2026-09-18: returning live bars as fast as the
            # engine asks makes the engine count them all as WARMUP — a 49/650 split was
            # reported back as "warmup phase complete — 699 bar(s)", live trading began only
            # after the stream was exhausted, and nothing ever traded live. A real feed
            # delivers a bar per minute, so the boundary between history and live is a GAP in
            # time; with no gap there is no boundary. This is also the root cause of the staged
            # probe placing nothing: its window kept opening inside a warmup that never ended.
            await asyncio.sleep(self._live_pace)
            bar = self._live_bars[self._idx]
            self._idx += 1
            self._day.replay_bar(self._live_orig_ts[self._idx - 1], into=self._venue)
            # Move the history cursor with the stream, so a history read never returns a bar the
            # replay has not reached (see venue_http's /price/ohlc clamp).
            self._server.catalogue["replay_cursor"] = bar.timestamp
            return bar
        await self._exhausted.wait()
        return self._live_bars[-1]

    # ----------------------------------------------------------------- venue shape

    @override
    def get_symbol_info(self, force_update: bool = False):
        """24/7 for a deterministic replay: an empty session calendar keeps the idle
        synthesiser and the market-closed stream pauses out of the way, as the replay provider
        contract already does (``replay_sandbox.py:141-148``)."""
        import dataclasses
        info = super().get_symbol_info(force_update=force_update)
        return dataclasses.replace(info, opening_hours=[], session_starts=[], session_ends=[])

    @override
    def resolve_contract(self, symbol: str | None = None) -> str:
        self._ensure_venue()
        return self._day.symbol

    @override
    def classify_market_type(self, symbol: str | None = None) -> "tuple[str, bool]":
        # AUTHORITATIVE, as the sandbox probe's env pin is: the fake's catalogue states the
        # market type, so the classifier is not guessing, and the price-unit guard (#119) needs
        # an authoritative answer before it will write.
        return "DERIVATIVE", True
