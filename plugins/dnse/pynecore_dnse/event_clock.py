"""#101 — EventClockBroker: event-paced bar delivery for STAGED PROBES.

TESTING ONLY. This subclass exists so order-lifecycle probes advance on
order-confirm round-trips (~1-2 s) instead of bar closes (15 s-1 m per
transition): the measured bar-clock floor was 2m00s @15S / ~9 min @1m for
a ladder whose venue confirmations total a few seconds. It is wired as
its OWN opt-in plugin entry point (``dnse_event``) — production
``dnse_broker`` runs are structurally untouched (the #80 panel's
default-OFF condition, and the repo's rule that test affordances live in
a subclass, never a config flag the live order path reads). The probes'
.pine files are byte-identical under either clock.

The three panel guards this design is built around (card #101, all
code-verified):

1. **Consumer handshake, never venue polling.** ``watch_ohlcv`` feeds an
   UNBOUNDED producer-side queue (live_runner:1295) — a venue-only
   "settled?" check is vacuously true at start and would free-run every
   bar into the queue before the first order confirms. A bar is released
   only after a plugin dispatch BEGAN AND RETURNED following the previous
   emission (proof the engine consumed it), with an idle-grace fallback
   for observation-only states.
2. **Positive-evidence settlement + loud timeout.** Absence from the book
   is the 101-154 ms visibility lag / ~10 s stale replica, never
   evidence. Settlement = every venue id newly tracked since the last
   bar positively observed stable-or-terminal via ``_last_seen``. The
   per-transition timeout lives on ``self`` (the runner cancels the
   coroutine every ≤2 s) and ABORTS the run loudly — a #51 park must
   never drift into idle-synth.
3. **Wall-clock-seeded, strictly monotone bar grid.** Client-order-ids
   are a pure function of ``bar_ts_ms`` (models.py:1557) and a repeated
   grid REOPENS a prior run's closed journal rows
   (store_helpers.py:658-676) — so the grid seeds from launch wall-clock
   milliseconds and only ever moves forward.

Bars are FLAT synthetics at the last real close (probes price levels as
percent offsets from close — a flat close keeps every level stable), and
are print-independent: event-clock probes run in venue phases where no
bars exist at all (lunch cancels, pre-open conditionals).
"""
import asyncio
import time

from pynecore.lib import log
from pynecore.types.ohlcv import OHLCV

from .broker import DNSEBroker


class EventClockBroker(DNSEBroker):
    """See module docstring. TESTING ONLY — never run a strategy on this."""

    plugin_name = "DNSEEventClock"

    #: Idle-grace: a state that dispatched nothing (observation-only bars,
    #: the DONE state) advances after this many seconds. Long enough that
    #: a genuinely-dispatching state's call always begins first.
    _EC_IDLE_GRACE_S = 4.0
    #: Per-transition settlement ceiling — a park/#51 window aborts loudly
    #: instead of hanging (never reaches the engine's idle-synth).
    _EC_SETTLE_TIMEOUT_S = 25.0
    _EC_POLL_S = 0.25

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # All clock state on self — the runner cancels/re-enters the
        # watch coroutine every <=2 s (the #81 lesson: coroutine locals
        # can never accumulate).
        self._ec_bar_ts_ms: "int | None" = None      # wall-seeded, monotone
        self._ec_base_price: "float | None" = None
        self._ec_dispatch_began = 0
        self._ec_dispatch_done = 0
        self._ec_done_at_emit = 0
        self._ec_began_at_emit = 0
        self._ec_emit_wall: float = 0.0
        self._ec_ids_at_emit: "frozenset[str]" = frozenset()

    # --- dispatch handshake instrumentation (consumer proof) -------------

    async def _ec_wrap(self, coro):
        self._ec_dispatch_began += 1
        try:
            return await coro
        finally:
            self._ec_dispatch_done += 1

    async def execute_entry(self, envelope):
        return await self._ec_wrap(super().execute_entry(envelope))

    async def execute_exit(self, envelope):
        return await self._ec_wrap(super().execute_exit(envelope))

    async def execute_cancel(self, envelope):
        return await self._ec_wrap(super().execute_cancel(envelope))

    async def execute_cancel_with_outcome(self, envelope):
        return await self._ec_wrap(super().execute_cancel_with_outcome(envelope))

    async def modify_entry(self, old, new):
        return await self._ec_wrap(super().modify_entry(old, new))

    async def modify_exit(self, old, new):
        return await self._ec_wrap(super().modify_exit(old, new))

    # --- settlement predicate --------------------------------------------

    def _ec_tracked_ids(self) -> "frozenset[str]":
        return frozenset(oid for ids in self._order_ids.values()
                         for oid in ids)

    def _ec_new_ids_settled(self) -> bool:
        """POSITIVE evidence only: every venue id newly tracked since the
        last emission has been observed by the poll ladder (`_last_seen`
        carries (fill, raw_status) written from real venue reads)."""
        new_ids = self._ec_tracked_ids() - self._ec_ids_at_emit
        return all(oid in self._last_seen for oid in new_ids)

    # --- the event clock ---------------------------------------------------

    async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
        from pynecore.lib import timeframe as tf_lib
        tf_ms = int(tf_lib.in_seconds(timeframe)) * 1000
        if self._ec_bar_ts_ms is None:
            # Guard 3: wall-clock seed — unique per run, strictly monotone,
            # so deterministic coids can never replay a prior run's rows.
            self._ec_bar_ts_ms = int(time.time() * 1000)
            status, body = await asyncio.to_thread(
                lambda: self.client.get_ohlc(self.market_type, {
                    "symbol": self.symbol, "resolution": "1",
                    "from": int(time.time()) - 600,
                    "to": int(time.time())}))
            closes = (body or {}).get("c") if isinstance(body, dict) else None
            if not closes:
                raise RuntimeError(
                    "event clock: cannot read a base price (venue 1m "
                    "closes empty) — refusing to run on a made-up price")
            self._ec_base_price = float(closes[-1])
            log.broker_info(
                "EVENT CLOCK active (#101, TESTING ONLY): flat bars @ %s, "
                "grid seeded %d, handshake+settlement paced",
                self._ec_base_price, self._ec_bar_ts_ms)
            return self._ec_emit(tf_ms)

        deadline = self._ec_emit_wall + self._EC_SETTLE_TIMEOUT_S
        while True:
            began = self._ec_dispatch_began > self._ec_began_at_emit
            done = self._ec_dispatch_done > self._ec_done_at_emit
            if began and done and self._ec_new_ids_settled():
                return self._ec_emit(tf_ms)
            if (not began and time.monotonic() - self._ec_emit_wall
                    >= self._EC_IDLE_GRACE_S):
                # Observation-only state: nothing dispatched — advance.
                return self._ec_emit(tf_ms)
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"event clock: transition unsettled after "
                    f"{self._EC_SETTLE_TIMEOUT_S:.0f}s (dispatch "
                    f"began={began} done={done} unsettled="
                    f"{sorted(self._ec_tracked_ids() - self._ec_ids_at_emit - set(self._last_seen))}) "
                    f"— aborting LOUDLY (#101 guard: a park/#51 window "
                    f"must never drift into idle-synth)")
            await asyncio.sleep(self._EC_POLL_S)

    def _ec_emit(self, tf_ms: int) -> OHLCV:
        assert self._ec_bar_ts_ms is not None and self._ec_base_price is not None
        bar_ts = self._ec_bar_ts_ms
        self._ec_bar_ts_ms = bar_ts + tf_ms
        self._ec_began_at_emit = self._ec_dispatch_began
        self._ec_done_at_emit = self._ec_dispatch_done
        self._ec_ids_at_emit = self._ec_tracked_ids()
        self._ec_emit_wall = time.monotonic()
        price = self._ec_base_price
        return OHLCV(timestamp=bar_ts, open=price, high=price, low=price,
                     close=price, volume=0.0, is_closed=True)
