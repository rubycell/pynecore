"""TESTING ONLY (#114): Sandbox Replay E2E broker — opt-in by NAME.

Serves REPLAY data (a JSON fixture of warmup + live bars, fed as fake-realtime) as the
provider AND routes ORDERS to the DNSE Sandbox as the broker — so a REAL transpiled strategy
drives the full engine loop (warmup -> live bars -> strategy -> engine -> sandbox order ->
fill -> engine) DETERMINISTICALLY and offline. Production ``dnse_broker`` runs are structurally
untouched (this is a separate class reached only via ``dnse_replay_sandbox:...``, exactly like
``dnse_event``).

- **Data** overrides the provider methods to replay the fixture at ``$REPLAY_SANDBOX_FIXTURE``.
- **Orders** inherit :class:`DNSEBroker` — config points at the sandbox (base_url/keys/token) —
  with the sandbox accommodations proven in ``testing/sandbox_arm_on_fill_probe.py``:
  ``get_position`` stubbed flat (the sandbox has no netting position model) and ``market_type``
  pinnable via ``$REPLAY_SANDBOX_MARKET_TYPE`` (the raw sandbox derivative code classifies STOCK).

The sandbox rejects conditional STOP/OCO and has no price sim, so strategies run here must use
ONLY market/limit (NORMAL) orders and drive SL/TP/PTP/OCA from strategy logic against the
replay price. See CLAUDE.md "Sandbox Replay E2E".
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

from pynecore.core.plugin import override
from pynecore.types.ohlcv import OHLCV

from .broker import DNSEBroker


class ReplaySandboxBroker(DNSEBroker):
    """DNSE broker whose DATA is a replayed fixture and whose ORDERS hit the sandbox."""

    _live_bars: list[OHLCV]
    _live_idx: int
    _exhausted: "asyncio.Event | None"

    def _fixture(self) -> dict:
        path = os.environ.get("REPLAY_SANDBOX_FIXTURE")
        if not path:
            raise RuntimeError(
                "REPLAY_SANDBOX_FIXTURE is unset — point it at the replay fixture JSON "
                "(warmup + live bars)."
            )
        return json.loads(Path(path).read_text())

    # --- DATA: replay the fixture (no venue, deterministic) -----------------

    @override
    def download_ohlcv(self, time_from: datetime, time_to: datetime,
                       on_progress: Callable[[datetime], None] | None = None,
                       limit: int | None = None, with_extra: bool = False):
        """Replay the fixture's warmup (all closed) into the capture buffer.
        ``time_from``/``time_to`` are ignored — the whole recorded warmup is authoritative."""
        for row in self._fixture().get("warmup", []):
            ts, o, h, low, c, v = row[:6]
            self.save_ohlcv_data(OHLCV(
                timestamp=int(ts), open=float(o), high=float(h),
                low=float(low), close=float(c), volume=float(v),
            ))
        if on_progress is not None:
            on_progress(time_to)

    @override
    async def connect(self) -> None:
        """Load the fixture's live bars; DNSE's own connect() is a no-op REST path."""
        self._live_bars = [
            OHLCV(timestamp=int(r[0]), open=float(r[1]), high=float(r[2]),
                  low=float(r[3]), close=float(r[4]), volume=float(r[5]),
                  is_closed=bool(r[6]) if len(r) > 6 else True)
            for r in self._fixture().get("live", [])
        ]
        self._live_idx = 0
        self._exhausted = asyncio.Event()
        self._connected = True

    @override
    async def disconnect(self) -> None:
        self._connected = False

    @property
    @override
    def is_connected(self) -> bool:
        return bool(getattr(self, "_connected", False))

    @override
    async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
        """Yield the next fixture live bar; park on an unset event once exhausted (the
        streamer's stop() cancels this await at shutdown — kill the run when the strategy is
        done)."""
        if self._live_idx < len(self._live_bars):
            bar = self._live_bars[self._live_idx]
            self._live_idx += 1
            return bar
        assert self._exhausted is not None
        await self._exhausted.wait()
        return self._live_bars[-1]

    # --- ORDER-path sandbox accommodations (see sandbox_arm_on_fill_probe.py) ---

    @property
    @override
    def market_type(self) -> str:
        forced = os.environ.get("REPLAY_SANDBOX_MARKET_TYPE")
        return forced or super().market_type

    @override
    def resolve_contract(self, symbol: str | None = None) -> str:
        # The sandbox's instrument catalog is empty, so the real resolve returns the
        # alias unchanged (e.g. `VN30F1M` -> `SYMBOL_NOT_EXIST` at order time). Supply the
        # sandbox's dated code via env (VN30F1M front-month is `41I1G9000` on the sandbox, #113).
        forced = os.environ.get("REPLAY_SANDBOX_CONTRACT")
        return forced or super().resolve_contract(symbol)

    @override
    def _band(self) -> tuple[float, float]:
        # The sandbox has no secdef price band; a market order (-> marketable LO at the band
        # edge) needs one. Supply a synthetic band (the sandbox ignores price / fills a LIMIT
        # at its own price). Override via env "ceiling,floor"; else +/-7% around the last live
        # bar's close (DNSE derivative daily band), rounded to a 0.1 tick.
        env = os.environ.get("REPLAY_SANDBOX_BAND")
        if env:
            ceiling, floor = (float(x) for x in env.split(","))
            return ceiling, floor
        bars = getattr(self, "_live_bars", None)
        idx = getattr(self, "_live_idx", 0)
        ref = bars[idx - 1].close if bars and idx else 2000.0
        return round(ref * 1.07, 1), round(ref * 0.93, 1)

    @override
    def get_symbol_info(self, force_update: bool = False):
        # 24/7 for deterministic replay: empty opening_hours disables the session calendar
        # (idle-synth throttling + "market closed" stream pauses) so the fixture streams
        # continuously regardless of the bars' real timestamps — the ReplayProvider contract.
        import dataclasses
        si = super().get_symbol_info(force_update=force_update)
        return dataclasses.replace(si, opening_hours=[], session_starts=[], session_ends=[])

    @override
    async def get_position(self, symbol):
        return None   # the sandbox has no netting position model (accumulating `deals`)
