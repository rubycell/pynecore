"""Pure tick→bar aggregation for sub-minute (and any intraday) timeframes.

Built for card #80 (DNSE LTF): venue REST floors at 1 minute, but the
market-data tick stream reconstructs venue bars EXACTLY (measured live
2026-09-09: per-minute OHLC and volume identical to the venue's own REST
bars, dv=0 on every complete minute). This module is the venue-agnostic
half of that synthesis: feed it ticks, it emits closed bars.

Boundary math: for INTRADAY timeframes the resampler grid is the
epoch-aligned grid — ``bar_time = ts - ts % tf_seconds`` — which is what
``Resampler.get_bar_time`` computes for intraday inputs; the session-grid
machinery there exists for D/W/M targets and is deliberately not dragged
into this pure path (#80 adjudication, seat-3 split: pure aggregator in
core, venue frame adapters in plugins). The parity tests pin the grid; if
a venue/TV intraday boundary ever disagrees, they fail loudly.

Empty windows emit NOTHING — matching the venue's own bar stream shape
(#80 adjudication; TV parity for zero-print windows is an operator-verify
on the wiring card).

Gap detection: venues that publish a cumulative session volume alongside
each print (DNSE: ``totalVolumeTraded``) get per-bar integrity for free —
a bar whose summed print volume disagrees with the cumulative delta saw a
MISSED print (WS drop, reconnect window) and is flagged ``suspect``, so a
consumer can refuse to trade a bar built on incomplete data instead of
trusting a wrong high/low.
"""
from dataclasses import dataclass


@dataclass
class TickBar:
    """One closed synthesized bar. ``suspect`` = built on provably
    incomplete prints (cumulative-volume mismatch) — consumers must treat
    it as damaged, never as a normal bar."""
    time: int                # bar OPEN, epoch seconds, grid-aligned
    open: float
    high: float
    low: float
    close: float
    volume: float
    ticks: int               # constituent prints
    suspect: bool = False


class TickAggregator:
    """Feed ticks in arrival order; closed bars come back as they complete.

    ``add`` returns the list of bars CLOSED by this tick (usually empty or
    one; more after a quiet gap spanning several boundaries — empty
    windows in between emit nothing). ``flush`` closes and returns the
    in-progress bar (end of session / shutdown); the flushed bar is by
    construction PARTIAL — callers decide whether a partial final bar is
    usable for their purpose.
    """

    def __init__(self, tf_seconds: int) -> None:
        if tf_seconds <= 0:
            raise ValueError(f"tf_seconds must be positive, got {tf_seconds}")
        self.tf_seconds = int(tf_seconds)
        self._bar: "TickBar | None" = None
        self._cum_anchor: "float | None" = None   # cumulative vol at bar open
        self._cum_last: "float | None" = None

    def _grid(self, ts: float) -> int:
        return int(ts) - int(ts) % self.tf_seconds

    def _close_current(self) -> "TickBar":
        bar = self._bar
        assert bar is not None
        if (self._cum_anchor is not None and self._cum_last is not None):
            expected = self._cum_last - self._cum_anchor
            # Float-tolerant: cumulative and per-print volumes are contract
            # counts at DNSE, but stay generic for fractional venues.
            if abs(expected - bar.volume) > 1e-9:
                bar.suspect = True
        self._bar = None
        self._cum_anchor = None
        return bar

    def add(self, ts: float, price: float, qty: float,
            cumulative: "float | None" = None) -> "list[TickBar]":
        """One print. Out-of-order ticks BEHIND the current bar's window are
        counted into the current bar rather than rewriting a closed one —
        a closed bar is immutable (consumers already saw it); the
        cumulative check then marks the affected bar suspect, which is the
        honest outcome for late data."""
        closed: list[TickBar] = []
        bar_time = self._grid(ts)
        if self._bar is not None and bar_time > self._bar.time:
            closed.append(self._close_current())
        if self._bar is None:
            self._bar = TickBar(time=bar_time, open=price, high=price,
                                low=price, close=price, volume=0.0, ticks=0)
            self._cum_anchor = (cumulative - qty
                                if cumulative is not None else None)
        bar = self._bar
        bar.high = max(bar.high, price)
        bar.low = min(bar.low, price)
        bar.close = price
        bar.volume += qty
        bar.ticks += 1
        if cumulative is not None:
            self._cum_last = cumulative
        return closed

    def flush(self) -> "TickBar | None":
        """Close the in-progress (partial) bar, if any."""
        if self._bar is None:
            return None
        return self._close_current()
