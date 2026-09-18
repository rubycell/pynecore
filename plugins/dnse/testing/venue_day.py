"""TESTING ONLY (#157) stage B: a replayable venue day — recorded from the venue, or synthesised.

A "venue day" is the input the fake venue replays: the per-print tick stream plus the 1m bars of
one session. It has exactly two provenances, and they must never be confusable:

* :attr:`DayLabel.RECORDED` — captured from the real venue during a live session.
* :attr:`DayLabel.SYNTHETIC` — derived from tracked ``.ohlcv`` bars, so work can proceed before
  any live recording exists.

**The label is load-bearing, and this module refuses to guess it.** A synthetic day that could be
read as a recorded one would let a measurement be claimed that never happened — the most
expensive kind of wrong this project produces. So :meth:`VenueDay.from_dict` raises on a missing
or unknown label rather than defaulting to either. Defaulting to RECORDED fabricates evidence;
defaulting to SYNTHETIC quietly discounts a real capture. Refusing is the only answer that cannot
mislead.

A day may also be PARTIAL: the live recording window can be joined late, and a partial day
presented as a whole one misrepresents the session's open, so it says so and carries the
timestamp where it joined.

This module contacts nothing. The live recorder that produces a RECORDED day takes its frame
source as an argument, so it is driven by the vendored WS client in production and by a fixture
in tests — the same seam discipline that #160 is about.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class DayLabel(str, Enum):
    """How this day came to exist. Never inferred, always stated."""

    RECORDED = "RECORDED"
    SYNTHETIC = "SYNTHETIC"


class MalformedDay(Exception):
    """A day whose provenance or structure cannot be trusted, so it is refused."""


@dataclass
class VenueDay:
    """One session's prints and bars, with its provenance attached."""

    symbol: str
    label: DayLabel
    prints: list[dict] = field(default_factory=list)
    bars: list[dict] = field(default_factory=list)
    partial: bool = False
    first_print_ts: int | None = None

    # ----------------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "label": self.label.value,
            "partial": self.partial,
            "first_print_ts": self.first_print_ts,
            "prints": self.prints,
            "bars": self.bars,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "VenueDay":
        """Rebuild a day, REFUSING anything whose provenance is absent or unrecognised."""
        if "label" not in raw:
            raise MalformedDay(
                "venue day carries no 'label': refusing to guess its provenance. A day must "
                "state whether it was RECORDED from the venue or SYNTHETIC; defaulting either "
                "way would either fabricate a measurement or discard a real one.")
        try:
            label = DayLabel(raw["label"])
        except ValueError as exc:
            raise MalformedDay(
                f"unknown venue-day label {raw['label']!r}; expected one of "
                f"{[d.value for d in DayLabel]}") from exc
        for required in ("symbol", "prints"):
            if required not in raw:
                raise MalformedDay(f"venue day is missing {required!r}")
        return cls(symbol=raw["symbol"], label=label, prints=list(raw["prints"]),
                   bars=list(raw.get("bars", [])), partial=bool(raw.get("partial", False)),
                   first_print_ts=raw.get("first_print_ts"))

    # ----------------------------------------------------------------- replay

    def replay_bar(self, bar_ts: int, *, into) -> int:
        """Feed one bar's prints into a venue, in order. Returns how many were fed."""
        fed = 0
        for tick in self.prints:
            if tick["bar_ts"] == bar_ts:
                into.feed_print(price=tick["price"], volume=tick["volume"])
                fed += 1
        return fed

    def replay_into(self, venue) -> int:
        """Feed the whole day, in recorded order."""
        for tick in self.prints:
            venue.feed_print(price=tick["price"], volume=tick["volume"])
        return len(self.prints)


# --------------------------------------------------------------------------- synthesis

def synthesise_day(bars: Iterable[dict], *, symbol: str,
                   session_open_ts: int | None = None) -> VenueDay:
    """Derive a SYNTHETIC day from 1m bars.

    Each bar becomes four prints — open, high, low, close — which is the smallest sequence that
    preserves everything an order can react to: the bar's first and last traded price and both
    extremes. A synthesiser that emitted only closes would never trigger a stop the real session
    triggered intrabar, so the replay would disagree with the market it came from.

    The bar's volume is CONSERVED across its prints, because fills are matched against print
    volume: inventing volume here would change how orders fill.

    The high-then-low ordering is a deliberate, stated simplification. The real tick path within
    a bar is unknowable from OHLC alone, so a synthetic day cannot answer questions about
    intrabar SEQUENCE; only a RECORDED day can. That is precisely why the label exists.
    """
    bar_list = [dict(b) for b in bars]
    prints: list[dict] = []

    for bar in bar_list:
        prices = [bar["open"], bar["high"], bar["low"], bar["close"]]
        total = float(bar.get("volume", 0.0))
        share = round(total / len(prices), 6)
        volumes = [share] * len(prices)
        # Put the rounding remainder on the last print so the bar's volume is conserved exactly.
        volumes[-1] = round(total - share * (len(prices) - 1), 6)
        for price, volume in zip(prices, volumes):
            prints.append({"bar_ts": bar["timestamp"], "price": float(price),
                           "volume": float(volume)})

    first_ts = bar_list[0]["timestamp"] if bar_list else None
    partial = bool(session_open_ts is not None and first_ts is not None
                   and first_ts > session_open_ts)

    return VenueDay(symbol=symbol, label=DayLabel.SYNTHETIC, prints=prints,
                    bars=bar_list, partial=partial, first_print_ts=first_ts)


def synthesise_day_from_ohlcv(path: str | Path, *, symbol: str,
                              limit: int | None = None) -> VenueDay:
    """Synthesise a day from a tracked ``.ohlcv`` file (the offline data already in the repo)."""
    from pynecore.core.ohlcv import OHLCVReader

    bars: list[dict] = []
    with OHLCVReader(Path(path)) as reader:
        for candle in reader:
            bars.append({"timestamp": int(candle.timestamp), "open": float(candle.open),
                         "high": float(candle.high), "low": float(candle.low),
                         "close": float(candle.close), "volume": float(candle.volume)})
            if limit is not None and len(bars) >= limit:
                break
    return synthesise_day(bars, symbol=symbol)


# --------------------------------------------------------------------------- disk

def save_day(day: VenueDay, path: str | Path) -> Path:
    """Write a day as gzipped JSON.

    Compressed because an instrument-day is tens of megabytes uncompressed and this repo is
    cloned into cloud sandboxes.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target, "wt", encoding="utf-8") as handle:
        json.dump(day.to_dict(), handle)
    return target


def load_day(path: str | Path) -> VenueDay:
    """Read a day back, refusing one whose provenance is missing or unrecognised."""
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as handle:
        try:
            raw = json.load(handle)
        except json.JSONDecodeError as exc:
            raise MalformedDay(f"{source}: not readable as JSON: {exc}") from exc
    return VenueDay.from_dict(raw)


# --------------------------------------------------------------------------- live recorder

class DayRecorder:
    """Builds a RECORDED day from a live frame source.

    The source is INJECTED, never constructed here: in production it is the vendored WS client,
    in tests a fixture. That is the same discipline #160 is about — a component that can only
    reach production is a component that can only be tested there.

    Start it whenever. If it joins after the session open it marks the day PARTIAL and records
    where it joined, so a late start can never be mistaken for a whole session.
    """

    def __init__(self, *, symbol: str, session_open_ts: int | None = None):
        self.symbol = symbol
        self.session_open_ts = session_open_ts
        self._prints: list[dict] = []
        self._bars: list[dict] = []

    def on_print(self, *, bar_ts: int, price: float, volume: float) -> None:
        self._prints.append({"bar_ts": int(bar_ts), "price": float(price),
                             "volume": float(volume)})

    def on_bar(self, bar: dict) -> None:
        self._bars.append(dict(bar))

    def finish(self) -> VenueDay:
        first_ts = self._prints[0]["bar_ts"] if self._prints else None
        partial = bool(self.session_open_ts is not None and first_ts is not None
                       and first_ts > self.session_open_ts)
        return VenueDay(symbol=self.symbol, label=DayLabel.RECORDED, prints=list(self._prints),
                        bars=list(self._bars), partial=partial, first_print_ts=first_ts)
