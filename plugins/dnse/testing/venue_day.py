"""TESTING ONLY (#157) stage B: a replayable venue day — recorded from the venue, or synthesised.

A "venue day" is the input the fake venue replays: the per-print tick stream plus the 1m bars of
one session. It has three provenances, and they must never be confusable:

* :attr:`DayLabel.RECORDED` — captured from the real venue during a live session.
* :attr:`DayLabel.SYNTHETIC` — derived from tracked ``.ohlcv`` bars, so work can proceed before
  any live recording exists.
* :attr:`DayLabel.DERIVED_FROM_1M` — built from 1m history downloaded from the venue for one
  named session, by ``venue_day_from_history.py``. Its prices are the venue's own, but its
  intrabar sequence is reconstructed, so it carries SYNTHETIC's limitation and RECORDED's
  dating. It must state where it came from, and is refused if it does not.

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
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class DayLabel(str, Enum):
    """How this day came to exist. Never inferred, always stated."""

    RECORDED = "RECORDED"
    SYNTHETIC = "SYNTHETIC"
    #: Built from 1m OHLCV downloaded from the venue's history endpoint for one named session.
    #: A third thing on purpose: nobody captured ticks (so it is not RECORDED) and the bars did
    #: not come from a file already in this repo (so it is not SYNTHETIC). It shares SYNTHETIC's
    #: limitation unchanged — intrabar SEQUENCE is unknowable from bar data — and differs only
    #: in where the bars came from, which is why it must say so. See :attr:`VenueDay.provenance`.
    DERIVED_FROM_1M = "DERIVED-FROM-1M"


#: What a DERIVED-FROM-1M day must state about itself before it will load. A provenance label
#: with no provenance behind it is just a longer string.
REQUIRED_PROVENANCE = ("symbol", "session_date", "downloaded_at", "source", "bar_count")


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
    #: Where the numbers came from. Empty for RECORDED and SYNTHETIC days, whose label already
    #: says everything there is to say; REQUIRED for DERIVED-FROM-1M, whose label claims a
    #: specific download of a specific session and is unverifiable without it.
    provenance: dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "label": self.label.value,
            "partial": self.partial,
            "first_print_ts": self.first_print_ts,
            "provenance": self.provenance,
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

        provenance = dict(raw.get("provenance") or {})
        if label is DayLabel.DERIVED_FROM_1M:
            # Specific to this label, deliberately. The other two describe themselves fully, and
            # a blanket requirement would invalidate every day already on disk.
            missing = [key for key in REQUIRED_PROVENANCE if key not in provenance]
            if missing:
                raise MalformedDay(
                    f"a {label.value} day must carry its provenance; missing "
                    f"{', '.join(missing)}. The label claims a named session downloaded at a "
                    f"named time, and that claim is unverifiable without it.")

        return cls(symbol=raw["symbol"], label=label, prints=list(raw["prints"]),
                   bars=list(raw.get("bars", [])), partial=bool(raw.get("partial", False)),
                   first_print_ts=raw.get("first_print_ts"), provenance=provenance)

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

def _prints_from_bars(bar_list: list[dict]) -> list[dict]:
    """Turn each bar into the four prints of its OHLC path, conserving the bar's volume.

    Shared by every provenance that reconstructs prints from bars, so the reconstruction cannot
    drift between them: a derived day and a synthetic day must replay a bar identically, or a
    parity result would depend on which builder produced the file.
    """
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
    return prints


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
    prints = _prints_from_bars(bar_list)

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


# --------------------------------------------------------------------------- derived from 1m

#: Vietnam has no daylight saving, so a fixed offset is exact rather than an approximation.
ICT = timezone(timedelta(hours=7))

MINUTE_MS = 60_000
#: Below this, an epoch value is seconds, not milliseconds (1e12 ms is the year 2001; the
#: seconds epoch will not reach 1e12 until the year 33658).
MILLISECONDS_FLOOR = 1_000_000_000_000


def group_bars_by_session(bars: Iterable[dict], *, tz: timezone = ICT) -> dict[str, list[dict]]:
    """Split a flat download into one list per LOCAL trading date.

    The venue answers a multi-day request as one array; a venue day is one session. The split
    must happen on the Vietnamese trading date rather than on UTC midnight, because a VN session
    runs 09:00-14:45 ICT — that is 02:00-07:45 UTC, so a UTC split happens to work today but a
    grouper written against UTC breaks the moment a bar lands before 07:00 ICT (pre-open,
    auction, or any future session extension), cutting one session into two files.
    """
    grouped: dict[str, list[dict]] = {}
    for bar in bars:
        stamp = int(bar["timestamp"])
        date_key = datetime.fromtimestamp(stamp / 1000, tz).strftime("%Y-%m-%d")
        grouped.setdefault(date_key, []).append(dict(bar))
    return {key: grouped[key] for key in sorted(grouped)}


def derive_day_from_1m_bars(bars: Iterable[dict], *, symbol: str, session_date: str,
                            downloaded_at: str, source: str,
                            session_open_ts: int | None = None,
                            extra: dict[str, Any] | None = None) -> VenueDay:
    """Build a DERIVED-FROM-1M day from downloaded 1m history for ONE session.

    The prints are reconstructed from each bar's OHLC path exactly as a synthetic day's are, so
    the two replay identically; what differs is the provenance, which names the symbol, the
    session, when it was downloaded and from which endpoint.

    Two boundary checks, because the label is a claim and an unchecked claim is worse than none:

    * **The timestamps must be milliseconds.** The venue's history endpoint answers in seconds
      and every consumer here replays milliseconds. An unconverted day replays its whole session
      inside a second of wall clock, the engine's wall-clock anchoring sees a missed timeframe
      boundary every real minute, and it substitutes flat synthetic bars for the entire run —
      which looks like a working run producing no trades.
    * **The dominant step must be one minute.** Not every step: a real session breaks for lunch
      and again before the closing auction, and a minute with no trades is simply absent from
      the venue's answer. Those are counted as gaps and reported. But a file whose steps are
      mostly five minutes is 5m data, and stamping it DERIVED-FROM-1M would claim a price path
      five times finer than the one actually carried.
    """
    bar_list = [dict(b) for b in bars]
    if not bar_list:
        raise MalformedDay(f"no bars for {symbol} on {session_date}: refusing to write an "
                           f"empty day, which would replay as a session that never traded")

    stamps = [int(b["timestamp"]) for b in bar_list]
    if min(stamps) < MILLISECONDS_FLOOR:
        raise MalformedDay(
            f"bar timestamps for {symbol} on {session_date} look like SECONDS "
            f"(min {min(stamps)}); this builder requires milliseconds, the unit the replay and "
            f"the engine's wall-clock anchoring both use.")

    steps = [later - earlier for earlier, later in zip(stamps, stamps[1:])]
    if any(step <= 0 for step in steps):
        raise MalformedDay(f"bars for {symbol} on {session_date} are not in ascending time order")
    one_minute = sum(1 for step in steps if step == MINUTE_MS)
    gaps = len(steps) - one_minute
    if steps and one_minute * 2 <= len(steps):
        raise MalformedDay(
            f"only {one_minute} of {len(steps)} steps are one minute apart, so this is not 1m "
            f"data for {symbol} on {session_date}; refusing to label it DERIVED-FROM-1M")

    first_ts = stamps[0]
    partial = bool(session_open_ts is not None and first_ts > session_open_ts)

    # Caller context first, measured facts second: `extra` adds context such as which alias
    # was asked for and what it resolved to, but it must never be able to restate a number
    # this builder measured, or the whole provenance block stops being evidence.
    provenance = dict(extra or {})
    provenance.update(
        {
            "symbol": symbol,
            "session_date": session_date,
            "downloaded_at": downloaded_at,
            "source": source,
            "bar_count": len(bar_list),
            "gaps": gaps,
            "first_bar_ts": first_ts,
            "last_bar_ts": stamps[-1],
        }
    )

    return VenueDay(
        symbol=symbol,
        label=DayLabel.DERIVED_FROM_1M,
        prints=_prints_from_bars(bar_list),
        bars=bar_list,
        partial=partial,
        first_print_ts=first_ts,
        provenance=provenance,
    )


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
