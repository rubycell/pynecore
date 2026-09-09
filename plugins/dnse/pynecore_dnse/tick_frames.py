"""DNSE market-data tick-frame adapter (#80) — WS frames → aggregator ticks.

The tick channel's frame schema, measured live 2026-09-09 (200s captures,
probe80): ``matchPrice`` / ``matchQtty`` per print, cumulative
``totalVolumeTraded`` (the gap detector), and a ``time`` field that arrived
in THREE formats across capture runs — a protobuf-style dict
(``{"Seconds": ..., "Nanos": ...}``), an ISO-8601 string, and (per the
docs' examples) a plain epoch number. Parse all three; an unparseable
frame returns ``None`` (callers count and move on — a control frame or a
schema drift must never kill the feed).

Pure functions, no I/O — offline-tested against the recorded frame corpus
(``plugins/dnse/tests/fixtures/tick_corpus_vn30f1m.jsonl``).
"""
from datetime import datetime


def parse_tick_time(raw) -> "float | None":
    """The three measured wire formats → epoch seconds (float, sub-second
    kept when the format carries it)."""
    if raw is None:
        return None
    if isinstance(raw, dict):                     # protobuf {"Seconds","Nanos"}
        seconds = raw.get("Seconds", raw.get("seconds"))
        if seconds is None:
            return None
        nanos = raw.get("Nanos", raw.get("nanos")) or 0
        try:
            return float(seconds) + float(nanos) / 1e9
        except (TypeError, ValueError):
            return None
    if isinstance(raw, str):                      # ISO-8601
        try:
            return datetime.fromisoformat(raw).timestamp()
        except ValueError:
            return None
    try:                                          # plain epoch (s or ms)
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value / 1000.0 if value > 1e11 else value


def parse_tick_frame(frame: dict) -> "tuple[float, float, float, float | None] | None":
    """``(ts, price, qty, cumulative_volume | None)`` for a tick print;
    ``None`` for control frames / non-prints / unparseable rows."""
    if not isinstance(frame, dict) or frame.get("action"):
        return None
    price = frame.get("matchPrice")
    if price is None:
        return None
    ts = parse_tick_time(frame.get("time"))
    if ts is None:
        return None
    try:
        price = float(price)
        qty = float(frame.get("matchQtty") or 0)
    except (TypeError, ValueError):
        return None
    cumulative = frame.get("totalVolumeTraded")
    try:
        cumulative = float(cumulative) if cumulative is not None else None
    except (TypeError, ValueError):
        cumulative = None
    return ts, price, qty, cumulative
