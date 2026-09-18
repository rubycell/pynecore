"""TESTING ONLY (#157): build DERIVED-FROM-1M venue days from the venue's 1m history.

Downloads N sessions of 1m OHLCV for one symbol through the plugin's own client and writes one
venue day per session, with prints reconstructed from each bar's OHLC path.

    .venv/bin/python plugins/dnse/testing/venue_day_from_history.py --symbol VN30F2M --sessions 5

**Read-only on the venue.** The only request it makes is ``GET /price/ohlc``. It places no
orders, needs no trading token, and touches no account endpoint.

**It never writes a bar store.** The provider has a persisting download path
(``download_ohlcv`` -> ``save_ohlcv_data``) which TRUNCATES AND REWRITES the shared ``.ohlcv``
file for that (provider, symbol, timeframe). That file is tracked, every offline backtest in the
repo reads it, and the rewrite is exactly the damage ``pyne run --from`` does. This helper calls
the client directly instead, keeps everything in memory, and refuses any output path under a
``workdir/data`` directory so a mistyped destination cannot reach one either.

**It does not read the trading token or any credential value.** The API key and secret are
loaded from the provider config straight into the client; nothing is printed, and the banner the
client logs names endpoints only.

What a day built this way can and cannot prove is set by its label, not by this file: the bars
are the venue's own, so prices and volumes are real, but the intrabar SEQUENCE is reconstructed
from each bar's OHLC path and is not what the market did. A question about the order in which
trades arrived still needs a RECORDED day.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from venue_day import (                                                # noqa: E402
    ICT, derive_day_from_1m_bars, group_bars_by_session, save_day,
)

#: The venue's history endpoint, named in every day's provenance so a file says where it is from.
SOURCE = "GET /price/ohlc resolution=1"

#: Calendar days requested per trading session asked for, plus a fixed cushion. Weekends and
#: holidays mean N sessions never fit in N days; asking short is the quiet way to end up with
#: four sessions when five were wanted.
DAYS_PER_SESSION = 2
WINDOW_CUSHION_DAYS = 5

_REQUIRED_ARRAYS = ("t", "o", "h", "l", "c", "v")


class HistoryDownloadError(Exception):
    """The venue's answer cannot be turned into a trustworthy set of sessions."""


class ProductionWriteRefused(Exception):
    """A destination that would damage state this helper does not own."""


# --------------------------------------------------------------------------- parsing

def bars_from_ohlc_body(body: Any) -> list[dict]:
    """Turn ``/price/ohlc``'s parallel arrays into bars in MILLISECONDS.

    The endpoint answers TradingView-UDF style: five arrays keyed only by POSITION, plus a
    status. Two refusals rather than best-effort reads:

    * arrays of different lengths mean the rows no longer line up, and zipping them would pair
      one bar's high with another bar's close — a corrupted day that looks well-formed on disk;
    * ``s: "no_data"`` is the venue saying it holds nothing for the window, which is not the
      same as a session in which nothing traded.
    """
    if not isinstance(body, dict) or not all(key in body for key in _REQUIRED_ARRAYS):
        raise HistoryDownloadError(
            f"unexpected /price/ohlc body: {type(body).__name__} without the "
            f"{', '.join(_REQUIRED_ARRAYS)} arrays; got {str(body)[:200]}")

    status = str(body.get("s", "")).lower()
    if status and status != "ok":
        raise HistoryDownloadError(
            f"the venue answered s={body.get('s')!r}: it holds no history for this window. "
            f"That is not an empty session and will not be written as one.")

    lengths = {key: len(body[key]) for key in _REQUIRED_ARRAYS}
    if not lengths["t"]:
        # MEASURED 2026-09-18 on production: this endpoint serves the rolling ALIASES only.
        # Asked for the dated contract 41I1GA000 it answers HTTP 200 with every array empty and
        # no error field, while the retired 41I1G9000 answers 400. The silent empty is how the
        # venue says "not a symbol I serve", so it must not read as a quiet session.
        raise HistoryDownloadError(
            "the venue answered 200 with no bars at all. That is how /price/ohlc reports a "
            "symbol it does not serve — it takes the rolling aliases (VN30F1M, VN30F2M), not "
            "the dated contract code — so it is refused rather than written as an empty day.")
    if len(set(lengths.values())) != 1:
        raise HistoryDownloadError(
            f"/price/ohlc parallel arrays differ in length ({lengths}); the rows no longer "
            f"line up and pairing them would silently corrupt every bar")

    return [
        {"timestamp": int(body["t"][index]) * 1000,
         "open": float(body["o"][index]), "high": float(body["h"][index]),
         "low": float(body["l"][index]), "close": float(body["c"][index]),
         "volume": float(body["v"][index])}
        for index in range(lengths["t"])
    ]


def select_last_sessions(grouped: dict[str, list], sessions: int) -> dict[str, list]:
    """The last ``sessions`` trading dates, or a refusal naming what was actually served."""
    dates = sorted(grouped)
    if len(dates) < sessions:
        raise HistoryDownloadError(
            f"the venue served {len(dates)} session(s), fewer than the {sessions} asked for "
            f"({', '.join(dates) or 'none'}); widen the window or ask for fewer rather than "
            f"letting a {sessions}-session claim rest on {len(dates)}")
    return {date: grouped[date] for date in dates[-sessions:]}


# --------------------------------------------------------------------------- destination

def refuse_bar_store_paths(target: Path) -> Path:
    """Refuse any destination inside a ``workdir/data`` directory, in any checkout.

    That directory holds the tracked ``.ohlcv`` files every offline backtest reads, and they are
    shared with the main checkout's working tree. A venue day written there would at best be
    clutter in a tracked directory and at worst overwrite a bar store.
    """
    parts = [part.lower() for part in Path(target).absolute().parts]
    for index in range(len(parts) - 1):
        if parts[index] == "workdir" and parts[index + 1] == "data":
            raise ProductionWriteRefused(
                f"{target}: refusing to write inside workdir/data. That directory holds the "
                f"tracked .ohlcv bar stores every offline backtest reads; venue days belong "
                f"under plugins/dnse/testing/fixtures/venue_day/.")
    return target


def day_filename(symbol: str, session_date: str) -> str:
    """``DERIVED-FROM-1M_<symbol>_<date>.json.gz`` — provenance visible without opening it."""
    return f"DERIVED-FROM-1M_{symbol}_{session_date}.json.gz"


# --------------------------------------------------------------------------- building

def build_days(*, fetch: Callable[..., tuple[int, Any]], symbol: str, market_type: str,
               sessions: int, out_dir: Path | str, downloaded_at: str,
               end: datetime | None = None,
               extra_provenance: dict[str, Any] | None = None) -> list[Path]:
    """Download, split into sessions, and write one day file per session.

    ``fetch`` is injected: in production it is the plugin client's ``get_ohlc``, in tests a
    fixture. Every day is BUILT before any is WRITTEN, so a refusal partway through leaves
    nothing on disk — a half-written set is worse than none, because the good files still look
    authoritative.
    """
    if not isinstance(market_type, str):
        raise HistoryDownloadError(
            f"market_type must be the bar-type STRING, got {market_type!r}. The provider's\n"
            f"classify_market_type answers a (type, authoritative) TUPLE; passing it whole\n"
            f"reaches the venue as HTTP 400 INVALID_BAR_TYPE.")

    finish = end or datetime.now(ICT)
    start = finish - timedelta(days=sessions * DAYS_PER_SESSION + WINDOW_CUSHION_DAYS)

    status, body = fetch(symbol=symbol, market_type=market_type, resolution="1",
                         time_from=int(start.timestamp()), time_to=int(finish.timestamp()))
    if status != 200:
        raise HistoryDownloadError(
            f"/price/ohlc for {symbol} answered HTTP {status}: {str(body)[:200]}")

    chosen = select_last_sessions(group_bars_by_session(bars_from_ohlc_body(body)), sessions)

    target_dir = refuse_bar_store_paths(Path(out_dir))
    built = [(target_dir / day_filename(symbol, session_date),
              derive_day_from_1m_bars(bars, symbol=symbol, session_date=session_date,
                                      downloaded_at=downloaded_at, source=SOURCE,
                                      extra=extra_provenance))
             for session_date, bars in chosen.items()]

    return [save_day(day, refuse_bar_store_paths(path)) for path, day in built]


# --------------------------------------------------------------------------- the venue seam

def _provider(symbol: str, config_path: Path):
    """A provider wired to the live config, WITHOUT ``ensure_config``.

    ``ensure_config`` regenerates and rewrites the config file it reads. That is fine for a
    normal run and wrong here: this file holds live credentials, the helper is meant to be
    read-only on everything it touches, and a rewrite driven by a parent schema is how a config
    loses keys it was not asked about. The values are read and handed straight to the provider.
    """
    import tomllib

    from pynecore_dnse.provider import DNSEConfig, DNSEProvider

    if not config_path.exists():
        raise HistoryDownloadError(f"{config_path} does not exist: no credentials to read")
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    known = {field for field in DNSEConfig.__dataclass_fields__}
    config = DNSEConfig(**{key: value for key, value in raw.items() if key in known})
    if not config.api_key or not config.api_secret:
        raise HistoryDownloadError(
            f"{config_path} carries no api_key/api_secret; copy the live provider config into "
            f"this checkout file-to-file rather than retyping it")
    return DNSEProvider(symbol=symbol, timeframe="1", config=config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", default="VN30F1M",
                        help="alias or dated contract; an alias is RESOLVED against the venue")
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--out-dir",
                        default=str(_HERE / "fixtures" / "venue_day"))
    parser.add_argument("--repo", default=str(_HERE.parents[2]))
    args = parser.parse_args(argv)

    repo = Path(args.repo)
    sys.path.insert(0, str(repo / "src"))

    provider = _provider(args.symbol, repo / "workdir" / "config" / "plugins" / "dnse.toml")
    contract = provider.resolve_contract()
    # ``classify_market_type`` answers (type, authoritative); ``[0]`` is the bar type the
    # endpoint wants. Passing the whole tuple is answered HTTP 400 INVALID_BAR_TYPE, which
    # is the venue catching a mistake the boundary guard in build_days now catches first.
    market_type, authoritative = provider.classify_market_type(contract)
    print(f"resolved {args.symbol} -> {contract} ({market_type}, "
          f"{'from the venue' if authoritative else 'GUESSED from the symbol prefix'})")

    def _fetch(*, symbol, market_type, resolution, time_from, time_to):
        """The one and only venue request this helper makes: read-only history."""
        return provider.client.get_ohlc(market_type, {
            "symbol": symbol, "resolution": resolution, "from": time_from, "to": time_to})

    now = datetime.now(ICT)
    # The history endpoint serves the rolling ALIAS, never the dated code (measured 2026-09-18:
    # the dated code answers 200 with no bars). So days are keyed by the alias, and the dated
    # contract it resolved to is recorded as context with the date that resolution was taken.
    # The alias series splices at each roll, so claiming the dated contract for a past session
    # would be a claim nobody measured.
    written = build_days(fetch=_fetch, symbol=args.symbol, market_type=market_type,
                         sessions=args.sessions, out_dir=Path(args.out_dir),
                         downloaded_at=now.isoformat(timespec="seconds"),
                         extra_provenance={
                             "resolved_contract": contract,
                             "resolved_on": now.strftime("%Y-%m-%d"),
                             "contract_source": ("venue /market/instruments" if authoritative
                                                 else "GUESSED from the symbol prefix"),
                         })

    for path in written:
        from venue_day import load_day
        day = load_day(path)
        print(f"{path.name}: {day.provenance['bar_count']} bars, "
              f"{day.provenance['gaps']} gap(s), {len(day.prints)} prints")
    return 0


if __name__ == "__main__":                                              # pragma: no cover
    raise SystemExit(main())
