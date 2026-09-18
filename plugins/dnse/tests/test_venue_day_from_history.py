"""#157 — the 1m-history downloader that builds DERIVED-FROM-1M venue days. Tests FIRST.

The helper's only venue contact is ``GET /price/ohlc`` through the plugin's own client, so the
network half is a single injected callable and every test here runs offline. What is pinned is
the part that can silently corrupt something:

* **It must never write a bar store.** The provider's own ``download_ohlcv`` persists through
  ``save_ohlcv_data``, which TRUNCATES AND REWRITES the shared ``.ohlcv`` file for that
  (provider, symbol, timeframe) — the same mechanism ``pyne run --from`` abuses to destroy
  accumulated history. This helper therefore does not use that path at all, and refuses an
  output path under ``workdir/data`` so a mistyped destination cannot land there either.
* **It must convert the venue's SECONDS to milliseconds.** ``/price/ohlc`` answers unix seconds;
  the replay and the engine's wall-clock anchoring use milliseconds.
* **It must not invent a session.** An empty or short answer is reported, never padded, and a
  request whose window the venue did not fully cover says so rather than labelling a truncated
  download as a whole session.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "testing"))
from venue_day import DayLabel, MalformedDay, load_day                 # noqa: E402
from venue_day_from_history import (                                   # noqa: E402
    HistoryDownloadError, ProductionWriteRefused, bars_from_ohlc_body, build_days,
    day_filename, refuse_bar_store_paths, select_last_sessions,
)

#: 2026-09-16 09:00 ICT.
SESSION_OPEN_S = 1_789_524_000


def _body(count=3, start=SESSION_OPEN_S, step=60):
    """The venue's TradingView-UDF-style parallel arrays, with ``t`` in SECONDS."""
    return {
        "s": "ok",
        "t": [start + index * step for index in range(count)],
        "o": [1980.0 + index for index in range(count)],
        "h": [1985.0 + index for index in range(count)],
        "l": [1979.0 + index for index in range(count)],
        "c": [1984.0 + index for index in range(count)],
        "v": [100.0 + index for index in range(count)],
    }


# --------------------------------------------------------------------------- parsing the answer

def __test_the_venue_seconds_become_milliseconds__():
    bars = bars_from_ohlc_body(_body())

    assert bars[0]["timestamp"] == SESSION_OPEN_S * 1000
    assert bars[1]["timestamp"] - bars[0]["timestamp"] == 60_000


def __test_every_price_and_volume_survives_the_parse__():
    body = _body(3)

    bars = bars_from_ohlc_body(body)

    assert [b["open"] for b in bars] == body["o"]
    assert [b["high"] for b in bars] == body["h"]
    assert [b["low"] for b in bars] == body["l"]
    assert [b["close"] for b in bars] == body["c"]
    assert [b["volume"] for b in bars] == body["v"]


def __test_parallel_arrays_of_unequal_length_are_REFUSED__():
    """The venue's answer is five parallel arrays keyed only by position. A short one means the
    rows no longer line up, and zipping them silently would pair one bar's high with another
    bar's close — a corrupted day that looks perfectly well-formed on disk."""
    body = _body(3)
    body["v"] = body["v"][:2]

    with pytest.raises(HistoryDownloadError) as excinfo:
        bars_from_ohlc_body(body)

    assert "length" in str(excinfo.value).lower()


def __test_a_no_data_answer_is_REFUSED_rather_than_read_as_an_empty_session__():
    """``s: "no_data"`` is the venue saying it has nothing, which is not the same as a session
    with no trades. Reading it as an empty day would write a file claiming a session that the
    venue never confirmed existed."""
    with pytest.raises(HistoryDownloadError) as excinfo:
        bars_from_ohlc_body({"s": "no_data", "t": [], "o": [], "h": [], "l": [], "c": [],
                             "v": []})

    assert "no_data" in str(excinfo.value)


def __test_a_body_that_is_not_the_expected_shape_is_REFUSED__():
    with pytest.raises(HistoryDownloadError):
        bars_from_ohlc_body({"error": "SYMBOL_NOT_EXIST"})


def __test_a_200_answer_with_no_bars_at_all_is_REFUSED__():
    """MEASURED 2026-09-18 against production: this endpoint serves the rolling ALIASES only.
    Asked for the dated contract code 41I1GA000 it answers HTTP 200 with every array EMPTY and
    no error field at all, while the retired 41I1G9000 answers 400. So the silent empty is the
    venue's real "I do not serve this symbol", and a helper that read it as a quiet day would
    write files for sessions that never happened."""
    empty = {"t": [], "o": [], "h": [], "l": [], "c": [], "v": [], "nextTime": 0}

    with pytest.raises(HistoryDownloadError) as excinfo:
        bars_from_ohlc_body(empty)

    assert "no bars" in str(excinfo.value).lower()


def __test_the_venue_sends_no_status_field_and_that_alone_is_not_an_error__():
    """The discriminating half. DNSE's answer carries no ``s`` key at all (measured: the keys
    are c, h, l, nextTime, o, t, v), so a parser that demanded one would refuse every real
    download. The status check must fire only on a status that is present and not ok."""
    body = _body(3)
    del body["s"]

    assert len(bars_from_ohlc_body(body)) == 3


# --------------------------------------------------------------------------- choosing sessions

def __test_the_last_n_sessions_are_chosen_by_trading_date_not_by_bar_count__():
    grouped = {"2026-09-14": [1], "2026-09-15": [1, 2], "2026-09-16": [1], "2026-09-17": [1]}

    chosen = select_last_sessions(grouped, 2)

    assert list(chosen) == ["2026-09-16", "2026-09-17"]


def __test_asking_for_more_sessions_than_the_venue_served_is_REPORTED_not_padded__():
    """A short answer is a real outcome — a holiday week, a new contract, a truncated window.
    Padding it or silently returning fewer would let a five-session claim rest on two."""
    grouped = {"2026-09-16": [1], "2026-09-17": [1]}

    with pytest.raises(HistoryDownloadError) as excinfo:
        select_last_sessions(grouped, 5)

    assert "2" in str(excinfo.value) and "5" in str(excinfo.value)


# --------------------------------------------------------------------------- the destination

def __test_an_output_path_under_the_shared_bar_store_is_REFUSED__():
    """``workdir/data`` holds the tracked ``.ohlcv`` files that every offline backtest in this
    repo reads. Writing a venue day there is the one mistake that would damage state this
    session does not own."""
    with pytest.raises(ProductionWriteRefused):
        refuse_bar_store_paths(Path("workdir/data/dnse_VN30F1M_1.ohlcv"))

    with pytest.raises(ProductionWriteRefused):
        refuse_bar_store_paths(Path("/home/mike/workspace/github/pynecore/workdir/data/x.json.gz"))


def __test_the_fixtures_directory_is_accepted__():
    """The discriminating half: a guard that refused every path would pass the test above and
    make the helper unable to write anything at all."""
    target = Path("plugins/dnse/testing/fixtures/venue_day/DERIVED-FROM-1M_x_2026-09-16.json.gz")

    assert refuse_bar_store_paths(target) == target


def __test_the_filename_carries_the_label_symbol_and_session_date__():
    """These files are read by a human choosing one to replay months from now; the provenance
    has to be visible without decompressing the file."""
    name = day_filename("41I1GA000", "2026-09-16")

    assert name.startswith("DERIVED-FROM-1M_")
    assert "41I1GA000" in name and "2026-09-16" in name
    assert name.endswith(".json.gz")


# --------------------------------------------------------------------------- building days

def _two_sessions_body():
    """Two consecutive VN sessions of four 1m bars each, as the venue would answer them."""
    day_one = _body(4, start=SESSION_OPEN_S)
    day_two = _body(4, start=SESSION_OPEN_S + 24 * 3600)
    merged = {"s": "ok"}
    for key in ("t", "o", "h", "l", "c", "v"):
        merged[key] = day_one[key] + day_two[key]
    return merged


def __test_build_days_writes_one_labelled_file_per_session__(tmp_path):
    written = build_days(
        fetch=lambda **kwargs: (200, _two_sessions_body()),
        symbol="41I1GA000", market_type="DERIVATIVE", sessions=2,
        out_dir=tmp_path, downloaded_at="2026-09-18T15:00:00+07:00")

    assert len(written) == 2
    for path in written:
        day = load_day(path)
        assert day.label is DayLabel.DERIVED_FROM_1M
        assert day.provenance["symbol"] == "41I1GA000"
        assert day.provenance["session_date"] in path.name
        assert day.bars and day.prints


def __test_build_days_never_calls_the_bar_store_writer__(tmp_path, monkeypatch):
    """The failure this guards is silent and destructive: one call to the provider's persisting
    download path truncates the shared 1m file for the symbol. A test that only checked the
    output files would pass while that happened."""
    import pynecore_dnse.provider as provider_module

    def _explode(*args, **kwargs):
        raise AssertionError("the downloader must not persist through the shared bar store")

    monkeypatch.setattr(provider_module.DNSEProvider, "save_ohlcv_data", _explode, raising=False)
    monkeypatch.setattr(provider_module.DNSEProvider, "download_ohlcv", _explode, raising=False)

    written = build_days(
        fetch=lambda **kwargs: (200, _two_sessions_body()),
        symbol="41I1GA000", market_type="DERIVATIVE", sessions=2,
        out_dir=tmp_path, downloaded_at="2026-09-18T15:00:00+07:00")

    assert len(written) == 2


def __test_a_non_200_answer_is_REFUSED_and_names_the_status__(tmp_path):
    with pytest.raises(HistoryDownloadError) as excinfo:
        build_days(fetch=lambda **kwargs: (429, {"error": "rate limited"}),
                   symbol="41I1GA000", market_type="DERIVATIVE", sessions=1,
                   out_dir=tmp_path, downloaded_at="2026-09-18T15:00:00+07:00")

    assert "429" in str(excinfo.value)


def __test_a_five_minute_download_is_REFUSED_before_anything_is_written__(tmp_path):
    """The label's claim is checked at the build boundary, and nothing reaches disk when it
    fails — a half-written set of files is worse than none, because the good ones look
    authoritative."""
    coarse = _body(8, start=SESSION_OPEN_S, step=300)

    with pytest.raises(MalformedDay):
        build_days(fetch=lambda **kwargs: (200, coarse),
                   symbol="41I1GA000", market_type="DERIVATIVE", sessions=1,
                   out_dir=tmp_path, downloaded_at="2026-09-18T15:00:00+07:00")

    assert list(tmp_path.iterdir()) == []


def __test_the_bar_type_tuple_is_REFUSED_before_it_reaches_the_venue__(tmp_path):
    """MEASURED 2026-09-18, against production: the provider's ``classify_market_type`` answers
    a ``(type, authoritative)`` TUPLE, and the first wiring of this helper passed it whole. The
    venue answered ``HTTP 400 INVALID_BAR_TYPE`` — a clear error, but only after a live request.
    The guard turns the same mistake into a refusal at the boundary, where it costs nothing."""
    def _never_called(**kwargs):
        raise AssertionError("the guard must refuse before any request is made")

    with pytest.raises(HistoryDownloadError) as excinfo:
        build_days(fetch=_never_called, symbol="41I1GA000",
                   market_type=("DERIVATIVE", True),        # the exact shape that reached prod
                   sessions=1, out_dir=tmp_path, downloaded_at="2026-09-18T15:00:00+07:00")

    assert "INVALID_BAR_TYPE" in str(excinfo.value)


def __test_the_request_window_covers_the_sessions_asked_for__(tmp_path):
    """A window too short is the quiet way to get four sessions when five were asked for. The
    fetch must be given a span comfortably wider than the sessions requested, because weekends
    and holidays mean N sessions span more than N days."""
    seen = {}

    def _fetch(**kwargs):
        seen.update(kwargs)
        return 200, _two_sessions_body()

    build_days(fetch=_fetch, symbol="41I1GA000", market_type="DERIVATIVE", sessions=2,
               out_dir=tmp_path, downloaded_at="2026-09-18T15:00:00+07:00")

    span_days = (seen["time_to"] - seen["time_from"]) / 86400
    assert span_days > 2, "N calendar days cannot contain N trading sessions"
    assert seen["resolution"] == "1"
    assert seen["symbol"] == "41I1GA000"


# --------------------------------------------------------------------------- the roll

def __test_a_day_records_the_alias_asked_for_and_the_contract_it_resolved_to__(tmp_path):
    """MEASURED 2026-09-18, and the reason this is not cosmetic.

    ``/price/ohlc`` serves only the rolling aliases, and the alias series SPLICES at each roll:
    asked over the same fortnight, VN30F1M and VN30F2M return different closes for the same past
    sessions, because each alias means whichever dated contract was front or next AT THE TIME.
    On this date VN30F1M resolved to 41I1GA000 and VN30F2M to 41I1GB000 — the roll happened that
    morning — so four of the five most recent VN30F1M sessions belong to the contract that
    expired the day before.

    A day therefore cannot claim a dated contract for a past session. It records what was
    actually asked (the alias), what that alias resolved to and WHEN that resolution was taken,
    and leaves the reader to draw the line rather than asserting one that may be false.
    """
    written = build_days(
        fetch=lambda **kwargs: (200, _two_sessions_body()),
        symbol="VN30F1M", market_type="DERIVATIVE", sessions=2,
        out_dir=tmp_path, downloaded_at="2026-09-18T15:00:00+07:00",
        extra_provenance={"resolved_contract": "41I1GA000", "resolved_on": "2026-09-18"})

    day = load_day(written[0])
    assert day.provenance["symbol"] == "VN30F1M"
    assert day.provenance["resolved_contract"] == "41I1GA000"
    assert day.provenance["resolved_on"] == "2026-09-18"


def __test_extra_provenance_cannot_overwrite_the_facts_the_builder_measured__(tmp_path):
    """The discriminating half. Caller-supplied provenance is context, not a way to restate the
    bar count or the session date — a day that could be told it holds 241 bars when it holds 66
    would make the whole provenance block untrustworthy."""
    written = build_days(
        fetch=lambda **kwargs: (200, _two_sessions_body()),
        symbol="VN30F1M", market_type="DERIVATIVE", sessions=2,
        out_dir=tmp_path, downloaded_at="2026-09-18T15:00:00+07:00",
        extra_provenance={"bar_count": 99999, "session_date": "1999-01-01"})

    day = load_day(written[0])
    assert day.provenance["bar_count"] == 4
    assert day.provenance["session_date"] == "2026-09-16"
