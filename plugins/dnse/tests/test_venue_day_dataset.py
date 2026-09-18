"""#157 — building the backtest arm's bar store from a venue day. Tests FIRST.

Parity compares two engines over the same day. The fake arm reads the venue-day file; the
backtest arm reads an ``.ohlcv`` bar store. Until now that store existed as a local artefact
nobody could rebuild, so "the same day" rested on someone's memory of how it was made. Building
it FROM the day makes the two arms provably the same series.

The one dangerous thing this module can do is write into ``workdir/data``, which holds the
accumulated shared history every offline backtest in the repo reads. So it may write only names
under its own reserved prefix, and that refusal is pinned before anything else.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "testing"))
from venue_day import DayLabel, VenueDay                              # noqa: E402
from venue_day_dataset import (                                       # noqa: E402
    SharedDatasetRefused, dataset_from_day, refuse_foreign_dataset,
)

MINUTE_MS = 60_000
SESSION_OPEN_MS = 1_789_524_000_000


def _day(count=5):
    bars = [{"timestamp": SESSION_OPEN_MS + index * MINUTE_MS,
             "open": 1980.0 + index, "high": 1985.0 + index, "low": 1979.0 + index,
             "close": 1984.0 + index, "volume": 100.0 + index}
            for index in range(count)]
    return VenueDay(symbol="VN30F1M", label=DayLabel.SYNTHETIC, bars=bars, prints=[])


def _with_template(tmp_path):
    """A data directory holding only the sidecar the builder is allowed to copy."""
    repo = Path(__file__).resolve().parents[3]
    source = repo / "workdir" / "data" / "fakeparity_VN30F1M_1.toml"
    if not source.exists():
        pytest.skip("no fakeparity sidecar in this checkout to copy from")
    (tmp_path / "fakeparity_VN30F1M_1.toml").write_text(source.read_text())
    return tmp_path


# --------------------------------------------------------------------------- the refusal first

def __test_a_shared_dataset_name_is_REFUSED__():
    """``dnse_VN30F1M_1`` is the accumulated 1m history for the live symbol. Overwriting it
    destroys data no session here owns and that no download can fully restore."""
    for name in ("dnse_VN30F1M_1", "dnsebroker_VN30F1M_1", "HPG_1D"):
        with pytest.raises(SharedDatasetRefused):
            refuse_foreign_dataset(name)


def __test_the_reserved_fixture_prefix_is_accepted__():
    """The discriminating half: a guard that refused everything would pass the test above and
    leave the module unable to write its own fixture."""
    assert refuse_foreign_dataset("fakeparity_derived_VN30F1M_1").startswith("fakeparity")


def __test_the_builder_itself_refuses_a_foreign_name_before_writing__(tmp_path):
    """The guard has to sit inside the write path, not only in a helper a caller may skip."""
    data = _with_template(tmp_path)

    with pytest.raises(SharedDatasetRefused):
        dataset_from_day(_day(), name="dnse_VN30F1M_1", data_dir=data,
                         template="fakeparity_VN30F1M_1")

    assert not (data / "dnse_VN30F1M_1.ohlcv").exists()


# --------------------------------------------------------------------------- the bars

def __test_the_store_holds_exactly_the_day_bars__(tmp_path):
    from pynecore.core.ohlcv import OHLCVReader

    data = _with_template(tmp_path)
    day = _day(5)

    path = dataset_from_day(day, name="fakeparity_x_VN30F1M_1", data_dir=data,
                            template="fakeparity_VN30F1M_1")

    with OHLCVReader(path) as reader:
        stored = [(int(c.timestamp), float(c.open), float(c.high), float(c.low),
                   float(c.close), float(c.volume)) for c in reader]
    assert stored == [(b["timestamp"], b["open"], b["high"], b["low"], b["close"], b["volume"])
                      for b in day.bars]


def __test_the_sidecar_is_copied_so_the_session_schedule_cannot_drift__(tmp_path):
    """A synthesised sidecar with the wrong sessions changes which bars the engine treats as
    tradable, and that appears as a parity difference between the engines when it is really a
    difference between two fixtures."""
    data = _with_template(tmp_path)

    dataset_from_day(_day(), name="fakeparity_x_VN30F1M_1", data_dir=data,
                     template="fakeparity_VN30F1M_1")

    assert ((data / "fakeparity_x_VN30F1M_1.toml").read_text()
            == (data / "fakeparity_VN30F1M_1.toml").read_text())


def __test_a_missing_template_is_REFUSED_rather_than_invented__(tmp_path):
    with pytest.raises(FileNotFoundError):
        dataset_from_day(_day(), name="fakeparity_x_VN30F1M_1", data_dir=tmp_path,
                         template="does_not_exist")
