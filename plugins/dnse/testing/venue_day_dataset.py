"""TESTING ONLY (#157): turn a venue day's bars into a file-mode ``.ohlcv`` dataset.

Trade-list parity runs the same strategy twice over the SAME day — once as a file-mode backtest
through the sim engine, once through the fake venue. The fake arm reads the venue-day file; the
backtest arm needs the identical bars as a bar store. This builds that store from the day, so
the two arms are provably the same series rather than two files someone believes match.

**It writes only its own dataset, never a shared one.** The name is fixed by a prefix reserved
for these fixtures, and any attempt to write a store whose name does not carry that prefix is
refused — the shared ``dnse_*`` stores are the accumulated history every offline backtest reads,
and a helper that could overwrite one has no business being convenient about it.

The sidecar ``.toml`` (tick grid, sessions, timezone) is COPIED from an existing dataset for the
same instrument rather than synthesised, because a wrong session schedule silently changes which
bars the engine treats as tradable and would show up as a parity difference that is really a
fixture bug.
"""
from __future__ import annotations

import shutil
from pathlib import Path

#: Every dataset this module writes starts with this. A store without it is somebody else's.
FIXTURE_PREFIX = "fakeparity"


class SharedDatasetRefused(Exception):
    """A dataset name that is not one of this module's own fixtures."""


def refuse_foreign_dataset(name: str) -> str:
    """Refuse any dataset name outside the reserved fixture prefix."""
    if not name.startswith(FIXTURE_PREFIX):
        raise SharedDatasetRefused(
            f"{name!r} is not a {FIXTURE_PREFIX}* fixture dataset. The shared stores in "
            f"workdir/data hold accumulated history every offline backtest reads; this module "
            f"writes only its own fixtures.")
    return name


def dataset_from_day(day, *, name: str, data_dir: Path | str, template: str,
                     period: str = "1", minmove: int = 1, pricescale: int = 10,
                     timezone: str = "Asia/Ho_Chi_Minh") -> Path:
    """Write ``day.bars`` as ``<data_dir>/<name>.ohlcv`` plus a copied ``.toml`` sidecar.

    ``template`` is the basename of an existing dataset for the same instrument whose sidecar is
    copied verbatim.
    """
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.types.ohlcv import OHLCV

    refuse_foreign_dataset(name)
    directory = Path(data_dir)
    target = directory / f"{name}.ohlcv"

    source_toml = directory / f"{template}.toml"
    if not source_toml.exists():
        raise FileNotFoundError(
            f"{source_toml} does not exist; the sidecar is copied rather than synthesised "
            f"because a wrong session schedule shows up as a parity difference that is really "
            f"a fixture bug")

    with OHLCVWriter(target, period, minmove=minmove, pricescale=pricescale,
                     truncate=True, timezone=timezone) as writer:
        for bar in day.bars:
            writer.write(OHLCV(timestamp=int(bar["timestamp"]), open=float(bar["open"]),
                               high=float(bar["high"]), low=float(bar["low"]),
                               close=float(bar["close"]), volume=float(bar["volume"])))

    shutil.copyfile(source_toml, directory / f"{name}.toml")
    return target
