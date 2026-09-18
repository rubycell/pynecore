"""Run an official DNSE SDK example against the offline fake venue.

    .venv/bin/python plugins/dnse/testing/examples/run_against_fake.py use-cases/market-data.py

The examples are the vendor's own, written by the people who run the venue, so they are an
EXTERNAL definition of the API contract. Every other conformance target this fake has had was our
own reading of it. An example that fails here is evidence about the fake, or about our reading,
in a way none of our own pins can be — those were written by the same people who wrote the thing
under test.

**The examples are not modified.** They already take ``DNSE_BASE_URL`` from the environment, so
pointing them at the fake is configuration, not a patch. This runner starts the venue, exports
the environment and hands over. Anything that still fails is a real difference between the fake
and the venue the vendor documented, which is the point.

**They run against the VENDORED SDK**, never a pip install: the vendored directory is put on the
path ahead of everything else, so ``import dnse`` resolves to ``_vendor/dnse`` at its pinned
version. A pip-installed SDK would silently test a different library.

Nothing here can reach production. The venue binds loopback, ``DNSE_BASE_URL`` is its own
address, and the credentials exported below are nonsense.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
TESTING = REPO / "plugins" / "dnse" / "testing"
VENDOR = REPO / "plugins" / "dnse" / "pynecore_dnse" / "_vendor"
UPSTREAM = Path(__file__).resolve().parent / "upstream"

DEFAULT_DAY = TESTING / "fixtures" / "venue_day" / "DERIVED-FROM-1M_VN30F1M_2026-09-18.json.gz"
ACCOUNT_NO = "0001000000"


def start_venue(day_path: Path, *, phase: str = "continuous"):
    """Boot a fake venue over loopback HTTP and feed it the day's prints.

    The prints are fed up front rather than paced: an example is a one-shot script, not a live
    strategy, and a venue with no price motion can never fill anything, which would make every
    order-placing example prove nothing.
    """
    sys.path.insert(0, str(TESTING))
    from venue_core import FakeVenue
    from venue_day import load_day
    from venue_http import VenueHTTP

    day = load_day(day_path)
    reference = float(day.bars[0]["close"])
    venue = FakeVenue(symbol=day.symbol, market_type="DERIVATIVE", last_price=reference,
                      seed=1157, phase=phase)
    server = VenueHTTP(venue, contract=day.symbol, bars=day.bars,
                       account_no=ACCOUNT_NO, final_trade_date="2026-12-17",
                       band=(round(reference * 1.07, 1), round(reference * 0.93, 1))).start()
    for tick in day.prints:
        venue.feed_print(price=tick["price"], volume=tick["volume"])
    return venue, server


def _allow_plain_http() -> None:
    """Let the vendored SDK reach a loopback ``http://`` endpoint.

    MEASURED 2026-09-18, and it is a property of the SDK, not of this fake. The vendored client
    builds its pool as ``urllib3.PoolManager(..., assert_hostname=False)``
    (``_vendor/dnse/api/client.py:29-35``). On urllib3 2.x that keyword is forwarded to the
    connection class: ``HTTPSConnection`` accepts it, ``HTTPConnection`` does not, and raises

        TypeError: HTTPConnection.__init__() got an unexpected keyword argument 'assert_hostname'

    So the SDK as shipped can talk to production over TLS and to nothing at all over plain HTTP.
    Every example dies on its first request against any local server.

    This drops that one keyword for plain-HTTP pools and changes nothing else. It is the same
    correction the plugin's own wrapper already makes for the same reason — ``client.py`` replaces
    the SDK's pool outright because it also ships ``cert_reqs=CERT_NONE``, i.e. unverified HTTPS,
    which is unacceptable for a live trading client. Neither the SDK nor the examples are edited:
    the vendored copy stays pristine and the examples stay the vendor's own code, which is the
    whole reason they are worth running.
    """
    import urllib3

    original = urllib3.PoolManager._new_pool

    def _new_pool(self, scheme, host, port, request_context=None):
        if scheme == "http" and request_context:
            request_context.pop("assert_hostname", None)
        return original(self, scheme, host, port, request_context)

    urllib3.PoolManager._new_pool = _new_pool


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("example", help="path under the upstream tree, e.g. use-cases/market-data.py")
    parser.add_argument("--day", default=str(DEFAULT_DAY))
    parser.add_argument("--phase", default="continuous",
                        choices=("continuous", "lunch", "atc", "closed"))
    args, passthrough = parser.parse_known_args(argv)

    script = UPSTREAM / args.example
    if not script.is_file():
        raise SystemExit(
            f"{script} does not exist. The upstream examples are fetched, not committed by "
            f"default — see this directory's README for where they come from.")

    venue, server = start_venue(Path(args.day), phase=args.phase)

    # The vendored SDK first, so `import dnse` is the pinned copy and never a pip install.
    sys.path.insert(0, str(VENDOR))
    sys.path.insert(0, str(UPSTREAM))            # the examples' own helper modules
    _allow_plain_http()

    os.environ.update({
        "DNSE_BASE_URL": server.base_url,
        "DNSE_WS_URL": f"ws://127.0.0.1:{server.port}",
        "DNSE_API_KEY": "fake-venue-key",
        "DNSE_API_SECRET": "fake-venue-secret",
        "DNSE_ACCOUNT_NO": ACCOUNT_NO,
    })
    print(f"[FAKE VENUE] {server.base_url} phase={args.phase} "
          f"day={Path(args.day).name} prints={len(venue.records()) and ''}"
          f"{len(server.catalogue['bars'])} bars")
    print(f"[RUNNER] {args.example} against the vendored SDK at {VENDOR}")

    sys.argv = [str(script)] + passthrough
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exit_code:
        return int(exit_code.code or 0)
    finally:
        server.stop()
    return 0


if __name__ == "__main__":                                              # pragma: no cover
    raise SystemExit(main())
