"""API flatten for the FILL tier — thin CLI over ``tools/flatten.py`` (#91).

Close-first, then sweep the bot's own protection (journal-rooted, read-only
attribution) so a flat account is never left with an armed entry-stop
(measured 2026-09-08, operator-caught). Foreign orders are reported, never
cancelled — the netting account is shared. No app involvement, so the #51
window never opens (operator decision 2026-09-07).

Exit codes (venue.py convention): 0 flat AND owned orders swept/resolved,
1 not flat / sweep unresolved, 2 could not determine.

Usage: .venv/bin/python plugins/dnse/testing/live_test/flatten_api.py [--dry-run]
"""
import argparse
import sys
from pathlib import Path

REPO = Path("/home/mike/workspace/github/pynecore")
sys.path.insert(0, str(REPO / "plugins" / "dnse" / "tools"))

from pynecore.core.config import ensure_config                       # noqa: E402
from pynecore_dnse.broker import DNSEBroker, DNSEBrokerConfig        # noqa: E402
from flatten import flatten, owned_live_ids                          # noqa: E402

STORE = REPO / "workdir" / "output" / "logs" / "broker.sqlite"


def main() -> int:
    # STRICT parsing, and it happens BEFORE anything touches the venue.
    # This used to be `dry = "--dry-run" in sys.argv`, which silently IGNORED
    # every other argument — so `--help`, typed expecting usage text, fell
    # straight through and FLATTENED A LIVE ACCOUNT (2026-09-16: it read a
    # stale position and sold, taking short 1 to short 2). An execution-capable
    # tool must never treat an unrecognised argument as "proceed".
    parser = argparse.ArgumentParser(
        description="API flatten for the FILL tier (#91). Closes the position, "
                    "then sweeps the bot's OWN protection. Foreign orders are "
                    "reported, never cancelled.",
        epilog="Exit codes: 0 flat and swept, 1 unresolved, 2 could not "
               "determine. --help NEVER contacts the venue.")
    parser.add_argument("--dry-run", action="store_true",
                        help="report attribution and exit; send nothing")
    # parse_args exits 2 on an unknown flag and 0 on --help, both BEFORE the
    # broker below is constructed.
    dry = parser.parse_args().dry_run

    cfg = ensure_config(DNSEBrokerConfig,
                        REPO / "workdir/config/plugins/dnse_broker.toml")
    broker = DNSEBroker(symbol="VN30F1M", timeframe="1", config=cfg)
    symbol = broker.symbol or "VN30F1M"
    owned = owned_live_ids(STORE, cfg.account_no or broker.account_id)
    if owned is None:
        print(f"attribution UNAVAILABLE (store: {STORE})")
    else:
        print(f"attribution: {len(owned)} journalled live id(s)")
    if dry:
        print("dry-run: no orders sent, no cancels")
        return 0
    return flatten(broker, symbol, owned)


if __name__ == "__main__":
    sys.exit(main())
