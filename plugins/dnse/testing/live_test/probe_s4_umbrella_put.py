"""#93-S4 measurement probe — can the OCO UMBRELLA's stopPrice be PUT?

UNMEASURED premise from the #93 panel (seat 1's S4 candidate): #18 proved
the STOP book 500s every amend, but the OCO umbrella was never PUT. If
the venue accepts a stopPrice amend on the UMBRELLA, real trailing-SL
support exists and dominates the rejected cancel+replace design.

RUN ONLY while an OCO bracket of OURS is resting (the F9 live window):
finds the newest journalled OCO umbrella on the account, PUTs a stopPrice
1 tick further from market, and reports the venue's answer VERBATIM.
One write, no retry (#58). Exit codes: 0 = venue ACCEPTED (S4 exists!),
1 = refused (records the code), 2 = no owned resting umbrella found.
"""
import sys
from pathlib import Path

REPO = Path("/home/mike/workspace/github/pynecore")
sys.path.insert(0, str(REPO / "plugins" / "dnse" / "tools"))

from pynecore.core.config import ensure_config                       # noqa: E402
from pynecore_dnse.broker import DNSEBroker, DNSEBrokerConfig        # noqa: E402


def main() -> int:
    cfg = ensure_config(DNSEBrokerConfig,
                        REPO / "workdir/config/plugins/dnse_broker.toml")
    b = DNSEBroker(symbol="VN30F1M", timeframe="1", config=cfg)
    # Newest OCO umbrella on the account that is OURS (journal-rooted).
    from flatten import owned_live_ids
    owned = owned_live_ids(REPO / "workdir/output/logs/broker.sqlite",
                            cfg.account_no or b.account_id)
    if owned is None:
        print("attribution unavailable — refusing to probe"); return 2
    status, body = b.client.get_orders(b.account_id, b.market_type,
                                       order_category="OCO",
                                       page_index=0, page_size=100)
    rows = (body or {}).get("orders") or [] if isinstance(body, dict) else []
    ours = [r for r in rows if str(r.get("id")) in owned
            and str(r.get("orderStatus", "")).upper() in ("NEW", "PENDINGNEW")]
    if not ours:
        print("no owned resting OCO umbrella — run during F9's hold window")
        return 2
    row = ours[-1]
    oid = str(row["id"])
    old_stop = float(row.get("stopPrice") or 0)
    if not old_stop:
        print(f"umbrella {oid} carries no stopPrice field — cannot probe")
        return 2
    new_stop = round(old_stop - 0.1, 1)   # protective long-SL: 1 tick lower
    print(f"S4 PROBE: PUT umbrella {oid} stopPrice {old_stop} -> {new_stop}")
    status, body = b._write(lambda tok: b.client.put_order(
        b.account_id, oid, b.market_type,
        {"stopPrice": new_stop}, tok, order_category="OCO"))
    print(f"venue answer: http={status} body={str(body)[:300]}")
    if status in (200, 201):
        print("S4 EXISTS — the umbrella accepts stopPrice amends. "
              "Real trailing-SL support is possible (#93 follow-up card).")
        return 0
    print("S4 refused — record the code on #93; cancel+replace stays the "
          "only trailing route.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
