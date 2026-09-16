#!/usr/bin/env python3
"""#116 SAME-DAY measurement — cancel a KNOWN-FILLED order id and record the EXACT reject.

Usage:  probe_116_same_day_cancel.py <numeric_order_id>

Run RIGHT AFTER a fill (same trading day) with the id that just filled — cross-day ids
answer 400 RESOURCE_NOT_FOUND (measured 2026-09-15) and settle nothing. Outcomes:
  * a TERMINAL_CODES member (ORDER_IS_DONE / ORDER_CANCEL_STATUS_REJECTED)
      -> prod uses STRUCTURED codes -> #116's message-blind free-text case is SANDBOX-ONLY
         -> shrink/close the card.
  * a generic code + free-text "already in terminal state Filled"
      -> #116 is REAL ON PROD -> the message-blind classification needs the fix.
This is a NO-OP class write: cancelling a FILLED order cannot execute anything; the venue
refuses it — the refusal's (http, code, message) IS the measurement.
Safety: refuses to run if the id is not numeric (NORMAL-book ids are integers) or if the
order's detail does not read back Filled/terminal first (never poke an id we can't read).
Prints raw (status, code, message) only — never token/account values.
"""
from __future__ import annotations
import sys
import json
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "plugins" / "dnse"))
from pynecore_dnse.client import DNSEClient  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2 or not argv[1].isdigit():
        print("usage: probe_116_same_day_cancel.py <numeric_order_id>  (NORMAL-book id, same day)")
        return 2
    order_id = argv[1]
    cfg = tomllib.loads((REPO / "workdir/config/plugins/dnse.toml").read_text())
    client = DNSEClient(cfg["api_key"], cfg["api_secret"],
                        base_url=cfg.get("base_url") or "https://openapi.dnse.com.vn")
    token = json.loads((REPO / "workdir/state/dnse_trading_token.json").read_text()).get("trading_token")
    if not token:
        print("no trading token — mint first"); return 2
    account = client.get_accounts()[1]["accounts"][0]["id"]

    status, detail = client.get_order_detail(account, order_id, "DERIVATIVE", order_category="NORMAL")
    order_status = detail.get("orderStatus") if isinstance(detail, dict) else None
    print(f"[read-back] http={status} orderStatus={order_status!r} "
          f"fillQuantity={detail.get('fillQuantity') if isinstance(detail, dict) else '?'}")
    if order_status is None:
        print("REFUSING — cannot read the order (same-day id required); nothing sent."); return 2
    if order_status not in ("Filled", "Rejected", "Canceled", "Expired"):
        # PartiallyFilled is WORKING (live remainder) — cancelling it is a REAL cancel.
        print(f"REFUSING — order is {order_status!r} (working, not terminal): cancelling it would "
              "be a REAL cancel, not the #116 no-op probe. Use a fully FILLED id."); return 2

    status, body = client.cancel_order(account, order_id, "DERIVATIVE", token, order_category="NORMAL")
    code = body.get("code") if isinstance(body, dict) else None
    message = body.get("message") if isinstance(body, dict) else str(body)[:160]
    print(f"[#116 MEASUREMENT] cancel of {order_status} id={order_id}: "
          f"http={status} code={code!r} message={message!r}")
    print("interpretation: TERMINAL_CODES member -> free-text case is sandbox-only; "
          "generic code + 'terminal state' free text -> #116 real on prod")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
