#!/usr/bin/env python3
"""PROD premise-confirmation probes (2026-09-14) — #116 and #117.

Two measurements, both NORMAL-book, both no-fill / no-op class:

  A (#116): cancel a KNOWN-FILLED order id and record the venue's EXACT
     (http, code, message). Safety: refuses to run unless the working book is
     EMPTY (NORMAL ids are reused per-day, #96 — an empty book means a stale id
     cannot address any live order). Expected outcomes:
       - TERMINAL_CODES member (ORDER_IS_DONE / ORDER_CANCEL_STATUS_REJECTED)
         -> #116 is SANDBOX-ONLY -> shrink the card
       - generic code + "terminal state Filled" message -> #116 is REAL ON PROD
       - NOT_FOUND -> cross-day ids don't resolve; re-measure after a same-day fill

  B (#117): STOCK amend semantics. Place a HPG LO at the FLOOR (~-7%, no-fill
     risk ~nil), then ONE PUT changing BOTH price and quantity, and record:
       - does the response carry the SAME id or a NEW id? (docs say cancel+replace)
       - is both-fields-in-one-PUT accepted for STOCK?
     Then cancel whatever rests (by the returned id, and the original if different)
     and VERIFY terminal. Board lot 100; only at-risk if filled, which a floor-priced
     LO in continuous session effectively cannot.

Prints raw (status, code, message) only — never token/account values.
Usage: prod_premise_probes.py A|B|AB
"""
from __future__ import annotations
import json
import sys
import time
import tomllib
from pathlib import Path

REPO = Path("/home/mike/workspace/github/pynecore")
sys.path.insert(0, str(REPO / "plugins" / "dnse"))
from pynecore_dnse.client import DNSEClient  # noqa: E402

CFG = tomllib.loads((REPO / "workdir/config/plugins/dnse.toml").read_text())
TOKEN_FILE = REPO / "workdir/state/dnse_trading_token.json"

#: known-FILLED NORMAL ids (ours/venue history, newest first) for probe A
FILLED_IDS = ["99996", "168116",            # 2026-09-14 (yesterday, most likely to resolve)
               "294106", "294096", "294086", "294066"]   # 2026-09-11
STOCK_SYMBOL = "HPG"
STOCK_QTY = 100


def client_and_token():
    c = DNSEClient(CFG["api_key"], CFG["api_secret"],
                   base_url=CFG.get("base_url") or "https://openapi.dnse.com.vn")
    tok = json.loads(TOKEN_FILE.read_text()).get("trading_token")
    if not tok:
        sys.exit("no trading token in token file — mint first")
    acct = c.get_accounts()[1]["accounts"][0]["id"]
    return c, tok, acct


def show(tag, st, body):
    code = body.get("code") if isinstance(body, dict) else None
    msg = (body.get("message") or body.get("error")) if isinstance(body, dict) else str(body)[:120]
    print(f"  [{tag}] http={st} code={code!r} message={str(msg)[:140]!r}")
    return st, code, msg


def assert_book_empty(c, acct, market):
    st, body = c.get_orders(acct, market, order_category="NORMAL", page_index=0, page_size=100)
    rows = (body.get("data") or body.get("orders") or []) if isinstance(body, dict) else []
    working = [r for r in rows if str(r.get("orderStatus")) in
               ("PendingNew", "New", "PartiallyFilled")]
    if st != 200:
        sys.exit(f"cannot read the {market} book (http={st}) — refusing to probe")
    if working:
        sys.exit(f"{market} book NOT empty ({len(working)} working) — refusing (id-reuse risk, #96)")
    print(f"  [safety] {market} working book EMPTY — safe")


def probe_a(c, tok, acct):
    print("\n=== PROBE A (#116): cancel a KNOWN-FILLED id — record the exact reject ===")
    assert_book_empty(c, acct, "DERIVATIVE")
    for oid in FILLED_IDS:
        st, body = c.cancel_order(acct, oid, "DERIVATIVE", tok, order_category="NORMAL")
        _, code, msg = show(f"cancel filled {oid}", st, body)
        if code not in ("RESOURCE_NOT_FOUND", "INVALID_ORDER_ID"):
            print("  -> decisive answer captured; stopping")
            return
    print("  -> all ids answered NOT_FOUND: cross-day ids don't resolve;"
          " re-measure right after a same-day fill")


def probe_b(c, tok, acct):
    print("\n=== PROBE B (#117): STOCK amend — id stability + both-fields-in-one-PUT ===")
    st, sd = c.get_security_definition(STOCK_SYMBOL)
    row = (sd[0] if isinstance(sd, list) and sd else sd) if st == 200 else None
    if not isinstance(row, dict):
        sys.exit(f"no STOCK secdef for {STOCK_SYMBOL} (http={st}) — cannot price the floor")
    floor = float(row.get("floorPrice"))
    tick = 0.05
    st, lt = c.get_latest_trade(STOCK_SYMBOL)
    last = None
    if st == 200 and isinstance(lt, dict):
        trades = lt.get("trades") or []
        row0 = trades[-1] if isinstance(trades, list) and trades else (trades if isinstance(trades, dict) else {})
        if isinstance(row0, dict):
            last = float(row0.get("price") or row0.get("matchPrice") or 0) or None
    if last is None:
        sys.exit(f"no last trade for {STOCK_SYMBOL} (http={st}, keys={list(lt) if isinstance(lt,dict) else lt}) — cannot anchor a safe no-fill price")
    # -5% below last (L1 no-fill convention), on-tick, and strictly above the floor
    price1 = max(round(round(last * 0.95 / tick) * tick, 2), round(floor + tick, 2))
    price2 = round(price1 + tick, 2)
    print(f"  last={last} (floor={floor}) -> anchor -5%")
    # #119 (measured live 2026-09-15): the STOCK order book takes ĐỒNG on the wire, not
    # thousands — 20.95 was rejected PRICE_MUST_GREATER_THAN_OR_EQUAL_TO_FLOOR_PRICE
    # against floor 19.45 (i.e. 19450đ). Convert via the plugin's own codec.
    from pynecore_dnse.price_units import STOCK_WIRE_SCALE, quantize_wire, to_wire
    wire_price1 = quantize_wire(to_wire(price1, STOCK_WIRE_SCALE), STOCK_WIRE_SCALE)
    wire_price2 = quantize_wire(to_wire(price2, STOCK_WIRE_SCALE), STOCK_WIRE_SCALE)
    price1, price2 = wire_price1, wire_price2
    print(f"  wire (đồng, HOSE-tick-snapped): place @ {price1}, amend to {price2}")
    st, lp = c.get_loan_packages(acct, "STOCK", symbol=STOCK_SYMBOL)
    pkgs = (lp.get("loanPackages") or lp.get("data") or []) if isinstance(lp, dict) else []
    loan = pkgs[0].get("id") if pkgs else None
    print(f"  floor={floor} -> place {STOCK_QTY} @ {price1}, amend to {STOCK_QTY + 100} @ {price2}")
    st, body = c.post_order(acct, "STOCK",
                            {"symbol": STOCK_SYMBOL, "side": "NB", "orderType": "LO",
                             "quantity": STOCK_QTY, "price": price1,
                             "loanPackageId": loan},
                            tok, order_category="NORMAL")
    show("place", st, body)
    oid = str(body.get("id")) if isinstance(body, dict) and body.get("id") else None
    if not oid:
        sys.exit("place failed — nothing to amend")
    time.sleep(1.0)
    st, body = c.put_order(acct, oid, "STOCK",
                           {"price": price2, "quantity": STOCK_QTY + 100},
                           tok, order_category="NORMAL")
    show("PUT both fields", st, body)
    new_id = str(body.get("id")) if isinstance(body, dict) and body.get("id") else None
    if new_id:
        print(f"  -> id after amend: {new_id}  ({'SAME' if new_id == oid else 'NEW — cancel+replace CONFIRMED'})")
    # cleanup: cancel every id we touched, then verify terminal
    for cid in dict.fromkeys(filter(None, [new_id, oid])):
        st, body = c.cancel_order(acct, cid, "STOCK", tok, order_category="NORMAL")
        show(f"cleanup cancel {cid}", st, body)
        time.sleep(0.8)
        st, det = c.get_order_detail(acct, cid, "STOCK", order_category="NORMAL")
        print(f"  [verify {cid}] http={st} status={det.get('orderStatus') if isinstance(det, dict) else det!r}")


def main(argv):
    which = (argv[1] if len(argv) > 1 else "AB").upper()
    c, tok, acct = client_and_token()
    if "A" in which:
        probe_a(c, tok, acct)
    if "B" in which:
        probe_b(c, tok, acct)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
