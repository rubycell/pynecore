#!/usr/bin/env python3
"""DNSE SANDBOX order-lifecycle + trading-WS probe.

Proves the trading WebSocket delivers order/position events end-to-end, using DNSE's
Sandbox (mock env — no real money, no market hours). Places a NORMAL order and captures
its auto-progression ``PendingNew -> New -> PartiallyFilled -> Filled`` on the trading WS.

WHY THIS EXISTS: the production trading WS could not be validated live (real money, market
hours, and — the trap — the order channel is CASE-SENSITIVE: ``order.DERIVATIVE.json``
UPPERCASE per the docs/SDK; a lowercase name is silently accepted but streams nothing,
producing a false "silent WS" verdict). Sandbox makes this a five-minute offline test.

SETUP (one-time, operator): register Sandbox on the LightSpeed API page, save the one-time
secret, then create ``workdir/config/plugins/dnse_sandbox.toml`` (gitignored) with:
    api_key = "<sandbox key>"
    api_secret = "<sandbox secret>"
    base_url = "https://sb-openapi.dnse.com.vn"
    ws_url = "wss://ws-sb-openapi.dnse.com.vn"
    token_file = "workdir/state/dnse_sandbox_trading_token.json"

Sandbox OTP is the fixed public test constant 666666 (no real email/SMS). Sandbox supports
NORMAL orders only — conditional STOP/OCO still need production.

Read-only w.r.t. production. Masks account/investor identifiers; never prints secrets.

Usage:  python plugins/dnse/testing/sandbox_lifecycle_probe.py [--seconds 30]
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import hmac
import json
import sys
import time
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "plugins" / "dnse"))
from pynecore_dnse.client import DNSEClient  # noqa: E402
import websockets  # noqa: E402

CONFIG = REPO / "workdir" / "config" / "plugins" / "dnse_sandbox.toml"
_PII = ("accountNo", "investorId", "custodyCode", "name", "id")


def _mask(obj):
    """Recursively blank identifier fields so nothing PII is ever printed."""
    if isinstance(obj, dict):
        return {k: ("<masked>" if k in _PII else _mask(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_mask(v) for v in obj]
    return obj


def _mint_token(cfg) -> str:
    tf = REPO / cfg["token_file"]
    if tf.exists():
        tok = json.loads(tf.read_text()).get("trading_token")
        if tok:
            return tok
    c = DNSEClient(cfg["api_key"], cfg["api_secret"], base_url=cfg["base_url"])
    st, body = c.create_trading_token("smart_otp", "666666")
    tok = body.get("tradingToken") or body.get("trading-token") if isinstance(body, dict) else None
    if st != 200 or not tok:
        sys.exit(f"sandbox token mint failed: HTTP {st}")
    tf.parent.mkdir(parents=True, exist_ok=True)
    tf.write_text(json.dumps({"trading_token": tok, "minted_at": int(time.time())}))
    return tok


async def _run(seconds: int) -> int:
    if not CONFIG.exists():
        sys.exit(f"sandbox config not found: {CONFIG} (see this file's docstring)")
    cfg = tomllib.loads(CONFIG.read_text())
    tok = _mint_token(cfg)
    c = DNSEClient(cfg["api_key"], cfg["api_secret"], base_url=cfg["base_url"])
    acct = c.get_accounts()[1]["accounts"][0]["id"]
    _, lp = c.get_loan_packages(acct, "DERIVATIVE", symbol="41I1G9000")
    pkgs = (lp.get("loanPackages") or lp.get("data") or []) if isinstance(lp, dict) else []
    loan_id = pkgs[0].get("id") if pkgs else None
    if not loan_id:
        sys.exit(f"no sandbox loan package: {str(lp)[:160]}")

    ws_base = cfg.get("ws_url", "wss://ws-sb-openapi.dnse.com.vn")
    url = f"{ws_base}/v1/stream?encoding=json"
    ts, nonce = int(time.time()), str(int(time.time() * 1_000_000))
    sig = hmac.new(cfg["api_secret"].encode(),
                   f'{cfg["api_key"]}:{ts}:{nonce}'.encode(), hashlib.sha256).hexdigest()

    order_frames = 0
    async with websockets.connect(url, open_timeout=15) as ws:
        ready = json.loads(await asyncio.wait_for(ws.recv(), 15))
        print(f"[ws] {ready.get('action')}")
        await ws.send(json.dumps({"action": "auth", "api_key": cfg["api_key"],
                                  "signature": sig, "timestamp": ts, "nonce": nonce}))
        auth = json.loads(await asyncio.wait_for(ws.recv(), 15))
        print(f"[ws] {auth.get('action')}")
        if auth.get("action") != "auth_success":
            sys.exit(f"WS auth failed: {auth.get('message')}")
        await ws.send(json.dumps({"action": "subscribe", "channels": [
            {"name": "order.DERIVATIVE.json", "symbols": []},
            {"name": "position.DERIVATIVE.json", "symbols": []}]}))
        await asyncio.sleep(1.5)

        payload = {"symbol": "41I1G9000", "side": "NB", "orderType": "LO",
                   "quantity": 1, "price": 1300, "loanPackageId": loan_id}
        ost, _ = c.post_order(acct, "DERIVATIVE", payload, tok, order_category="NORMAL")
        print(f"[order] placed -> HTTP {ost}")

        end = time.time() + seconds
        while time.time() < end:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 4))
            except (asyncio.TimeoutError, ValueError):
                continue
            if not isinstance(msg, dict):
                continue
            act = msg.get("action")
            if act == "ping":
                await ws.send(json.dumps({"action": "pong"}))
                continue
            if act in ("subscribed", "error"):
                continue
            order = msg.get("order")
            if isinstance(order, dict) and order.get("orderStatus"):
                order_frames += 1
                print(f"  [ORDER] status={order['orderStatus']} "
                      f"fillQty={order.get('fillQuantity')} qty={order.get('quantity')}")
            elif isinstance(msg.get("position"), dict):
                print("  [POSITION] update received")

    print(f"\n>>> {order_frames} order lifecycle frame(s) on order.DERIVATIVE.json "
          f"-- {'DELIVERY OK' if order_frames else 'NONE'}")
    return 0 if order_frames else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=30)
    return asyncio.run(_run(ap.parse_args().seconds))


if __name__ == "__main__":
    raise SystemExit(main())
