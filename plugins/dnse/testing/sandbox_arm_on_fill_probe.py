#!/usr/bin/env python3
"""#111 SANDBOX INTEGRATION probe — arm-on-fill fires against a REAL sandbox fill.

The venue-level twin of the unit flip test
(``test_025 :: __test_111_arm_protection_on_fill_arms_bracket_at_drain__``): instead of a
MockBroker + injected fill, it drives the REAL ``DNSEBroker`` against the DNSE Sandbox and a
REAL auto-fill, and confirms the engine's ``_arm_protective_exits_after_fill`` path places a
REAL protective exit on the sandbox venue.

Design (per the operator's "lean engine-driven probe" choice):
  * ``arm_protection_on_fill`` is turned on via a PROBE SUBCLASS of DNSEBroker
    (``get_capabilities`` override), NEVER a plugin config flag — the live order path stays
    untouched (rule: no-test-hooks-in-plugin-code).
  * The engine runs with ``event_loop=None`` -> each broker REST call is its own
    ``asyncio.run`` (no background thread, no deadlock). The fill is detected by driving the
    REAL ``watch_orders`` inside a single ``asyncio.run`` (cross-cycle poll state preserved).
  * TP-only exit (``exit(limit=...)`` -> NORMAL LO) so the sandbox accepts it (sandbox rejects
    the conditional OCO category).

NOT window-closing evidence (no mid-bar wake exists; #111 is off-by-default parked infra) —
this proves only that the arm-on-fill CODE PATH executes end-to-end against a real venue fill
and dispatches a real exit. Read-only w.r.t. production; masks identifiers; never prints secrets.

Usage:  python plugins/dnse/testing/sandbox_arm_on_fill_probe.py
"""
from __future__ import annotations
import asyncio
import dataclasses
import json
import sys
import time
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "plugins" / "dnse"))
sys.path.insert(0, str(REPO / "tests" / "t00_pynecore" / "core"))

from pynecore_dnse import broker as dnse_broker  # noqa: E402
from pynecore_dnse.config import DNSEBrokerConfig  # noqa: E402
from pynecore_dnse.client import DNSEClient  # noqa: E402
from pynecore.core.broker.sync_engine import OrderSyncEngine  # noqa: E402
from pynecore.core.broker.position import BrokerPosition  # noqa: E402
# Reuse the unit-test module's order-builder helpers (authoritative order_type values)
# so the probe constructs orders exactly as the flip test does.
import test_025_order_sync_engine as T  # noqa: E402

CONFIG = REPO / "workdir" / "config" / "plugins" / "dnse_sandbox.toml"
SANDBOX_SYMBOL = "41I1G9000"        # the sandbox derivative code (see lifecycle probe)
ENTRY_PX = 1300.0
TP_PX = 1310.0                       # sell TP above entry -> resting NORMAL LO, not marketable
BAR_TS = 1_700_000_000_000


def _mint_token(cfg: dict) -> str:
    tf = REPO / cfg["token_file"]
    if tf.exists():
        tok = json.loads(tf.read_text()).get("trading_token")
        if tok:
            return tok
    c = DNSEClient(cfg["api_key"], cfg["api_secret"], base_url=cfg["base_url"])
    st, body = c.create_trading_token("smart_otp", "666666")   # sandbox OTP = public constant
    tok = (body.get("tradingToken") or body.get("trading-token")) if isinstance(body, dict) else None
    if st != 200 or not tok:
        sys.exit(f"sandbox token mint failed: HTTP {st}")
    tf.parent.mkdir(parents=True, exist_ok=True)
    tf.write_text(json.dumps({"trading_token": tok, "minted_at": int(time.time())}))
    return tok


class _ArmDNSEBroker(dnse_broker.DNSEBroker):
    """Probe-only subclass: report ``arm_protection_on_fill=True`` and pin the sandbox
    contract as a DERIVATIVE.

    ``market_type`` classifies by symbol prefix (only ``VN30F*`` -> DERIVATIVE), so the raw
    sandbox contract code ``41I1G9000`` would default to STOCK — and stock lot-sizing rejects
    qty=1 (``INVALID_ORDER_QUANTITY``). We KNOW it is a derivative on the sandbox, so pin it.
    Neither override touches the live plugin (rule: no-test-hooks-in-plugin-code)."""
    @property
    def market_type(self) -> str:
        return "DERIVATIVE"

    async def get_position(self, symbol):
        """Report FLAT at reconcile.

        The DNSE Sandbox is a pure order-lifecycle + WS-event simulator with NO
        matching / netting (CLAUDE.md): its ``/positions`` returns accumulated
        ``deals`` that no opposing order can flatten, and the plugin's netting-based
        reader cannot consume that shape. ``get_position`` is only used at the startup
        reconcile to CONFIRM the read view; the arm-on-fill path under test runs off the
        FILL EVENT + in-memory position state (which the sandbox fully supports). So the
        probe treats the account as flat here — an accommodation for the sandbox's absent
        position model, NOT a shortcut on the arm path (entry, fill and exit are all real)."""
        return None

    def get_capabilities(self):
        return dataclasses.replace(super().get_capabilities(),
                                   arm_protection_on_fill=True)


async def _await_entry_fill(broker, pine_id: str, *, timeout_s: float = 25.0):
    """Drive the REAL watch_orders in ONE loop until the entry's fill event arrives."""
    agen = broker.watch_orders()
    deadline = None
    try:
        while True:
            ev = await asyncio.wait_for(agen.__anext__(), timeout_s)
            if ev.event_type in ("filled", "partial") and ev.pine_id == pine_id:
                return ev
    finally:
        await agen.aclose()


def main() -> int:
    # record_fill reads self.equity -> lib._script.initial_capital; stub it exactly as
    # the unit tests do (test_025). No real strategy runs in this engine-driven probe.
    from types import SimpleNamespace
    from pynecore import lib
    lib._script = SimpleNamespace(initial_capital=500_000_000.0)

    if not CONFIG.exists():
        sys.exit(f"sandbox config not found: {CONFIG}")
    sb = tomllib.loads(CONFIG.read_text())
    _mint_token(sb)   # ensure a sandbox trading token exists (OTP 666666)

    cfg = DNSEBrokerConfig(
        api_key=sb["api_key"], api_secret=sb["api_secret"],
        base_url=sb["base_url"], ws_url=sb.get("ws_url", "wss://ws-sb-openapi.dnse.com.vn"),
        token_file=sb["token_file"],
    )
    broker = _ArmDNSEBroker(symbol=SANDBOX_SYMBOL, timeframe="1", config=cfg)
    broker._client = DNSEClient(sb["api_key"], sb["api_secret"], base_url=sb["base_url"])

    caps = broker.get_capabilities()
    print(f"[caps] arm_protection_on_fill={caps.arm_protection_on_fill} "
          f"exit_orders_execute_standalone={caps.exit_orders_execute_standalone}")
    assert caps.arm_protection_on_fill, "probe subclass must report the flag ON"

    pos = BrokerPosition()
    engine = OrderSyncEngine(broker, pos, SANDBOX_SYMBOL, run_tag="arm1",
                             event_loop=None, store_ctx=None, mintick=0.1)
    print(f"[engine] arm flag seen by engine = {engine._arm_protection_on_fill}")

    # Long entry (NORMAL LO) + TP-only protective exit (sell LO above entry -> NORMAL,
    # sandbox-accepted). Negative exit size -> 'sell' (protects a long); #82b defers it
    # until the entry fills, then the arm dispatches it.
    pos.entry_orders["E"] = T._entry_order("E", 1.0, limit=ENTRY_PX)
    pos.exit_orders[("X", "E")] = T._exit_order("E", -1.0, "X", limit=TP_PX)

    # Startup reconcile: adopt the venue's authoritative state + CONFIRM the read
    # view, exactly as the live runner's start_broker() does before the first bar.
    # Without it the engine defers every EntryIntent (may_open_exposure=False) and
    # nothing is placed.
    print("[step] engine.reconcile() -> confirm the broker read view")
    engine.reconcile()

    print("[step] engine.sync -> place entry (exit must be #82b-skipped, no position yet)")
    engine.sync(BAR_TS)
    print(f"[after-sync] active_intents={list(engine.active_intents.keys())} "
          f"order_mapping={ {k: v for k, v in engine.order_mapping.items()} }")
    if "E" not in engine.active_intents:
        print("FAIL: entry was NOT dispatched (deferred or rejected) — cannot test the fill path.")
        return 1

    print("[step] driving REAL watch_orders until the sandbox auto-fills the entry ...")
    fill = asyncio.run(_await_entry_fill(broker, "E"))
    print(f"[fill] entry filled on sandbox: type={fill.event_type} qty={fill.fill_qty} "
          f"price={fill.fill_price} pine_id={fill.pine_id}")

    exits_before = list(engine.active_intents.keys())
    print("[step] engine.on_order_event + apply_async_events -> DRAIN + ARM-ON-FILL")
    engine.on_order_event(fill)
    engine.apply_async_events()

    armed = [k for k in engine.active_intents if k not in exits_before]
    print(f"[after-drain] position.size={pos.size} newly-armed intents={armed}")
    exit_key = None
    for k in engine.active_intents:
        if "\x00" in k or k not in ("E",):   # exit keys are 'X<sep>E'
            if k != "E":
                exit_key = k
    print(f"[result] exit intent registered = {exit_key!r}  "
          f"venue order ids = {engine.order_mapping.get(exit_key) if exit_key else None}")

    ok = bool(exit_key) and bool(engine.order_mapping.get(exit_key))
    print("\n=== VERDICT ===")
    if ok:
        print("PASS: a REAL sandbox fill drove _arm_protective_exits_after_fill, which placed "
              f"a REAL protective exit on the sandbox (order id(s) {engine.order_mapping[exit_key]}).")
    else:
        print("FAIL: the arm-on-fill path did not register/dispatch the protective exit.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
