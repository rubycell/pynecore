#!/usr/bin/env python3
"""#111 SANDBOX INTEGRATION probe — arm-on-fill fires against a REAL sandbox fill.

The venue-level twin of the unit flip test
(``test_025 :: __test_111_arm_protection_on_fill_arms_bracket_at_drain__``): instead of a
MockBroker + injected fill, it drives the REAL ``DNSEBroker`` against the DNSE Sandbox and a
REAL auto-fill, and confirms the engine's ``_arm_protective_exits_after_fill`` path places a
REAL protective exit on the sandbox venue.

Runs BOTH instrument types with their NATURAL symbols (no market_type override):
  * VN30F1M — a DERIVATIVE (qty 1, price ~1300 index points)
  * HPG     — a STOCK      (qty 100 = one board lot, price ~26.x thousand VND)

Design (per the operator's "lean engine-driven probe" choice):
  * ``arm_protection_on_fill`` is turned on via a PROBE SUBCLASS of DNSEBroker
    (``get_capabilities`` override), NEVER a plugin config flag — the live order path stays
    untouched (rule: no-test-hooks-in-plugin-code).
  * The engine runs with ``event_loop=None`` -> each broker REST call is its own
    ``asyncio.run`` (no background thread, no deadlock). The fill is detected by driving the
    REAL ``watch_orders`` inside a single ``asyncio.run`` (cross-cycle poll state preserved).
  * TP-only exit (``exit(limit=...)`` -> NORMAL LO) so the sandbox accepts it (sandbox rejects
    the conditional OCO category).
  * ``get_position`` is stubbed FLAT: the sandbox is a pure order-lifecycle + WS-event
    simulator with NO matching/netting (its ``/positions`` returns accumulating ``deals`` that
    no opposing order flattens, and the netting reader cannot consume that shape).
    ``get_position`` is only the reconcile view-confirm; the arm path runs off the FILL EVENT +
    in-memory position state, which the sandbox fully supports. NOT a shortcut on the arm path
    (entry, fill and exit are all real).

NOT window-closing evidence (no mid-bar wake exists; #111 is off-by-default parked infra) —
this proves only that the arm-on-fill CODE PATH executes end-to-end against a real venue fill
and dispatches a real exit. Read-only w.r.t. production; masks identifiers; never prints secrets.

Usage:  python plugins/dnse/testing/sandbox_arm_on_fill_probe.py [SYMBOL ...]
        (default: VN30F1M HPG)
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
BAR_TS = 1_700_000_000_000

#: One entry per instrument type. entry/tp are on a valid tick; qty is one board unit
#: (derivative = 1 contract, HOSE stock = 100-share lot). A TP above entry rests as a
#: NORMAL sell LO (not marketable) so #82b defers it until the fill, then the arm places it.
#: NOTE the sandbox CATALOG: its VN30 front-month derivative is coded ``41I1G9000`` — the
#: ``VN30F1M`` alias is NOT a sandbox symbol (``SYMBOL_NOT_EXIST``), and ``resolve_contract``
#: passes it through unchanged. So the derivative is tested via ``41I1G9000`` (the sandbox's
#: VN30F1M-equivalent), which needs ``force_mt=DERIVATIVE`` because it does not match the
#: plugin's ``VN30F*`` prefix rule. HPG (stock) uses its natural classification.
INSTRUMENTS = {
    "41I1G9000": dict(kind="derivative (sandbox VN30F1M)", qty=1, entry_px=1300.0,
                      tp_px=1310.0, mintick=0.1, force_mt="DERIVATIVE"),
    "HPG":       dict(kind="stock", qty=100, entry_px=26.5, tp_px=26.7, mintick=0.05),
    "VN30F1M":   dict(kind="derivative alias (NOT a sandbox symbol -> SYMBOL_NOT_EXIST)",
                      qty=1, entry_px=1300.0, tp_px=1310.0, mintick=0.1),
}


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
    """Probe-only subclass: report ``arm_protection_on_fill=True``, stub ``get_position``
    flat, and OPTIONALLY force ``market_type`` (``_forced_mt``, per-instance). By default the
    plugin's NATURAL classification is used (VN30F* -> DERIVATIVE, else STOCK) — only the raw
    sandbox derivative code ``41I1G9000`` needs a force, since it does not match the prefix.
    None of these touch the live plugin (rule: no-test-hooks-in-plugin-code)."""
    _forced_mt = None

    @property
    def market_type(self) -> str:
        return self._forced_mt or super().market_type

    async def get_position(self, symbol):
        return None   # sandbox has no netting position model — see module docstring

    def get_capabilities(self):
        return dataclasses.replace(super().get_capabilities(),
                                   arm_protection_on_fill=True)


async def _await_entry_fill(broker, pine_id: str, *, timeout_s: float = 25.0):
    """Drive the REAL watch_orders in ONE loop until the entry's fill event arrives."""
    agen = broker.watch_orders()
    try:
        while True:
            ev = await asyncio.wait_for(agen.__anext__(), timeout_s)
            if ev.event_type in ("filled", "partial") and ev.pine_id == pine_id:
                return ev
    finally:
        await agen.aclose()


def _run_instrument(sb: dict, symbol: str, spec: dict) -> bool:
    print(f"\n{'='*66}\n=== {symbol}  ({spec['kind']}, qty={spec['qty']}, "
          f"entry={spec['entry_px']} tp={spec['tp_px']}) ===\n{'='*66}")
    cfg = DNSEBrokerConfig(
        api_key=sb["api_key"], api_secret=sb["api_secret"],
        base_url=sb["base_url"], ws_url=sb.get("ws_url", "wss://ws-sb-openapi.dnse.com.vn"),
        token_file=sb["token_file"],
    )
    broker = _ArmDNSEBroker(symbol=symbol, timeframe="1", config=cfg)
    broker._client = DNSEClient(sb["api_key"], sb["api_secret"], base_url=sb["base_url"])
    broker._forced_mt = spec.get("force_mt")
    print(f"[classify] market_type={broker.market_type} resolve_contract={broker.resolve_contract()}"
          + ("  (market_type forced)" if spec.get("force_mt") else "  (natural)"))
    assert broker.get_capabilities().arm_protection_on_fill

    pos = BrokerPosition()
    engine = OrderSyncEngine(broker, pos, symbol, run_tag="arm1",
                             event_loop=None, store_ctx=None, mintick=spec["mintick"])

    pos.entry_orders["E"] = T._entry_order("E", float(spec["qty"]), limit=spec["entry_px"])
    pos.exit_orders[("X", "E")] = T._exit_order("E", -float(spec["qty"]), "X", limit=spec["tp_px"])

    engine.reconcile()                                   # confirm read view (flat)
    engine.sync(BAR_TS)                                  # place the real entry
    if "E" not in engine.active_intents:
        print("FAIL: entry NOT dispatched (deferred / rejected).")
        return False
    entry_ids = engine.order_mapping.get("E")
    print(f"[entry] dispatched -> venue id(s) {entry_ids}")

    try:
        fill = asyncio.run(_await_entry_fill(broker, "E"))
    except (asyncio.TimeoutError, TimeoutError):
        print("FAIL: no fill observed within timeout.")
        return False
    print(f"[fill] filled qty={fill.fill_qty} price={fill.fill_price}")

    before = set(engine.active_intents)
    engine.on_order_event(fill)
    engine.apply_async_events()                          # DRAIN + ARM-ON-FILL
    armed = [k for k in engine.active_intents if k not in before and k != "E"]
    exit_key = armed[0] if armed else None
    exit_ids = engine.order_mapping.get(exit_key) if exit_key else None
    print(f"[arm] position.size={pos.size} exit intent={exit_key!r} -> venue id(s) {exit_ids}")

    ok = bool(exit_ids)
    print(f"[{symbol}] {'PASS' if ok else 'FAIL'}: arm-on-fill "
          f"{'placed a real protective exit' if ok else 'did NOT place the exit'}"
          + (f' (id {exit_ids})' if ok else ''))
    return ok


def main(argv) -> int:
    from types import SimpleNamespace
    from pynecore import lib
    lib._script = SimpleNamespace(initial_capital=500_000_000.0)   # record_fill reads self.equity

    if not CONFIG.exists():
        sys.exit(f"sandbox config not found: {CONFIG}")
    sb = tomllib.loads(CONFIG.read_text())
    _mint_token(sb)

    # Default: the derivative (sandbox code) + the stock. VN30F1M is runnable on request
    # (``... VN30F1M``) but is not in the sandbox catalog — see INSTRUMENTS note.
    symbols = argv[1:] or ["41I1G9000", "HPG"]
    results = {}
    for symbol in symbols:
        spec = INSTRUMENTS.get(symbol)
        if spec is None:
            print(f"[skip] {symbol}: no instrument spec")
            continue
        try:
            results[symbol] = _run_instrument(sb, symbol, spec)
        except Exception as exc:                          # noqa: BLE001
            print(f"[{symbol}] ERROR: {type(exc).__name__}: {exc}")
            results[symbol] = False

    print(f"\n{'='*66}\n=== VERDICT ===")
    for symbol, ok in results.items():
        print(f"  {symbol:10s} {'PASS' if ok else 'FAIL'}")
    return 0 if results and all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
