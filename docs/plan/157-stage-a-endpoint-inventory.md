# #157 stage A — endpoint inventory and the measured offline-coverage baseline

Measured 2026-09-18 in worktree `pynecore-worker4` at `f88c7e43`. Every row below was read
from the tree; nothing here is quoted from the docs mirror or from memory. Re-run the three
greps at the end to reproduce it.

## 1. The REST surface the plugin can reach

`client.py:84` delegates by `__getattr__` to the vendored SDK, so the plugin's REST surface
IS the SDK's surface. The vendored SDK (11 files under
`plugins/dnse/pynecore_dnse/_vendor/dnse/`) builds these 24 paths:

| group | paths |
|---|---|
| account | `/accounts`, `/accounts/{a}/balances`, `/accounts/{a}/ppse`, `/accounts/{a}/loan-packages`, `/accounts/{a}/corporate-action-history` |
| orders | `/accounts/{a}/orders`, `/accounts/{a}/orders/{id}`, `/accounts/{a}/orders/history`, `/accounts/{a}/executions/{order_id}` |
| positions | `/accounts/{a}/positions` (the only path spelled in the plugin itself, `client.py:57`) |
| market | `/market/instruments`, `/market/trading-session`, `/market/working-dates` |
| price | `/price/ohlc`, `/price/{s}/quotes`, `/price/{s}/quotes/latest`, `/price/{s}/trades`, `/price/{s}/trades/latest`, `/price/{s}/trades/volume-profile`, `/price/{s}/close`, `/price/{s}/expected-price`, `/price/{s}/foreign-trading`, `/price/{s}/secdef`, `/price/{index}/market-index` |

Not in this list and still required, because they are not SDK paths: the token-mint endpoints
(`dnse-2-fa-verification.md` in the mirror) and `POST /accounts/{a}/positions/{id}/close`.
**Stage A action:** confirm each against the mirror and mark which the plugin actually calls
today versus which exist for completeness. The fake serves the called set first.

## 2. The WebSocket surface

The plugin calls exactly 7 client methods (`grep '_client\.'`): `connect`, `disconnect`,
`on` (×3), `subscribe_trades`, `subscribe_order_event`, `subscribe_broker_order_event`,
`subscribe_broker_position_event`. The connection path exists only in the SDK
(`_vendor/dnse/websocket/client.py`), never in the mirror, so the fake's WS contract must be
read from the SDK.

## 3. The measured baseline: what the two existing offline seams actually cover

This is the gap the card asserts. Measured, not assumed:

| seam | what it is | covers | does NOT cover |
|---|---|---|---|
| `plugins/dnse/tests/conftest.py` `_FakeClient` | `__getattr__` stub returning canned `(status, body)`, default `(200, {})`, injected as `broker._client` | every method name, at the WRAPPER level, per test | the wire (no HTTP, no URL, no headers), cross-call state, id allocation, WS frames, activation, fills, time |
| `plugins/dnse/pynecore_dnse/replay_sandbox.py` (#114) | real engine + replayed bar fixture, orders routed to the REAL DNSE sandbox | the full engine loop for NORMAL orders, deterministically on the data side | conditionals — its own docstring (lines 16–18) records that the sandbox rejects STOP/OCO and has no price simulation; also needs the network, and stubs `get_position` flat |

**Conclusion, stated as the gap:** of the 24 REST paths, **zero** are served by an offline
wire-level server holding venue state, and the conditional lifecycle (rest → `Activated` →
NORMAL child via `externalOrderId`), price-driven fills, partial fills, and WS order frames
alongside market data are reachable through **neither** seam. The card's premise is therefore
CONFIRMED, with the correction below.

## 4. Correction to the card body (1b.2 false-positive challenge)

The card says "nothing exercises the real plugin plus the real engine offline". That is too
strong as written: `replay_sandbox.py` already drives the real plugin and the real engine, and
it predates this card (#114). The accurate statement, which the body should carry:

> The engine loop is already exercised for NORMAL orders by the Sandbox Replay E2E, but that
> path needs the network and cannot reach conditional orders or price-driven fills. What has
> no coverage at all is the CONDITIONAL lifecycle, partial fills and WS order events driven by
> replayed market data, offline.

This narrows the card rather than weakening it, and it identifies `replay_sandbox.py` as the
structural precedent the new fake should follow: a separate class reached by its own provider
name, leaving production `dnse_broker` untouched.

## 5. Reproduce

```bash
grep -rhoE '"/(accounts|market|price)[a-zA-Z0-9_{}./-]*"|f"/(accounts|market|price)[a-zA-Z0-9_{}./-]*"' plugins/dnse/pynecore_dnse/_vendor/dnse/ | sort -u
grep -rhoE '_client\.[a-z_]+\(' plugins/dnse/pynecore_dnse/*.py | sed 's/_client\.//;s/(//' | sort | uniq -c
sed -n '1,20p' plugins/dnse/pynecore_dnse/replay_sandbox.py
```
