# #157 stage A — endpoint inventory and the measured offline-coverage baseline

Measured 2026-09-18 in worktree `pynecore-worker4` at `f88c7e43`. Every row below was read
from the tree; nothing here is quoted from the docs mirror or from memory. Re-run the three
greps at the end to reproduce it.

## 1. The REST surface the plugin can reach

> **CORRECTED 2026-09-18** after the #157 panel (reviewer 1, correctness lens). The first
> version of this section said 24 paths and claimed the token-mint endpoints and
> `positions/{id}/close` were "not SDK paths". Both were wrong: a `sort -u` over the whole
> vendored SDK gives **31** unique path literals, and the position and pnl-config endpoints are
> in the SDK (`_vendor/dnse/api/client.py:81`). They all reach the wire through `_request` and
> the `_http` PoolManager, so the in-process seam covers them. The undercount came from a grep
> anchored on three path prefixes rather than on every literal. Re-run:
> `grep -rhoE '"/[^"]*"|f"/[^"]*"' plugins/dnse/pynecore_dnse/_vendor/dnse/ | sort -u | wc -l`

`client.py:84` delegates by `__getattr__` to the vendored SDK, so the plugin's REST surface
IS the SDK's surface. The vendored SDK (11 files under
`plugins/dnse/pynecore_dnse/_vendor/dnse/`) builds **31** unique paths; the prefix-anchored
subset below is the part the first pass found, and stage A's remaining work is to enumerate all
31 against the mirror:

| group | paths |
|---|---|
| account | `/accounts`, `/accounts/{a}/balances`, `/accounts/{a}/ppse`, `/accounts/{a}/loan-packages`, `/accounts/{a}/corporate-action-history` |
| orders | `/accounts/{a}/orders`, `/accounts/{a}/orders/{id}`, `/accounts/{a}/orders/history`, `/accounts/{a}/executions/{order_id}` |
| positions | `/accounts/{a}/positions` (the only path spelled in the plugin itself, `client.py:57`) |
| market | `/market/instruments`, `/market/trading-session`, `/market/working-dates` |
| price | `/price/ohlc`, `/price/{s}/quotes`, `/price/{s}/quotes/latest`, `/price/{s}/trades`, `/price/{s}/trades/latest`, `/price/{s}/trades/volume-profile`, `/price/{s}/close`, `/price/{s}/expected-price`, `/price/{s}/foreign-trading`, `/price/{s}/secdef`, `/price/{index}/market-index` |
| positions (missed by the first pass) | `/positions/{id}`, `/positions/{id}/close`, `/positions/{id}/pnl-configs` |
| registration (missed by the first pass) | `/registration/send-email-otp`, `/registration/trading-token` |
| broker (missed by the first pass) | `/brokers/accounts/care-by` |
| root | `/` |

### The CALLED set — what the fake must serve first

The plugin reaches the SDK by name, so only the methods it actually calls matter for stage C.
Measured with `grep -oE 'client\.[a-z_]+\(' broker.py provider.py`, 16 distinct methods:

| calls | method | serves |
|---|---|---|
| 7 | `get_order_detail` | the poll ladder — the single most exercised endpoint |
| 4 | `get_ohlc` | bars over REST, history and latest |
| 2 each | `put_order`, `get_instruments`, `get_accounts` | amend; catalogue and roll; account discovery |
| 1 each | `post_order`, `cancel_order`, `get_orders`, `get_order_history`, `get_positions`, `get_execution_detail`, `get_balances`, `get_loan_packages`, `get_latest_trade`, `get_expected_price`, `get_security_definition` | the rest of the order and account surface |

Everything else among the 31 exists in the SDK but is not called by the plugin today, so the
fake may answer those with an explicit "not implemented" rather than a plausible fiction — a
fake that invents a response for an uncalled endpoint would pin a shape nothing produces.

**Stage A remaining work:** cross-check these 16 against the mirror page for each, and confirm
the response shape the fake must serve comes from `MEASURED_FACTS.md` where the venue deviates
from its own documentation.

## 2. The WebSocket surface

The plugin calls exactly 7 client methods (`grep '_client\.'`): `connect`, `disconnect`,
`on` (×3), `subscribe_trades`, `subscribe_order_event`, `subscribe_broker_order_event`,
`subscribe_broker_position_event`.

> **CORRECTED 2026-09-18.** This section said the connection path "exists only in the SDK, never
> in the mirror". That is **false**, verified at the source: `sdk-build_websocket.md:65` gives
> `wss://ws-openapi.dnse.com.vn/v1/stream?encoding={encoding}` and `:72-145` the full HMAC
> handshake, and `MEASURED_FACTS.md:124-136` already records that the docs agree. What is true,
> and what the original claim garbled, is narrower: the three *guide* pages give only the BASE
> url, so a probe built from those alone 404s. `CLAUDE.md:327-330` carries the same stale
> sentence and is leader-owned — proposed for correction on the card, not edited here.

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
