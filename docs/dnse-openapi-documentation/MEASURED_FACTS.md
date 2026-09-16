# MEASURED FACTS — our live measurements vs. the mirrored vendor docs

**This file is LOCAL. It is never fetched and never overwritten by
`fetch_docs.py`** (that script only writes `<flattened-doc-slug>.md` files
discovered from DNSE's sitemap, plus `openapi-spec-<version>.yaml`; every such
name is lowercase-hyphenated, so `MEASURED_FACTS.md` matches nothing it writes).

Everything in the rest of this directory is the **vendor's claim**, refreshed
verbatim on every sync. This file is **our ground truth where the two differ**.
Each entry states what was MEASURED, when, under which card, and where the
re-runnable evidence lives. Never "fix" a fetched page to match a measurement —
reference the page by filename here instead, so the next sync cannot silently
erase the correction.

---

## 1. The prod order WebSocket streams NORMAL-book events ONLY

**MEASURED 2026-09-16 (cards #130 / #107 / #121).** Conditional-book lifecycles
never appear on the order stream: neither the `orderCategory=STOP`/`OCO`
umbrella nor the conditional record itself — not on place, not on cancel, not on
`Activated`.

**Discriminating test** (this is the point — an earlier capture saw silence, but
every order in it was conditional, so "silent channel" and "conditional events
don't stream" were indistinguishable): one capture window touching **both**
books — a NORMAL LO place → book-visible → cancel, then a conditional STOP
place → book-visible → cancel, ~4 s apart.

Result: **4 `do` frames, all four keyed to the NORMAL order id; zero frames for
the conditional order id**, while the REST record confirms the conditional leg
did place, become visible, and cancel inside the same window.

- Runner: `plugins/dnse/testing/live_test/run_ws_book_discriminator.sh`
  (payload = `direct_probes_t14_t15_t17.py --case t18`)
- Evidence: `plugins/dnse/testing/live_test/logs/ws_book_disc_112233.log`
  (WS side) + `…/logs/ws_book_disc_t18_112233.log` (venue side); an earlier
  run at `…_111914.log` shows the same 4-frames-NORMAL / 0-frames-conditional split.

**Consequences for the plugin**

- Conditional-order state is **POLL-ONLY**. That includes venue-initiated
  cancels of protective exits (our SL/TP legs are conditional orders) — no WS
  event will ever announce them; only the REST poll can see them.
- A **triggered** conditional is not an exception to the rule and not a counter-
  example: the trigger creates a *NORMAL-book child* (see the two-order-books
  note in `README.md`), and that child **does** stream. The parent conditional's
  own `Activated` transition still does not.
- The WS order feed is therefore an additive **latency failsafe over the REST
  poll for NORMAL-book events**, never a replacement for polling.

**Vendor pages this qualifies:** `guide-market-data-trading_connect.md` presents
`order.{market_type}.{encoding}` as "real-time order data … updated whenever an
order on the account changes", with no book distinction. Read that page as
NORMAL-book-only.

---

## 2. The broker order channel is refused on a retail account

**MEASURED 2026-09-16, reproduced twice (card #130).** Subscribing
`order.broker.{market_type}.{investorId}.{encoding}` — with a correct
`investorId` (masked `******5317`), on an authenticated connection whose
`auth_success` arrived normally — answers:

```json
{"action": "error", "code": "SUBSCRIBE_FAILED", "message": "internal error"}
```

On the *same* connection pair, the short channels subscribe fine
(`{"action":"subscribed","channel":"order.DERIVATIVE.json","status":"active"}`),
so this is the channel, not the session.

**Reason UNMEASURED.** The error names no channel and is not a permissions code
— notably it is **not** the documented authorization refusal
(`guide-market-data-broker_connect.md`: "user does not have permissions for
investorId"), so "retail account lacks the broker role" is a plausible reading
but is **not** what the venue actually said. Do not record a cause until one is
measured.

**Working transport:** the SHORT channel `order.{MARKET_TYPE}.{encoding}` —
with the market type **UPPERCASE**. A lowercase name is silently accepted
(`status: active`) and streams nothing; that quirk produced an earlier false
"the trading WS is silent" verdict.

Evidence: `plugins/dnse/testing/live_test/logs/ws_book_disc_112233.log` and
`…/logs/ws_book_disc_111914.log` (both show `[broker] … SUBSCRIBE_FAILED` beside
`[short] … subscribed`, `[broker] rc=0 frames=0` vs `[short] rc=0 frames=4`).

**Vendor page this qualifies:** `guide-market-data-broker_connect.md` documents
the channel as a broker-role feed keyed by an `investor_id` obtained from the
broker-only "Get list careby" endpoint. Our account is not a broker account; the
channel is documented, reachable, and refused.

---

## 3. `/accounts`: `investorId` is TOP-LEVEL, not inside `accounts[0]`

**MEASURED 2026-09-16 (card #129).** The live `GET /accounts` body has:

- **top level:** `accounts`, `custodyCode`, `investorId`, `name`
- **`accounts[i]`:** `dealAccount`, `derivative`, `derivativeAccount`, `id`
  (no `investorId`)

`dnse-get-accounts.md` is ambiguous enough to be misread: its response-schema
table uses one `»` prefix for `name` / `custodyCode` / `investorId` / `accounts`
and `»»` for the array members, but the flat markdown table makes the nesting
easy to lose — and the example body is the only unambiguous statement of it.

**What this cost us:** the plugin's resolver read
`accounts[0].get("investorId")` (`plugins/dnse/pynecore_dnse/broker.py`,
`_resolve_investor_id`) and therefore returned `None` for **every** real body —
so the broker WS channel was never subscribed on any live run, and the plugin
silently ran poll-only while appearing to have a WS failsafe. This is the #129
fix; it is also why fact 2 above needed a probe that parses the body itself
(`plugins/dnse/testing/live_test/probe_ws_market_data.py::resolve_investor_id`
deliberately uses the plugin's *transport* but its own *parse*, and says so).

**Read this as the general lesson:** a `None` behind a `hasattr`/`.get()` guard
is not an answer — it is a question that never ran.

---

## 4. Context — WS connect + auth (agrees with the docs; recorded so it is co-located)

- **Endpoint:** `wss://ws-openapi.dnse.com.vn/v1/stream?encoding=json|msgpack`.
  The guide pages (`guide-market-data-connect.md`,
  `…-trading_connect.md`, `…-broker_connect.md`) give only the **base** URL —
  the full path appears in `sdk-build_websocket.md` and in the vendored SDK
  (`plugins/dnse/pynecore_dnse/_vendor/dnse/websocket/client.py`). A probe built
  from the guide pages alone cannot connect; use the vendored `TradingClient`.
- **Auth is mandatory within 30 s** of the `welcome` frame: an `auth` message
  carrying `HMAC-SHA256(api_secret, "{api_key}:{timestamp}:{nonce}")`.
  **Subscribes are refused before `auth_success`** — a probe that skips or
  fumbles this handshake sees "silent channels" and reports a false verdict
  (this is exactly how the original "the DNSE WS is silent" claim was produced).
- **Empty ≠ conclusive** on event-driven channels (`order.*`, `position.*`):
  zero frames with zero account activity in the window is the EXPECTED result,
  not evidence of silence. Only a capture with a *confirmed* account event
  inside it can grade the channel — which is why fact 1 required both books in
  one window.
