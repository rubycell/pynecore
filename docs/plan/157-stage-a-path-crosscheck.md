# #157 stage A — every vendored-SDK REST path cross-checked against the docs mirror

Measured 2026-09-18 in worktree `pynecore-worker4` at `61154d80`. Read-only: no code change,
no venue contact, no GitHub write. Every claim below names the file and line it was read from;
where the tree cannot settle a question the row says UNDETERMINED and what would settle it.

This file completes the "stage A remaining work" left open at the end of
`docs/plan/157-stage-a-endpoint-inventory.md:52-54` (enumerate all 31 against the mirror). It
does not repeat that file's seam analysis.

## 0. Counts

| question | answer | how measured |
|---|---|---|
| unique path literals in the vendored SDK | **31** (matches the brief) | `grep -rhoE '"/[^"]*"\|f"/[^"]*"' plugins/dnse/pynecore_dnse/_vendor/dnse/ \| sort -u \| wc -l` → 31 |
| of which are request paths | 30 — the 31st is the bare `"/"` in `base_url.rstrip("/")` (`_vendor/dnse/api/client.py:23`), not an endpoint | table §1 |
| mirror files | 53 `.md` (plus 2 OpenAPI yaml, `fetch_docs.py`) | `ls docs/dnse-openapi-documentation/*.md \| wc -l` → 53 |
| request paths with a dedicated endpoint page at the SAME path | 26 | table §1 |
| request paths documented under a DIFFERENT path | 1 — SDK `/market/instruments`, mirror `GET /instruments` | §2 D2 |
| request paths with no endpoint page but named in a guide | 1 — `/brokers/accounts/care-by` (`guide-broker.md:34`; only in the 2026-05-07 spec, absent from 2026-07-23) | table §1 |
| request paths with **NO MIRROR PAGE** | **2** — `/price/{index_name}/market-index`, `/price/{symbol}/trades/volume-profile` | table §1 |
| SDK methods the plugin calls (broker.py + provider.py) | **16** | `grep -rhoE 'client\.[a-z_]+\(' plugins/dnse/pynecore_dnse/broker.py plugins/dnse/pynecore_dnse/provider.py \| sort -u` |
| distinct paths those 16 methods reach | 13 (orders GET+POST share one path; orders/{id} GET+PUT+DELETE share one) | table §1 |
| called methods whose path could not be determined | **0** | every one resolved in `client.py`; `get_positions` resolved through the wrapper override (§1 note) |

The 2026-08-04 prod capture `plugins/dnse/testing/dnse_fixtures.json` (22 entries; recorder
`plugins/dnse/testing/record_fixtures.py:60-94`; committed in `d63aa156`, unchanged in the tree —
`git status --short -- plugins/dnse/testing/dnse_fixtures.json` is empty; body timestamps read
`2026-08-04`) is the only real REST capture in this tree and is cited below as `fixture [n]`
(n = index in that JSON list).

## 1. All 31 literals

Columns: verb(s) the SDK uses; SDK method with its line in
`plugins/dnse/pynecore_dnse/_vendor/dnse/api/client.py`; mirror page (line of the `### VERB /path`
heading); called by the plugin (call-site lines in `plugins/dnse/pynecore_dnse/broker.py` /
`provider.py`).

| # | path literal | verb | SDK method (client.py line) | mirror page | called by plugin? |
|---|---|---|---|---|---|
| 1 | `"/"` | — | not a request path: `self._base_url = base_url.rstrip("/")` (:23) | n/a | n/a |
| 2 | `/accounts` | GET | `get_accounts` (:38) | `dnse-get-accounts.md:8` | YES — broker.py:401, :635 |
| 3 | `/accounts/{account_no}/balances` | GET | `get_balances` (:41) | `dnse-get-account-balances.md:8` | YES — broker.py:3312 |
| 4 | `/accounts/{account_no}/corporate-action-history` | GET | `get_corporate_action_history` (:164) | `dnse-get-corporate-action-history.md:8` | no |
| 5 | `/accounts/{account_no}/executions/{order_id}` | GET | `get_execution_detail` (:127) | `dnse-get-executions.md:8` | YES — broker.py:3184 |
| 6 | `/accounts/{account_no}/loan-packages` | GET | `get_loan_packages` (:44) | `dnse-get-loan-packages.md:8` | YES — broker.py:1393 |
| 7 | `/accounts/{account_no}/orders` | GET, POST | `get_orders` (:93), `post_order` (:422) | `dnse-get-orders.md:8`, `dnse-post-accounts-account-no-orders.md:8` | YES — get_orders broker.py:2586; post_order broker.py:1525 |
| 8 | `/accounts/{account_no}/orders/history` | GET | `get_order_history` (:138) | `dnse-get-orders-history.md:8` | YES — broker.py:1933 |
| 9 | `/accounts/{account_no}/orders/{order_id}` | GET, PUT, DELETE | `get_order_detail` (:116), `put_order` (:437), `cancel_order` (:460) | `dnse-get-order-detail.md:8`, `dnse-replace-order.md:8`, `dnse-cancel-order.md:8` | YES — get_order_detail broker.py:1604, :1688, :1698, :1899, :2384, :2928, :2969; put_order :2456, :2553; cancel_order :2098 |
| 10 | `/accounts/{account_no}/positions` | GET | `get_positions` (:55) — **overridden** by the wrapper `plugins/dnse/pynecore_dnse/client.py:57-67`, which builds the same path itself via `_sdk._request` and adds `pageSize` | `dnse-get-positions.md:8` | YES — broker.py:3246 (wrapper override; `POSITIONS_PAGE_SIZE = 500`, `page_completeness.py:24`) |
| 11 | `/accounts/{account_no}/ppse` | GET | `get_ppse` (:192) | `dnse-get-ppse.md:8` | no (captured in fixture [9]) |
| 12 | `/brokers/accounts/care-by` | GET | `get_list_care_by` (:402) | **no endpoint page**; named at `guide-broker.md:34`; present in `openapi-spec-2026-05-07.yaml:13252`, absent from `openapi-spec-2026-07-23.yaml` (grep count 0) | no |
| 13 | `/market/trading-session` | GET | `get_lastest_session` (:409) | `dnse-get-session.md:8` | no (fixture [16]) |
| 14 | `/market/working-dates` | GET | `get_working_dates` (:395) | `dnse-get-market-working-dates.md:8` | no (fixture [17]) |
| 15 | `/positions/{position_id}` | GET | `get_position_by_id` (:63) | `dnse-get-positions-position-id.md:8` | no |
| 16 | `/positions/{position_id}/close` | POST | `close_position` (:496) | `dnse-post-positions-position-id-close.md:8` | no |
| 17 | `/positions/{position_id}/pnl-configs` | GET, POST | `get_position_pnl_configs` (:72), `post_position_pnl_configs` (:81) | `dnse-get-positions-position-id-pnl-configs.md:8`, `dnse-post-positions-position-id-pnl-configs.md:8` | no |
| 18 | `/price/{index_name}/market-index` | GET | `get_market_index` (:321) | **NO MIRROR PAGE** — the only `market-index` hits are the WebSocket channel anchor in `guide-market-data-connect.md:77`, `:449`; absent from both yaml specs | no |
| 19 | `/price/{symbol}/close` | GET | `get_close_price` (:384) | `dnse-get-price-symbol-close.md:8` | no (fixture [13]) |
| 20 | `/price/{symbol}/expected-price` | GET | `get_expected_price` (:258) | `dnse-get-expected-price.md:8` | YES — provider.py:413 |
| 21 | `/price/{symbol}/foreign-trading` | GET | `get_foreign_trading` (:300) | `dnse-get-foreign-trading.md:8` | no |
| 22 | `/price/{symbol}/quotes` | GET | `get_quotes` (:279) | `dnse-get-quotes.md:8` | no |
| 23 | `/price/{symbol}/quotes/latest` | GET | `get_latest_quote` (:373) | `dnse-get-latest-quotes.md:8` | no (fixture [14]) |
| 24 | `/price/{symbol}/secdef` | GET | `get_security_definition` (:205) | `dnse-get-symbol-secdef.md:8` | YES — provider.py:384 |
| 25 | `/price/{symbol}/trades` | GET | `get_trades` (:226) | `dnse-get-history-trades.md:8` | no |
| 26 | `/price/{symbol}/trades/latest` | GET | `get_latest_trade` (:362) | `dnse-get-latest-trades.md:8` | YES — broker.py:1006 |
| 27 | `/price/{symbol}/trades/volume-profile` | GET | `get_trades_volume_profile` (:247) | **NO MIRROR PAGE** — zero hits for `volume-profile` / `volume profile` / `volumeProfile` anywhere under `docs/dnse-openapi-documentation/`, yaml included (`_vendor/VENDOR_INFO.txt:5` records this method arrived with the v2.2.0 bump) | no |
| 28 | `/market/instruments` | GET | `get_instruments` (:340) | **documented under a different path**: `dnse-get-instruments.md:8` is `GET /instruments`; both specs list `/instruments` (`openapi-spec-2026-07-23.yaml:7886`, `openapi-spec-2026-05-07.yaml:7634`); the SDK's `/market/instruments` string appears nowhere in the mirror | YES — provider.py:280, :339 (also `tools/naked_watch.py:131`) |
| 29 | `/price/ohlc` | GET | `get_ohlc` (:216) | `dnse-get-ohlc-history.md:8` | YES — provider.py:556; broker.py:893, :976, :1175 |
| 30 | `/registration/send-email-otp` | POST | `send_email_otp` (:489) | `dnse-send-email-otp.md:8` | no (broker/provider); `tools/refresh_token.py:81` |
| 31 | `/registration/trading-token` | POST | `create_trading_token` (:481) | `dnse-2-fa-verification.md:8` | no (broker/provider); `tools/refresh_token.py:364`, `tools/token_status.py:148`, `testing/sandbox_e2e_runner.py:123` |

Method-to-path map for the 16 called methods (all resolved): `get_accounts`→#2,
`get_balances`→#3, `get_execution_detail`→#5, `get_loan_packages`→#6, `get_orders`/`post_order`→#7,
`get_order_history`→#8, `get_order_detail`/`put_order`/`cancel_order`→#9, `get_positions`→#10
(wrapper override), `get_expected_price`→#20, `get_security_definition`→#24, `get_latest_trade`→#26,
`get_instruments`→#28, `get_ohlc`→#29.

Transport facts common to every path (the fake's front door): every request goes through
`_request` (`client.py:508`) → `_build_url` (`:559`, `base_url + path + urlencode(query)`), with
headers `<date header>`, `X-Signature`, `x-api-key`, `version` (`:515-520`), `Content-Type:
application/json` when a body is sent (`:522-523`), and `trading-token` added by the write methods
(`:423`, `:447`, `:469`, `:82`, `:497`). The plugin pins `version` to `"2026-07-23"`
(`plugins/dnse/pynecore_dnse/client.py:32`; SDK default `common.py:11` is the same) and swaps the
SDK's non-verifying pool for a certifi-verifying one (`client.py:51-56`).

## 2. Divergences: measured vs documented (called paths only)

Authority order used here: `docs/dnse-openapi-documentation/MEASURED_FACTS.md` first; then the
other measured records in this tree (`plugins/dnse/testing/live_test/README.md` §"Measured venue
facts" lines 385-482, the tracked `CLAUDE.md`, `dnse_fixtures.json`, code comments that name a
measurement). Where two of OUR records disagree and `MEASURED_FACTS.md` is silent, the row says so
and marks the point UNDETERMINED rather than picking one. Vendor pages are the "documented" side.

Only three of `MEASURED_FACTS.md`'s four entries touch REST at all (fact 3, `/accounts`); facts
1, 2, 4 are WebSocket and are covered in §3.

### D1 — the signing date header is `Date` on the wire, `X-Aux-Date` in the endpoint docs (all paths)

- Documented: every endpoint page lists `X-Aux-Date` as a required header (e.g.
  `dnse-get-accounts.md:17`, `dnse-post-accounts-account-no-orders.md:27`) and the current spec
  names it `X-Aux-Date` (`openapi-spec-2026-07-23.yaml:21`). The auth guide instead says `Date`
  "or a separately configured header name" (`guide-intro-authentication.md:10`, `:65`) and the
  FAQ says `Date`, RFC1123, ±1 min (`guide-faq.md:99-103`). The vendor documents both names.
- What the SDK sends: `Date` unless `DATE_HEADER` is set (`_vendor/dnse/api/common.py:14-15`);
  the plugin never sets it (`grep -rn 'DATE_HEADER\|X-Aux-Date' plugins/dnse --include='*.py'`
  outside `_vendor` → no hits). The signature string covers `(request-target)` and the lowercased
  header name (`common.py:24-27`).
- Measured: all 22 fixtures were captured through the SDK with `Date` and every well-formed
  query answered 200 (fixture [0]-[5], [7], [9]-[19]); `docs/dnse-openapi-documentation/README.md:66-67`
  records (2026-08-06) that `X-Aux-Date` and `Date` behave the same on the conditional endpoints.
- Fake must: accept `Date` (and `X-Aux-Date`) as the signed date header and verify the signature
  over whichever name the client used.

### D2 — `/market/instruments` (SDK) vs `/instruments` (docs, both specs, real capture)

- Documented: `GET /instruments` (`dnse-get-instruments.md:8`; specs as in §1 row 28).
  Response `data[] / total / page / pageSize`, `page` is 1-based (`:163-182`, `:201`).
- Measured: `GET /instruments?limit=20` → 200, `total: 3217`, rows carrying `symbolType`
  (`41I1G8000`=VN30F1M, `41I1G9000`=VN30F2M on 2026-08-04) — fixture [10], recorded by
  `record_fixtures.py:75`. The response shape equals the doc.
- The SDK path `/market/instruments` (`client.py:356`) is what the plugin actually sends
  (`provider.py:280`, `:339`; `probe_113_roll_cache.py:96`). `CLAUDE.md:538` states it answered on
  prod 2026-09-12 with the expected `symbolType` rows, and `test_gtd_expiry_clamp_starved.py:5`
  relies on it, but **no log artifact in this tree shows the SDK path answering** (a `resolv`
  grep over `live_test/logs/l0_118_vn30f1m_0730z.log` and `l2b_fill_133954.log` returned
  nothing).
- UNDETERMINED: whether the venue serves both paths as one resource or `/market/instruments` is an
  undocumented alias. Settled by one read-only GET of each with the same query (no writes). Until
  then the fake should serve BOTH paths from the same handler.

### D3 — `GET /accounts/{a}/orders`: required paging params are optional on the wire; unknown `orderCategory` is accepted; and our own records disagree about the category filter

- Documented: `pageIndex` and `pageSize` REQUIRED (`dnse-get-orders.md:24-25`); `orderCategory`
  REQUIRED with values NORMAL/STOP/OCO (`:23`, `:39-42`); response `orders[] / pageIndex /
  pageSize / totalPages / totalRecords` (`:161-189`).
- Measured (fixture [4]): with NO page params the venue answers 200 and the body is only
  `{"orders": []}` — no paging meta at all. The plugin always sends `page_index`/`page_size=100`
  and reads `totalPages` (`broker.py:2586-2597`), so the fake must include the meta whenever page
  params are sent and may omit it when they are not.
- Measured (fixture [5]): `orderCategory=CONDITIONAL` — a value the doc does not list — answers
  200 `{"orders": []}`, not 400. (Empty book only; the fake should not 400 an unknown category.)
- Conflict in our records, `MEASURED_FACTS.md` silent: `docs/dnse-openapi-documentation/README.md:70-73`
  (2026-08-06) says the list "silently ignores the `orderCategory` filter (NORMAL/STOP/OCO all
  return the same NORMAL rows)". The plugin's design scans the NORMAL and STOP books as distinct
  books (`broker.py:190-193`) and `MEASURED_FACTS.md:27-32` (2026-09-16) describes a STOP order
  becoming "book-visible" on placement. UNDETERMINED which is current (the 08-06 note predates the
  `version: 2026-07-23` pin at `client.py:32`). The fake must serve distinct rows per
  `orderCategory` because that is what the plugin depends on; a capture of
  `GET orders?orderCategory=STOP` while a STOP rests settles it. Conditional rows carry string
  ids and `stopPrice`/`conditionOperator`/`durationType`/`durationDateTime` (`dnse-get-orders.md:227-264`),
  OCO rows add `stopOrderPrice` (`:290`).

### D4 — `GET /accounts/{a}/orders/history`: undocumented paging that the venue honours; NORMAL-book only; a 400 code the docs never list

- Documented: params `marketType`, `from`, `to` only, no paging (`dnse-get-orders-history.md:16-23`);
  yet the response carries `total/start/end` (`:157-159`, `:191-193`); ids are date-prefixed
  strings (`"20260312_241"`, `:163`); covers "lệnh thường" = normal orders (`:10`).
- SDK sends `pageSize`/`pageIndex` when given (`client.py:153-156`); the plugin sends
  `page_size=200`, `page_index=n` and drains until `len(rows) >= total` (`broker.py:1933-1945`).
- Measured: pagination honoured — "page_index yields distinct rows, total stable across pages"
  (`live_test/README.md:411-417`); NORMAL-book only — 276/276 rows numeric ids over 17 trading
  days full of placed-and-cancelled conditionals (`README.md:411-414`); closed-session day books
  answer `200, rows=0, totalPages=0` (`:418-419`).
- Measured: omitting `from` → HTTP 400 `{"status": 400, "code": "INVALID_INPUT", "message":
  "from is required"}` (fixture [6]). `INVALID_INPUT` is not in `guide-error_codes.md` (grep → no
  hit); the page's own 400 example uses `OA-003` (`dnse-get-orders.md:305-309`). The error
  envelope shape `{status, code, message}` matches `guide-error_codes.md:30-35`.
- Fake must: page on `pageSize`/`pageIndex`, never return a conditional string id here, answer the
  `INVALID_INPUT` body when `from` is absent.

### D5 — `GET /accounts/{a}/orders/{id}`: conditional string ids on an "integer" path; an undocumented field the plugin cannot live without; stale reads

- Documented: NORMAL only (`dnse-get-order-detail.md:10`), `orderId` path typed **integer**
  (`:23`; same on the cancel page `dnse-cancel-order.md:30`), `version` REQUIRED (`:21`); `error`
  = "error code if rejected" (`:205`).
- Measured: the same endpoint is read for conditional STRING ids with `orderCategory=STOP|OCO`
  — the cancel page itself shows one (`"id": "d9ol9kq0cvks72pfqiug"`, `dnse-cancel-order.md:195`)
  despite typing the param integer. Unknown numeric id → HTTP **400** `{"status":400,"code":
  "RESOURCE_NOT_FOUND","message":"OrderRepository: cannot find object with id: 999999999"}`
  (fixture [20]; code documented at `guide-error_codes.md:102`). Cross-day numeric ids do not
  resolve on the cancel endpoint either (`CLAUDE.md:209-210`). NOTE the HTTP status conflict in our
  records: `live_test/README.md:469-470` says a cancel against the wrong book "404s
  (`RESOURCE_NOT_FOUND`)" while the detail capture is 400 — different verb; UNDETERMINED whether
  DELETE really answers 404; a capture of that reject settles it.
- Measured, **undocumented anywhere in the mirror** (grep `externalOrderId` → no hit): an
  `Activated` conditional's DETAIL carries `externalOrderId` = the NORMAL-book child that does the
  work (`CLAUDE.md:226-229`; `broker.py:1593` "Only the DETAIL carries externalOrderId (the list
  view omits it)"; live evidence
  `plugins/dnse/testing/live_test/logs/t15_t16_t17_evidence_20260818.txt`: "externalOrderId=437346
  (child on the NORMAL book) -> FILLED"). The conditional detail carries
  orderStatus/side/quantity/price/stopPrice/createdDate/modifiedDate/externalOrderId/orderCategory
  and **no duration field** (`broker.py:108-111`).
- Measured vs `:205`: a `Rejected` detail carries **no reject reason** (`CLAUDE.md:210-212`) —
  the documented `error` field does not explain a venue reject.
- Measured: detail reads can be stale / non-monotonic — a Canceled order served as `New` ~10 s
  later (`live_test/README.md:472-474`).
- Status spelling: the doc's conditional enum says `Cancelled` while its NORMAL enum and its own
  conditional example say `Canceled` (`dnse-get-orders.md:332`; `dnse-cancel-order.md:200`,
  `:233`). The plugin maps both (`broker.py:169`). UNDETERMINED which the venue emits for
  conditionals (no conditional row in the fixtures).
- JSON type of NORMAL `id`: doc examples show a string on POST (`"id": "1631"`,
  `dnse-post-accounts-account-no-orders.md:224`) and an integer on GET/PUT (`141`,
  `dnse-get-orders.md:164`; `1626`, `dnse-replace-order.md:187`). The plugin compares as strings
  (`broker.py:2487`). UNDETERMINED per endpoint on the live venue (the fixtures hold no order row).

### D6 — `PUT /accounts/{a}/orders/{id}`: documented semantics hold; the reject codes are not documented

- Documented: stock amend = cancel old + place new, price and quantity together allowed
  (`dnse-replace-order.md:12`, `:41`); derivative: price OR quantity, new quantity > filled
  (`:14`, `:42`, `:45`); response id typed integer (`:211`).
- Measured, agrees: stock PUT answers 200 with a **new id** and the old id reads back `Canceled`
  (`CLAUDE.md:195-201`; the plugin re-maps its tracked id from the PUT body, `broker.py:2474-2499`).
  Derivative: one field per call, otherwise `400 INVALID_INPUT "Only allow edit order quantity or
  price"`; the payload must still carry BOTH keys or `400 EDIT_ORDER_QUANTITY_NOT_ENOUGH`
  (`live_test/README.md:441-448`; `broker.py:2394`). Both codes absent from `guide-error_codes.md`.
  PUT on a done order → `ORDER_IS_DONE` (`CLAUDE.md:203-204`; documented `guide-error_codes.md:92`).
  Conditional amend → HTTP 500, venue keeps the old level (`README.md:449-450`) — undocumented.
  Sandbox has no PUT at all (`HTTP-405`, `CLAUDE.md` sandbox section) — sandbox only.

### D7 — `DELETE /accounts/{a}/orders/{id}`: a 2xx is an acknowledgement, and the reject codes differ per book

- Documented: response is the order record with `PendingCancel` (NORMAL, `dnse-cancel-order.md:173`)
  or `Canceled` (conditional, `:200`); cancellable states listed at `:14-16`; `version`
  REQUIRED (`:28`).
- Measured: the 2xx is an ACK — the venue can still read `New` >12 s after `200 OK`
  (`live_test/README.md:424-425`); the plugin re-reads until terminal (`broker.py:2101-2104`)
  and accepts 200 or 204 (`:2101`; the sandbox answers 204, `CLAUDE.md` sandbox section).
- Measured reject codes: second cancel of a NORMAL order → `ORDER_CANCEL_STATUS_REJECTED`
  (`README.md:234`, `CLAUDE.md:204`) — **not** in `guide-error_codes.md` (it lists
  `ORDER_STATUS_REJECTED`, rows at `:85-145`); cancel of an already-Canceled CONDITIONAL is
  idempotent 200 + record, not a reject (`README.md:474-475`); cancel racing `PendingNew` →
  `CAN_NOT_CANCEL_PENDINGNEW_ORDER_IN_OPEN_SESSION`, retryable (`CLAUDE.md:205`; documented
  `guide-error_codes.md:95`); cancel of an `Activated` OCO umbrella → `CO-ORD-013` ("order status
  is not new" / "order is done", `CLAUDE.md:240`, `:248`, `:257`) — code absent from the mirror;
  ATC phase (14:30-14:45 ICT) refuses cancels (`README.md:453-456`) — not on the endpoint page;
  `400 INVALID_TRADING_TOKEN` on conditional writes with a VALID token after the operator's app
  trades (`CLAUDE.md:354-358`) — the code is documented with a different meaning
  (`guide-error_codes.md:133`).

### D8 — `POST /accounts/{a}/orders`: status codes as documented; the OCO birth state and the GTD ceiling are not

- Documented: 200 for NORMAL with `PendingNew`, 201 for STOP/OCO with `New`
  (`dnse-post-accounts-account-no-orders.md:315`, `:326`, `:334`, `:347`); STOP needs
  `stopPrice`+`conditionOperator`+`durationType=GTD`+`durationDateTime` (`:37-41`, `:72-74`);
  OCO is `DAY` (`:74`).
- Measured, agrees: a GTD stop answers 201 and rests `New` (`docs/dnse-openapi-documentation/README.md:58-59`).
- Measured, contradicts `:326`/`:347`: an OCO umbrella is `Activated` at the FIRST read with a
  NORMAL-book TP child already spawned (`CLAUDE.md:236-241`), and `Activated` is terminal for the
  conditional row itself (`CLAUDE.md:244-257`).
- Measured, undocumented: `durationDateTime` past the ceiling → `CO-ORD-006 Validate Order
  Failed` (`live_test/README.md:451-452`; `broker.py:1230`, `:1264`, `:1298`); the ceiling on
  expiry day is 14:30 ICT = 07:30Z, derived from the operator's app display, not read back from
  the venue (`broker.py:100-111`); a rejected-at-entry STOCK buy answers HTTP 200 then flips
  `Rejected` in ~3 ms with `canceledQuantity=qty` (`CLAUDE.md:213-214`); session phases gate
  placement (`README.md:453-456`); sandbox rejects the category with
  `400 UNSUPPORTED_ORDER_CATEGORY` (`CLAUDE.md` sandbox section) — code absent from the mirror.

### D9 — `GET /accounts/{a}/positions`: measured agrees with the doc; the wrapper adds a param the SDK lacks

- Documented: `pageSize` optional (`dnse-get-positions.md:17`); paging meta "derivatives only"
  (`:199-202`); `side` NB/NS (`:188`); `status` OPEN/PENDING_CLOSE/CLOSED/ODD_LOT (`:186`).
- Measured, agrees: DERIVATIVE without `pageSize` → `pageSize: 50`, `total: 0` (fixture [2]);
  STOCK → `positions` only, no meta (fixture [3]). The wrapper sends `pageSize=500`
  (`client.py:57-67`, `page_completeness.py:24`) and refuses to conclude when `len(rows) < total`
  (`broker.py:3258-3261`); an absent `total` (STOCK) never infers truncation (`:3259-3260`).
- Measured, undocumented: a single read can lag the truth — 0 while the position was 1
  (`CLAUDE.md:184`, #124-OBS; two-read rule at `:185-188`). The fake should be able to inject one stale
  read.
- Sandbox only: the key is `deals`, not `positions` (`CLAUDE.md:518`).

### D10 — `GET /accounts/{a}/executions/{id}`: what the venue answers for a conditional id is unmeasured

- Documented: NORMAL orders only (`dnse-get-executions.md:10`); `orderCategory` REQUIRED (`:17`);
  response = order record + `reports[]` (`:149-308`).
- Plugin passes the TRACKED category, which may be STOP/OCO (`broker.py:3180-3186`), degrades to
  the cumulative VWAP on any non-200 and holds a 60 s cooldown after 429 (`:3193-3200`).
- UNDETERMINED: no prod capture of this endpoint exists in the tree (no fixture; grep of
  `live_test/logs` for `executions` → no hit); the sandbox answers 404 (`CLAUDE.md:525-526`).
  A capture of one real fill's executions read (NORMAL id, then a conditional child id) settles it.

### D11 — `GET /accounts/{a}/loan-packages`: `symbol` is required for STOCK only, optional for DERIVATIVE

- Documented: `symbol` REQUIRED (`dnse-get-loan-packages.md:26`); `initialRate` typed integer
  (`:301`); derivative example answers `symbolType` in place of `symbol` (`:189`).
- Measured: DERIVATIVE without `symbol` → 200, `symbolType: "VN30F1M"`, `initialRate: 0.1848`
  (fixture [7]); STOCK without `symbol` → `400 INVALID_INPUT "symbol is required"` (fixture [8]).
  The plugin calls without a symbol (`broker.py:1393`) and takes `loanPackages[0].id` (`:1396`),
  so a STOCK run would 400 here — an observation, not this card's fix.

### D12 — `GET /accounts/{a}/balances`: no divergence found

Measured shape (fixture [1]) equals the doc (`dnse-get-account-balances.md:137-163`); the plugin
reads `derivative.remainSecure` else `stock.availableCash` (`broker.py:3316-3319`).

### D13 — `GET /accounts`: `investorId` is top-level (doc example agrees; the schema table misleads)

`MEASURED_FACTS.md:97-108` (fact 3) — the live body has `investorId` beside `accounts`, not
inside `accounts[0]`; the doc's example already shows this (`dnse-get-accounts.md:139`) while its
flat schema table (`:159-167`) is easy to misread. Fixture [0] agrees. Not a venue-vs-doc
divergence; recorded because the fake must NOT nest it.

### D14 — `GET /price/{s}/secdef`: an undocumented status value, multi-board lists, and a field that comes and goes

- Documented: `securityStatus` HALT / NO_HALT (`dnse-get-symbol-secdef.md:185`); example is a
  one-row list (`:148-167`); `finalTradeDate` typed `any`, example `null` (`:189`, `:163`).
- Measured: `securityStatus: "UNSPECIFIED"` on both a derivative and HPG (fixtures [11], [12]);
  the list has 2 rows for `41I1G8000` and 7 for `HPG`, and HPG's FIRST row is board `T6`, not
  `G1` (fixture [12]) — the plugin takes `body[0]` (`provider.py:385-386`), so row order matters
  and the fake must reproduce multi-board lists. `finalTradeDate`: `null` on 2026-08-04 for the
  expiring front month (fixture [11]), ISO `2026-08-20` on 2026-08-14, `None` again on 2026-09-14
  (`plugins/dnse/tests/test_gtd_expiry_clamp_starved.py:3-6`); `CLAUDE.md:553-555` adds the
  documented compact form `20260416`. `expiry.parse_venue_date` (`expiry.py:110`) accepts both;
  the fake must be able to serve all three (null / ISO / compact).

### D15 — `GET /price/ohlc`: shape as documented; two venue defects and an error code the docs never list

- Documented: `type` optional (`dnse-get-ohlc-history.md:17`), values STOCK/DERIVATIVE/INDEX
  (`:28-31`), resolutions `1,3,5,15,30,1h,1D,1W` (`:34`); arrays `t/o/h/l/c/v` + `nextTime`
  (`:153-185`).
- Measured, agrees: fixtures [18], [19] (34 and 22 bars, `nextTime: 0`).
- Measured, undocumented: `type=STOCK` on an index → 400; indices need `type=INDEX`
  (`CLAUDE.md:561`; provider routes via `_INDEX_SYMBOLS`, `provider.py:74`, `:448`); invalid symbol
  → `400 {"status":400,"code":"BAD_REQUEST","message":"invalid symbol"}` (fixture [21]) —
  `BAD_REQUEST` absent from `guide-error_codes.md`; the 14:45 INDEX bar publishes C outside
  `[L,H]` with `O==H==L` (`CLAUDE.md:588-590`; repaired from `provider.py:571` on); no bars during ATC
  and the session-final candle is withheld until the 14:45 auction print (+903 s measured,
  `live_test/README.md:256`, Live-L4-T03).

### D16 — `GET /price/{s}/trades/latest`: shape as documented; 429 body unmeasured

Fixture [15] equals the doc (`dnse-get-latest-trades.md:148-167`), several board rows per symbol.
The plugin keeps only `_tick_board` rows and excludes T1 (`broker.py:1003-1004`) and has hit 429
here (`:1007-1012`). `guide-ratelimits.md` gives counts only; the 429 body shape is UNDETERMINED
(no capture).

### D17 — `GET /price/{s}/expected-price`: the plugin omits two required params

Documented `from`/`to` REQUIRED (`dnse-get-expected-price.md:15-16`); the SDK sends them only if
given (`client.py:262-265`) and the plugin calls with the symbol alone (`provider.py:413`).
UNDETERMINED what the venue answers without them (no fixture, no log). One read-only GET settles it.

## 3. No documentation, capture required

Things the fake will need that the mirror does not describe, so they can only be built from a real
capture. The right-hand column names the only evidence in this tree today.

| # | needed behaviour | documented? | evidence in tree |
|---|---|---|---|
| C1 | `externalOrderId` on an `Activated` conditional detail; the child's id shape; the conditional detail's field set (no duration field) | no (`grep externalOrderId` mirror → none) | `t15_t16_t17_evidence_20260818.txt` one line; `broker.py:108-111`, `:1593`; `CLAUDE.md:226-235` |
| C2 | OCO umbrella `Activated` from birth with a live TP child; `CO-ORD-013` body on cancelling it; umbrella row after the child dies (`Activated`, populated `stopPrice`, `modifiedDate` = child cancel instant) | no (`CO-ORD-013` absent) | `CLAUDE.md:236-257` (operator-observed 2026-09-15 / 09-18, ids `damadq2vfqkc7397o0tg`) — no raw capture file |
| C3 | `CO-ORD-006` body; the 07:30Z GTD ceiling on expiry day | no | `broker.py:100-111`, `:1264`, `:1298`; `live_test/README.md:451-452`; `logs/l0_118_*` (raw logs, not parsed here) |
| C4 | cancel ACK lag (`New` >12 s after 200); stale/non-monotonic detail (`Canceled`→`New`); idempotent conditional cancel vs `ORDER_CANCEL_STATUS_REJECTED` on NORMAL | no | `live_test/README.md:424-425`, `:472-475`, `:234` |
| C5 | error bodies for `INVALID_INPUT` (two variants), `EDIT_ORDER_QUANTITY_NOT_ENOUGH`, `BAD_REQUEST`, `ORDER_CANCEL_STATUS_REJECTED`, `UNSUPPORTED_ORDER_CATEGORY`, `CO-ORD-*` | none of these codes are in `guide-error_codes.md` | real bodies exist ONLY for `INVALID_INPUT` (fixtures [6], [8]), `RESOURCE_NOT_FOUND` ([20]), `BAD_REQUEST` ([21]); the rest are quoted in prose only |
| C6 | executions read for a conditional id (or its child) on prod | doc says NORMAL only | none (D10) |
| C7 | `GET orders?orderCategory=STOP` with a resting STOP: rows or not (the D3 conflict) | doc: filter exists | conflicting prose only |
| C8 | placement/cancel rejects in ATC and closed phases (which code, which HTTP status) | no | `live_test/README.md:453-456` prose |
| C9 | DAY-order `Expired` arrives as a deferred batch ~15:04, not at 14:45 | no | `live_test/README.md:465-468` |
| C10 | position replica lag (a 0 read while the position is 1) | no | `CLAUDE.md:184`; no capture file |
| C11 | JSON type of `id` per endpoint on NORMAL records; `Canceled` vs `Cancelled` on conditionals | doc self-inconsistent (D5) | none — fixtures hold no order rows |
| C12 | `/market/instruments` answering on prod (vs `/instruments`) | no (D2) | `CLAUDE.md:538` prose |
| C13 | 429 bodies on `/trades/latest` and `/executions` | counts only (`guide-ratelimits.md`) | none |
| C14 | `expected-price` without `from`/`to` (D17) | doc says required | none |
| C15 | session-final candle withheld until the 14:45 print; no bars during ATC | no | `live_test/README.md:256` (Live-L4-T03 summary) |

### WebSocket — the connection/auth path IS in the mirror (the brief's expectation does not hold)

The brief expected the WS connect/auth path to be absent from the mirror. It is present:

- Full connect URL `wss://ws-openapi.dnse.com.vn/v1/stream?encoding={encoding}` —
  `sdk-build_websocket.md:65` (base URL `:42`; also `:406`). The vendored client builds the same
  string (`_vendor/dnse/websocket/client.py:80`, `:138`).
- Handshake — `welcome` frame with "Please authenticate within 30 seconds" (`:72-85`,
  `AUTH_TIMEOUT` on miss), HMAC-SHA256 over `"{api_key}:{timestamp}:{nonce}"` (`:89-105`),
  ±5 min timestamp window and 10-min nonce replay guard (`:108-109`), `auth` message shape
  (`:115-119`), `auth_success` (`:129`), `AUTH_FAILED` (`:139-145`), "server does not accept
  subscribe before auth" (`:91`). The vendored `AuthManager` matches
  (`_vendor/dnse/websocket/auth.py:25-53`).
- `MEASURED_FACTS.md:124-136` (fact 4) says the same and already names `sdk-build_websocket.md`
  as where the full path lives.

So three of OUR records are stale on this point and should be corrected in a later stage, not
here: `CLAUDE.md:327-330` ("exists ONLY in the SDK, not the docs"),
`docs/plan/157-stage-a-endpoint-inventory.md:60-62` ("never in the mirror"), and the brief.
The guide pages that give only the base URL (`guide-market-data-connect.md:12`,
`guide-market-data-trading_connect.md:14`, `guide-market-data-broker_connect.md:10`) are what
produced the false "404" verdict.

What the WS side genuinely lacks documentation for (capture-required, all from `MEASURED_FACTS.md`
and `CLAUDE.md`):

| # | needed behaviour | documented? | evidence |
|---|---|---|---|
| W1 | order stream carries NORMAL-book events only; nothing for a conditional id on place/cancel/`Activated`; a triggered conditional's CHILD does stream | no — `guide-market-data-trading_connect.md` presents the channel as all order changes | `MEASURED_FACTS.md:17-55`; logs `ws_book_disc_112233.log`, `ws_book_disc_t18_112233.log` |
| W2 | `order.broker.{mt}.{investorId}.{encoding}` refused on a retail account with `{"action":"error","code":"SUBSCRIBE_FAILED","message":"internal error"}`; reason unmeasured | no (doc gives a different refusal text) | `MEASURED_FACTS.md:59-93` |
| W3 | lowercase market type in a channel name is accepted (`status: active`) and streams nothing | no | `MEASURED_FACTS.md:81-84` |
| W4 | frame envelope: the order nested under `msg["order"]` with `T:"do"` (positions `T:"dp"`) — the guide shows a FLAT order object (`guide-market-data-trading_connect.md:91-105`) | no | `CLAUDE.md:480` (sandbox capture) and the prod `do` frames of `MEASURED_FACTS.md:30` — UNDETERMINED whether prod and sandbox envelopes are identical; a prod frame file settles it |

Record conflict to resolve in a later stage: `live_test/README.md:400-409` says prod REQUIRES
`subscribe_broker_order_event`; `MEASURED_FACTS.md:59-93` and `CLAUDE.md:346-352` say the SHORT
channel works on prod and the broker channel is refused. Per this card's rule `MEASURED_FACTS.md`
wins; the README paragraph is stale.

### Stale statements in our own records found on the way (not vendor errors)

- `docs/dnse-openapi-documentation/README.md:37`, `:43` — "neither SDK uses" the account-scoped
  order endpoints. The vendored SDK does (`client.py:429`, `:453`, `:475`), and it is v2.2.0
  (`_vendor/VENDOR_INFO.txt:4-5`), not the v2.0.0 that `CLAUDE.md:160` names.
- `docs/dnse-openapi-documentation/README.md:63-75` (2026-08-06) — STOP detail/cancel answer 500
  and native STOP is "place-only". Later measurements read and cancel conditional ids routinely
  (`MEASURED_FACTS.md:27-32`; D5, D7). Nothing in `MEASURED_FACTS.md` retires the 08-06 text.

## 4. Reproduce

```bash
cd /home/mike/workspace/github/pynecore-worker4

# 1. the 31 literals (expect 31)
grep -rhoE '"/[^"]*"|f"/[^"]*"' plugins/dnse/pynecore_dnse/_vendor/dnse/ | sort -u | wc -l
grep -rhoE '"/[^"]*"|f"/[^"]*"' plugins/dnse/pynecore_dnse/_vendor/dnse/ | sort -u

# 2. the 16 called methods (expect 16)
grep -rhoE 'client\.[a-z_]+\(' plugins/dnse/pynecore_dnse/broker.py plugins/dnse/pynecore_dnse/provider.py | sort -u | wc -l

# 3. SDK method -> path (defs, paths and verbs in one listing)
grep -nE '^\s*def |"/|"(GET|POST|PUT|DELETE)"' plugins/dnse/pynecore_dnse/_vendor/dnse/api/client.py

# 4. the wrapper override that bypasses the SDK's get_positions
sed -n '57,67p' plugins/dnse/pynecore_dnse/client.py

# 5. endpoint heading of every mirror page (match on the path, not the filename)
for f in docs/dnse-openapi-documentation/dnse-*.md; do echo "$(basename $f): $(sed -n '8p' $f)"; done

# 6. the two paths with no mirror page, and the instruments path mismatch
grep -rn 'volume-profile' docs/dnse-openapi-documentation/            # expect nothing
grep -rn 'market-index'   docs/dnse-openapi-documentation/            # expect only the WS-guide anchor
grep -rn 'market/instruments' docs/dnse-openapi-documentation/        # expect nothing
grep -nE '^  /' docs/dnse-openapi-documentation/openapi-spec-2026-07-23.yaml   # the spec's path list

# 7. which measured reject codes the vendor documents
for c in ORDER_IS_DONE ORDER_CANCEL_STATUS_REJECTED CAN_NOT_CANCEL_PENDINGNEW_ORDER_IN_OPEN_SESSION CO-ORD-013 CO-ORD-006 RESOURCE_NOT_FOUND INVALID_TRADING_TOKEN INVALID_INPUT EDIT_ORDER_QUANTITY_NOT_ENOUGH BAD_REQUEST UNSUPPORTED_ORDER_CATEGORY externalOrderId; do
  printf '%s -> ' "$c"; grep -l -- "$c" docs/dnse-openapi-documentation/*.md | grep -v MEASURED_FACTS | xargs -r -n1 basename | tr '\n' ' '; echo
done

# 8. the real capture (statuses, keys, error bodies)
.venv/bin/python -c "
import json; d=json.load(open('plugins/dnse/testing/dnse_fixtures.json'))
for i,e in enumerate(d): print(i, e['method'], e['path'], e['query'], e['status'], e['body'] if e['status']!=200 else sorted(e['body']) if isinstance(e['body'],dict) else type(e['body']).__name__)"
git log -1 --format='%h %ci' -- plugins/dnse/testing/dnse_fixtures.json   # d63aa156

# 9. the WS path in the mirror (the brief expected none)
grep -n 'v1/stream' docs/dnse-openapi-documentation/*.md plugins/dnse/pynecore_dnse/_vendor/dnse/websocket/client.py

# 10. the date header the SDK sends
sed -n '14,15p' plugins/dnse/pynecore_dnse/_vendor/dnse/api/common.py
```
