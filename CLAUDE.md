# pynecore — repo notes for Claude

> This file is force-tracked (`git add -f`) past the repo's `.gitignore` so that
> cloud/sandbox sessions — which only receive tracked files — still get these rules.
> Upstream deliberately ignores CLAUDE.md and will never modify it, so rebases don't
> conflict. Keep secrets out: this file is public in the repo.

## If you are running in a CLOUD SANDBOX (claude.ai/code, remote agent)

Only tracked files exist there. Consequences:

- **The pine2pyne transpiler installs from PyPI** — `pip install opencode-pine2pyne`,
  then `python -m pine2pyne in.pine -o out.py`. (Transpiled `.py` files are also
  committed next to every `.pine`.) Still NEVER `pyne compile` / `pyne run x.pine` —
  that is the CLOUD compiler and needs an API key we don't have.
- **OHLCV data IS tracked** in `workdir/data/` (force-added `.ohlcv` + `.toml` pairs:
  dnse VN30F1M @1/3/5/15/1D, dnsebroker @1/3/5, HPG 1D, a ccxt BTC sample, replay
  DEMO). Backtests therefore work offline in FILE MODE — pass the dataset name, e.g.
  `pyne run script.py dnse_VN30F1M_1`. Provider mode (`dnse:VN30F1M@1`) still needs
  API credentials and re-downloads, so avoid it in the sandbox. Note the tracked
  files are snapshots from the dev machine; a provider-mode run there rewrites them.
- **Credentials and the DNSE trading token are absent** → anything live (`--broker`,
  the L0 gate) is IMPOSSIBLE. What works offline: `pytest plugins/dnse/tests/`
  (incl. the fake-venue e2e), transpiling, and the staged probes in backtest/oracle
  mode over the tracked data.
- **The dev machine's memory/notes are absent.** The durable venue facts (cancel-ACK,
  no cascade, amend-500, GTD clamp, session phases) live in
  `plugins/dnse/testing/live_test/README.md` — read it before touching the plugin.

## Conventions (distilled from the dev machine's global rules)

- **Never delete** — move unwanted files to `backup/deleteable/` and untrack
  (`git add -A <path>` to stage the deletion); `backup/` is gitignored.
- **Back up before destructive commands** on files with uncommitted changes.
- **Verify you are testing NEW code** — editable installs must resolve to the repo
  (`python -c "import pynecore; print(pynecore.__file__)"`), not site-packages.
- **Fork policy:** `main` = upstream (PyneSys/pynecore) + a linear fork stack,
  maintained by REBASE, never merge. NEVER use GitHub's "Sync fork"/update-branch
  PR on this repo — it merges (and conflicts); upstream updates go through the
  worktree-rebase flow (scratch worktree, skip superseded fork commits, replay
  the feature branch, full suites vs a pristine-upstream control, land by
  reset + force-with-lease). Dist name is `opencode-pyneruntime`; the import
  stays `pynecore`.
- Tests use `__test_*__` naming (`pytest.ini`); full-suite runs need
  `--ignore=tests/t00_pynecore/ast/test_045_lib_import_normalizer_invalid_alias.py`
  (a `@pyne` file that self-triggers at collection — long-standing upstream)
  AND `--deselect tests/t01_lib/t04_math/test_009_math_log_correctly_rounded.py`
  (pre-existing upstream red since the 6.9.1 sync, 818b63a; pytest.ini carries
  `-x`, so without the deselect the run stops early and incomplete).

## Strategy code is AUTHORED IN `.pine`, never hand-written in Python (CRITICAL)

Any strategy — and any TEST or probe that exercises strategy behaviour — is
written as a **`.pine` file first**, then transpiled with pine2pyne. Do NOT
hand-write the `@pyne` Python, even when it looks quicker and even for a
throwaway probe.

Why: Pine is the source of truth. A hand-written `@pyne` file is a second
dialect nobody validates — it can use idioms the transpiler would never emit, so
it proves things about PyneCore that no real (transpiled) strategy would ever
hit, and it cannot be pasted into TradingView to check what the answer SHOULD
be. Authoring in `.pine` keeps every test runnable on both engines, which is the
only way a "PyneCore disagrees with TradingView" claim can be settled.

```bash
# author            edit foo.pine
cd /home/mike/workspace/github/pine2pyne
.venv/bin/python -m pine2pyne /abs/path/to/foo.pine -o /abs/path/to/foo.py
# then run the generated foo.py (see below)
```

Keep the `.pine` next to its generated `.py` and re-transpile after every edit —
the `.py` is a build artifact, so never edit it by hand (the change is lost on
the next transpile, and the two silently disagree until then).

The exception is engine/runtime tests that are not strategy behaviour at all
(AST transforms, storage, plugin plumbing) — those are ordinary Python tests.

## Pine → Python: use the LOCAL pine2pyne transpiler, NOT the cloud API

To compile a `.pine` strategy to the `.py` that `pyne run` executes, use the
**local pine2pyne transpiler**. Do NOT use `pyne compile` or `pyne run script.pine`
— those call the PyneSys **cloud** compiler and require an API key we don't have
(`workdir/config/api.toml` is empty, and `PYNESYS_API_KEY` is unset).

pine2pyne is a separate repo with its own venv:

```bash
cd /home/mike/workspace/github/pine2pyne
.venv/bin/python -m pine2pyne /abs/path/to/in.pine -o /abs/path/to/out.py
# globs work too: .venv/bin/python -m pine2pyne "sample/pinescript/*.pine" -o workdir/scripts/
```

Then run the transpiled `.py` (no key needed):

```bash
cd /home/mike/workspace/github/pynecore
.venv/bin/pyne run path/to/out.py <data> [--broker]     # e.g. data: dnse:VN30F1M@5
```

Notes:
- The tracked `*.py` test strategies can be **stale** — after editing a `.pine`,
  re-transpile it (they were not regenerated automatically).
- `pyne run` auto-compiles a `.pine` via the cloud API — avoid it; transpile
  locally first, then run the `.py`.

## NEVER use `pyne run` with `--from` (CRITICAL)

Always run **without** `--from`:

```bash
.venv/bin/pyne run path/to/out.py dnse:VN30F1M@1 [--broker]    # correct
.venv/bin/pyne run path/to/out.py dnse:VN30F1M@1 --from -30    # NEVER
```

Why: in provider mode the warmup download **truncates and rewrites the shared
`.ohlcv` file** (`cli/commands/run.py:611`). A run with `--from -30` therefore
*destroys* the accumulated local history for that `(provider, symbol, timeframe)`,
leaving only 30 bars — the next run must re-download, and any deeper history is
gone. `--from` is data-destructive, not a read-only window.

It also silently starves indicators: warmup replays exactly the bars `--from`
fetched (no lookback introspection — `max_bars_back` is a no-op stub), so
`ta.sma(close, 600)` under a small `--from` is `NaN`, every comparison is
`false`, and the strategy quietly never trades while the run looks healthy.

Omitting `--from` uses the built-in default (`-500` bars **requested**, gap-retried
up to 4×), which is safe for the cache. But it does NOT deliver 500 bars: measured
2026-09-10, `VN30F1M@15` warmup got **275 real bars of the 500 asked** — the window
spans ~27 days and the ATC/holiday slots inside it are empty. The venue is not the
limit (DNSE serves ~4,219 bars/year at 15m); the default's *window* is. So a strategy
using `ta.sma(close, 200)` starts with only ~75 valid bars.

If a script genuinely needs real depth, raise the default in code — pre-downloading
does NOT survive, because provider-mode warmup rewrites the shared `.ohlcv` (that is
the same mechanism `--from` abuses). Gate the strategy on `not na(<series>)` so
insufficient warmup fails loudly instead of silently. See #17.

## `--live` vs `--broker` — the one-plugin model (how a live test runs)

- **`--live`** streams live DATA after the historical warmup (no order routing).
- **`--broker`** adds live ORDER routing and **implies `--live`**. The data-source provider
  MUST subclass `BrokerPlugin` and **IS the broker** — ONE plugin instance serves BOTH data
  and orders (`cli/commands/run.py:1602`). Consequence: the data source and the broker cannot
  differ — `pyne run <out>.py dnse_broker:VN30F1M@1 --broker` has `dnse_broker` serve both, and
  you canNOT pair a different data source (e.g. the data-only `replay` fixture provider) with a
  real broker via `pyne run`.
- A `--broker` run needs a GOOD trading token + the mandatory L0 gate FIRST, at 5m/15m — see the
  DNSE testing section and `plugins/dnse/testing/live_test/README.md`.

## Plugins in this repo (`plugins/`)

Fork-specific venue plugins, editable-installed (so they import as
`pynecore_<name>`) and discovered via the `pyne.plugin` entry-point group:

- **`plugins/dnse/`** — DNSE (Vietnamese broker) plugin. Two entry points:
  `dnse` (`DNSEProvider` — OHLCV history + metadata) and `dnse_broker`
  (`DNSEBroker` — native STOP/OCO conditional orders). REST order path (no WS
  order-event transport yet, #107; sub-minute bars are WS per-print, #100), built on the
  **vendored** DNSE openapi-sdk at `plugins/dnse/pynecore_dnse/_vendor/dnse/` (v2.2.0
  since 2026-09-11 — `_vendor/VENDOR_INFO.txt` is the ground truth, not this line). The
  SDK's own `python/examples` are vendored SEPARATELY and never edited, at
  `plugins/dnse/testing/examples/upstream/` (its own VENDOR_INFO.txt, pinned to an upstream
  commit); five of the seven run against the fake venue and are part of the suite, because
  the vendor's examples ask questions we did not write — 2026-09-18 they found three fake
  defects that answered CLEANLY and were simply not answers to the question asked, a class
  our own pins cannot find.
  (do NOT pip-install the SDK). Run: `pyne run <out>.py dnse:VN30F1M@5` for data,
  `… dnse_broker:VN30F1M@5 --broker` for live orders. Tests: `plugins/dnse/tests/`
  — `pytest` with functions named `__test_*__` (see `pytest.ini`); mock via the
  fake-client seam in `conftest.py` — pytest itself NEVER hits the live venue.
  Live testing exists but is its own gated suite (see "DNSE testing" below). Docs mirror + sync tool:
  `docs/dnse-openapi-documentation/` (`fetch_docs.py`); plans in `docs/plan/`.

## A DNSE position read is NOT authoritative alone (replica lag) — AND its size is UNSIGNED

TWO SEPARATE HAZARDS. They were conflated for a day; keep them apart.

**1. `ExchangePosition.size` is a MAGNITUDE — the sign lives in `.side`.** `broker.py`
builds it as `size=abs(net)` with `side="long" if net > 0 else "short"`. Read `.size`
alone and every non-flat position tests as positive, so a `size > 0` branch NEVER takes
its negative arm. That is what made `tools/flatten.py` print `long 1.0` for an account
that was really SHORT 1 and SELL into it — short 1 -> short 2, exit code 0 (2026-09-16;
fixed 2026-09-17, pinned by the short/long order-side tests in `test_flatten_tool.py`).
It is a pure sign-derivation bug, NOT a stale read: both reads agreed, and no
confirmation discipline can catch it. **Any reader of `ExchangePosition` derives the sign
from `.side` and refuses to guess an unrecognised label** — `sync_engine.py` does this at
three sites (~3849, ~4763, ~5098); `tools/flatten.py` now mirrors them.

**2. A single read can still be STALE (lagging replica).** Measured independently: a
`get_position` answered 0 while the position was really 1 (#124-OBS), and a stale FLAT
nearly retired protection for a live position (#122 post-close fix 6abe04c6). **Any code
deciding something IRREVERSIBLE from a position snapshot (its SIGN, or its emptiness)
requires TWO AGREEING READS; disagreement resolves to could-not-determine — never to the
newer snapshot — and the fail-closed action.** Enforced in `tools/flatten.py` (bf5096e3)
and `sync_engine.py` (6abe04c6); new call sites must follow.

The two-read rule STAYS — hazard 2 is real and orthogonal. But it was credited with
catching hazard 1, which it never could: both reads return the same unsigned magnitude and
agree with each other.

## STOCK amend is CANCEL+REPLACE with a NEW id; write rejects are CODED (measured 2026-09-15)

Live-measured on prod (#117 probe, funded stock account):
- `PUT /orders/{id}` on a resting STOCK LO answers 200 with a **NEW order id** —
  the venue cancels the old id itself (reads back `Canceled` untouched) and mints a
  replacement. **Both price AND quantity in one PUT are accepted.** Consequence:
  after any stock amend the engine MUST re-map its tracked id from the PUT response
  or it goes blind (#39 family) — and must EXPECT the predecessor's `Canceled` push.
- Same-day write rejects are STRUCTURED codes, not free text: PUT on a done order →
  `ORDER_IS_DONE`; cancel on a terminal order → `ORDER_CANCEL_STATUS_REJECTED`;
  cancel racing `PendingNew` → `CAN_NOT_CANCEL_PENDINGNEW_ORDER_IN_OPEN_SESSION`
  (retry after a beat, NOT terminal). The sandbox's free-text "already in terminal
  state Filled" was NOT observed on prod → #116 looks sandbox-only (Filled-status
  case still unmeasured — needs a same-day fill).
- **Cross-day numeric ids do NOT resolve on the cancel endpoint** (`RESOURCE_NOT_FOUND`,
  even for yesterday's fills). And an order detail carries **NO reject-reason field**
  (`orderStatus='Rejected'` with nothing else) — a venue reject can never be explained
  from the record alone.
- A rejected-at-entry STOCK buy (no buying power) goes `Rejected` in ~3 ms with
  `canceledQuantity=qty`; the HTTP place still answers 200.

## DNSE has TWO order books — and a triggered conditional MOVES between them (CRITICAL)

Venue mechanic, operator-confirmed 2026-08-18:

- **Conditional book** — STOP / STOP-LIMIT orders. Ids are **long strings**
  (`da203hg6p09g1n1vipog`).
- **Normal book** — plain LOs for derivatives and stocks. Ids are **integers**
  (`437346`).

A stop order rests on the CONDITIONAL book until its TTL expires or its trigger
fires. On trigger the conditional order is **`Activated` — i.e. CLOSED, not
filled** — and the venue creates a **NEW order on the NORMAL book** which is what
actually executes. The activated conditional's detail carries metadata naming the
child normal-order id (`externalOrderId`).

**Consequence for the plugin:** a stop entry's fill NEVER appears on the id we
placed. Anything that maps venue records to Pine ids by the placed id alone goes
blind the moment a stop triggers — the strategy keeps believing `pos=0` while the
account holds a real position (measured live, Live-L3-F11 → issue #39; the OCO
exit path already tracks its child via `externalOrderId`, the stop-entry path does
not). When reading order state: `Activated` on the conditional book means *look up
the child on the normal book* — never treat it as terminal-without-fill.
**Measured 2026-09-15:** an OCO umbrella is `Activated` FROM BIRTH — placement
immediately spawns the normal-book TP child (`Activated` at the FIRST read, no
trigger involved) — so a cancel of the umbrella answers `CO-ORD-013 "order status
is not new"` from second one; cancel the CHILD id instead. `Activated` never means
"triggered" by itself.

**`Activated` IS TERMINAL FOR THE CONDITIONAL ITSELF (operator, 2026-09-18).** On
BOTH the STOP and the OCO book, `Activated` means exactly one thing: *it has
created its normal-book order, and nothing further can be done with it.* The
conditional row is a spent shell from that moment — it cannot be cancelled
(`CO-ORD-013 "order is done"`), cannot be amended, and will never act again. All
remaining behaviour lives on the NORMAL child.

Two consequences that have each cost a live session:
- **An `Activated` OCO umbrella with a populated `stopPrice` is NOT an armed
  stop.** Once its child is cancelled the bracket is dead, however live the row
  looks. Measured 2026-09-18: after `flatten_api.py` swept TP child 39356, the
  umbrella `damadq2vfqkc7397o0tg` still read `Activated stopPrice=1980.8` on a
  FLAT account; its `modifiedDate` was the exact instant of the child's cancel,
  and the venue refused the cancel with `CO-ORD-013 "order is done"`. It was
  spent, not armed. Do not chase it, and do not report it as a naked stop.
- **The stop leg is only real while the umbrella's child is alive.** So "is this
  position stopped?" is answered by the umbrella's `stopPrice` AND a live child —
  never by the row's status, which reads `Activated` in both the protected and
  the spent case.

## DNSE positions are VENUE-DERIVED from fills — we create ORDERS, not positions (confirmed 2026-09-12)

No create-position API exists. You `POST .../orders`; a fill makes the **venue** create/update the
position, with a venue-assigned `id` (e.g. `177410795472387`, NOT our coid) + a venue `createdDate`.
The whole position surface is read + lifecycle: `GET .../positions`, `GET /positions/{id}`,
`POST /positions/{id}/close`, `.../pnl-configs`. A position is a lifecycle object: `status`
(OPEN/PENDING_CLOSE/CLOSED/ODD_LOT) + five quantities (accumulate/trade/closed/open/overNight),
netting `openQuantity = accumulate - closed`.

**Per-strategy position isolation is BUILT and green (`#73`) — do NOT "fix" it away.**
The venue holds ONE net position per asset (`broker.py get_position` sums `openQuantity x side` over
ALL non-CLOSED rows for the symbol = the whole-ACCOUNT net; `None` at net==0), and we run many
strategies x many users on that one asset. The engine already isolates each run from that shared net
(so my earlier "reconcile naively adopts the account-net" was WRONG):
- **Startup adoption is CLAMPED to the run-owned slice** (`sync_engine.py _clamp_adoption_to_owned`,
  #73/C1): a run adopts only the exposure its OWN journal fills produced (`_durable_owned_signed_size`
  — signed sum of this run's `filled_qty` cursors; foreign `ADOPTED_STARTUP_EXTRA` legs excluded),
  never another run's slice. BOTH startup branches clamp (plain + replayed-close, `sync_engine.py`
  ~4546/~4869).
- **Periodic external-flatten reads the RAW account net BY DESIGN** (#73/C2, ~4633): `net==0` on a
  netting account is proof of ABSENCE (nobody holds anything), so clearing is safe; an owned-*belief*
  there would destructively clear a live position.
- Partial divergence mid-run is warn-only, never adopted (#48).
Per-run identity: `run_tag` (restart-STABLE SHA256 of strategy_id+source+symbol+tf+account+label,
`run_identity.py`) + the per-run journal (`run_instance_id`). Green: `test_journal_wiring` +
`test_divergence_matrix` + `test_079` (27 pass); live `Live-L1-T10-DualStrategy` (order-level, ✅).
**INVARIANT (pinned by `plugins/dnse/tests/test_get_position_account_net.py`): `get_position` MUST
return the whole-account net, UNFILTERED by run** — #73/C2 uses it as the absence-proof. "Fixing" it
to return "our share" would SILENTLY break external-flatten detection.
**Limitation (venue physics, not a bug):** two OPPOSING-direction strategies cannot coexist on one
netting account (they net away); an external PARTIAL close cannot be attributed to a specific run.
The one real gap is coverage — T10 is NO-FILL (order-level); a FILL-level DNSE isolation test (two
engines, one sandbox account, both fill, each `_durable_owned_signed_size` reflects only its own
fills) is #115, and is where the sandbox harness (#114) lands.

## Protective-exit arm timing + #124 re-arm (measured live 2026-09-15)

- **PRE-PLACE the protective exit on the ENTRY bar for same-bar protection.** A `strategy.exit`
  placed on the entry bar (withheld by #82b until the fill, so it is already in the exit book when
  the fill lands) arms on the **SAME bar and SAME timestamp as the fill** — #121's arm-on-fill wake
  dispatches + creates the SL/TP at the fill bar, no naked window (l2b, entry `id=214806` bar 502
  14:17:00 -> exit same bar 502 -> `215286`). This SUPERSEDES the "#107 ~1-bar window, ACCEPTED not
  fixed" verdict **only for a pre-placed bracket**. A **REACTIVELY-placed** exit (gated on
  `strategy.position_size > 0`, as in l2b_entry_update) is NOT in the book at fill time, so the wake
  has nothing to arm and it still lands **a bar late** (fill bar 502 -> exit bar 503, same morning).
  Same-bar protection is a STRATEGY-AUTHORING requirement, NOT an engine guarantee.
- **#124 (EXTERNAL cancel of a protective exit) now RE-ARMS instead of quarantining** (fixed live
  2026-09-15, first prod confirmation): when our conditional bracket is cancelled out-of-band while the
  position is still open, the fix logs `#124: ... re-arming (N/3), no quarantine` and places a fresh
  exit rather than leaving a naked position. The fix makes the cancel SAFE; the venue-cancel ROOT
  CAUSE is CLOSED 09-16: ALL 'venue cancel' events (09-14 F9/l3, 09-15 13:42) were the
  OPERATOR'S manual app cancels (his confirmation; #128 closed) — no venue phenomenon exists.
  The fix matters regardless of actor (a human cancelling mid-position is the documented case). The
  #124-OBS `get_position` read can LAG (read 0 while the position was 1) — a racy read worth tightening.

## DNSE WebSocket testing rules — how the "silent WS" false verdict happened (CRITICAL)

Two venue facts were wrongly recorded as "WS is silent" for ~2 weeks because
probes were built from the DOCS alone. Rules (measured 2026-08-26):

- **The WS requires an explicit auth message within 30 s** (HMAC-SHA256 over
  `"{api_key}:{timestamp}:{nonce}"`); subscribes are REFUSED before auth.
  A probe that skips or fails this handshake sees "silent channels".
- **The connection path is in the SDK guide page, NOT in the three REST guide pages**:
  `wss://ws-openapi.dnse.com.vn/v1/stream?encoding=json|msgpack`
  (`_vendor/dnse/websocket/client.py`; also `docs/dnse-openapi-documentation/sdk-build_websocket.md:65`
  with the full HMAC handshake at :72-145). The REST guide pages give only the base URL, so a
  probe built from those alone 404s. (Corrected 2026-09-18; the earlier "SDK only" claim was false.)
- **Use the vendored `TradingClient`** (`_vendor/dnse/websocket/` — full
  client + AuthManager, subscribe helpers for every channel). Operator
  mandate: never hand-roll a WS client for production paths.
- **Event-driven channels (order./position.) prove delivery only when an
  account event occurs during the capture** — zero frames with zero account
  activity is the EXPECTED result, not silence (empty ≠ conclusive).
  Market-data channels (tick/quotes/ohlc) prove themselves in seconds during
  trading hours (measured: ~136 prints + ~650 quote frames per 30 s).
- **The engine's "WS connected and subscribed" banner is generic live-runner
  text** — the DNSE plugin's `connect()` is a no-op (the order path is REST), so
  the banner prints even on a 1m run where no order socket is opened. A real WS
  DOES exist for sub-minute market data (#100, started lazily in `watch_ohlcv`),
  but the banner is never evidence of an ORDER-event socket. Never take it as such.
- Working probe: `plugins/dnse/testing/live_test/probe_ws_market_data.py`
  (`--trading` for order/position channels). Findings live on card #50.
- **PROD ORDER EVENTS ARE DELIVERED — captured 2026-09-15 (first time ever)** on the
  SHORT channel `order.DERIVATIVE.json` (`subscribe_order_event`): 4 `do` frames
  (incl. `pendingnew`) for a real OCO place+cancel — the same channel that works on
  the sandbox. This CORRECTS the earlier "prod needs subscribe_broker_order_event"
  claim (017e2bd): the short channel works on prod; the BROKER channel
  (`order.broker.{mt}.{investor}` — what `ws_order_source.py` subscribes, #121)
  remains UNCAPTURED and is the open question on #107/#121.

## INVALID_TRADING_TOKEN on conditional writes is usually NOT a token problem (CRITICAL)

Measured 2026-08-24/25 (cards #46, #51): DNSE rejects VALID, in-TTL trading
tokens on **conditional-book writes** (place AND cancel) with
`400 INVALID_TRADING_TOKEN` — a 30-second-old token got the same rejection.
Both measured windows began right after the operator traded in the EntradeX
app; before that boundary the same token worked (T32 placed+cancelled a
conditional fine at 09:00). The token-mint flow itself (request → email OTP →
token, 8h TTL) has never been the failing part.

Rules:
- Do NOT re-mint on this error — re-minting was measured NOT to help (both days).
- `token_status.py` liveness stays GOOD through the breakage (its bogus-id
  probe dies at order lookup, before the deep check) — it is NOT a proxy for
  conditional-write ability.
- Schedule L0 and any conditional-book live work BEFORE the operator's first
  app trade of the day (pre-open ~08:20), or after they log out / go idle.
- During a broken window: cancels go through the operator's app; DAY orders
  expire at 14:45; do not keep retrying.

## DNSE venue toolkit — use it instead of hand-writing probes (CRITICAL)

`plugins/dnse/tools/venue.py` answers the questions every live session needs.
**Use it. Do not re-type a `DNSEBroker(...)` heredoc to ask them** — that habit is
what produced a false "account is FLAT" report on 2026-08-19 while the account
held +2 contracts: the improvised probe called `fetch_position`, which is a
CAPABILITY FLAG, not a method (the method is `get_position`), guarded it with
`hasattr`, and read the resulting `None` as "no position".

```bash
.venv/bin/python plugins/dnse/tools/venue.py status      # phase, token, position, orders
.venv/bin/python plugins/dnse/tools/venue.py flat        # exit 0 ONLY if truly flat+clean
.venv/bin/python plugins/dnse/tools/venue.py order <id>  # detail, auto book, history fallback
.venv/bin/python plugins/dnse/tools/venue.py cancel <id> # cancel AND verify terminal
.venv/bin/python plugins/dnse/tools/venue.py sweep --yes # cancel everything (destructive)
.venv/bin/python plugins/dnse/tools/venue.py history [--date YYYY-MM-DD]
```

**Exit codes are the point** — `0` answered affirmative, `1` answered negative,
`2` COULD NOT DETERMINE (the read failed). A failed read never looks like "flat".
Gate a live launch on `venue.py flat` rather than on anyone's summary.

Two behaviours worth knowing:
- **Phantom shells are separated from live orders.** A triggered conditional stays
  `Activated` on the STOP book forever while its NORMAL-book child does the work
  (#41), so it would otherwise make every post-fill cleanliness check cry wolf.
  `status` lists them as `phantom -> child N did the work`; `flat` ignores them.
- **`order` falls back to `/orders/history`** for previous-day ids, because the
  detail endpoint answers `None` for them (rows are date-prefixed, under `data`).

`sweep` cancels EVERY working order on the account, including the operator's own —
DNSE nets per symbol and the venue has no notion of "this run's" orders. Hence
`--yes`.

## DNSE testing (read before touching the plugin or running anything live)

**Standing authorisation (operator, 2026-09-18): agents may run L0, L1 and L2 test cases
autonomously** — no per-run operator word is needed to launch them. The preconditions still
apply every time: the L0 gate passes first (exit 0), `venue.py flat` answers 0 before any
L2 launch, the run is flat again by 14:25, and results are graded from the venue record.
**L3 (uncertain stop distance and/or size ≥ 2) still needs the operator's explicit word per
run**, and entering an OTP or any credential remains prohibited for every agent. This
supersedes the "never execute an order-placing command" line in earlier executor prompts
for L0–L2 only.

**A claim about a live event travels with the log line that shows it, or it is labelled
UNVERIFIED** (rule adopted 2026-09-18 after two false claims crossed three sessions and reached
a card body; trace in `docs/plan/false-claim-propagation-2026-09-18.md`). The line must show the
claim itself, not its neighbourhood: a `[NAKED]` verdict shows detection, not an alarm; if the
quoted line supports a weaker claim than the sentence, the weaker claim is the one you may make.
When repeating a peer's claim about a live event, quote THEIR line or run the check yourself: a
peer's claim about what THEY did is testimony; a peer's claim about what the SYSTEM did is a
measurement, and measurements are checked no matter who took them.

Test cases are named **`Live-L<level>-<case>`** (e.g. `Live-L1-T11-OcaCancelMember`,
`Live-L3-F05-LongStopLimit`) — use ONLY these IDs in plans, cards and conversation;
the canonical registry with live status is the first table in
`plugins/dnse/testing/live_test/README.md`. The suite's rules are documented in
that same README — four types: `pytest plugins/dnse/tests/`
(unit + fake-venue e2e), the L0 venue-semantics gate (MANDATORY, exit 0, before EVERY
live run), the staged no-fill probe (T1–T13 + the `run_t10_dual.sh` dual-strategy
runner), and the staged fill test (F1–F8; its backtest mode over a past window IS the
oracle). Both staged probes are driven by `winStart`/`winEnd`/`startState` in their
`.toml` — for a live run `winStart` must be AFTER launch or warmup consumes the stages.
Grade live results from the VENUE record, never the run log alone. Measured venue facts
(cancel-ACK, no cascade, amend-500, GTD clamp, session phases) are listed in that README.
Trading-token workflow (OTP mint, ~8h TTL, status check): `plugins/dnse/tools/README.md`
— live runs need a GOOD token first (`tools/token_status.py`).

### The five test levels (operator decision 2026-09-18; the README above is the source)

| Level | What it is | Where it runs | Tier | Precondition | Purpose |
|---|---|---|---|---|---|
| L0 | venue-semantics gate (`level0_venue_semantics/l0_order_semantics.py`) | direct client, no engine | no-fill | GOOD token; any hour except ATC and post-close (pre-open needs `--allow-conditionals-when-closed`; 08:45-09:00 unmeasured, #156) | proves auth, both books, place/rest/cancel today |
| L1 | staged no-fill probes T01–T33 + direct probes + runners | `pyne run … --broker` or direct | no-fill | open session + GOOD token; safe with an open user position | engine→plugin→venue order paths, ≥4.5 % away |
| L4 | data parity / latency / ATC / tick delivery (`level4_data_parity/`) | passive recorder | no-fill, no orders | no token | bar feed correctness |
| L2 | fill with TIGHT risk control (`l2b_fill_protect_flatten`) | `--broker`, FLAT account, supervised | FILL | flat account or sub-account, L0 + no-fill smoke green today | the protected fill→bracket→flatten chain |
| L3 | fill with UNCERTAIN stop distance and/or size ≥2 (`live_staged_fill` F01–F12, F10/F11 vehicles) | `--broker`, FLAT account, supervised | FILL | as L2, plus operator-in-the-loop close protocol | order-type fill semantics, cascades, races |

L2/L3 split criterion: L2 means the protective stop's distance from the REAL fill price is
known and bounded before the run AND the position size is exactly 1. L3 means the stop's
distance from the real fill is uncertain (reactive exits, trailing stops, brackets armed a
bar late, crossed-at-placement, prune/adoption races) and/or the size can reach 2 or more.
Three groups: no-fill ORDER tests (L0, L1), no-fill DATA tests (L4), FILL tests (L2, L3).
Mandatory run order every session: L0 (exit 0) → no-fill (L1 smoke T01–T03, L4) → fill LAST.

### Live-run session mechanics (Claude Code specifics)

- Background bash jobs die at ~10 min — a 1m staged run fits; 3m+ does not.
  Relaunch with `startState` to resume instead of stretching one job.
- Watch logs with `timeout N bash -c 'until grep -aq "X" f.log; do sleep 5; done'`
  — bare or chained `sleep` is blocked by the harness.
- Kill runs via `pgrep -f "live_[s]taged" | xargs -r kill` (bracket avoids matching
  your own wrapper shell). Exit 144 after your own kill is EXPECTED, not a failure.
- Raw `--broker` logs are ~500K ANSI spinner noise. Commit only stripped evidence:
  `sed 's/\x1b\[[0-9;]*m//g' f.log | grep -aoE '\[(L1|F|BROKER)\][^[]*' > f_evidence.txt`
  and park the raw log in `backup/deleteable/`.

## DNSE Sandbox — free fill/WS testing (no real money, no market hours, no OTP dance)

DNSE has a **Sandbox** (mock env, separate keys) that AUTO-runs the order lifecycle
`PendingNew -> New -> PartiallyFilled -> Filled` and pushes it on the trading WS — so fill
and WS-order-event testing costs nothing and needs no session. Proven end-to-end 2026-09-11
(#107): `plugins/dnse/testing/sandbox_lifecycle_probe.py` places a NORMAL order and captures
the full lifecycle on `order.DERIVATIVE.json`.

- **Endpoints:** REST `https://sb-openapi.dnse.com.vn`, WS `wss://ws-sb-openapi.dnse.com.vn`
  (the docs also list a `-uat` WS host; the plain one works). Same auth/signature/paths as prod.
- **OTP is mocked:** the trading-token passcode is the fixed public constant **666666** (any accepted OTP type). So the sandbox token mint is NOT real-2FA entry — mint it freely.
- **Config:** `workdir/config/plugins/dnse_sandbox.toml` (gitignored) with sandbox
  key/secret + the sandbox base_url/ws_url + a SEPARATE token_file under `workdir/state/` so it never clobbers the prod token.
  Keys come from `.env` (`DNSE_SANDBOX_API_KEY`/`_SECRET`) — move file-to-file, never echo.
- **The WS order channel is CASE-SENSITIVE:** `order.DERIVATIVE.json` / `order.STOCK.json`
  UPPERCASE market_type (docs + SDK default). A lowercase name is silently accepted
  (`status: active`) but streams NOTHING — this produced a false "trading WS is silent"
  verdict for a while. The payload nests the order under `msg["order"]` (`T:"do"`;
  positions `T:"dp"`), carrying `orderStatus`/`fillQuantity`/`quantity`.
- **LIMIT (measured):** Sandbox accepts `orderCategory=NORMAL` only — but ALL order TYPES
  within it (LO limit, MTL/MOK/MAK market, ATO auction all return 200); only the conditional
  CATEGORY (`orderCategory=STOP`/`OCO`) is rejected. AND there is **NO market-price
  simulation** — it is a pure order-lifecycle + WS-event simulator: a LIMIT fills at its OWN
  price (unchecked), a MARKET fills at averagePrice=0. So it tests order-state + WS delivery +
  engine event handling, NOT SL/TP triggering, matching, or P&L (use tracked `.ohlcv`
  backtests for price behaviour). Conditional STOP/OCO placement still needs production. Data resets.
- **Order ops + fill cadence (measured 2026-09-12, live).** The fill is a fixed server-side
  TIMER, not a matcher: every order walks `PendingNew -> New -> [PartiallyFilled] -> Filled` at
  ~0.3 s per transition (Filled in ~1.2 s), independent of size/price/side. `qty=1` fills in one
  step (no partial); `qty>1` gets exactly ONE `PartiallyFilled` tick with a VARIABLE chunk
  (measured 9/30, 80/100 — not a fixed fraction) then the remainder on the next tick — ideal for
  exercising the engine's partial-fill + WS-event paths. Per-op: **place** NORMAL LO/MTL works
  (200); **cancel a RESTING order** works (204 -> `Canceled` — the ~0.6 s `New` window beats the
  ~1.2 s fill, so a prompt cancel lands); **amend/modify** does NOT (`PUT /orders/{id}` ->
  `HTTP-405`, no endpoint) — a strategy that re-places/moves a resting order (limit chasing
  `low[1]`, trailing stops) CANNOT run on the sandbox; **conditional STOP/OCO** does NOT
  (`400 UNSUPPORTED_ORDER_CATEGORY`). The `/sandbox-e2e` runner flags the amend-405 + STOP cases.
- **Sandbox testing does NOT go through `pyne run --broker`** — there is no `dnse_sandbox`
  entry point, the one-plugin model (above) means data+orders share one plugin, and the sandbox
  has no market data. Drive it from a standalone probe: `sandbox_lifecycle_probe.py` (raw client
  lifecycle + WS) or `sandbox_arm_on_fill_probe.py` (full engine + real broker). For fake-realtime
  bars in an engine probe, wire the data-only `replay` provider (`providers/replay.py`) + a real
  broker together IN the probe. Name this shape a **Sandbox Replay E2E**: replay bars driving a
  real strategy through the engine while orders route to the sandbox — the only DETERMINISTIC,
  offline exercise of the REAL broker + engine order/fill path (backtest uses the sim engine;
  live needs real money + market hours). `sandbox_arm_on_fill_probe.py` is the engine-driven
  half; adding the replay-bar feed completes it.
- **The full engine runs against the sandbox, not just the raw client.**
  `plugins/dnse/testing/sandbox_arm_on_fill_probe.py` drives the real `DNSEBroker` +
  `OrderSyncEngine` and a real auto-fill to prove the engine's arm-on-fill path places a real
  protective exit — graded from the venue order record, for BOTH a derivative (`41I1G9000`,
  qty 1) and a stock (`HPG`, qty 100). Engine-driven probes need `event_loop=None` (per-call
  `asyncio.run`), `lib._script` stubbed (`SimpleNamespace(initial_capital=...)`, for
  `record_fill`), a 4-alphanumeric `run_tag`, and a flat `get_position` stub (next bullet).
- **The sandbox does NOT net/match** (pure order-lifecycle sim): `/positions` returns
  accumulating `deals` (key `deals`, NOT `positions`/`data`), an opposing order never flattens
  a position, and the plugin's netting `get_position` can't read that shape (0 rows vs
  `total>=1` -> "truncated, refusing to conclude"). Stub `get_position -> None` for engine
  tests — it is only the reconcile view-confirm; fills drive position state via the event.
- **Catalog / classification gotchas.** `VN30F1M` is NOT a sandbox symbol (`SYMBOL_NOT_EXIST`,
  #113) — the VN30 front-month is coded `41I1G9000`, which classifies STOCK (only `VN30F*` ->
  DERIVATIVE) so needs a `market_type` pin, and qty is per-instrument (derivative 1, HOSE
  stock 100 = one board lot, price ~26.x thousand VND). Harmless: a fill logs `executions read
  http=404` (no executions endpoint) -> booked at cumulative VWAP.

## DNSE VN30-futures symbol taxonomy — an alias is NOT a "type"

Three notations name the SAME rolling VN30 futures contracts:

| Notation | What it is |
|---|---|
| `VN30F1M` / `VN30F2M` | conventional **aliases** for the month-1 / month-2 (front / next) contract |
| `HNX:VN301!` / `HNX:VN302!` | TradingView's continuous front / next-month notation for the same |
| `41I1G9000` | the actual **dated contract** — KRX's deterministic naming for one specific expiration |

**Confirmed on prod (2026-09-12):** `/market/instruments` carries `symbolType=VN30F1M symbol=41I1G9000`
and `symbolType=VN30F2M symbol=41I1GA000`, so `resolve_contract` (which matches `symbolType`) ALREADY
resolves BOTH month aliases to their dated code. DNSE's field is literally named `symbolType`, but the
VALUES are rolling **aliases**, not a classification type — the only real "type" is `market_type`
(STOCK/DERIVATIVE; all of the above are DERIVATIVE) — and they **move**: each expiration `VN30F1M`
repoints to a new dated code (the roll). `resolve_contract` reads that mapping from the venue (never
computes KRX codes) — BUT its per-instance cache can serve a STALE dated code across a roll (the real
remaining concern, #113). TradingView keys (`HNX:VN30x!`) are NOT DNSE `symbolType`s -> they need a
`symbol_map` line (`"HNX:VN301!" = "dnse:VN30F1M"`).

**Operator-confirmed roll mechanics (2026-09-14):** the `VN30F1M`/`F2M` aliases repoint to the next
dated contract on the **morning AFTER expiry** (Friday), not on expiry day — so a process started
before that morning holds a stale alias->contract mapping at Friday's open (#113: re-resolve per
trading day, never per-instance-forever). When the 3rd Thursday is a **holiday**, the final trade
date moves to the **preceding trading day** (confirms #118's walk-back). The API's documented
`finalTradeDate` shape is the compact `20260416` (doc-typed "string"; derivatives + covered
warrants only — stocks never carry it), while the venue has ALSO served ISO `2026-08-20` live
(2026-08-14): both forms must parse (`expiry.parse_venue_date` does).

## request.security() on DNSE — indices work, wired by symbol_map (not `--security`)

DNSE serves market INDICES (`VNINDEX`, `VN30`) on `/price/ohlc?type=INDEX` — ~1 year of
15m history, current to the session. The provider routes them via `_INDEX_SYMBOLS` (#104);
`type=STOCK` on an index answers 400.

The FRAMEWORK owns Pine-key → native-symbol translation (`script_runner.py` calls
`resolve_symbol`; all three official PyneSys plugins inherit it rather than implement it),
so a TradingView-style symbol is a CONFIG entry, never code:

```toml
# workdir/config/symbol_map.toml
[symbol_map]
"HOSE:VN30" = "dnse:VN30"
"VNINDEX"   = "dnse:VNINDEX"
```

**Backtest**: with that map plus the `.ohlcv` files, contexts resolve with **no `--security`
flags** (verified 2026-09-10).

**Live**: verified only WITH explicit flags so far —
`--security 'VNINDEX:15=VNINDEX' --security 'HOSE:VN30:15=VN30'` delivered real, moving,
chart-distinct index values. The framework documents a live fallback that resolves through
`resolve_symbol` and builds the `PluginSymbol` itself (`script_runner.py`), so the map alone
*should* be enough — but that has NOT been run, so pass the flags until someone measures it.
Live warms up and streams each security in its own subprocess with its own provider instance.

Two things that look like feed bugs and are NOT:
- **No index bar at 09:00.** Derivatives ATO is 08:45–09:00, stock ATO 09:00–09:15, so index
  bars legitimately start **09:15** — `na` on the 09:00 futures bar is correct. (Backtest
  carries the prior close forward there; live reports `na` — they differ on the day's first bar.)
- **The 14:45 index bar is malformed at the source**: the auction close is published outside
  `[low, high]` with `O==H==L` (~0.5% of bars, all 14:45; futures unaffected). The provider
  widens the range and logs it; O and C are never altered.

## Pine behaviours that LOOK like bugs and are NOT (cross-checked on TradingView)

Measured 2026-09-10 after a long wrong turn — **do not re-investigate**:

- **`qty=1` everywhere can still leave `position_size = 2`.** A price-based entry freezes its
  reversal augmentation at PLACEMENT (`lib/strategy/__init__.py:6136`), so a long entry stop
  placed while short 1 carries qty **2**. If a protective exit closes the short before that
  entry fills, the frozen 2 opens from flat. **TradingView does the same** — same setup,
  TV-portable port, VN301! 15m. Not a PyneCore defect; it is inside TV-validated backtests too.
  The cause is the ENTRY's stale flip quantity, **not** an orphaned exit opening a position —
  that theory is refuted by `flip_exit_entry_first` (#105). So the outcome is
  **order-dependent**: exit level crossed first → 2 contracts, entry level first → 1, and the
  orphaned exit opens nothing.
- **The plain flip idiom is safe**: two `strategy.entry` calls in opposite directions with NO
  `strategy.exit` never exceeds 1 contract (measured: 50 trades, max held 1, including bars
  where both conditions fire). The hazard above needs an ARMED EXIT, not just a flip.
- **`pyramiding` IS enforced** by PyneCore, for market and stop entries alike.
- **Reversal sizing is correct**: short 1 then long `qty=1` → **+1** (not +2, not 0).
- **Do NOT try to cap size by guarding entries on `strategy.position_size`.** Order commit is
  deferred and stop entries fill intrabar, so the guard reads a stale `0`. Measured: it made
  things WORSE (2 → 3 contracts). A hard size ceiling can only live in the broker — and
  **`strategy.risk.max_position_size` is not that ceiling in live**: it bounds the position in
  backtest (cap applied at FILL time) but only gates the order at SUBMIT time live, so the
  breach above survives it (measured 2026-09-11, #105). A backtest proving "never exceeds 1"
  does not transfer to live.

Evidence is re-runnable, not just recorded: the probes live in `docs/probes/flip_exit/`
(with a README table of what each answers), and the flip/exit result is pinned by
`tests/t01_lib/t30_strategy/test_130_flip_does_not_cancel_an_armed_exit.py` — a
discriminating test (remove the armed exit and the same bars give 1, not 2), so an
upstream rebase that changes this behaviour FAILS rather than silently falsifying this
section.

## Pine format strings: NEVER use a lone apostrophe

`log.*` / `str.format` follow ICU quoting — a single `'` opens a LITERAL section,
so `"the operator's close {1}"` drops the apostrophe AND prints `{1}` raw
(placeholders before it still work, so it looks like a partial bug). A balanced
pair (`close('T26') phase={0}`) is harmless — quotes are consumed cosmetically and
later placeholders still substitute. Escape a real quote as `''`. Measured
2026-08-19; symptom: literal `{n}` in a log line.

## Pine Script language reference

Authoritative Pine v6 reference (syntax, built-ins, `strategy.*` semantics):
https://www.tradingview.com/pine-script-reference/v6/
Use it when writing test `.pine` files or checking what TradingView-compatible
behaviour SHOULD be (e.g. `strategy.exit` has no `oca_type`; exits form a reduce
group; OCA fires on fill, not on cancel).

## Reference repos (workspace siblings — read for patterns, don't edit)

Cloned next to this repo under `/home/mike/workspace/github/`:

- **`pine2pyne/`** — the local Pine→Python transpiler (see above); its own venv.
- **Official PyneSys broker-plugin samples** — the canonical examples the DNSE
  plugin's design was ground-checked against; copy their module split
  (`execution.py` / `activity.py` / `reconcile.py` / `recovery.py`), the
  `store_ctx` persistence + `DisappearanceTracker` patterns, and the
  `broker_lab/` conformance-lab `VenueProfile`:
  - `pynecore-plugin-bybit/` — native `reduceOnly`, bar-close fills
  - `pynecore-plugin-capitalcom/` — deal-id model, `recovery.py` reopen-on-retry
  - `pynecore-plugin-ctrader/` — cTrader
