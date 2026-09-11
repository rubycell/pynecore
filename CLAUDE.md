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
  **vendored** DNSE openapi-sdk v2.0.0 at `plugins/dnse/pynecore_dnse/_vendor/dnse/`
  (do NOT pip-install the SDK). Run: `pyne run <out>.py dnse:VN30F1M@5` for data,
  `… dnse_broker:VN30F1M@5 --broker` for live orders. Tests: `plugins/dnse/tests/`
  — `pytest` with functions named `__test_*__` (see `pytest.ini`); mock via the
  fake-client seam in `conftest.py` — pytest itself NEVER hits the live venue.
  Live testing exists but is its own gated suite (see "DNSE testing" below). Docs mirror + sync tool:
  `docs/dnse-openapi-documentation/` (`fetch_docs.py`); plans in `docs/plan/`.

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

## DNSE WebSocket testing rules — how the "silent WS" false verdict happened (CRITICAL)

Two venue facts were wrongly recorded as "WS is silent" for ~2 weeks because
probes were built from the DOCS alone. Rules (measured 2026-08-26):

- **The WS requires an explicit auth message within 30 s** (HMAC-SHA256 over
  `"{api_key}:{timestamp}:{nonce}"`); subscribes are REFUSED before auth.
  A probe that skips or fails this handshake sees "silent channels".
- **The connection path exists ONLY in the SDK, not the docs**:
  `wss://ws-openapi.dnse.com.vn/v1/stream?encoding=json|msgpack`
  (`_vendor/dnse/websocket/client.py`). Docs give only the base URL — a
  docs-faithful probe 404s.
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
