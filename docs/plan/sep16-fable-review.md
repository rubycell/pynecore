# Card-state audit — 2026-09-16 (Fable review)

Scope: open cards the operator suspected are quietly done or outdated —
**75, 79, 84, 88, 89, 97, 99, 102, 103, 114, 115, 116, 117**.
Excluded (state already known): 113 (Friday roll probe), 120/122/127/132/133 (active),
129 (closed today — fixed `58787aaa`, live-verified), 130/131 (measured-fact references),
135 (new CRITICAL — folded in below only where it interacts).

Base: branch `dnse-broker-v2`. The audit started at `9c149a84`; HEAD moved to `099bf3fe`
(`chore(hooks): mechanical code-review gate`) mid-audit — a hooks-only commit that touches no
code cited here. Every verdict below is a code read at HEAD.
Nothing here was closed or commented on GitHub — the leader does that.

---

## 1. NEEDS REAL ORDER BOOK — NEXT LIVE SESSION (Friday piggyback candidates)

The market is closed (audit written ~16:08 ICT). These are the items a real order book would
settle, sized as piggybacks on a session that is already happening. **Two qualify**; a third
is listed as a free passive observation. Everything else on the audit list was settleable
from code and is graded below without needing the venue.

### P1 — #116: the same-day cancel-of-FILLED reject signature (needs a FILL)

The only unmeasured premise on the card. Existing probe, already written and gated:

```bash
cd /home/mike/workspace/github/pynecore
.venv/bin/python plugins/dnse/testing/live_test/probe_116_same_day_cancel.py <numeric_order_id>
```

- Run it **immediately after a fill, same trading day**, with the id that just filled.
  Cross-day ids answer `400 RESOURCE_NOT_FOUND` and settle nothing — that is exactly what the
  09-15 attempt hit (`logs/session0915_p116_090540.log`: six reads, all `RESOURCE_NOT_FOUND`).
- Safety is built in, not assumed: the probe refuses a non-numeric id, refuses if the detail
  does not read back terminal, and explicitly refuses `PartiallyFilled` (a working remainder —
  cancelling it would be a real cancel, not the no-op measurement). Cancelling a fully FILLED
  order cannot execute anything; the refusal itself is the datum.
- Needs: a trading token, and a fill-tier run that actually fills. **No extra order is placed
  for this probe** — it reuses whatever id the session's own fill produced.

**Decision rule**
- Reject code is a `TERMINAL_CODES` member (`ORDER_IS_DONE` / `ORDER_CANCEL_STATUS_REJECTED`,
  `plugins/dnse/pynecore_dnse/errors.py:101-108`) → prod uses structured codes → the
  message-blind free-text loop is **sandbox-only** → shrink #116 to a hardening note and close.
- Generic code (e.g. `ERROR`) plus free text like "already in terminal state Filled" → **#116 is
  real on prod** → the message-gated classification fix is required, panel-gated.

**If Friday produces no fill:** #116 can still be shrunk without it. The 09-15 session measured
prod answering *structured* codes in three distinct same-day terminal-write situations
(`ORDER_IS_DONE` on a PUT to a done order; `ORDER_CANCEL_STATUS_REJECTED` on a cancel of a
terminal order, two instances; `CAN_NOT_CANCEL_PENDINGNEW_ORDER_IN_OPEN_SESSION` on a race),
with **zero** free-text observed. `ORDER_CANCEL_STATUS_REJECTED` is documented in
`errors.py:104-107` as the venue's generic "order status is not valid to cancel" answer —
Filled is in that same class. That is a strong inference, not proof: a venue may special-case
Filled. So the honest fallback is *retitle-shrink to a hardening note carrying the inference
explicitly*, not *close as proven*.

### P2 — #102: idle-synth fabricating V=0 bars during calendar-open ATC (NO orders, passive)

Confirmed STILL-REAL from code below, but a live confirmation is free if any `--live` run is
already up across 14:30-14:45 ICT. **Requires no `--broker`, places nothing.** Just read the
log of whatever run exists:

```bash
cd /home/mike/workspace/github/pynecore
sed 's/\x1b\[[0-9;]*m//g' <the-running-log>.log | grep -a "idle-bar synth emitted"
```

**Decision rule**
- One or more `idle-bar synth emitted` lines with timestamps inside 14:30-14:45 → #102
  confirmed live a second time at HEAD → keep, raise priority (a strategy sees fabricated
  prices in exactly the phase where DNSE refuses cancels and fills what rests).
- Zero such lines across a full 14:30-14:45 window on a run that was streaming before 14:30 →
  something suppresses it that the code read did not find → re-open the code question before
  acting.

*Caveat that makes this weaker than it looks:* with #135 open, the operator + leader have
concurred on **no `--broker` runs until the watermark fix lands**. A data-only `--live` run is
still safe (no order routing), so P2 stays viable — but only if someone is running one anyway.
Do not launch a run solely for this; the code evidence below is already sufficient to keep the card.

### ~~P3 — #103~~ — WITHDRAWN, settled from code

Listed as a candidate while it was still open; it is now **closed out offline** and needs no
venue time. The mechanism was located exactly (the duplicate runs in the *warmup* loop, whose
`bar_index += 1` is unconditional — see §3), so a live repro would only re-observe a known
cause. Do not spend a Friday slot on it.

### Explicitly NOT needing the venue

**#75, #79, #84, #88, #89, #97, #99, #103, #114, #115, #117** — all settleable from code, tests,
or already-recorded measurements, and all graded below. This is a real result, not a gap: of
thirteen cards, exactly **one** (#116) genuinely needs a live order book — and it needs a
*fill*, not a placement, so it costs the session nothing extra.

---

## 2. Verdict table

| Card | Verdict | Evidence (commit / file:line) | Recommended action |
|---|---|---|---|
| **75** | STILL-REAL | `residue_detector.py` `ResidueTracker.observe` takes no session input at all; `broker.py:1923-1960` `_residue_step` gates only on `all_books_readable`; zero `session_phase` refs anywhere in `pynecore_dnse/`. Premise re-confirmed: the L0 phase module is still in `testing/live_test/`, loaded by path (`tests/test_session_phase.py:17-22`), not importable from the plugin. | keep (low severity — noise, never a false cancel) |
| **79** | OUTDATED | Plugin facts still literally true (`plugins/binance/.../broker.py:838-850`, `:235`; no `UnexpectedCancelError`), but attribution is the **engine's** job: `sync_engine.py:20098` registers bot cancels, `:6670-6677` consumes them, `:7051` `_apply_unexpected_cancel_policy` is the real policy path. Binance already feeds it — `broker.py:673-674` calls `native_cancel_all_expected_sink` *before* `cancel_all_orders`. | close-with-comment; **spin off** the real residual (below) |
| **84** | STILL-REAL (all 3 sub-claims) | (a) `live_runner.py:1108-1113` states "No attempt limit" in a comment; no `max_reconnect_attempts` in `src/`; watchdog `:1496-1501`/`:1555-1567` raises into that handler. (b) one `raise_if_halted` site, `script_runner.py:2499`, inside `for bar_update in live_stream:` (#121 WAKE at `:2508-2511` narrows but does not close it). (c) `run.py:2072-2082` swallows `BrokerManualInterventionError` with no re-raise → **exit 0**. | keep, raise priority (blocks unattended operation) |
| **88** | OUTDATED as written (severity dissolved); label wart remains | `sync_engine.py:6640-6643` is the terminal `else` **outside** the `if key is not None` guard that reaches `_apply_unexpected_cancel_policy` (`:6668`) → INFO-only, `policy=stop` cannot fire. Pinned: `test_025:12316-12338` asserts `quarantined is False and halted is False`. | retitle-shrink to a cosmetic log-label nit, **or** fold the unexplained duplicate into #135 (see §3) |
| **89** | STILL-REAL | `recovery_ladder.py:88-99` reports **every** id in `strand_ids` with no `terminal_status` filter; the source, `storage.py:1924-1951`, filters only on `closed_ts_ms IS NULL` — and #73 deliberately keeps FILL-terminal rows live, so they qualify forever. `git log --since=2026-09-08 -- storage.py` → empty. | keep; mechanism now precisely named (fix the SQL predicate or the ladder filter) |
| **97** | STILL-REAL | `broker.py:2304` `_amend_normal` is still a plain `def`; `broker.py:2198` `return self._amend_normal(...)` un-awaited from an `async def`; `_write` (`:1323`) and `_order_detail_dict` (`:2299`) are both sync, so 3-6 venue round-trips block the loop. The cancel-path invariant test exists (`test_cancel_disposition.py:239`) with **no** modify equivalent. | keep — real, and #135's no-`--broker` freeze is a good window to land it |
| **99** | STILL-REAL | `validation.py:221-408` validates ~9 capability surfaces (types, idempotency, `watch_orders`, `amend_order`, coid length, residual pair, position/spot ports, short-selling, account sentinel) — `entry_stop_limit_native` is **not** among them, nor checked anywhere at startup. `git log --since=2026-09-09 -- validation.py` → empty. Only test pins the engine's disarm (`test_025:14204`), not the plugin's honouring of `intent.stop`. | keep (latent — DNSE honours it; it is a contract-validation gap) |
| **102** | STILL-REAL | `live_runner.py:1376` `_in_feed_quiet_phase()` is still nested inside `if not _market_open_at(synth_ts):` (`:1348`). DNSE declares `feed_quiet_phases = (("14:30","14:45"),)` (`provider.py:175`) — calendar-**open**, so the gate is unreachable and the synth emitter at `:1506-1541` fabricates V=0 bars. | keep, raise priority (real-money hazard in ATC) |
| **103** | STILL-REAL (real mechanism found; card's hypothesis wrong) | Seeded dedup works (`script_runner.py:2476`/`:2522`/`:2585`; pin `test_072_…::__test_warmup_bar_continued_live_counts_once__` passes). Duplicate never reaches it — both bars ran in the **warmup** loop, `script_runner.py:2307-2319`, whose `bar_index += 1` at `:2313` is **unconditional**; pre-`LIVE_TRANSITION` queued bars arrive there (`live_runner.py:519-521`, sentinel at `:1680`/`:1688`). | keep, retitle to the real mechanism; fix at `script_runner.py:2313` or in the generator |
| **114** | STILL-REAL, but much further along than the card reads | Harness LANDED (`647fbb0`, `dnse_replay_sandbox` entry point at `plugins/dnse/pyproject.toml:36`, `replay_sandbox.py:36`). Three sandbox cases now exist — `sb_sl_tp_scaleout`, `sb_tier2_limit_entry_tp`, `sb_tier2_arm_wake` (`.pine`/`.py`/`.toml` each) — but **all are UNTRACKED** (`git status` `??`). No OCA sandbox case. | retitle-shrink to "commit the sb_* cases + add the OCA case"; do **not** close |
| **115** | QUIETLY-DONE (for its stated scope) | Substantive work committed `e2f1a92`; invariant pinned red-first (`plugins/dnse/tests/test_get_position_account_net.py`); clamp covered offline (`test_divergence_matrix.py:99` matrix_a, `:238` matrix_c). The card's own final comment says "NO further test needed". Only a live-money capstone remains, explicitly deferred. | close-with-comment (move the live capstone to its own low-priority card) |
| **116** | NEEDS-VENUE-CONFIRM (§1 P1) | 09-15 prod evidence gives structured codes in 3 terminal-write situations, zero free-text; the exact Filled case still unmeasured (`logs/session0915_p116_090540.log` = six `RESOURCE_NOT_FOUND`). Probe ready: `probe_116_same_day_cancel.py`. | hold for a Friday fill; fallback = retitle-shrink on the inference |
| **117** | QUIETLY-DONE in its load-bearing part; a stated sub-ask remains | Re-map landed `47a3b575`: `broker.py:2390` `_remap_amended_order_id`, called at `:2377` and `:2473`; tested by `plugins/dnse/tests/test_amend_id_remap.py` (5 tests incl. `__test_second_amend_leg_targets_the_new_id__` and `__test_old_ids_cancel_push_is_expected_not_unowned__`). | retitle-shrink to the leftovers (below) — do not close outright |

---

## 3. Per-card notes

### #75 — residue detector stamps off-session — **STILL-REAL**

`ResidueTracker.observe(tracked_ids, present_ids, all_books_readable, now)` has no session
parameter and no caller supplies one. `_residue_step` (`broker.py:1923-1960`) computes
readability as `all(o is None for o in book_outcomes.values())` (`:2675-2677`) — and the card's
own weekend measurement showed closed books answer `200, rows=0`, i.e. *readable and empty*.
So the stamp/age path runs all night exactly as described.

The card's parenthetical premise also still holds: there is no plugin-importable session-phase
module. `plugins/dnse/tests/test_session_phase.py:8` says so in as many words ("The module lives
in the testing tree") and loads it via `spec_from_file_location` from
`testing/live_test/level0_venue_semantics/`. So the fix still needs either a plugin-local phase
helper or a promotion of the L0 module out of `testing/`.

Severity is unchanged from the card's own assessment: **noisy, not dangerous** — a conditional
can never falsely conclude CANCELLED because history is NORMAL-only. Keep, low priority.

### #79 — binance unexpected-cancel policy — **OUTDATED**

Every literal claim on the card is still true of the plugin file, which is why a code-read of
the plugin alone re-confirms it. The premise dissolves one level up: attribution was never meant
to live in the plugin. The engine keys it on `event.order.id` — bot single cancels register at
`sync_engine.py:20098` and are consumed one-shot at `:6670-6677`; bulk cancels go through
`enqueue_native_cancel_all_expected` (`:7689`) → `_handle_expected_native_cancel_all` (`:7733`),
which explicitly tears down without firing the policy; only a genuinely external cancel of a
still-mapped order reaches `_apply_unexpected_cancel_policy` (`:7051`).

And Binance does participate: `plugins/binance/pynecore_binance/broker.py:673-674` calls
`self.native_cancel_all_expected_sink(symbol)` *before* `cancel_all_orders`, with a comment
saying otherwise "the pushed CANCELED events read as external and trip quarantine". That is the
card's proposed fix, already present, in the right layer.

**Spin off, do not lose:** a genuine residual the card does not mention — `_poll_once` maps
`EXPIRED` to `'cancelled'` without `cancel_reason=CANCEL_REASON_VENUE_EXPIRED`, so a venue
expiry takes the external-cancel branch instead of the #94 exemption
(`sync_engine.py:6649-6668`). Narrower and more actionable than the original card.

### #84 — feed-liveness HALT undeliverable — **STILL-REAL (all three sub-claims)**

All three verified at HEAD; the card's line numbers had drifted but every claim re-anchors.

**(a) Reconnect is unbounded — the code says so in a comment.** `_handle_connection_error`
(`live_runner.py:1061`) carries, inside its retry `while`:

> `live_runner.py:1108-1113` — "No attempt limit: a live session must ride out an arbitrarily
> long outage (router restart, ISP drop, provider maintenance) and resume on its own. The
> exponential backoff saturates at `provider.max_reconnect_delay` …"

`attempts` is a logging/backoff counter only (`:334-342`); there is no `max_reconnect_attempts`
anywhere in `src/`. The loop exits only on `stop_event`/market-closed or on success. The
feed-staleness watchdog (`:1496-1501`, `:1555-1567`) raises straight into this handler — so
blindness escalates to nothing.

**(b) `raise_if_halted` is per delivered item only.** One call site in the runner:
`script_runner.py:2499`, the first statement inside `for bar_update in live_stream:`. Partial
mitigation has landed since the card was written — `4637b27b` (#121) added the out-of-band
`WAKE` sentinel, which is also an item and therefore also trips the check
(`script_runner.py:2508-2511`). But with no bar **and** no fill-wake, the halt is never raised.
Still real, now with a narrower window.

**(c) The halt path exits 0.** The real site is `run.py:2072-2082`: `except
BrokerManualInterventionError:` → `progress.stop()`, `stop_reason = "manual intervention
required"`, one `broker_warning`. No re-raise, no `Exit`. `stop_reason` feeds only the summary
text (`:2096`). Every `raise Exit(1)` in the file is startup validation (all ≤ `:1784`); the one
other exit is `raise Exit(0)` at `:1856`. **Exit code 0** — a halted bot is indistinguishable
from a clean stop to a supervisor, which is precisely the unattended-operation hazard the card
names.

Keep, and note the composition: (a) means the halt is rarely reached, (b) means it may never be
checked, (c) means that if it *is* reached it looks like success. Any one of the three alone
would be survivable.

### #103 — warmup→live re-delivers the last warmup bar — **STILL-REAL (card's mechanism is wrong; the real one is located)**

The seeded dedup the module docstring advertises **works**, and was proved working at HEAD
rather than assumed: `script_runner.py:2476` seeds `last_bar_timestamp = last_warmup_timestamp`,
`:2522` computes `is_new_bar`, and `:2585` `self.bar_index += 1` is gated on it. `:2838` is the
same variable in the bar-close branch of the *same* loop (not a second unseeded consumer), and
`:3063` is `_run_iter_magnified`, backtest only. The existing pin —
`tests/t00_pynecore/core/test_072_live_transition_no_every_tick.py::__test_warmup_bar_continued_live_counts_once__`
— **passes** when invoked directly.

So the duplicate never reached that loop. The evidence log shows why: bars 274 and 275 carry no
`[OHLCV]` line, while every post-handoff bar (276+) does —

```
plugins/dnse/testing/live_test/logs/idx_feed_probe.log:299-301
[09:30] bar: 274 [IDX] ...
[09:30] bar: 275 [IDX] ...
[09:45] bar: 276 [OHLCV] O=1965.5 ...   <- first live-loop bar
```

Both duplicates ran in the **warmup** loop, which has **no timestamp dedup at all**:
`script_runner.py:2307-2319` — `while next_item is not LIVE_TRANSITION:` … `self.bar_index += 1`
at `:2313`, **unconditional**, with `last_warmup_timestamp = candle.timestamp` set afterwards.

That path is reachable because `live_ohlcv_generator` delivers bars queued while the warmup loop
was still replaying, *before* it yields `LIVE_TRANSITION` (docstring `live_runner.py:519-521`;
the sentinel is yielded at `:1680`/`:1688` only once the pre-transition queue drains). The
generator's only filter is the deliberate strict `<` at `:1246-1250` — so the equal-timestamp
bar passes, by design, expecting `script_runner`'s seeded dedup to refine it. It never gets
there; it lands in the unguarded warmup loop instead. (Gap-recovery at `:1013-1045` is not the
source — DNSE has no own `backfill_closed_bars`; only the base `live_provider.py:207` exists.)

No commit since 2026-09-10 touches this (`4637b27b` WAKE and `f830a622` redaction only).

**Fix belongs at `script_runner.py:2313`** (skip or refine a warmup bar whose timestamp equals
the previous one) **or in the generator** (do not emit pre-transition bars at or below the
warmup cutoff). This also explains the card's unexplained 15m-vs-1m asymmetry: the wider the
window between warmup's end and the first live poll, the more pre-transition bars queue up.

**Separate finding, worth its own card.** `test_019_live_mode.py`, `test_070_*` and `test_072_*`
— all `@pyne`-docstring modules — **collect 0 tests under pytest at HEAD**
(`pytest <file> --collect-only` → "collected 0 items"; the core directory collects 1612 with
none of their names present), though the functions exist on a plain import. The no-double-bump
pin therefore **exists but never runs in the suite** — it had to be invoked manually to get the
pass above. That is a silent coverage hole affecting more than this card.

### #88 — late duplicate CANCELLED logs "external cancel observed" — **OUTDATED as written**, with a live loose end

The card asked two questions. The dangerous one is answered **no**:
`sync_engine.py:6640-6643` is the terminal `else` of a chain whose policy arm requires a
non-`None` key; `_apply_unexpected_cancel_policy` is only reachable at `:6668` inside
`if key is not None`. The fallback does nothing but `_blog_info(...)`. This is pinned by a test
that asserts precisely the feared outcome cannot happen —
`test_025_order_sync_engine.py:12316-12338`, `quarantined is False and halted is False`. The
sibling test at `:10262-10303` pins the *first* echo taking the `"strategy cancel confirmed"`
arm and notes the id is consumed **one-shot**, which is exactly why a *second* duplicate falls
through to the wrong label. So the residue is a cosmetic log string.

**A correction worth carrying, red-checked:** it is tempting to say the plugin's `_last_seen`
equality dedup (`broker.py:2907-2909`) explains the incident away. It does not — that dedup
landed in `d4240605` on **2026-08-18**, three weeks *before* the 2026-09-08 duplicate got
through. So the observed duplicate must have differed in `cumulative` or `raw_status`, and the
mechanism that produced it is **still unexplained**. (The #117 consumption at `:2910-2921` does
not cover it either — it is gated on `_superseded_amend_order_ids`, i.e. amend predecessors only.)

**Interaction with #135 — the reason this loose end matters.** #135's mechanism is a
non-monotonic `_last_seen` watermark under out-of-order transport arrival: a stale frame
rewrites `(cumulative, raw_status)` backwards, after which a already-seen frame reads as fresh.
That is a mechanism which *would* let a byte-identical late terminal push past the `:2908`
equality guard. So the unexplained 09-08 duplicate is plausibly an **early sighting of the #135
mechanism** — hypothesis, not a finding.

Recommended: retitle-shrink #88 to the cosmetic label nit, **and** add its 09-08 evidence
(`logs/t19_21_evidence.txt`) to #135 as a possible earlier instance. Do **not** merge them —
blast radii are disjoint (#88 is log text; #135 is position accounting), and #135 must not
inherit a cosmetic scope.

Note for whoever fixes #135: its stated fix rule ("status-only transitions at EQUAL cumulative
still pass — Canceled/Filled at same cum must flow") means a monotonic watermark will **not**
suppress a duplicate CANCELLED echo. #88's label defect survives the #135 fix by design.

### #89 — recovery report re-lists prior-session FILL-terminal rows — **STILL-REAL**

Precisely located. `classify_recovery` (`recovery_ladder.py:88-99`) emits the
"Reported only — never adopted, never cancelled" verdict for **every** member of `strand_ids`,
unconditionally. `strand_ids` comes from `store_ctx.foreign_live_exchange_order_ids`
(`broker.py:550-557`), whose SQL (`storage.py:1938-1948`) filters on
`owned.closed_ts_ms IS NULL` and nothing else — no `terminal_status` predicate.

That is the whole defect, and it composes with a deliberate design choice: #73 **keeps
FILL-terminal exposure rows live** (stated in `residue_detector.py`'s own docstring: "exposure
rows (`filled_qty>0` / terminal extras — the #73 keep-live ledger)"). So every prior session's
fills stay `closed_ts_ms IS NULL` forever, qualify as strands forever, and the report grows by
one line per fill per session — exactly the 8 lines the card measured.

`git log --since=2026-09-08 -- src/pynecore/core/broker/storage.py` → empty. Unfixed.
The card's proposed predicate (skip rows with `terminal_status` set) is still the right shape.

### #97 — `_amend_normal` blocks the event loop — **STILL-REAL**

Verified line by line at HEAD. `_amend` is `async def` (`broker.py:2145`) and ends
`return self._amend_normal(old, new, order_id)` (`:2198`) with no `await` — because
`_amend_normal` is a plain `def` (`:2304`). Inside it, `_order_detail_dict` (`:2299`) and
`_write` (`:1323`) are both synchronous, and the loop at `:2366-2380` issues one blocking PUT
per changed field. So the card's "3-6 synchronous venue round-trips per modify" stands.

The contrast the card draws is also still exact: the cancel path is `await asyncio.to_thread(...)`
throughout (e.g. `:1549`, `:1814`, `:2012`), and the invariant is pinned by
`test_cancel_disposition.py:239 __test_event_loop_stays_live_during_cancel_verify__` — for
which there is **no modify counterpart** (grep for `event_loop` in `plugins/dnse/tests/` returns
that one hit only).

Note this got *worse*, not better, since the card was written: `47a3b575` added
`_remap_amended_order_id` calls inside the same synchronous loop, so the STOCK path now does
strictly more blocking work per amend.

### #99 — `entry_stop_limit_native` unvalidated — **STILL-REAL**

`validate_plugin_contract` (`validation.py:221-408`) is thorough about everything *else*:
capability field types (`:232-253`), `idempotency` not UNSUPPORTED (`:255`), `watch_orders`
declared-vs-overridden and async-generator-ness (`:264-289`), `amend_order` NATIVE/PARTIAL_NATIVE
vs `modify_entry`/`modify_exit` (`:291-301`), `client_order_id_max_len` (`:304-317`), the
residual-orders/cancel-ref override pair (`:321-328`), position and spot port surfaces,
short-selling vs spot port (`:388-399`), account sentinel. That list is what makes the card's
"unlike other capability levels" comparison fair.

`entry_stop_limit_native` appears only at `models.py:485` (default `False`),
`sync_engine.py:1124-1125` (read), `:1948` / `:11032-11041` (disarm the software watch), and
`plugins/dnse/pynecore_dnse/broker.py:416` (DNSE declares `True`). Nothing validates that a
plugin declaring it actually honours `intent.stop`. `git log --since=2026-09-09 -- validation.py`
is empty. The existing test (`test_025:14204` plus its control at `:14226`) pins the *engine's*
disarm, which is the opposite direction.

Latent, as the card says — DNSE honours the stop (T21 live-proven). Keep as a contract-hardening
item; it becomes load-bearing the moment a second plugin declares the flag.

### #102 — quiet-phase gate dead during calendar-open ATC — **STILL-REAL**

The nesting the card describes is intact at HEAD. `_in_feed_quiet_phase()` is defined at
`live_runner.py:784-795` and referenced on the synth path at exactly one site, `:1376` — inside
the `if not _market_open_at(synth_ts):` branch opened at `:1348`. DNSE's ATC is calendar-**open**
(`feed_quiet_phases = (("14:30","14:45"),)`, `provider.py:175`), so control never enters that
branch during ATC. It falls through to `:1490-1541`, whose only guards are
`provider.is_connected` and `feed_stale_after` — neither of which knows about a quiet phase —
and then emits `OHLCV(open=…=close=last_close, volume=0.0, is_closed=True)` and queues it.

The branch docstring at `:1384-1385` still promises the opposite ("this branch also keeps the
idle-synth filler from fabricating bars for withheld slots"), so doc-vs-code drift confirms the
defect rather than excusing it.

Worth separating for whoever fixes it: the **plugin-local** quiet-phase check
(`broker.py:742`, used by the LTF feed) works and is tested
(`tests/test_ltf_feed.py:91`, `:143-164`). The defect is exclusively the framework's idle-synth
filler leg. And the card's own fix warning still applies — `:1385-1391` explains the branch is
deliberately keyed on *current* session state so a rebase cannot disarm the staleness watchdog
forever; a naive hoist reintroduces that.

### #103 — warmup→live re-delivers the last warmup bar — **UNVERIFIED at time of writing**

What I established directly: the dedup design the module docstring advertises
(`live_runner.py:29-40`) is real — the `last_historical_timestamp` filter deliberately uses
strict `<` (`:1246-1250`) so the equal-timestamp bar passes through to `script_runner`, which is
supposed to absorb it by seeding `last_bar_timestamp = last_warmup_timestamp`
(`script_runner.py:2476`) and gating on `is_new_bar = (candle.timestamp != last_bar_timestamp)`
(`:2522`).

The problem with calling this DONE: that seeding arrived in `e99fed11`, dated **2026-04-30** —
it already existed on 2026-09-10 when the duplicate was measured. So either the seeding is inert
on the path a live run takes (e.g. `last_warmup_timestamp` never assigned at `:2319` for that
path, leaving it `None`), or a second consumer loop (`:2838`) keeps its own unseeded
`last_bar_timestamp`. I ran out of time to finish that trace.

**Settle offline.** The card's own discriminating check — one venue timestamp appearing under
two `bar_index` values — is readable from a saved log, and the 15m-vs-1m asymmetry it noted
(1 duplicate at 15m, 0 at 1m) points at launch-offset-within-bar as the variable. A controlled
offline harness beats a venue slot here. Recommend: keep, flagged UNVERIFIED-at-audit, with the
trace above as the next step.

### #114 — Sandbox Replay E2E NORMAL-order suite — **STILL-REAL, but the card understates progress**

Two of the four build-plan steps are done and the card does not reflect it in its title:

- Step 1 (harness) and step 3 (prove on `l2_fill_flatten`) LANDED in `647fbb0` — the
  `dnse_replay_sandbox` entry point is registered (`plugins/dnse/pyproject.toml:36` →
  `replay_sandbox.py:36 ReplaySandboxBroker`), with the 24/7 syminfo override at
  `replay_sandbox.py:143` and the flat `get_position` stub at `:151`.
- Step 4 (the cases) is **partly written but entirely uncommitted**: `sb_sl_tp_scaleout`,
  `sb_tier2_limit_entry_tp`, and `sb_tier2_arm_wake` each exist as `.pine` + transpiled `.py` +
  `.toml`, and `git status` shows every one of them as `??`. `sb_tier2_arm_wake` even has a
  recorded `.evidence.txt` and `.sandbox_e2e.log`.

So the real remaining work is (a) commit the three `sb_*` cases with their evidence, and (b)
author the missing **OCA** case — the only one of the card's four behaviours (SL / TP / PTP /
OCA) with no sandbox counterpart. The existing OCA assets (`live3_oca_cancel.pine`,
`live_oca_entry_group.pine`) are prod live-tier, not sandbox.

One dependency note that is now stale: the card's 2026-09-12 comment blocks this on #115. #115
is done (below), so #114 is unblocked. The `get_position` question it raised was also resolved
in that card's own later comment — fill-derived tracking is the prod-correct model, not a crutch.

### #115 — per-strategy position isolation — **QUIETLY-DONE**

The card converged after three reframings and its own final comment states the conclusion:
"NO further test needed". Verified at HEAD:

- `plugins/dnse/tests/test_get_position_account_net.py` exists — the load-bearing invariant
  (`get_position` returns the whole-account net, unfiltered) is pinned red-first, which is what
  #73/C2's absence-proof depends on.
- The clamp coverage the card claims is real: `plugins/dnse/tests/test_divergence_matrix.py:99`
  `__test_matrix_a_startup_adoption_is_clamped_to_nothing__` and `:238`
  `__test_matrix_c_nothing_prevents_two_engines_on_one_account__`.
- Committed `e2f1a92`.

The task brief framed #115's ask as "a fill-level dual-engine sandbox test". The card itself
already refuted that as the wrong instrument: the sandbox does **not** net, `get_position` is
stubbed `None`, and `_clamp_adoption_to_owned` only runs when `exch_pos is not None` — so a
sandbox dual-engine test would never exercise the clamp. That reasoning holds against the code.

Close with a comment; open a separate low-priority card for the live-money two-run capstone so
it is not lost in a closed card's body.

### #116 — message-blind cancel disposition — **NEEDS-VENUE-CONFIRM**

See §1 P1 for the command, gates, and decision rule. Summary of state: the classification is
still code-only (`errors.py:200-201` `if code in TERMINAL_CODES`, with the set at `:101-108`),
the sandbox 166x loop is pinned by an `xfail(strict)` repro in `test_cancel_disposition.py`, and
prod has answered structured codes in every same-day terminal-write situation measured so far —
but never yet for the exact Filled-status cancel, because cross-day ids die at
`RESOURCE_NOT_FOUND` first (six of them in `logs/session0915_p116_090540.log`).

### #117 — per-market order semantics — **QUIETLY-DONE in its load-bearing part**

The dangerous half landed in `47a3b575` and is well tested. `_remap_amended_order_id`
(`broker.py:2390-2430`) adopts the id a successful amend returns and moves every tracked
reference; it is invoked on both write paths (`:2377` in the per-leg loop, `:2473` in the qty
path), and guarded on the id actually changing so the derivatives path is byte-identical.
`plugins/dnse/tests/test_amend_id_remap.py` pins five behaviours, including the two that matter
most: `__test_second_amend_leg_targets_the_new_id__` (the latent #86 bug the card predicted as
risk 1) and `__test_old_ids_cancel_push_is_expected_not_unowned__` (the predecessor's `Canceled`
push, consumed in `_scan_row` at `broker.py:2910-2921`).

**Three stated asks remain**, which is why this should shrink rather than close:

1. **The one-PUT-both-fields STOCK branch was not built.** The card proposed "add a
   `market_type == STOCK` branch — one PUT with BOTH fields". `_amend_normal` (`:2304`) still
   has no `market_type` branch; a STOCK amend changing both price and qty still issues **two**
   PUTs (`:2344-2348` builds two payloads). It is now *correct* (each leg re-maps), but on a
   cancel-replace venue that means two sequential replacements — two lost queue positions and
   two predecessor `Canceled` pushes where one would do. Measured 09-15: STOCK accepts both
   fields in one PUT. Worth doing, no longer urgent.
2. **The BROADER audit was not done.** The card explicitly asks to audit *all* market-type
   order paths (place / cancel / positions), not just amend. `47a3b575` covers amend only.
3. Interaction to state when shrinking: `_remap_amended_order_id` mints a new id mid-run, which
   is exactly the case #135's monotonic-watermark fix must be validated against (the #135 card
   already names "#117 replace-mints-NEW-id" as a required validation case). Keep #117 open so
   that link is visible.

---

## 4. Confidence notes

**High confidence (read the code myself, at HEAD, this session):** 75, 89, 97, 102, 114, 115,
116, 117. Each verdict above cites the line I read, not a line the card claimed.

**High confidence, delegated + spot-checked:** 79, 84, 88, 99, 103.

Two spot-checks are worth recording because one of them changed a verdict:

- **#88 — a delegated claim I red-checked and rejected.** The result argued the plugin's
  `_last_seen` equality dedup (`broker.py:2907-2909`) now suppresses the duplicate. `git log -S`
  puts that dedup at `d4240605`, **2026-08-18** — three weeks *before* the 09-08 incident, so it
  cannot be what fixed it. The verdict survives on a different, verified fact (the INFO-only
  branch at `:6640-6643` plus its pinning test at `test_025:12316`), but the "already
  suppressed" explanation is struck and replaced with the open question recorded in §3.
- **#84 — spot-checked and confirmed.** I re-read `live_runner.py:1108-1113` (the "No attempt
  limit" comment) and `run.py:2072-2082` (the swallowed `BrokerManualInterventionError`) myself.
  Both cites are accurate as quoted.

I did not independently re-verify every line cite in the #79, #99 and #103 results.

**One verdict rests on a manually-invoked test, not a suite run.** #103's "the seeded dedup
works" leg was established by invoking
`test_072_live_transition_no_every_tick.py::__test_warmup_bar_continued_live_counts_once__`
directly, because that module — like `test_019_live_mode.py` and `test_070_*` — **collects zero
tests under pytest at HEAD**. Treat the pass as real but unguarded by CI, and see the
collection-hole finding in §3 (#103), which deserves its own card.

**Two things I deliberately did not do:** close or comment on any card (the leader does that),
and recommend launching a live run purely to satisfy an audit item — with #135 open and
`--broker` frozen by operator/leader concurrence, only #116 (which piggybacks on a fill that a
session produces anyway) and #102 (passive log read, data-only) are worth venue time at all.

**Base-rate caution on the "quietly done" hypothesis:** the operator suspected several of these
were quietly done or outdated. Of thirteen: **one** is fully done (#115), **one** is done in its
load-bearing part with stated sub-asks left (#117), **two** are dissolved without a fix — #79 by
architecture (the engine always owned attribution) and #88 by construction (the feared policy
cannot fire). That leaves **eight still real** (75, 84, 89, 97, 99, 102, 103, 114) and **one
needing the venue** (116). The suspicion was right in kind and wrong in degree — roughly a third
of the list moved, not most of it.

**Two cards got *worse* since they were written**, which is the opposite of quietly-done and
worth surfacing: #89 grows by one report line per fill per session (unbounded), and #97's
blocking amend path gained more synchronous work in `47a3b575`.

**Three findings here are not on any card yet** and need capturing so they are not lost with
this document: the Binance `EXPIRED`→`cancelled` mapping that misses the #94 exemption (§3 #79),
the pytest collection hole swallowing `test_019`/`test_070_*`/`test_072_*` (§3 #103), and the
still-unexplained 09-08 duplicate push that may be an early #135 sighting (§3 #88).
