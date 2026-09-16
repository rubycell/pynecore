# Live session plan — 2026-09-16 (Wed)

Operator supervises; operator fires every order-placing command; Fable preps, monitors,
grades from the venue record, and does read-only venue calls + cancels-of-our-own-test-orders.
All fill work at 1m unless noted. Account must be FLAT + token GOOD + L0 exit 0 before any
live step (same gates as 09-15).

**Calendar (shapes today):** Thu 09-17 = expiry (final trade date). Fri 09-18 morning = the
alias roll (#113 test is FRIDAY, not today). Today is the last normal day of the contract
month: any GTD conditional clamps to TOMORROW (#118's clamp at its tightest — observe for
free), and expiry-week-only measurements (B3 family) exist only this week. Do NOT hold a
position into expiry — every fill test self-flattens or is operator-closed same session.

**Context from 09-15 (what today builds on):** #124 CLOSED — a venue-cancelled protective
exit now RE-ARMS (bounded 1/3), no quarantine, no naked position (live-proven). FAST ARM
proven: a bracket PRE-PLACED on the entry bar arms SAME-bar/timestamp as the fill; reactive
placement still costs a bar. #18 closed (cancel+replace clean). #128 opened: venue-cancel
root cause — OCO `currentAction` leg-transition LEAD (from order metadata) + the unexplained
bare-STOP-entry cancel (10:56, ~200s, no OCO machinery).

---

## Morning prep (offline, before 09:00 — Fable, no venue writes)

### 1. The OCO-transition instrument (#128 thread 1 — the multiplier)
Extend `_observe_oco_cancel` (plugins/dnse/pynecore_dnse/broker.py, the #124-OBS line).
NOTE (double-checked 09-16): `currentAction`'s VALUE is NOT exposed by any read we have —
the umbrella's OCO detail carries no such field and no metadata; `currentAction` appears only
as a VARIABLE inside the child's metadata `condition` string. So log what is provably
readable: on an OCO-child cancel, (a) the umbrella's `orderStatus` + `externalOrderId`
(a NEW child id = leg-transition/respawn evidence — the actual discriminator), and (b) the
cancelled CHILD's metadata fields `cancel_ip`, `originCategory`, `eventNo` (was 4 on the
09-15 cancel — possibly a transition counter) and `condition` (once per key is enough).
The transition is INFERRED from child succession + eventNo, which is measurable. + fake-venue test.
WHY FIRST: it turns every bracketed fill today into a #128 root-cause measurement for free.
Verify: plugins/dnse tests green; the new field appears in the fake-venue observation test.

### 2. Prep the vehicles
- Re-transpile `l3_fill_fixed_sl_tp.pine` (local pine2pyne) + oracle spot-check (backtest
  places zero trades live-shape; oracleMode replays round-trips offline).
- Write `probe_bare_stop_lifecycle.py` (#128 thread 2): place ONE bare STOP entry (no OCO)
  ~5% from market (no-fill), watch >=5 min (yesterday's cancel came at ~200s — the 120s
  watch was too short), log every transition + final metadata (`cancel_ip`, `originCategory`),
  auto-cancel + verify terminal on exit. Dry-run default, `--yes` to place. Model:
  probe_124_oco_lifecycle_isolation.py.
- Extend `probe_ws_market_data.py` for step 5: a `--broker-channel` mode (or per-channel
  frame COUNTERS when both are subscribed) — verified 09-16 that `--trading` subscribes the
  SHORT channels only and normalized frames don't record which channel delivered, so without
  this step 5 cannot discriminate. Keep the change probe-local.
- One-command runners for steps 3 and 6, same shape as run_l2b_orig_fill.sh
  (pre-flight flat+token -> L0 -> run -> teardown -> summary).

---

## Market hours (operator fires; gates: FLAT + token GOOD + L0 exit 0)

### 3. THE HEADLINE — `l3_fill_fixed_sl_tp` live: does the SL actually FIRE? (fill tier)
The one thing never yet proven live: every position so far was closed by a backstop or a
venue cancel — the SL/TP itself has never done its job. l3 = market entry qty=1, bracket
FROZEN at the fill (SL below / TP above, OCO), no trailing, no chasing. Unblocked by #124.
**RAISE `maxHoldBars` for this run** (double-checked 09-16): the default 3 bars ≈ 3 min at 1m
almost guarantees the FLATTEN backstop fires before a 0.20% leg is reached — defeating the
test. Set it via the script's `.toml` inputs to 30-60 at 1m (or run 5m with 6-10). Exposure
stays bounded by the backstop + supervision; the point is giving a leg room to resolve.
Grade from the venue record:
- SL/TP rests on the conditional book after the fill. EXPECT a ONE-BAR arm lag: l3 derives
  its levels from position_avg_price, so it is REACTIVE by design — acceptable here, the
  QUESTION is firing, not latency (fast-arm-via-pre-placement was proven 09-15 on l2b).
- If price crosses a leg: the conditional Activates -> normal-book child fills -> position
  closes at ~the level. THAT is the missing proof.
- If the venue cancels the bracket: the #124 re-arm covers it AND the step-1 instrument
  logs the umbrella child-succession + child metadata = the #128 thread-1 measurement.
- Backstops bound exposure (maxHoldBars flatten, adverse double-check). Supervised, app open.

### 4. #116 — same-day cancel-of-a-FILLED id (~30 s, piggybacks step 3's fill)
Immediately after any fill today, cancel THAT id (probe_116_117 A pattern, or venue.py).
Yesterday cross-day ids answered RESOURCE_NOT_FOUND; a same-day filled id gives the real
reject signature. TERMINAL_CODES member -> #116 is sandbox-only -> shrink/close the card.

### 5. #121 — WS BROKER-channel dual capture (~5 min, read-only + one order event)
Run the WS watcher subscribing BOTH order channels (short `order.DERIVATIVE.json` — proven
09-15 — AND broker `order.broker.{mt}.{investor}` — never captured) across one place+cancel
(step 3's entry, or a far LO). Whichever delivers decides what `ws_order_source.py` should
trust; both-deliver validates the dual-subscribe as redundant. Closes the WS half of #121/#107.

### 6. #128 thread 2 — the bare-STOP-entry cancel probe (no-fill, >=5 min watch)
Run the step-2 probe: does a bare, in-GTD, untriggered STOP entry get venue-cancelled again
(reproducible) or not (one-off)? If cancelled: the final metadata (`cancel_ip`,
`originCategory`, timing) is the discriminator — a bare stop has NO OCO machinery, so
whatever cancels it is a SECOND phenomenon. Variant worth one extra run if time allows:
same probe with durationType=DAY vs GTD (yesterday's survivor was DAY, the cancelled ones GTD).

### 7. Opportunistic — #124 re-arm escalation (no dedicated run)
If the venue cancels a re-armed bracket AGAIN during step 3: watch the ladder — 2nd cancel
-> WARN (2/3), 3rd -> LOUD quarantine with the counter. Validates the bound end-to-end.
Do not force it; it only happens if the venue obliges. If it never fires today, the
fake-venue tests remain the coverage.

---

## Housekeeping (Fable, anytime)
- Close/annotate #118 and #119 if their shipped fixes + 09-15 live evidence complete them.
- Board hygiene: #108/#109/#110 — done-in-practice check.
- After the session: grade everything from the venue record, update cards (#128 with the
  currentAction datum, #116, #121), CLAUDE.md/memory only if a NEW venue fact was measured,
  commit in one pass.

## Stop rules (unchanged)
- L0 non-zero -> nothing runs. Account not provably flat -> nothing runs.
- No position held into ATC (14:30) or into expiry day.
- Any naked-position state: flatten via app first, diagnose second.
- venue.py flat must exit 0 at session end; park raw logs, commit stripped evidence only.

---

## EXECUTION HANDOFF — artifacts, exact commands, state (written 2026-09-16 ~10:45 ICT)

For an agent (or the operator) executing this plan WITHOUT the authoring session's context.
HARD RULES for any executing agent: the OPERATOR fires every order-placing command (steps
3, 6 --yes, and the L0-bearing runners); the agent preps, monitors logs, grades from the
venue record, and may do read-only venue calls + cancels of OUR OWN test orders only.
NEVER sweep. NEVER hold into ATC (14:30) or expiry (tomorrow). Grade from the VENUE record.

### Pre-flight checklist (verify these EXIST before executing; all paths repo-relative)
| Artifact | Path | State |
|---|---|---|
| This plan | docs/plan/sep16-fable-plan.md | you are here |
| l3 strategy (.pine source of truth) | plugins/dnse/testing/live_test/l3_fill_fixed_sl_tp.pine | ready |
| l3 transpiled (regen 09-16, current) | plugins/dnse/testing/live_test/l3_fill_fixed_sl_tp.py | ready |
| l3 inputs (maxHoldBars=45 set 09-16) | plugins/dnse/testing/live_test/l3_fill_fixed_sl_tp.toml | ready |
| Step-3 runner | plugins/dnse/testing/live_test/run_l3_sep16.sh | ready |
| Step-4 helper (#116 same-day cancel) | plugins/dnse/testing/live_test/probe_116_same_day_cancel.py | ready |
| Step-5 runner (WS dual capture) | plugins/dnse/testing/live_test/run_ws_dual_capture.sh | ready |
| Step-5 probe (needs broker-channel ext) | plugins/dnse/testing/live_test/probe_ws_market_data.py | EXTENSION IN FLIGHT — verify `--trading` subscribes BOTH channels + per-channel counters before step 5; if absent, the extension agent has not landed: check `git log`/working tree |
| Step-6 probe (bare stop, --duration A/B) | plugins/dnse/testing/live_test/probe_bare_stop_lifecycle.py | IN FLIGHT — must exist + `--help` render before step 6 |
| Step-6 runner | plugins/dnse/testing/live_test/run_bare_stop_probe.sh | ready |
| #128-OBS instrument (broker.py extension) | plugins/dnse/pynecore_dnse/broker.py (`_observe_oco_cancel`) | IN FLIGHT — verify a `#128-OBS` line exists in the code + plugins/dnse tests green before step 3 |
| Venue tools | plugins/dnse/tools/{venue.py,token_status.py} | ready (status/flat/cancel/order; exit 2 = could-not-determine) |
| L0 gate | plugins/dnse/testing/live_test/level0_venue_semantics/l0_order_semantics.py | ready |
| Logs land in | plugins/dnse/testing/live_test/logs/ | park raw logs to backup/deleteable/ after grading |

If any IN-FLIGHT artifact is missing, run offline: `.venv/bin/python -m pytest plugins/dnse/tests/ -q --maxfail=20`
must be green (>=630 passed / 1 xfailed baseline) before ANY live step.

### Exact commands, in session order (operator fires; `!` prefix in Claude Code)
```bash
# STEP 3 (headline; fire 13:00-13:10 ONLY — with maxHoldBars=45 a morning fill cannot
# reach the backstop before the 11:30 lunch and would strand an OPEN position mid-lunch
# (worker-caught 09-16 10:51); latest fire ~13:30 so 45 bars + teardown end before 14:30 ATC):
bash plugins/dnse/testing/live_test/run_l3_sep16.sh
#   grade: FILLED leg=entry -> next-bar EXIT dispatch (reactive lag EXPECTED) -> SL/TP rests
#   -> a leg FIRES (conditional Activated -> child fills -> position closes) = THE PROOF.
#   If venue cancels bracket: expect '#124 ... re-arming (N/3)' + '#128-OBS' metadata lines.

# STEP 4 (within the SAME day as step 3's fill; <id> = the FILLED entry's numeric id from the log):
.venv/bin/python plugins/dnse/testing/live_test/probe_116_same_day_cancel.py <id>
#   record http/code/message verbatim onto card #116.

# STEP 5 (anytime; ~4 min; L0 inside it generates the order events):
bash plugins/dnse/testing/live_test/run_ws_dual_capture.sh
#   verdict per the runner's guide -> card #121.

# STEP 6 (quiet stretch; GTD first, DAY variant if time):
bash plugins/dnse/testing/live_test/run_bare_stop_probe.sh            # dry run first
bash plugins/dnse/testing/live_test/run_bare_stop_probe.sh --yes
bash plugins/dnse/testing/live_test/run_bare_stop_probe.sh --yes --duration DAY
#   verdict + cancel-evidence -> card #128 (thread 2).

# TEARDOWN (always, end of session):
.venv/bin/python plugins/dnse/tools/venue.py status
.venv/bin/python plugins/dnse/tools/venue.py flat     # MUST exit 0
```

### After the session (agent work, no venue needed)
Update cards: #128 (step 3's #128-OBS data + step 6 verdicts), #116 (step 4 code), #121
(step 5 verdict). CLAUDE.md/memory ONLY for genuinely new venue facts. Commit in ONE pass
(exclude raw logs; stripped evidence only). Scrub-gate every card publish:
`python3 ~/.claude/skills/backlog-startcard/scrub_check.py <file> --strict --repo <repo-root>`
and ALWAYS `--repo rubycell/pynecore` on gh commands (fork default = upstream trap).

### Known facts the executor must not re-derive (measured, do not re-investigate)
- Reactive bracket (l3's design) arms ONE BAR after the fill — expected, not a bug (fast
  arm needs entry-bar pre-placement; proven 09-15 on l2b — different design, not l3's).
- A venue cancel of the bracket is SAFE: engine re-arms (1/3->3/3 loud quarantine). #124 closed.
- Conditional re-placement is a clean cancel+replace (#85); the #18 HTTP-500 loop is stale.
- Cross-day numeric ids don't resolve on the cancel endpoint (RESOURCE_NOT_FOUND).
- `Activated` on an OCO umbrella is its from-birth state, NEVER "triggered" by itself.
- INVALID_TRADING_TOKEN on conditional writes after the operator's first EntradeX app trade
  is the #51 session-binding, NOT a token problem — do not re-mint, reschedule.

---

## AFTERNOON ADDITIONS (operator-approved ~13:00) — parallel to step 3, offline only

Fable-side (agents), none touching the running step-3 engine or the frozen tree's live code:
- A1. **#120 red-first repro** — refused protective EXIT re-raises fatally -> process dies
  holding a naked position. Repro test (test_025 style, xfail idiom) + panel; FIX lands
  tomorrow (expiry day = fix day), not today.
- A2. **#122 red-first repro** — flatten sweep-cancel on a FLAT account -> quarantine +
  phantom re-place (explicitly NOT covered by the #124 open-position guard). Same shape:
  repro + panel today, fix tomorrow.
- A3. **Token-ops hardening diagnosis** — why the 08:00 refresh cron silently missed
  (empty log), plus the gate nit (token_status exits 0 on NOT GOOD; runners echo without
  gating). Diagnose + propose; crontab changes are operator-side.
- A4. **docs/dnse-openapi-documentation/ update** — record the measured WS contract
  (#130 NORMAL-book-only streaming, #131 broker-channel refusal, the auth/channel facts)
  WITHOUT editing fetched mirror files (fetch_docs.py would overwrite): follow the dir's
  convention for local annotations, or add a MEASURED_FACTS-style local notes file.
- A5. Wrap-up card list grows: WS-vs-poll fill-latency A/B (enabled by #129),
  differential-poll design review (conditional book is poll-only per #130 -> its cadence
  is a protection parameter), engine-attached bare-stop discriminator (the next #128
  thread-2 measurement); add the DNSE-support question (broker-channel role?) to #131 —
  operator sends the ticket.

## FRIDAY 2026-09-18 SESSION OUTLINE (the roll morning — measurements only it can host)
1. **#113 roll measurement** (AT OPEN, before anything else): does resolve_contract's
   per-instance cache serve the STALE dated code after the Fri-morning alias repoint?
   Fresh process vs long-lived process A/B if possible.
2. **WS-vs-poll fill-latency A/B** (first live run with the #129 fix applied): one fill,
   measure fill-event arrival WS push vs 0.5s poll; quantifies the arm-on-fill gain.
3. **Engine-attached bare-stop discriminator** (#128 thread 2): minimal strategy places
   ONE far stop entry and idles, full instrumentation on (every-cancel log + #128-OBS);
   cancelled like 09-15's 10:56 -> the logs now name the actor; rests -> the discriminator
   moves to "something the 09-15 run DID" (its chasing replaces).
Order matters: #113 first (it exists only at open), then the fill A/B, then the idle probe.
