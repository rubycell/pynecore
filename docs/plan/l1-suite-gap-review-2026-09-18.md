# L1 live-test suite — coverage gap review (2026-09-18)

Read-only review. Question answered: which L1 (no-fill) cases can we add. Every
MEASURED cell below cites a registry row with a date, an evidence file under
`plugins/dnse/testing/live_test/logs/`, or a card. Nothing is stated from memory.
Worktree: `/home/mike/workspace/github/pynecore-worker2` at `a16e977d`.

Naming: the registry's highest used L1 id is **T33** (`Live-L1-T33-CrossProcessCancel`,
planned row in README "Planned cases (v3 queue)"). New proposals start at **T34**.

## 1. Coverage matrix

Cell values: **MEASURED LIVE** (case id + date), **OFFLINE ONLY** (backtest / fake venue /
unit test), **NOT COVERED**. "Direct probe" means a script that drives `DNSEBroker`
without Pine or the engine (the L0 pattern).

### 1.1 Timeframe (engine-driven, bar clock)

| Timeframe | Status | Evidence |
|---|---|---|
| 1m | MEASURED LIVE | T01–T13 (registry 08-13/14/17; re-run 09-07 per registry, `nofill_rerun*_evidence.txt`); T19–T21 09-08 (`t19_21_evidence.txt`, "2026-09-08" inside); T22 09-09 (`combo_t22_evidence.txt`) |
| 15s | MEASURED LIVE, ladder only | `Live-L1-T19/20/21@15S-LtfSynthesis` 09-09 (`t19_21_15s_evidence.txt`: "WS connected and subscribed: DNSEBroker VN30F1M@15S", "LTF tick source subscribed: 41I1G9000 board=G1 (WS per-print, #100)"). Tick-delivery grading (Live-L4-T04) has no log and its registry row is "planned" |
| 5s | OFFLINE ONLY | #100 body names "closed 15S/5S bars"; `plugins/dnse/tests/test_ltf_feed.py`, `test_tick_aggregation.py`. No live log names `@5S` (grep of `logs/` for `@5S` returns nothing) |
| 3m | one untracked-in-registry smoke | `logs/staged_3m.log`: `VN30F1M@3`, 2026-08-13 13:46–13:52, one PLACE line; no registry row |
| 1m/3m/5m passive bars | MEASURED LIVE (L4) | Live-L4-T01/T02/T03 08-17 (`level4_data_parity/logs/l4_20260817_*.json`) |

### 1.2 Order type (derivative, L1 no-fill unless stated)

| Order type | Status | Evidence |
|---|---|---|
| Limit (NORMAL book) | MEASURED LIVE | T01/T02 08-13, 08-17, 09-07 (registry + `regression_a_evidence.txt`, `nofill_rerun_evidence.txt`); L0 `part_limit` pre-open 09-18 (`l0_20260918_preopen_limit_0815.log`: ids 166/176 Canceled filled=0) |
| Market | N/A at L1 (fills) | L0 market part runs only at lunch (`l0_order_semantics.py` docstring, measured 2026-08-13); fills at L3: F01/F02 09-07 (registry) |
| Stop (conditional) | MEASURED LIVE | T03 exit stop 08-13/17; T19 conditional entry cancel+replace 09-08 (`t19_21_evidence.txt`); L0 part 2 every session, pre-open 09-18 (`l0_20260918_preopen_0811.log`) |
| Stop-limit | MEASURED LIVE | T21 both-set → one conditional stop-limit 09-08 (`t19_21_evidence.txt`); L0 part 3 09-18 pre-open |
| OCO (native umbrella) | MEASURED LIVE, cancel path only | T05 08-14/17 (registry); L2-BracketFill "⚠️ 08-12 partial" (registry). Note #152 (open): `venue.py status/flat` never scan the OCO book, so any L1 grading of an umbrella must read it by id |
| OCA groups (engine) | MEASURED LIVE | T04 08-13/14/17, T11/T12/T13 08-17 (`regression_c/d_evidence.txt`) |

### 1.3 Asset type

| Asset | Status | Evidence |
|---|---|---|
| Derivative VN30F1M (qty 1) | MEASURED LIVE | every L1 row |
| Stock HPG-style (lot 100) — direct probe | MEASURED by direct probe, no registry row | CLAUDE.md "STOCK amend is CANCEL+REPLACE with a NEW id ... measured 2026-09-15" (#117 probe, `probe_116_117_prod_premises.py` part B). No file under `logs/` carries it; no `Live-L1` id exists for it |
| Stock — engine-driven (Pine → engine → plugin) | NOT COVERED | no `.pine` under `live_test/` names a stock symbol; #117 open: "REMAINING: per-market amend field rules + routing audit"; #119 (closed) stock price unit; #125 (closed) venue.py stock book |
| Stock — offline | OFFLINE ONLY | `tests/test_stock_price_unit.py`, `test_stock_price_guards.py`, `test_amend_id_remap.py` |

### 1.4 Contract aliasing, roll, expiry

| Item | Status | Evidence |
|---|---|---|
| VN30F1M alias → dated code | MEASURED LIVE | every L0 log prints the contract (08-12 `41I1G8000`; 09-18 `41I1GA000`); #113 body: prod `/market/instruments` read 2026-09-12 |
| VN30F2M alias | MEASURED LIVE (L0 `--symbol`) | commit f8cef967 (2026-09-17): "VN30F2M, any / next month ACCEPTED x4" |
| VN100F1M alias | NOT COVERED | not named in the registry, any staged script, or L0 (`SYMBOL = "VN30F1M"`) |
| Monthly roll (morning after third Thursday) | NOT COVERED (attempted, lost) | `FRIDAY_2026-09-18_RUNBOOK.md`: "the roll went unmeasured", probe died 09-17 08:04; at 06:40 Friday the alias already resolved to `41I1GA000`; "Next chance: the October expiry" |
| Holiday walk-back of expiry | OFFLINE ONLY | CLAUDE.md (operator-confirmed 09-14); `tests/test_expiry_dates.py` |
| Stale per-instance cache (#113) | NOT COVERED | `probe_113_roll_cache.py` exists; no completed run (runbook above); #113 open |
| Expiry-day GTD clamp (#118) | MEASURED LIVE | commit f8cef967 2026-09-17: L0 on the expiring contract, 07:30Z ACCEPTED x4, 07:45Z REFUSED x4, next-day REFUSED x4. Offline: `test_gtd_expiry_clamp_starved.py`, `test_gtd_expiry_resolution.py` |

### 1.5 Transport

| Item | Status | Evidence |
|---|---|---|
| REST polling of both books | MEASURED LIVE | every L1 row; T18 08-18 sub-second round trip (`t15_t16_t17_evidence_20260818.txt`) |
| WS order events, NORMAL-book id, captured by a probe | MEASURED LIVE 2026-09-16 | `sep16_evidence.txt` STEP 5 run 2: 4 `do` frames for NORMAL id 99256, zero for the conditional id; card #130 |
| WS order events for a conditional id | MEASURED ABSENT 2026-09-16 | same file; #130: "conditional lifecycles never appear" |
| Broker-role WS channel | MEASURED REFUSED 2026-09-16 | `sep16_evidence.txt`: `SUBSCRIBE_FAILED`; #131 |
| WS order events consumed BY THE ENGINE live (`ws_order_source`) | NOT COVERED | #129 (closed): "WS order source has NEVER started live (fix banked)"; #146: F13 WS arm planned. No F13 log exists in the worktree (F13 logs are untracked files in the main checkout and outside this review's scope) |
| Sub-minute per-print market-data WS feeding bars | MEASURED LIVE 09-09 (alive, cadence) | `t19_21_15s_evidence.txt`; registry: 18 bars, clean 15 s cadence. Who-closed-each-bar grading (L4-T04) NOT RUN |

### 1.6 Order modification rules

| Rule | Status | Evidence |
|---|---|---|
| Derivative NORMAL amend, one field per PUT | MEASURED LIVE | T06 08-14/17; T20 dual-field split, same id 174556, 09-08 (`t19_21_evidence.txt`); #86 |
| Derivative conditional amend → PUT 500 | MEASURED LIVE | T07 08-14/17 (#18) |
| Derivative conditional modify = cancel+replace | MEASURED LIVE | T15 08-18 (gap 99 ms, `t15_t16_t17_evidence_20260818.txt`); T19 09-08 (#85) |
| Replace inside the ack-lag window (NORMAL) | MEASURED LIVE | T17 08-18 |
| Stock amend = cancel+replace, NEW id, both fields in one PUT | MEASURED by direct probe 09-15 (CLAUDE.md, #117); engine path NOT COVERED | #117 open; `test_amend_id_remap.py` offline |
| Stock structured reject codes (`ORDER_IS_DONE`, `ORDER_CANCEL_STATUS_REJECTED`, `CAN_NOT_CANCEL_PENDINGNEW_ORDER_IN_OPEN_SESSION`) | MEASURED by direct probe 09-15 (CLAUDE.md) | no registry row, no log in the tree |
| Bracket (OCO) exit modify | fill tier (F9 planned) | registry "Planned fill-tier additions"; #93 |

### 1.7 Session phases (ICT)

| Phase | Status | Evidence |
|---|---|---|
| Pre-open (before 08:45) — direct | MEASURED LIVE 2026-09-18 | `l0_20260918_preopen_0811.log` (4 conditionals accepted, cancelled), `l0_20260918_preopen_limit_0815.log` (NORMAL 166/176 + 4 conditionals); commit a16e977d; #156 |
| Pre-open — Pine-driven | NOT COVERED (structurally) | no bars arrive before the first slot; README T32 row: "bars don't arrive during pure ATO" |
| ATO 08:45–09:00 | NOT COVERED | #156: "Decide the ATO window 08:45-09:00. NOT measured today." T32 placed on the 09:00 POST-ATO bar |
| POST-ATO 09:00–09:15 | MEASURED LIVE | T32 08-25 (`t32b25_evidence.txt`) |
| Continuous | MEASURED LIVE | all L1 rows |
| Lunch 11:30–13:00 — direct | MEASURED (L0 docstring, 2026-08-13; no separate evidence file) | `l0_order_semantics.py` header |
| Lunch — Pine-driven cancel across the boundary | one incidental observation | registry 09-07 note: T9 cancel ACKed 11:29, venue held `PendingCancel` through the break |
| ATC 14:30–14:45 | MEASURED LIVE | T14 08-18 (`t14_rerun.log`: 8 `ATC_SESSION` refusals); L4-T03 08-17 no bars in ATC; T22 09-09 expiry routing |
| Post-close (after 14:45) | MEASURED informally 2026-08-13 14:51 (L0 docstring); T33 script exists, never run | `t33_closed_hours.py`; registry planned row |
| Weekend / closed day books | MEASURED 2026-09-06 | README venue facts ("readable-and-EMPTY") |

## 2. Proposed L1 cases

Common safety envelope unless stated: 1 contract (derivative) or 1 board lot (stock);
every price ≥4.5 % from the last close so it cannot fill; L0 gate exit 0 first; grade from
`venue.py order <id>` and `/orders/history`, never the run log alone; account flat and
clean (`venue.py flat` exit 0) before and after; for any case that pre-places a bracket,
read the OCO umbrella by id (#152).

### T34 — Live-L1-T34-ColourAlternatorChurn (the operator's proposed case)

Procedure. A `.pine` state machine, `calc_on_every_tick=false`: on each closed bar inside
the window, first cancel whatever rested from the previous bar, then if the bar closed
green place a far limit long (close × 0.95, NORMAL book), if red a far limit short
(close × 1.05). Run N=10 bars. Two legs: **T34a at 1m** and **T34b at 15S** (the `@15S`
feed proven 09-09). Optionally a third arm with the WS order source enabled (post-#129)
if the leader wants the engine-consumption measurement inside the same run.

What it measures that nothing else does. Sustained one-order-per-bar NORMAL-book churn
under the bar clock. T01/T02 place once and cancel once; T19–T21@15S placed three
conditionals and one NORMAL order. No case has yet driven a placement on every bar,
alternating side, at 15 s cadence. It also exercises the cancel-then-place ordering
inside one bar, which the engine's diff loop must sequence (#140 names an unguarded
diff-removal loop; this run gives it a live sample).

Grading rule (venue record). `/orders/history` for the window shows exactly N NORMAL LO
ids, each `Canceled` with `filled=0`, one per bar, and the side of each matches the
colour of the bar that placed it (bar colours read from the tracked `.ohlcv`, or from
the 15S LTF store for T34b). No working order remains at the end; `venue.py flat` exit 0.
For the WS arm: every placed id has an "order frame via WS" line naming it, and the
CANCELLED event for each id arrives at most one poll interval after the frame.

Does T34b double as Live-L4-T04? **Partly, and it must not be recorded as T04.** It proves
the per-print WS feed is alive and that the aggregator closes bars at 15 s cadence (already
shown 09-09), and it adds that the engine keeps dispatching on every such bar. It does NOT
give T04's grading: who closed each bar (official within `tick_close_timeout` versus a
loud SYNTH fallback), forming-bar updates, 429 transitions, or any parity of the
synthesized bars, since the venue has no 15 s candles to compare against. About #100's
per-print feed it proves liveness and cadence, not completeness (no capture ratio versus
`/trades/latest`).

Safety. Continuous phase only, 09:20–11:20 or 13:05–14:15; flat-by 14:20. Prices ±5 %.
Cost: T34a ≈ 3 min warmup + 10 min; T34b ≈ 0 warmup + 3 min for 10 bars (run 40 bars
= 10 min for a fair sample). 0 fills, 1 resting contract at any time.

### T35 — Live-L1-T35-StockNoFillLadder (HPG, engine-driven)

Procedure. A `.pine` run on `dnse_broker:HPG@1 --broker` (needs a `market_type` pin and a
funded stock sub-account, as the 09-15 probe used): state 0 far limit buy at −5 %, lot
100, hold one bar, cancel; state 1 same entry re-issued next bar with BOTH a new price
(−4.6 %) and qty 200 in one modify; state 2 cancel by whatever id the engine now tracks.

What it measures that nothing else does. The first engine-driven stock order path. The
09-15 direct probe measured the venue (NEW id on PUT, both fields accepted) but not
that the engine re-maps its tracked id (#117: id re-map "DONE (47a3b575)" — not live-proven)
nor that the cancel targets the replacement.

Grading rule. Venue record shows the original numeric id `Canceled` by the venue at the
PUT, a second id with price −4.6 % and qty 200, and that second id `Canceled` by our
cancel; no order working; if any write is refused the structured code is recorded
verbatim.

Safety. Continuous phase, stock ATO excluded (09:00–09:15 is stock ATO per CLAUDE.md);
flat-by 14:20. Cost ≈ 8 min, 0 fills, exposure if it somehow filled ≈ 200 shares at
about 26 thousand VND (per CLAUDE.md sandbox note), so a few million VND.

### T36 — Live-L1-T36-AtoWindowProbe (direct probe, 08:45–09:00)

Procedure. Extend `t33_closed_hours.py`'s pattern: at 08:47 place a far NORMAL limit,
a STOP and a STOP-LIMIT (±5 %); confirm each rests; at 08:50 attempt to cancel each
during ATO; record accept/refuse and codes; at 09:01 (POST-ATO) cancel whatever remains
(T32 proved cancels work at 09:15; the probe should also try 09:01).

What it measures that nothing else does. The only session phase with no measurement
(#156). It decides whether ATO behaves like ATC (cancels refused) or like pre-open
(everything accepted), which the L0 phase ladder split needs.

Grading rule. Venue record per id: `Canceled` with the cancel timestamp inside ATO, or
the refusal code plus `Canceled` at POST-ATO.

Safety. Prices non-marketable at ±5 %; the worst case is a far order resting until 09:01.
Cost ≈ 6 min, 0 fills.

### T33 — run the existing Live-L1-T33 post-close probe

`t33_closed_hours.py` exists (four refusal probes plus the ordering probe) and has never
run. Running it on a weekday at 15:10 gives the post-close half of the ladder split and
the "cancel while closed" answer if anything is accepted. Cost ≈ 3 min, 0 fills.

### T37 — Live-L1-T37-FiveSecondLadder

Procedure. T19–T21 ladder (`live_staged_place_cancel` startState=12) on
`dnse_broker:VN30F1M@5S --broker`, exactly as the 09-09 15S run.

What it measures. The 5S synthesized feed live (never run), its warmup from the LTF
store, and whether the engine's per-bar sync keeps up at 5 s with the 0.5 s order poll.

Grading. Same as the 15S row: three conditional string ids each Canceled, one NORMAL id
amended in place then Canceled; bar cadence 5 s ±1 s in the log; venue flat.
Safety as T19–T21. Cost ≈ 3 min, 0 fills.

### T38 — Live-L1-T38-LunchBoundaryCancel

Procedure. Pine places a far NORMAL limit on the 11:27 bar and issues the cancel on the
11:29 bar; a second state does the same with a conditional STOP. The run stays alive
until 13:02.

What it measures. Whether a cancel issued on the last pre-lunch bar finalises during the
break or only at 13:00 (the 09-07 T9 observation was incidental: `PendingCancel` held
through the break), for both books, and whether the engine's cancel-tentative grace (#64)
misreads it.

Grading. Venue `modifiedDate` of the `Canceled` transition versus 11:30/13:00; no
quarantine line; both ids terminal. Cost ≈ 10 min of wall time (mostly waiting), 0 fills.

### T39 — Live-L1-T39-RollMorningContract (October expiry)

Procedure. Day before expiry: pre-flight row "probe PROCESS alive" (runbook), launch
`probe_113_roll_cache.py --hours 20` detached. Morning after: pre-open 08:11 (measured
acceptable 09-18) run L0 on the alias, then one Pine-driven far limit on `VN30F1M`.

What it measures. The repoint timestamp (lost on 09-18), and that an engine started after
the repoint orders on the NEW dated code (venue record `symbol`), plus the AGED arm for
the stale-cache hazard.

Grading. Probe verdict line plus the venue record's `symbol` on the placed id equal to
the FRESH arm's answer. Cost: overnight read-only probe (no token) + 5 min, 0 fills.

### T40 — Live-L1-T40-VN100Resolve

Procedure. Read-only first: `resolve_contract("VN100F1M")` against prod instruments; if
it resolves, run L0 with `--symbol VN100F1M` pre-open.

What it measures. Whether the second alias family exists on this venue and whether the
plugin's `symbolType` match covers it. Nothing in the tree names VN100F1M today.
Grading. Instruments row present; L0 VERDICT PASS on that contract. Cost ≈ 3 min, 0 fills.

## 3. Ranking by information gained per minute of live time

| Rank | Case | Live minutes | Why this rank |
|---|---|---|---|
| 1 | T36 AtoWindowProbe | ~6 | Closes the only unmeasured phase; needed by #156's ladder split; cheapest of all |
| 2 | T34b 15S colour alternator (+ WS arm) | ~10 | First sustained per-bar churn on synthesized bars, first live engine-side WS NORMAL-id observation if the arm is on; partial L4-T04 signal |
| 3 | T35 stock engine ladder | ~8 | An entire asset class has zero engine-driven live coverage; the id re-map (#117) is only offline-proven |
| 4 | T33 post-close run | ~3 | Script already exists; gives the other half of the phase split |
| 5 | T37 5S ladder | ~3 | Cheap, but 15S already proved the mechanism; adds cadence stress only |
| 6 | T34a 1m colour alternator | ~13 | Same design as T34b with less novelty: 1m NORMAL place/cancel is already T01/T02; value is the churn and the WS arm |
| 7 | T40 VN100 resolve | ~3 | Cheap, but may answer "not listed" and end there |
| 8 | T38 lunch boundary | ~10 wall | Real gap (#64 grace) but mostly waiting time, and timing-locked |
| 9 | T39 roll morning | ~5 + overnight | Highest one-off value (#113), but available once a month and needs the day-before discipline that failed on 09-17 |

## 4. Registry versus logs — disagreements and unreconcilable rows

1. **T-number collision (T19–T26).** The registry rows `Live-L1-T19-ConditionalTrailingReplace`,
   `T20-DualFieldAmendSplit`, `T21-BothSetSingleOwner`, `T22-ExpiryRouting`,
   `T23-BothSetRestartAdoption`, `T24-FlattenDrill` reuse ids that `live_staged_params.pine`
   (states 0–9, card #22) assigns to different cases (T19 both-set entry, T20 short sell-stop,
   T21 qty omitted, T22 tick exits, T23 trail-only, T24 qty_percent, T25 two exits,
   T26a/b close). `params_a_evidence.txt` carries `[PRM] T19 … T26a` lines for the #22
   meaning. The registry has no rows at all for the #22 set (the "Registry backfill: T19–T31"
   planned row is still open). Any id T19–T26 is ambiguous until one set is renumbered.
2. **T22 combo evidence lacks the graded event.** `combo_t22_evidence.txt` names order
   244006 twice but contains no `Expired`, `T22` or quarantine line; the registry's
   "venue flipped Expired at ~15:04" grade is not in the tracked file.
3. **Nofill re-run 09-07 is undated in its evidence.** `nofill_rerun_evidence.txt` and
   `nofill_rerun2_evidence.txt` carry no 2026-09 date token (the only date inside is
   2026-08-10, a warmup/data line); the 09-07 date exists only in the registry text.
4. **params_a/params_b evidence undated.** The registry planned row says "measured
   08-17/18"; the files carry no run timestamp (only a 2026-08-12 token). Not reconcilable.
5. **WS NORMAL-only measurement date.** Card #130 and `sep16_evidence.txt` date it
   2026-09-16; the assignment text and the repo memory line say 2026-09-17. The README venue
   fact paragraph still states prod requires `subscribe_broker_order_event` and that no prod
   frame was ever captured (verified 2026-09-14), which #130/#131 (09-16) and CLAUDE.md
   (first prod frames 2026-09-15 on the short channel) contradict.
6. **L2-BracketFill row stale.** Registry says "⚠️ 08-12 partial"; the README venue-facts
   section describes an l2b live run on 2026-09-15 arming same-bar (#121). No tracked log
   under `logs/` carries the 09-15 run (`l2b*_evidence.txt` are all 2026-08-12).
7. **T10 re-verified 08-17.** Only `dual_a/dual_b_evidence.txt` (2026-08-14) exist; no 08-17
   dual evidence.
8. **T16 09-07 / 09-08 grades.** Evidence is inline in the row; the only T16 file is the
   2026-08-18 "BUG FOUND" run in `t15_t16_t17_evidence_20260818.txt`.
9. **T32 row title overstates.** "AtoProbe" measured the 09:00 POST-ATO bar (file
   `t32b25_evidence.txt`); pure ATO 08:45–09:00 is unmeasured (#156 agrees).
10. **L0 docstrings stale after a16e977d.** `session_phase()` and the module header still
    say "closed — NOTHING can be placed" while the 09-18 logs show pre-open acceptance;
    the commit message defers the ladder split to a follow-up (#156).
11. **`staged_3m.log`** (2026-08-13, `VN30F1M@3`) has no registry row and only one
    placement line; it is not evidence of a 3m case.
12. **T14 logs.** `t14.log` has zero `ATC_SESSION` lines; the 8 refusals are in
    `t14_rerun.log`. The registry row cites "8/8" without naming which file.

## 5. Sources read

- `plugins/dnse/testing/live_test/README.md` (registry, tiers, venue facts)
- `plugins/dnse/testing/live_test/live_staged_place_cancel.{toml}`, `live_staged_params.{pine,toml}`,
  `live_t27_tick_dedup.pine`, `live_t31_direction_gate.pine`, `run_t10_dual.sh`,
  `direct_probes_t14_t15_t17.py`, `probe_113_roll_cache.py`, `probe_116_117_prod_premises.py`,
  `FRIDAY_2026-09-18_RUNBOOK.md`
- `plugins/dnse/testing/live_test/level0_venue_semantics/l0_order_semantics.py`,
  `t33_closed_hours.py`, `logs/l0_run.log`; commit `a16e977d` (stat + diff); commits
  `f8cef967`, `302a897d` (#118)
- `plugins/dnse/testing/live_test/level4_data_parity/l4_bar_parity.py` and `logs/`
- `plugins/dnse/testing/live_test/logs/`: all `*_evidence*.txt`, `l0_20260918_preopen_0811.log`,
  `l0_20260918_preopen_limit_0815.log`, `staged_3m.log`, `staged_1m.log`, `t14.log`, `t14_rerun.log`
- `plugins/dnse/tests/` (file names only, for offline coverage)
- Repo `CLAUDE.md` DNSE sections
- Cards (read only): #22, #28, #37, #51, #92, #100, #107, #113, #117, #118, #121, #130,
  #131, #132, #146, #152, #154, #156; full issue list

# Second pass

Question widened: what can go wrong live that no case exercises. Same evidence rules as the
first pass. New ids continue after T40. "No-fill" means every price is non-marketable
(≥4.5 % away) and the account stays flat; "FILL" means real money and the FILL-tier
preconditions from the README (flat account, supervised, `run_fill_case.sh`).
Additional cards read for this pass: #17, #23, #36, #39, #46, #48, #54, #64, #65, #84, #96,
#102, #103, #104, #105, #116, #122, #124, #128, #135, #141, #143, #151.

## (a) Failure injection on the live path

Existing partial coverage.
- Token expiry mid-run: NOT COVERED. `tools/README.md` states the ~8 h TTL; commit 1d963bd4
  (2026-09-18) moves the daily refresh to 06:45, so a token minted then expires ~14:45, at the
  close. No case has run into an expiry. #54 (closed) pins offline that a persistent 401 on the
  order poll escalates rather than going silent; the live behaviour is unmeasured.
- INVALID_TRADING_TOKEN on a conditional write after an operator app trade: MEASURED as an
  incident (cards #46 2026-08-24, #51 2026-08-25; CLAUDE.md "usually NOT a token problem").
  No case reproduces it deliberately; the L0 tooling row "classify the #51 signature" is still
  planned (README v3 queue).
- REST timeout / 5xx on place, cancel, poll: OFFLINE ONLY. #67 (closed, transport split) and
  `tests/test_transport_errors.py`; the lost-reply POST park is fake-venue only. Live: T07
  08-14/17 measured a real 500 on the conditional PUT only.
- WS disconnect and reconnect mid-position: NOT COVERED. `ws_order_source.py` passes
  `auto_reconnect=True` to the vendored client (line 104); #129 says the WS order source never
  started live; #135's sandbox re-run (2026-09-16) exercised WS frames, not a reconnect.
- Process restart with a resting conditional: PARTLY. T16 (09-07/09-08, NORMAL LO only) and
  #78 (closed: a re-owned STOP reconstructed as a limit — fixed offline). T23-BothSetRestart is
  planned. Restart with an OCO umbrella: NOT COVERED (needs a position, so FILL tier). On 6.9.4:
  #151 is OPEN ("READY TO LAND, waiting for the operator"); the registry has no T16 grade on a
  6.9.4 tree, and #151 names the restart sweep (45bc8103) as the one re-derived collision, so
  the restart case is unproven on the tree about to land.
- Stale replica position read (0 while holding 1): MEASURED as an observation only
  (CLAUDE.md "#124-OBS ... get_position answered 0 while the position was really 1"); the
  two-agreeing-reads rule is enforced in `tools/flatten.py` and `sync_engine.py` (CLAUDE.md,
  bf5096e3 / 6abe04c6) but never exercised live on purpose.

Proposals.

**T41 — Live-L1-T41-TokenExpiryInSession (no-fill).** Mint a token at 06:45 as the cron does,
launch a Pine run that places one far NORMAL limit and one far STOP at 13:30 and holds them;
at ~14:45 the token reaches its TTL (per `tools/README.md`, 8 h). The run stays alive to 15:10.
Measures: what the plugin does when the trading token dies with orders resting — order poll
(api-key read) versus writes (token) — and whether the #54 escalation fires live. Grading:
venue record shows both ids `Expired`/`Canceled` at the venue's own batch (T22 measured ~15:04),
and the run log's halt/warn classification is compared with the venue timestamps; the run must
not report the orders as cancelled by itself. Safety: ±5 %, orders expire at the close, flat-by
is the venue's expiry; ~100 min wall, 0 fills, 2 resting contracts.

**T42 — Live-L1-T42-ConditionalWriteAfterAppTrade (no-fill, operator-in-the-loop).** Pre-open
08:11 (measured acceptable 2026-09-18) L0 passes; the operator then makes one app trade on a
different symbol or a far conditional in the app and cancels it; L0 conditional parts are re-run
immediately. Measures: whether the #46/#51 signature reproduces on demand, and what the L0
classifier prints (planned "blocked by venue window" row). Grading: venue record shows the
attempted place absent (refused) or present-and-cancelled; the HTTP code is recorded verbatim.
Safety: ±5 %, pre-open or lunch; ~10 min, 0 fills. Needs the operator's app.

**T43 — Live-L1-T43-RestartWithRestingConditional (no-fill, 6.9.4 tree).** `run_t16_restart.sh`
extended with a state that places ONE far STOP entry (string id) and holds; SIGTERM then SIGKILL
variants; relaunch same label. Measures: re-ownership of a conditional id and its reconstruction
as a STOP (not a limit, #78) on the landed 6.9.4 tree (#151), and that `cancel_all` reaches it.
Grading: venue record shows the string id `Canceled` after the relaunch's cancel, no second
placement, no order working. Safety: +5 % trigger, continuous, flat-by 14:20; ~15 min per
variant, 0 fills.

**T44 — Live-L1-T44-TwoReadPositionDiscipline (FILL, read-only after the fill).** During any
L3 fill case, a sidecar script reads `get_position` at 100 ms cadence for 30 s after the fill
event and after the flatten, logging every disagreement between consecutive reads. Measures:
the replica lag window that produced #124-OBS, as a distribution rather than one anecdote.
Grading: the venue's `/positions` rows with `createdDate`/`modifiedDate` versus the read
timestamps. Safety: rides an existing fill case; adds 0 orders; ~0 extra minutes.

## (b) Cross-book lifecycle

Existing partial coverage.
- Stop-entry activation and child tracking: MEASURED F03/F04 2026-09-07 and F11 2026-08-19
  (registry; #39 closed). FILL tier only; no L1 case observes an `Activated` transition because
  a far trigger never fires.
- OCO umbrella born `Activated`: MEASURED 2026-09-15 (CLAUDE.md; #124 body "isolation probe
  2026-09-15: a standalone OCO RESTED UNTOUCHED 120s"). Requires a position: FILL tier.
- Cancelling the child versus the umbrella (`CO-ORD-013`): MEASURED 2026-09-15 (CLAUDE.md
  "cancel the CHILD id instead"). No case pins it; it is a note.
- What `venue.py status/flat` cannot see: MEASURED 2026-09-17 (#152 open). No case grades
  the umbrella by id today; the README venue-fact section still lists the umbrella-cancel path
  as unscanned (#124 "S3 visibility: add OCO to the scanned categories" — not done).

Proposals.

**T45 — Live-L1-T45-UmbrellaVisibilityGate (FILL, minimal).** After any l2b bracket fill,
before flattening, run three reads and record them side by side: `venue.py status`,
`venue.py order <umbrella string id>`, `venue.py order <child numeric id>`. Then cancel the
CHILD only and re-read the umbrella. Measures: whether the umbrella respawns a child after a
child cancel (the #128 open question: "NO respawn = world-2" observed once), and whether
`status` after the fix for #152 lists the umbrella. Grading: umbrella `orderStatus`,
`externalOrderId` before and after, child `Canceled`. Safety: rides an l2b run; the position
is protected by the SL leg until the child cancel, so the cancel is immediately followed by the
flatten; ~3 extra minutes, 1 contract already held by the parent case.

**T46 — Live-L1-T46-ActivatedShellHygiene (no-fill).** A far STOP placed pre-open is left
resting; at 09:01 it is cancelled; the run also lists the STOP book at 09:15. Measures: that a
never-triggered conditional does not leave an `Activated` phantom (#41 open: "a consumed
Activated conditional shows as a working order forever") and that `venue.py flat` ignores
phantoms but counts a still-New shell. Grading: STOP-book row `Canceled`, no `Activated` row
for that id. Safety: +5 %, pre-open placement (measured acceptable 09-18), flat-by 09:20;
~5 min, 0 fills.

## (c) Concurrency on one account

Existing partial coverage.
- Two runs placing: T10 2026-08-14 (`dual_a/b_evidence.txt`) aligned both on the same 1m bar;
  order-level isolation proven, fill-level isolation is #115 (closed: "fill-level DNSE coverage
  remains").
- External app cancel of our exit: MEASURED 2026-09-15 (#124 closed, re-arm N/3; #128 closed:
  all cancels were the operator's).
- External partial close unattributable: stated as venue physics in CLAUDE.md (#73/C2); never
  measured.
- Opposing strategies netting to zero: CLAUDE.md limitation; never measured.
- Foreign order id below ours: the assignment attributes this to #135; the #135 body I read
  (watermark walks backwards on out-of-order transport arrival, 2026-09-16) does not mention a
  foreign id — recorded in the disagreements section.

Proposals.

**T47 — Live-L1-T47-SameSecondDualPlace (no-fill).** `run_t10_dual.sh` variant on 15S bars so
both engines place inside the same second (15S cadence makes alignment tight), A a far limit
long, B a far limit short, both cancel next bar. Measures: id attribution when two of our
orders are created in the same second and the WS short channel (if enabled) streams both;
whether either run's journal claims the other's id. Grading: history shows two ids, each
`Canceled` by its own run (journal `run_tag` join), zero "owned by ANOTHER run label" lines
for live ids. Safety: ±5 %, continuous, flat-by 14:20; ~8 min, 0 fills.

**T48 — Live-L1-T48-OperatorCancelsOurEntry (no-fill, operator-in-the-loop).** Pine places a
far NORMAL limit and a far STOP and holds; the operator cancels each in the app; the run stays
alive 3 bars. Measures: the #74 residue detector and the #124 classification on ENTRY orders
(the 09-15 measurement was on a protective EXIT); the engine must log the external cancel and
not re-place. Grading: venue shows both `Canceled` with the app's origin; no new id placed
after the cancel. Safety: ±5 %, continuous; ~8 min, 0 fills.

**T49 — Live-L1-T49-OpposingRunsNetZero (FILL).** Two engines, one long market entry and one
short market entry on the same bar, each with its own pre-placed bracket. Measures: what each
run's `_durable_owned_signed_size` reports when the account nets to zero, and whether the
external-flatten read (#73/C2, raw net = 0) clears a live belief. Grading: `/positions` shows
no OPEN row; each journal shows its own fill; neither run quarantines. Safety: brackets ±1 %,
lunch-adjacent continuous, flatten both by `flatten_api.py` at once; ~10 min, 2 contracts
round-tripped. Real money; needs operator approval of the FILL tier.

## (d) Time boundaries

Existing partial coverage.
- Lunch with a pending cancel: one incidental observation (registry 09-07, T9 `PendingCancel`
  through the break). Lunch with a resting order: L0 lunch runs (docstring 08-13).
- 14:25–14:30 with an in-flight replace: NOT COVERED. T14 08-18 covered cancel refusal in ATC;
  T19 09-08 covered replace in CONT-PM at 13:15.
- Cross-day ids on the cancel endpoint: MEASURED 2026-09-15 (CLAUDE.md `RESOURCE_NOT_FOUND`);
  #96 (closed) pins the day-scoping offline. No case.
- GTD clamp expiry day vs day before: expiry day MEASURED 2026-09-17 (commit f8cef967); day
  before: the 2026-09-16 bare STOP carried GTD 2026-09-17T00:00Z (`sep16_evidence.txt`), which
  the 09-17 commit later showed is above the 07:30Z ceiling for the FINAL day — so the
  day-before value was accepted only because it was not yet the final day.
- Holiday walk-back: OFFLINE ONLY (`test_expiry_dates.py`).

Proposals.

**T50 — Live-L1-T50-LunchStartReplace (no-fill).** Pine re-issues a far conditional entry at
11:29 (cancel+replace, #85) and again at 13:00, and a far NORMAL amend at 11:29. Measures: a
cancel ACK landing at the break with the replacement placed during it (#64: the 10 s grace vs
>12 s settle), for both books. Grading: old ids `Canceled` with venue timestamps, new ids `New`
then `Canceled` by the 13:02 sweep. Safety: ±5 %; ~95 min wall, mostly idle; 0 fills.

**T51 — Live-L1-T51-PreAtcReplaceRace (no-fill).** Pine re-issues a far conditional at 14:28
and 14:29; the flat-by sweep fires on the 14:29 bar. Measures: whether an in-flight replace can
straddle 14:30 and leave a conditional that ATC refuses to cancel (T14 measured refusal). Grading:
STOP book at 14:35 shows nothing `New`; anything left is graded `Expired` next morning via
`/orders/history`. Safety: +5 % trigger so a leftover cannot fill in the auction; ~8 min, 0 fills.

**T52 — Live-L1-T52-DayBeforeExpiryGtd (no-fill, expiry week).** On the day before the final
trading day, L0 on the front month with the default clamp, then the same at 14:20 with a GTD set
to the final day 07:30Z and 07:45Z. Measures: the ceiling's dependence on the placement day (the
09-17 table measured only the final day itself). Grading: venue accept/refuse per GTD value.
Safety: L0 envelope; ~5 min, 0 fills. Once a month.

## (e) Quantity and price edges

Existing partial coverage.
- qty 2 on a derivative: T24/T25/T30 (#22 numbering) measured 08-17/18 per the planned
  backfill row (undated evidence, see disagreement 4); the frozen flip quantity (CLAUDE.md,
  #105 2026-09-11) is pinned by `test_130_flip_does_not_cancel_an_armed_exit.py` and measured
  live once (#105: 2 contracts live). No no-fill case can reach it (needs a fill).
- Partial fills on a stock: sandbox only (CLAUDE.md, one partial tick); prod NOT COVERED.
- Price tick rounding: #87 floor-quantizer (closed); T33's off-tick probe never run.
- Limit at the band edge: L0 market part (band-edge, lunch); a LIMIT exactly at ±7 % never
  placed on purpose. #119 (closed) stock floor check.
- Thousands-of-VND stock prices: #119 closed; measured once by the 09-15 probe.

Proposals.

**T53 — Live-L1-T53-BandEdgeLimit (no-fill).** Direct probe: NORMAL limit at exactly the ceiling
(+7 %) sell and floor (−7 %) buy, then one tick beyond each. Measures: the venue's band
validation code and whether our `_place` pre-rounds. Grading: accepted ids `Canceled`; refused
ones carry the code verbatim. Safety: sell at the ceiling / buy at the floor are non-marketable;
lunch; ~4 min, 0 fills.

**T54 — Live-L1-T54-StockOffTickAndBandEdge (no-fill).** Same as T53 on HPG plus an off-tick
price (x.x5 on a 0.1-tick, from `t33_closed_hours.py` probe 4) in continuous. Measures: which
check fires first (tick vs band) on a stock, and the đồng-vs-thousands unit (#119) on a live
place. Grading: codes verbatim; accepted ids `Canceled`. Safety: floor buy, lot 100; ~5 min,
0 fills.

**T55 — Live-L3-F14-StockPartialFill (FILL).** Buy 300 HPG at the ask in continuous; observe
fills. Measures: prod partial-fill slices on a stock (#56 executions endpoint: sandbox
`http=404`), and the #135 watermark under out-of-order poll/WS arrival with a multi-slice fill.
Grading: `/positions` row quantities (accumulate/trade) versus the journal's `filled_qty`.
Safety: flatten immediately by `flatten_api.py`; ~5 min; ~8 million VND notional. Real money.

## (f) Data path

Existing partial coverage.
- Warmup depth on 15S: `t19_21_15s_evidence.txt` 09-09 ("warmup 0 bars" per registry, store
  accumulated 19 bars). 5S: never run.
- Bar gap across lunch: L4-T01/T02 08-17 "PM-reopen chunk" (registry); 1m only.
- 14:45 malformed index bar: MEASURED 2026-09-10 (#104, provider clamps); never re-observed
  in a `--broker` run.
- `request.security` live with the symbol map only: NOT COVERED (CLAUDE.md: "has NOT been
  run"). Live with explicit `--security` flags: measured 2026-09-10 (`idx_feed_probe.log` per
  #103).
- OHLCV cache splice across the roll: NOT COVERED; #17 open (truncate on every run), and the
  09-18 data snapshots were committed (4fb803db) without a roll measurement.
- Warmup→live duplicate bar: MEASURED 2026-09-10 at 15m (#103 open).

Proposals.

**T56 — Live-L4-T06-LtfWarmupDepth (passive).** Launch `@15S` and `@5S` runs 30 min after a
previous run accumulated the LTF store; log warmup bar count and the first live bar timestamp.
Measures: whether the LTF store feeds warmup at all and whether the #103 duplicate appears at
sub-minute. Grading: first live bar timestamp strictly after the last warmup bar; count
equals the store. ~10 min, 0 orders.

**T57 — Live-L4-T07-SecurityMapOnly (passive).** A `request.security('VNINDEX')` strategy at
15m `--live` with only `symbol_map.toml`, no `--security` flags. Measures: the documented
fallback path (`script_runner.py resolve_symbol`) live; the 09:00 `na` bar; the 14:45 clamp
line if the run spans the close. Grading: the index values match a fresh `/price/ohlc?type=INDEX`
read for the same slots. ~20 min, 0 orders.

**T58 — Live-L4-T08-RollCacheSplice (passive, October expiry).** On the roll morning, run
`pyne run` on `dnse:VN30F1M@1` and compare the `.ohlcv` tail before and after: does the file
splice the new dated contract's bars onto the old alias history, and does the provider log it.
Measures: the #17 truncate behaviour across a contract change. Grading: `.ohlcv` timestamps
continuous, last-old and first-new bar both present. ~5 min, 0 orders. Back up the `.ohlcv`
first (house rule).

## (g) Operator interaction

Existing partial coverage.
- Operator placing a stop in the app during a run: the assignment says "measured 09-17 as a
  wake signal"; I found no log, card or commit in the tree that records it (grep of
  `docs/plan` and `live_test/*.md` for "wake signal"/"app stop" is empty). Commit f8cef967
  records the operator's app conditional on 09-17 13:40 as the #118 evidence, not as a wake.
  Recorded in the disagreements section.
- Operator cancelling our order: MEASURED 2026-09-15 for an EXIT (#124/#128); entries: no case.
- Operator flattening in the app: MEASURED 2026-08-19 protocol (README "the OPERATOR closes
  the position"; F10/F11 08-18/19); the API-flatten protocol replaced it 09-07. #48 external
  close detection measured F1 09-07.

Proposals.

**T59 — Live-L1-T59-ForeignConditionalDuringRun (no-fill, operator-in-the-loop).** Pine holds
a far limit; the operator places a far STOP in the app and cancels it 2 bars later. Measures:
that the run reports the foreign id (#60/#71 strand report) without adopting or cancelling it,
and whether any engine wake/sync is triggered by a foreign conditional. Grading: the foreign id
is `Canceled` by the app origin only; our id untouched and `Canceled` by us. Safety: ±5 %;
~8 min, 0 fills. T48 above covers the cancel-of-ours half.

**T60 — Live-L3-F15-AppFlattenWithBracket (FILL).** l2b fill, then the operator flattens in the
app instead of `flatten_api.py`. Measures: #48 external-close detection when the bracket's
umbrella is still resting, and whether the engine retires the umbrella (the CHILD id, per
CLAUDE.md `CO-ORD-013`) rather than re-arming (#124 N/3). Grading: `/positions` no OPEN row;
umbrella and child terminal; no new conditional placed after the flatten. Safety: FILL tier;
~8 min, 1 contract round-tripped.

## (h) Observability — grades that rest on the run log alone

Rows whose registry grade cites only run-log lines, with the venue read that would make them
venue-graded:

| Case | Today's grade source | Venue read to add |
|---|---|---|
| T21 09-08 | log lines "sole owner", zero "armed entry-stop watch" | STOP book listing at placement time showing exactly one string id; NORMAL book empty |
| T19/20/21@15S 09-09 | log cadence "18 bars, 0 suspect" | no venue equivalent for bars; for orders the same reads as T19–T21 (registry says "venue FLAT", not per-id) |
| T22 09-09 | "engine's poll ladder observed it", journal `terminal_status` | `/orders/history` row for 244006 with `Expired` and its timestamp (not in `combo_t22_evidence.txt`) |
| T16 09-07/09-08 | "re-owned 1 venue id" log line | `venue.py order 148286` / `189786` showing `Canceled` after the relaunch's cancel |
| T32 08-25 | "ACCEPTED at the earliest POST-ATO bar" | detail reads with `createdDate` inside 09:00–09:15 |
| T10 08-14 | `dual_a/b_evidence.txt` `[BROKER]` lines | both ids from `/orders/history` with `Canceled` and creation times |
| T15/T17/T18 08-18 | probe prints "poll A=Canceled B=New" (these are venue reads inside the probe) | already venue-based; only the T16 "BUG FOUND" section is log-only |
| L1-EC-Ladder | planned; grading text says "venue record grades identically" | keep |
| Live-L2-BracketFill 08-12 | `l2b_evidence.txt` `[BROKER]` lines | `/positions` closed row + umbrella/child ids |
| F13 (not in registry) | #146 grading is defined from `venue.py order` | keep; add the row to the registry when run |

## Re-ranking — first pass plus second pass

Information gained per live minute; "$" marks real-money (fills). One line each.

| Rank | Case | Money | Live min | Reason |
|---|---|---|---|---|
| 1 | T36 AtoWindowProbe | no-fill | 6 | only unmeasured phase; needed by #156 |
| 2 | T43 RestartWithRestingConditional (6.9.4) | no-fill | 15 | restart is the one #151 collision and the tree lands today; conditional never re-owned live |
| 3 | T34b 15S colour alternator + WS arm | no-fill | 10 | first per-bar NORMAL churn on synthesized bars; first engine-side WS observation |
| 4 | T53 BandEdgeLimit | no-fill | 4 | validation codes at the band never captured; feeds the flatten arm-price reject (#144) |
| 5 | T35 StockNoFillLadder | no-fill | 8 | whole asset class has no engine-driven live case; #117 remap only offline |
| 6 | T46 ActivatedShellHygiene | no-fill | 5 | #41 phantom is open and cheap to observe pre-open |
| 7 | T33 post-close run | no-fill | 3 | script exists; closes the phase split |
| 8 | T48 OperatorCancelsOurEntry | no-fill | 8 | #124 measured on exits only; entries untested; needs the operator for 2 minutes |
| 9 | T51 PreAtcReplaceRace | no-fill | 8 | a replace straddling 14:30 leaves an uncancellable order; one slot per day |
| 10 | T45 UmbrellaVisibilityGate | $ | 3 extra | rides an existing l2b fill; answers #128's respawn question and #152's read |
| 11 | T47 SameSecondDualPlace | no-fill | 8 | tightens T10 to same-second; low probability of a new finding |
| 12 | T54 StockOffTickAndBandEdge | no-fill | 5 | stock unit + tick ordering; depends on the stock sub-account |
| 13 | T37 5S ladder | no-fill | 3 | cadence only |
| 14 | T42 ConditionalWriteAfterAppTrade | no-fill | 10 | reproduces a known incident; value is the classifier row |
| 15 | T34a 1m colour alternator | no-fill | 13 | same design as T34b, less novelty |
| 16 | T59 ForeignConditionalDuringRun | no-fill | 8 | strand report is already measured on stale ids (#60); new only for a live foreign conditional |
| 17 | T44 TwoReadPositionDiscipline | $ (rides) | 0 extra | distribution of replica lag; no new orders |
| 18 | T57 SecurityMapOnly | passive | 20 | documented-but-unrun path; no order risk |
| 19 | T56 LtfWarmupDepth | passive | 10 | store feeding warmup; #103 at sub-minute |
| 20 | T40 VN100Resolve | no-fill | 3 | may end at "not listed" |
| 21 | T50 LunchStartReplace | no-fill | 95 wall | real (#64) but mostly idle time |
| 22 | T41 TokenExpiryInSession | no-fill | 100 wall | one shot per day, long idle, but the failure is certain to occur in production |
| 23 | T38 LunchBoundaryCancel | no-fill | 10 wall | subset of T50 |
| 24 | T60 AppFlattenWithBracket | $ | 8 | real money for a path the API-flatten protocol replaced |
| 25 | T55 StockPartialFill | $ | 5 | real money; prod partial slices unknown, but stock trading is not yet a production target |
| 26 | T49 OpposingRunsNetZero | $ | 10 | 2 contracts for a documented limitation |
| 27 | T52 DayBeforeExpiryGtd | no-fill | 5 | monthly slot; refines a ceiling already usable |
| 28 | T39 RollMorningContract | no-fill | 5 + night | monthly; high value but needs the day-before discipline that failed 09-17 |
| 29 | T58 RollCacheSplice | passive | 5 | monthly; #17 is a known defect already |

## Disagreements — additions from the second pass

13. **"Foreign order id below ours (#135)".** The #135 body I read describes the `_last_seen`
    watermark walking backwards on out-of-order transport arrival (2026-09-16) and says nothing
    about a foreign id. The lens (c) attribution could not be reconciled with the card.
14. **"Operator placing a stop in the app during a run, measured 09-17 as a wake signal".** No
    log, card or commit in the worktree records a wake; commit f8cef967 records the operator's
    09-17 13:40 app conditional as the #118 GTD evidence only.
15. **Restart on 6.9.4.** The registry's T16 grades (09-07/09-08) predate #151; #151 is open and
    names the restart sweep (45bc8103) as the re-derived collision, so the registry's ✅ on T16
    does not apply to the tree about to land.
16. **Day-before-expiry GTD.** `sep16_evidence.txt` shows a GTD of 2026-09-17T00:00Z accepted on
    09-16 ("#118-clamped"), while commit f8cef967 (09-17) states the midnight-UTC value was the
    wrong ceiling; both are true for their day but the registry's venue-fact bullet "clamped to
    midnight UTC of the final trade date" is now stale.
