# Live-L3-F13 (l2b) — Friday 2026-09-18 live test plan

Supersedes the F13 section of `plugins/dnse/testing/live_test/FRIDAY_2026-09-18_RUNBOOK.md`
(vehicle + grading only; every other section of that runbook still applies — roll grade at open,
standing rules, known-and-expected, the passive captures added 09-17).

## 0. Decision (operator, 2026-09-17 evening)

F13 runs on **`l2b_fill_protect_flatten`** as the primary vehicle, **`l2_fill_flatten`** as the
fallback. Rationale: `l2` (market entry, no `strategy.exit`) measures only fill-event *transport*
latency; `l2b` pre-places the bracket on the entry bar, so each run measures the chain that moves
money — **venue fill → engine fill event → `EXIT dispatched` → SL/TP resting on the venue** — per
transport, AND answers the open #130 question for the entry type real strategies use (a STOP
entry: does the WS arm deliver the normal-book *child's* frame and attribute it to our entry?).
Either answer is a measurement we need before trusting WS as the primary fill transport.

Costs accepted: a stop entry fills only on a breakout (mitigated by the per-run window + fallback
below); the contract is held ~2 bars at 5m; worst case 6 contract round-trips, typical 4.

## 0c. RESULT of the Thursday attempt (13:26): L0 gate FAIL on #118 — no conditional order is possible on expiry day

The runner started 13:26:43 (`--window-bars 2`), passed config/token/window/flat, and the **L0 gate
refused**: all four conditionals (STOP buy/sell, STOP-LIMIT buy/sell) rejected `CO-ORD-006 "Validate
Order Failed"`, each preceded by the engine's `GTD floored: final trade date 2026-09-17 is not in the
future — using 2026-09-18 … may still be refused (#118)`. Nothing reached the book; config restored
byte-identical; account flat ×2. **Cause (operator-corrected 13:37: "it can — just use the right date
time"): OUR clamp, not the venue.** `_clamp_gtd_to_expiry` (broker.py:1212) ceilings the GTD to
MIDNIGHT UTC of the final trade date (00:00Z = 07:00 ICT), which at 13:26 was already in the past, so the
"next open day" floor won and produced a GTD past the contract's life → CO-ORD-006. The venue reads a
GTD as a date-TIME: a GTD later the same day (end of session, 07:45Z = 14:45 ICT; the 08-14 measurement
already showed 04:00Z on the final date accepted) satisfies both "in the future" and "not past the final
trade date". Fix under #118 (Worker2): ceiling = final-day end of the CONTINUOUS session, floor only when the
final trade date is genuinely past (stale secdef, #113). **Measured 13:36–13:46:** 07:45Z (14:45 ICT)
was REFUSED too; VN30F2M (next month) PASSED all four conditional cases; then the operator's own app
STOP on the expiring contract with expiry **14:30 today** was seen RESTING (`daloml2vfqkc7397mdk0`) —
so the boundary is 14:30 ICT = 07:30Z, the ATC start. Clamp constant → 07:30Z; **L0 rerun on VN30F1M at 13:52 PASSED** (four conditionals
placed/rested/cancelled on the expiring contract on its final day). Boundary lies in (07:30Z, 07:45Z]. Until it passes, **Friday runs BOTH arms on the new front month** — ws arm 09:15–11:10 with
the default `--window-bars 6` after the roll grade, poll arm 13:00–14:15. W0 shadow ran through the
expiry-day session as evidence (graded on #132). The L0 gate did its job: the plan was wrong, the gate
refused, nothing was placed.

## 0b. Thursday 09-17 afternoon (operator decision 12:05): ws arm runs TODAY, expiry day — SUPERSEDED by 0c

Operator moved the ws arm to Thursday 13:00 (Worker3 executing; poll arm + roll grade stay
Friday). Expiry-day deltas: no roll grade; thin liquidity → NO-SAMPLE likely; flat by 14:25 is
absolute (14:30 ATC is the final settlement auction). **Window deviation:** `--window-bars 3`
today, not the default 6 — with 6, a NO-SAMPLE run 1 (13:00 + 2700 s) ends after the 13:40
latest-start, so the l2 fallback AND run 2 are both refused by the deadline gate and the arm
yields nothing; with 3, NO-SAMPLE ends ~13:30 and the fallback fits. Today's rows are graded
with `window-bars=3` written on them; Friday's rows use 6 and are not comparable as equals.
Pre-flight 12:20: commits present, 742 passed/1 xfailed, token GOOD (hand-minted, no cron log —
expected), venue flat ×2, W0 sight = 41I1G9000 (the expiring contract, correct today).

## 1. Thursday evening prep — runner change (card, Worker3 after #132-W0; Fable reviews)

`plugins/dnse/testing/live_test/run_f13_latency.sh` gains:

| Flag | Default | Meaning |
|---|---|---|
| `--vehicle l2b\|l2` | `l2b` | which transpiled `.py` to launch (`l2b_fill_protect_flatten.py` / `l2_fill_flatten.py`) |
| `--window-bars N` | `6` | l2b only: bars to wait for the stop to fill; unfilled → cancel, log `NO-SAMPLE`, count 0 |
| `--fallback` | on | if an l2b slot yields NO-SAMPLE, run ONE `l2` in its place so the arm still gets a transport sample |
| `--fills N` | unchanged | runs per arm (operator's n) |

Grading additions (per run, per arm), all read from the venue record + `[BROKER]` lines:

- `T(venue fill)` from `venue.py order <id> --json` (NEW in #146 — createdDate/modifiedDate, copied RAW;
  the venue serves them as **ISO-8601 strings** (`"2026-09-16T04:22:43.958Z"`, up to 9-digit fractions —
  docs + prod capture), NOT epoch ms; the grader parses ISO first (#146 round-3 F1). The runner captures
  it per entry id + the CHILD named by the activated conditional's `externalOrderId` into the run's
  venue.json for the grader).
- `T(event FILLED … leg=entry)` and `T([BROKER] dispatched EXIT …)` from the log.
- `venue.py order <umbrella id>` showing the bracket **resting** (umbrella is `Activated` from birth —
  never read that as triggered; the TP child on the normal book is the visible cover).
- ws arm only: whether an `order frame via WS` line names the **child id** (answers #130-for-stops);
  the existing `WS ORDER SOURCE FIRST LIVE FRAME` milestone stays the delivery gate (#134).
- Timeframe **5m** (the runner was on `@1` before #146 — corrected). `timeout` per run = (window + 3) × 300 s = 2700 s at the default window.
- NO-SAMPLE cleanup cancels EVERY entry id OUR OWN log put on the book — both `dispatched ENTRY … -> ['id']` AND `event CREATED … leg=entry` lines (a chased/replaced entry has more than one id) — via `venue.py cancel`, each; a cancel exit 2 = failure ("FLATTEN NOW"); `venue.py flat` is checked AFTER the cancels. If no id can be read the runner says so and does NOT cancel — never a sweep, never an inferred id.

Definition of done for the prep: `bash -n` clean; `--help` never constructs a broker; a dry run
against the fake seam prints the grading table with NO-SAMPLE rows; `.venv/bin/python -m pytest
plugins/dnse/tests/ -q` unchanged; committed; hash sent to Fable; reviewed before Friday's open.

## 2. Friday session — exact sequence (operator types every order-placing command)

Time windows are Asia/Ho_Chi_Minh. **No EntradeX app trade before step 4 finishes** (#51/#46:
conditional-book writes die after the first app trade of the day).

| # | When | Who | Command / action | Gate |
|---|---|---|---|---|
| 1 | 08:00 cron (TEST) then ~08:20 | cron / operator | #133: the 08:00 cron (installed 09-17, append-only in the shared user crontab) mints via the dedicated mailbox and appends to `workdir/state/refresh_token.log`. At 08:20 the executor runs `.venv/bin/python plugins/dnse/tools/token_status.py --require-cron`: exit 0 = the schedule ran AND the token is GOOD; exit 1 with "NOTHING DATED <today>" = the cron did not fire → the operator mints manually (`refresh_token.py`, auto mode reads the OTP itself) and the plain `token_status.py` must then read GOOD. Either outcome is a #133 measurement — record which | `token_status.py` → GOOD (exit 0) |
| 2 | ~08:25 | executor | `venue.py status` · `venue.py flat` | flat exit **0**; exit 2 = stop, could-not-determine |
| 3 | 08:45–09:05 | executor | **roll grade** per runbook §1 (probe log bracketing the repoint) | new venue fact recorded |
| 4 | 09:15–11:10 | operator | `bash plugins/dnse/testing/live_test/run_f13_latency.sh --arm ws --fills 2` — **every run costs the full 45 min unless the #146 terminator lands** (no vehicle self-terminates; `timeout` is the only stop). The runner (post-#146 fixes) REFUSES to start a run whose timeout would cross 11:25 / 14:25 or outside `continuous` | token exit read (not grep'd), flat 0, L0 0, session continuous |
| 5 | 13:00–14:15 | operator | `bash plugins/dnse/testing/live_test/run_f13_latency.sh --arm poll --fills 2` (afternoon). The terminator landed (#146): a run ends ~2 bars after its fill, ~15–30 min typical; a no-fill slot costs ~57 min (45-min window + fallback l2). Run 2 is REFUSED by the window gate unless run 1 ends by ~13:40 — that refusal is the design, not a failure | same; the runner's deadline gate enforces it |
| 6 | ≤11:20 | executor | `venue.py flat` — must exit 0 before lunch; cancel OUR leftovers with `venue.py cancel <id>` | never `sweep` |
| 7 | 13:00 (before step 5) | executor | `probe_116_same_day_cancel.py <filled entry id from step 4>` (refuses working orders; takes seconds) | record http/code/message verbatim |
| 8 | any run | executor | passive captures: one `venue.py status` while a position is open (prod `/positions` frame); first venue order id of the day vs Thursday's range (#135 id-reuse) | evidence file |
| 9 | ≤14:25 | executor | `venue.py flat` exit 0 — **never hold into 14:30 ATC** | |
| 10 | W0 reviewed CLEAN (b2cbb6c5, round 4) — RUN IT | executor | shadow sidecar: `naked_watch.py --interval 15 --bar-period 300` started AFTER the roll repoint (confirm the `sight:` line shows the NEW dated code — a stale cached code reads as "nothing bot-owned", #132 F3); alarm-only, exit 2 = could-not-determine incl. UNATTRIBUTED (a venue position our journal never claimed — expected beside the operator's own position); exit 1 = some cycle graded NAKED (a sub-window NAKED that recovered still exits 1 with NO alarm line — read the alarm log, not the exit code); ONE store per symbol (a stock run sharing the store grades UNDETERMINED every cycle); an OCO-bracket vehicle reads `[UNSTOPPED]` as healthy (the umbrella's stop leg is invisible) | never a decision input on Friday; evidence only |

**Operator duties the runner cannot perform** (from the #146 review): on `NOT FLAT … STOPPING the
ladder` or after Ctrl-C, the runner STOPS — it does not flatten; **flatten in the app immediately**,
then `venue.py flat` must exit 0 before anything else runs. A fallback l2 run happens ONLY after the
resting stop's cancel returned 0 AND `venue.py flat` exited 0 (post-#146); if the runner prints
"FLATTEN NOW", do it.

If step 4's ws arm shows **no `FIRST LIVE FRAME`** on the first fill: that is the delivery
milestone missing — do NOT infer from the subscribe line; finish the arm anyway (the poll fallback
inside the engine still fills and flattens), and grade it "WS delivered nothing for a stop entry"
with the child id evidence — that result is worth the run.

## 3. Grading rules (formal)

**Clock correction (Worker3, 2026-09-17 evening — stop-the-line):** `[BROKER]` log lines are
stamped with the PINE BAR time (`lib/log.py:66-81`: `lib._time` once bar 1 exists), not the
event arrival time — measured: PendingNew/New/Filled of one order all carry the identical bar
stamp. So NO latency may be computed from log-line timestamps; at 5m it would measure where in
the bar the fill landed, and the arm latency would read exactly 0 by construction. Therefore:
- **T(venue fill)** comes from the VENUE record (`venue.py order <id>` createdDate/modifiedDate,
  ISO-8601 strings on the venue clock, sub-second fraction) — never from our log.
- **T(engine arrival)** comes from a WALL-CLOCK PREFIX the runner adds to the log stream as it
  tees (a pipeline element, not code on the order path; pyne run under `PYTHONUNBUFFERED=1`),
  caveat stated in every table: "when OUR PROCESS PRINTED the line — an upper bound on arrival
  incl. the logging path". Proof before Friday: three status lines of one order carry three
  DIFFERENT prefix stamps.
- The grader (`f13_grade.py`, read-only, offline-testable) **REFUSES bar-stamped logs**: exit 2,
  "no usable event clock" — never a 0. Two clocks (venue vs host) are compared with the host's
  NTP state recorded in the evidence.
- Prior figures derived from log stamps (e.g. `logs/fill_f9_latency_evidence.txt`) are
  bar-quantised and marked UNVERIFIED on #121; not re-derived.


- **Fill latency** = `T(event FILLED leg=entry)` − `T(venue fill)`; **arm latency** as the grader
  reports it = `T(dispatched EXIT)` − `T(our FILLED print)` (labelled so in the table — it is the
  engine's fill→arm cost, not venue→arm; add the fill latency to get venue→arm). Report per run;
  with n=2 report both values, never a mean.
- ws arm: the grader does NOT split venue→frame vs frame→event; report `FIRST LIVE FRAME`
  present/absent and whether a frame names the child id.
- `NO-SAMPLE` rows are reported as such; a fallback `l2` run is labelled `transport-only`.
- Exit codes from venue tools: 0 yes / 1 no / **2 could-not-determine — never "no"**.
- `flatten.py` (teardown only; the vehicle flattens in-script) now signs from `.side`; a BUY-arm
  ceiling reject reads as "NOT FLAT after 25 s" — check the app before believing it.

## 4. Stop rules

- Any naked or can't-determine state: operator flattens in the app FIRST, diagnose second.
- `INVALID_TRADING_TOKEN` on a conditional write: do NOT re-mint (measured useless); reschedule.
- Position must be flat by 11:20 and by 14:25. Never into ATC, never overnight.
- Executor never types an order-placing command and never handles the OTP.

## 5. After the session (executor, offline)

Cards: F13 measurements → README registry row + a comment on #121 (arm-on-fill) and #130
(child-frame answer); #116 → its exact code; #113 → the repoint timestamp; #135 → id-reuse
verdict. Each comment scrub-gated (`scrub_check.py --strict`) and posted with
`--repo rubycell/pynecore`. CLAUDE.md only for a genuinely NEW venue fact. Commit stripped
evidence (`sed 's/\x1b\[[0-9;]*m//g' f.log | grep -aoE '\[(L1|F|BROKER)\][^[]*'`), never raw
logs, never workdir/ data; raw logs → `backup/deleteable/`.
