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

## 1. Thursday evening prep — runner change (card, Worker3 after #132-W0; Fable reviews)

`plugins/dnse/testing/live_test/run_f13_latency.sh` gains:

| Flag | Default | Meaning |
|---|---|---|
| `--vehicle l2b\|l2` | `l2b` | which transpiled `.py` to launch (`l2b_fill_protect_flatten.py` / `l2_fill_flatten.py`) |
| `--window-bars N` | `6` | l2b only: bars to wait for the stop to fill; unfilled → cancel, log `NO-SAMPLE`, count 0 |
| `--fallback` | on | if an l2b slot yields NO-SAMPLE, run ONE `l2` in its place so the arm still gets a transport sample |
| `--fills N` | unchanged | runs per arm (operator's n) |

Grading additions (per run, per arm), all read from the venue record + `[BROKER]` lines:

- `T(venue fill)` from `venue.py order <entry id>` (for l2b: the **child** normal-book id named by the
  activated conditional's `externalOrderId`; `venue.py order` follows it).
- `T(event FILLED … leg=entry)` and `T([BROKER] dispatched EXIT …)` from the log.
- `venue.py order <umbrella id>` showing the bracket **resting** (umbrella is `Activated` from birth —
  never read that as triggered; the TP child on the normal book is the visible cover).
- ws arm only: whether an `order frame via WS` line names the **child id** (answers #130-for-stops);
  the existing `WS ORDER SOURCE FIRST LIVE FRAME` milestone stays the delivery gate (#134).
- Timeframe **5m** (the runner was on `@1` before #146 — corrected). `timeout` per run = (window + 3) × 300 s = 2700 s at the default window.
- NO-SAMPLE cancel uses the entry id from OUR OWN log's `dispatched ENTRY … -> ['id']` line, passed to `venue.py cancel`; if that id cannot be read the runner says so and does NOT cancel — never a sweep, never an inferred id.

Definition of done for the prep: `bash -n` clean; `--help` never constructs a broker; a dry run
against the fake seam prints the grading table with NO-SAMPLE rows; `.venv/bin/python -m pytest
plugins/dnse/tests/ -q` unchanged; committed; hash sent to Fable; reviewed before Friday's open.

## 2. Friday session — exact sequence (operator types every order-placing command)

Time windows are Asia/Ho_Chi_Minh. **No EntradeX app trade before step 4 finishes** (#51/#46:
conditional-book writes die after the first app trade of the day).

| # | When | Who | Command / action | Gate |
|---|---|---|---|---|
| 1 | ~08:20 | operator | mint the trading token (`tools/refresh_token.py`, OTP by hand) | `token_status.py` → GOOD |
| 2 | ~08:25 | executor | `venue.py status` · `venue.py flat` | flat exit **0**; exit 2 = stop, could-not-determine |
| 3 | 08:45–09:05 | executor | **roll grade** per runbook §1 (probe log bracketing the repoint) | new venue fact recorded |
| 4 | 09:15–11:10 | operator | `bash plugins/dnse/testing/live_test/run_f13_latency.sh --arm ws --fills 2` — **each run can take up to 45 min** (5m × (6-bar window + 3); typical 15–20 min on a breakout) | L0 exit 0 inside the runner; executor watches the named log |
| 5 | 13:00–14:15 | operator | `bash plugins/dnse/testing/live_test/run_f13_latency.sh --arm poll --fills 2` (moved to the afternoon: two 45-min worst cases do not fit one morning) | same; flat by 14:25 |
| 6 | ≤11:20 | executor | `venue.py flat` — must exit 0 before lunch; cancel OUR leftovers with `venue.py cancel <id>` | never `sweep` |
| 7 | 13:00 (before step 5) | executor | `probe_116_same_day_cancel.py <filled entry id from step 4>` (refuses working orders; takes seconds) | record http/code/message verbatim |
| 8 | any run | executor | passive captures: one `venue.py status` while a position is open (prod `/positions` frame); first venue order id of the day vs Thursday's range (#135 id-reuse) | evidence file |
| 9 | ≤14:25 | executor | `venue.py flat` exit 0 — **never hold into 14:30 ATC** | |

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
  venue clock, ms) — never from our log.
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


- **Latency** = `T(event FILLED leg=entry)` − `T(venue fill)`; **arm latency** = `T(dispatched
  EXIT)` − `T(venue fill)`. Report per run; with n=2 report both values, never a mean.
- ws arm splits: venue→frame (delivery) vs frame→event (our drain cost).
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
