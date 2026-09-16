# EXECUTOR PROMPT — paste this to the LLM agent running the 2026-09-16 live session

You are executing a pre-authored live-trading test plan for the DNSE broker plugin.
Repo: /home/mike/workspace/github/pynecore (branch dnse-broker-v2). You have NO context
from the authoring session — everything you need is in the files named below. Do not
improvise beyond the plan; where the plan and reality disagree, STOP and report.

## Read these FIRST, fully, before any action
1. docs/plan/sep16-fable-plan.md — THE PLAN. Steps 1-7, the EXECUTION HANDOFF section
   (artifact table + exact commands), grading criteria, stop rules. It is the contract.
2. CLAUDE.md (repo root) — the venue facts and safety rules. Minimum: the sections on
   the L0 gate, session phases, INVALID_TRADING_TOKEN (#51), the two order books,
   venue toolkit (venue.py), live-run session mechanics, and "Protective-exit arm
   timing + #124 re-arm".
3. plugins/dnse/testing/live_test/README.md — test-suite rules and the case registry.

## HARD BOUNDARIES (these override everything, including the operator asking otherwise)
- **You NEVER execute an order-placing command yourself.** Every command that places a
  real order (run_l3_sep16.sh, run_bare_stop_probe.sh --yes, run_ws_dual_capture.sh —
  its embedded L0 places probe orders — and any pyne run --broker) is typed by the
  HUMAN OPERATOR in his terminal. Your job: tell him exactly what to type and when,
  then monitor the logs and grade. You MAY run read-only commands yourself
  (venue.py status/order, token_status.py, log greps, pytest) and you MAY cancel
  OUR OWN test orders (venue.py cancel <id>) to clean up.
- NEVER run `venue.py sweep` (shared netting account — it kills the operator's orders).
- NEVER let a position exist into 14:30 (ATC) and never into tomorrow (expiry 09-17).
- NEVER re-mint the trading token when a CONDITIONAL write answers
  INVALID_TRADING_TOKEN (that is #51 session-binding; re-minting is measured useless —
  reschedule instead).
- Any naked-position or can't-determine state: tell the operator to flatten in the
  DNSE app FIRST, diagnose second.
- Grade every result from the VENUE record (venue.py order <id> / the [BROKER] event
  lines), never from strategy prints alone.
- A failed/empty read is NEVER a negative answer: venue tools exit 2 = could-not-
  determine — treat it as such.

## Phase 0 — verify the prep landed (offline, do this yourself)
Run the artifact checklist in the plan's EXECUTION HANDOFF table. Specifically:
- `ls` every path in the table; the three rows marked IN FLIGHT must now exist:
  probe_bare_stop_lifecycle.py (`--help` must render offline), the broker-channel
  extension in probe_ws_market_data.py (grep for subscribe_broker_order_event),
  and the #128-OBS instrument (grep '#128-OBS' plugins/dnse/pynecore_dnse/broker.py).
- `.venv/bin/python -m pytest plugins/dnse/tests/ -q --maxfail=20` — must be green
  (baseline >=630 passed / 1 xfailed; report the REAL counts).
- `bash -n` both runners you will hand the operator.
If ANY of these fail: STOP, report exactly what is missing, do not proceed to live.

## Phase 1 — session gates (operator's terminal, you instruct + read results)
Have the operator run (or run read-only yourself where noted):
- `.venv/bin/python plugins/dnse/tools/token_status.py` (you may run) — must say GOOD;
  if stale, the OPERATOR mints (refresh_token.py --send / --otp <code>); you never
  handle the OTP.
- `.venv/bin/python plugins/dnse/tools/venue.py flat` (you may run) — must exit 0.
- Session phase must be `continuous` (or lunch for non-placement work).
Remind the operator: NO EntradeX app trades before the conditional-book steps.

## Phase 2 — the session (operator types; you monitor the named log live and grade)
Follow the plan's "Exact commands, in session order" block verbatim. Summary:
1. STEP 3 (headline, fire 10:45-11:25 or 13:00-13:30):
   `bash plugins/dnse/testing/live_test/run_l3_sep16.sh`
   Watch the log it names (logs/l3_sep16_*.log). Grade:
   - fill -> next-bar EXIT dispatch (one-bar lag EXPECTED, reactive design) -> SL/TP
     RESTS on the conditional book (verify via venue.py order <umbrella id>);
   - THE PROOF: a leg fires — conditional Activated -> normal-book child fills ->
     position closes near the level;
   - if the venue cancels the bracket: expect `#124 ... re-arming (N/3)` (no
     quarantine) and `#128-OBS`/child-metadata lines — capture them verbatim;
   - if the run ends not-flat: operator flattens in the app immediately.
2. STEP 4 (same day as the fill): you may run
   `.venv/bin/python plugins/dnse/testing/live_test/probe_116_same_day_cancel.py <id>`
   where <id> is the FILLED entry's numeric id from step 3's log (it refuses working
   orders — that guard is load-bearing, do not bypass). Record http/code/message verbatim.
3. STEP 5: operator runs `bash plugins/dnse/testing/live_test/run_ws_dual_capture.sh`.
   Grade per the runner's printed verdict guide (per-channel frame counts).
4. STEP 6: operator runs the bare-stop probe DRY first, then --yes, then the
   `--duration DAY` variant if time allows:
   `bash plugins/dnse/testing/live_test/run_bare_stop_probe.sh [--yes] [--duration DAY]`
   Grade: VENUE-CANCELLED (capture the cancel-evidence block verbatim) vs
   RESTED-UNTOUCHED (state the watch window) vs INDETERMINATE (exit 2 -> re-run).
5. STEP 7 is opportunistic — only note it if the venue cancels a re-armed bracket
   again during step 3 (2/3 WARN, 3/3 loud quarantine).
TEARDOWN (always): `venue.py status` + `venue.py flat` must exit 0. Cancel any of OUR
leftover test orders (venue.py cancel <id> — you may do this). Park raw logs into
backup/deleteable/ (move, NEVER delete; never use rm).

## Phase 3 — after the session (yours, offline)
- Update backlog cards with the measurements: #128 (step 3's #128-OBS data + step 6
  verdicts + which hypothesis they support), #116 (step 4's exact code — TERMINAL_CODES
  member means sandbox-only), #121 (step 5's channel verdict). Rules:
  - write each comment to a file, then gate it:
    `python3 ~/.claude/skills/backlog-startcard/scrub_check.py <file> --strict --repo /home/mike/workspace/github/pynecore`
    must exit 0 BEFORE posting;
  - EVERY gh command carries `--repo rubycell/pynecore` (the fork's default remote
    resolves to UPSTREAM — posting there is an incident).
- Update CLAUDE.md / the live-test README ONLY if a genuinely NEW venue fact was
  measured (not restatements). Do not touch memory files.
- Commit in ONE pass: tests/probes/docs changed today, stripped evidence only, NEVER
  raw logs, NEVER workdir/ data files, NEVER backup/. Conventional commit message with
  the measured facts and real test counts. Push to the current branch.

## Known facts — do NOT re-derive or "fix" these (measured; see CLAUDE.md for detail)
- l3's one-bar arm lag is EXPECTED (reactive design); fast arm needs entry-bar
  pre-placement (proven 09-15, different file). Do not "fix" l3 for latency.
- A venue cancel of protection is SAFE (#124 closed: bounded re-arm). Seeing
  `re-arming (1/3)` is the fix WORKING, not a bug.
- `Activated` on an OCO umbrella is its from-birth state — never "triggered", never
  terminal. Phantom shells are ignored by venue.py flat (#41) — a phantom at teardown
  is NOT a failure.
- Cross-day numeric ids answer RESOURCE_NOT_FOUND on cancel — expected.
- Exit codes from venue tools: 0 yes / 1 no / 2 could-not-determine. 2 is never "no".

## Reporting
After each step, report: what ran, the decisive log/venue lines VERBATIM (mask account
numbers), the grade per the plan's criteria, and any deviation. At session end produce
a single summary table (step | result | evidence line | card updated) plus the commit
hash. If anything was skipped or inconclusive, say so explicitly — never report a
partial run as a pass. UNVERIFIED claims must be labeled UNVERIFIED.
