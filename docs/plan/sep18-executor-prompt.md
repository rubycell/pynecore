# EXECUTOR PROMPT — paste to the worker session running the 2026-09-18 live session

You are executing a pre-authored live-trading test plan for the DNSE broker plugin. Repo:
/home/mike/workspace/github/pynecore (branch dnse-broker-v2). You have NO context from the
authoring sessions — everything you need is in the files below. Do not improvise beyond the
plan; where plan and reality disagree, STOP and report to Fable (session DNSEPlugin) and the
operator.

## Read these FIRST, fully, before any action
1. docs/plan/sep18-f13-l2b-plan.md — THE PLAN (vehicle decision, sequence table, grading, stop rules).
2. plugins/dnse/testing/live_test/FRIDAY_2026-09-18_RUNBOOK.md — roll grade at open, standing
   rules, known-and-expected, the 09-17 passive captures. The plan supersedes only its F13 section.
3. CLAUDE.md (repo root): the L0 gate, session phases, INVALID_TRADING_TOKEN (#51), the two order
   books, the venue toolkit (`venue.py`), "A DNSE position read is NOT authoritative alone — AND its
   size is UNSIGNED", "Protective-exit arm timing + #124 re-arm".
4. plugins/dnse/testing/live_test/README.md — suite rules and the case registry.
5. ~/.claude/CLAUDE.md house rule: every card is processed with the backlog-startcard skill; a
   fix on a Done/closed card moves it back to In progress first.

## HARD BOUNDARIES (override everything, including the operator asking otherwise)
- You NEVER execute an order-placing command. `run_f13_latency.sh` (both arms) and any `pyne run
  --broker` are typed by the HUMAN OPERATOR. You tell him exactly what to type and when, then
  monitor the named log and grade. You MAY run read-only commands (`venue.py status/order/flat`,
  `token_status.py`, log greps, pytest) and MAY cancel OUR OWN test orders (`venue.py cancel <id>`).
- You NEVER handle the OTP / trading-token mint. The operator does.
- NEVER `venue.py sweep` (shared netting account — it kills the operator's orders).
- NEVER let a position exist into 11:30 lunch, 14:30 ATC, or overnight.
- NEVER re-mint the token on `INVALID_TRADING_TOKEN` from a conditional write (measured useless).
- Any naked / can't-determine state: operator flattens in the app FIRST, diagnose second.
- Grade from the VENUE record (`venue.py order <id>`, `[BROKER]` lines), never strategy prints.
- Exit code 2 from any venue tool = could-not-determine. It is NEVER "no" and NEVER "flat".
- Every publish is scrub-gated (`python3 ~/.claude/skills/backlog-startcard/scrub_check.py <file>
  --strict --repo /home/mike/workspace/github/pynecore` exit 0) and carries `--repo rubycell/pynecore`.
  Mask account numbers. Never echo secrets.

## Phase 0 — verify the prep landed (offline, yourself, before 08:20)
- `git log --oneline -5` shows the runner change commit (`--vehicle`, `--window-bars`, `--fallback`)
  reviewed by Fable; `bash -n plugins/dnse/testing/live_test/run_f13_latency.sh`;
  `bash plugins/dnse/testing/live_test/run_f13_latency.sh --help` renders WITHOUT touching the venue.
- `.venv/bin/python -m pytest plugins/dnse/tests/ -q` green (report REAL counts; a gate whose exit
  status comes from a pipe tail is not a gate — read pytest's own summary line).
- The l2b vehicle files exist and match: `l2b_fill_protect_flatten.pine` / `.py` / `.toml`.
- Roll probe alive: `tail -3 plugins/dnse/testing/live_test/logs/roll113_*.log` ticking.
If ANY fails: STOP, report exactly what is missing. Do not proceed to live.

## Phase 1 — gates (08:20–09:05)
Operator mints the token; you run `token_status.py` (must say GOOD), `venue.py flat` (exit 0),
confirm session phase. Then the ROLL GRADE per runbook §1 — the repoint timestamp is a new venue
fact; "no divergence" is NOT a pass if the AGED arm equalled current at start.

## Phase 2 — the session (09:15–11:20, then 13:00–14:25)
Follow the plan's sequence table verbatim. Per run, capture and report:
- the entry id, the child id (activated conditional's `externalOrderId`), the umbrella id;
- `T(venue fill)`, `T(event FILLED leg=entry)`, `T(dispatched EXIT)`; SL/TP resting per
  `venue.py order`; ws arm: `FIRST LIVE FRAME` present/absent and whether an `order frame via WS`
  names the child id;
- NO-SAMPLE / fallback rows labelled as such.
Lunch: `venue.py flat` exit 0 by 11:20. Afternoon: #116 probe on a filled id (it refuses working
orders — that guard is load-bearing, do not bypass); the two passive captures; flat by 14:25.

## Phase 3 — after the session (offline)
Update cards per plan §5 (README registry row; #121, #130, #116, #113, #135 comments), CLAUDE.md
only for a genuinely new venue fact, commit stripped evidence only, raw logs → backup/deleteable/.
Report: a single table (step | result | evidence line | card updated) + the commit hash. Skipped
or inconclusive steps are stated as such — never report a partial run as a pass; UNVERIFIED
claims are labelled UNVERIFIED.
