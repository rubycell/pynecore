# Live session runbook — 2026-09-15 (Tue)

Goal: clear every question that ONLY the real DNSE order book can answer.
Operator monitors the DNSE account throughout. **The operator fires every
order-placing command himself.**

## Binding constraints (read before starting)

1. **#51 — conditional writes break after your first EntradeX app trade.**
   Steps ① and ⑤ are conditional-book writes. **Do NOT trade in the app until
   ① is done.** If `INVALID_TRADING_TOKEN` appears on a conditional write, do
   NOT re-mint (measured: does not help) — reschedule pre-open.
2. **Session phases** — derivatives ATO 08:45-09:00; continuous from 09:00.
   ATC refuses cancels and fills what rests. Lunch is safest for market probes.
3. **Roll** — expiry Thu 2026-09-17, alias repoint Fri 09-18. The #113 roll test
   is NOT today.
4. **Every step is NO-FILL class** (orders >=4.5-5% from market, or a no-op).
   Stop before the fill tier unless ① earns it.
5. Grade from the VENUE RECORD, never the run log alone.

## Step 0 — Gates (run first, every session)

```bash
cd /home/mike/workspace/github/pynecore
.venv/bin/python plugins/dnse/tools/token_status.py          # token GOOD?
.venv/bin/python plugins/dnse/tools/venue.py status          # phase, position, orders
.venv/bin/python plugins/dnse/tools/venue.py flat            # exit 0 == truly flat+clean
.venv/bin/python plugins/dnse/testing/live_test/level0_venue_semantics/l0_order_semantics.py
```
**L0 must exit 0.** If not, stop — nothing below is valid.

## Step ① — #124 part B: OCO lifecycle isolation  [HIGHEST VALUE]

Answers: does the VENUE cancel a resting conditional/OCO on its own, with no
engine running? Discriminates suspect (i) venue-lifecycle from (ii) wake
re-dispatch / (iii) add-a-leg. **Decides the entire #124 fix direction.**

```bash
# DRY RUN first — prints the intended order, touches nothing:
.venv/bin/python plugins/dnse/testing/live_test/probe_124_oco_lifecycle_isolation.py
# then, after reading the pre-flight summary:
.venv/bin/python plugins/dnse/testing/live_test/probe_124_oco_lifecycle_isolation.py --yes
```
- `VERDICT: VENUE-CANCELLED` -> suspect (i) CONFIRMED; fix must stop/tolerate a
  venue-driven cancel (tag `cancel_reason` so the existing guard at
  sync_engine.py:6320/6504 fires).
- `VERDICT: RESTED-UNTOUCHED` -> (i) REFUTED for that window; the cancel is OURS
  (wake re-dispatch or add-a-leg) -> fix is engine-side.
- `VERDICT: INDETERMINATE` (exit 2) -> a read failed; re-run, do NOT interpret.

## Step ② — prod trading-WS order events (NEVER once captured)

Answers: does the prod WS actually deliver ORDER events, or is our "dual
transport" really just the 0.5s poll?

```bash
# terminal A — start the watcher FIRST:
.venv/bin/python plugins/dnse/testing/live_test/probe_ws_market_data.py --trading --seconds 120
# terminal B — generate ONE account event (place + cancel a far LO):
#   use the L1 place/cancel path or venue.py; the order must be >=4.5% away.
```
Zero frames with ZERO account activity proves nothing (empty != conclusive) —
the event MUST occur inside the capture window.

## Step ③ — #116: cancel a KNOWN-FILLED order id  [no-op, safest]

```bash
.venv/bin/python plugins/dnse/testing/live_test/probe_116_117_prod_premises.py A     # see note below
```
Refuses unless the working book is EMPTY (id-reuse safety, #96).
- TERMINAL_CODES member -> #116 is SANDBOX-ONLY -> shrink the card.
- generic code + "terminal state Filled" -> #116 is REAL ON PROD.
- NOT_FOUND -> cross-day ids do not resolve; re-measure after a same-day fill.

## Step ④ — #117: STOCK amend semantics

Answers: does `PUT` return the SAME id or a NEW id (docs say cancel+replace),
and is both-price-and-qty-in-one-PUT accepted for STOCK?
```bash
.venv/bin/python plugins/dnse/testing/live_test/probe_116_117_prod_premises.py B
```
Places a HPG LO at -5% (no-fill), ONE PUT changing both fields, then cancels
every id it touched and verifies terminal.

## Step ⑤ — #118 GTD (free, while ①'s order rests)

Read the GTD on the resting conditional's venue record; confirm the clamp landed
on the real `finalTradeDate` (2026-09-17 this week).
```bash
.venv/bin/python plugins/dnse/tools/venue.py order <id-from-①>
```

## STOP HERE

Do not enter the fill tier (⑥ #123 live partial, ⑦ #124 in situ) unless ①
produced something that makes the exposure worth it. Re-running #124 blind just
reproduces a naked position a third time.

## Teardown (always)

```bash
.venv/bin/python plugins/dnse/tools/venue.py status
.venv/bin/python plugins/dnse/tools/venue.py flat     # must exit 0
```
Never `sweep` — it cancels the operator's own orders too.

## After the session

- Grade every step from the venue record.
- Update: CLAUDE.md (measured venue facts), memory, and cards
  #124 (root cause + part-B verdict), #116, #117, #118, #107/#121 (WS).
