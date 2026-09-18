# Live-L2c — measuring an OCO leg EXECUTE (2026-09-18)

## The operator's one line

**Two runs, size 1, one round-trip each, one at a time, flat by 14:25.**

| run | tpPct | slPct | near leg, should execute | far leg, venue should cancel | backstop | worst case, 1 contract |
|---|---|---|---|---|---|---|
| A, stop-favoured | 0.5 | **0.05** | SL | TP | -0.30% | ~6 pts, ~600,000 VND |
| B, TP-favoured | **0.05** | 0.3 | TP | SL | -0.55% | ~11 pts, ~1,100,000 VND |

Points at a ~1985 index level, 1 point = 100,000 VND per contract. The backstop is
`-(slPct + 0.25)`: loss-side only, bounded by the SL leg and deliberately NOT by
`max(tp, sl)`, because that let the take-profit setting decide the loss limit and bounded
run A at -1.5%, roughly 30 points, in exactly the scenario this run exists to detect. The
worst case is the bound, not the expectation: if the stop leg fills as designed run A loses
about 1 point. **A backstop fire is the gap-through measurement and should be reported as a
result, not treated as a loss to explain.**

Never concurrently: this account nets per symbol, so two vehicles would net against each
other and neither result would mean anything. 14:25 is the standing authorisation's
deadline for an L2 run, not a number chosen here — continuous trading ends 14:30 and the
ATC auction that follows REFUSES cancels while filling whatever rests.

## The gap this closes

Worker3's census: **no OCO umbrella leg, TP or SL, has ever been observed executing on
DNSE in any run we hold.** l2b proved the bracket is PLACED and that it ARMS on the fill
bar (#121, measured 09-15 and again this morning at 1.278 s from the WS fill frame to the
exit on the wire). What happens when a leg actually TRIGGERS has never been recorded.

Every fill-tier case depends on that path. Until it is measured it is a design we believe,
not a behaviour we have seen.

## Vehicle

`l2c_oco_execution.pine` — the l2b shape (breakout stop entry at `higherHigh2`,
trigger-derived bracket, pre-placed `strategy.exit`, size 1, `pyramiding=0`,
`barstate.isrealtime` gate) with three changes, each so a leg CAN execute:

1. **Enter-once latch**, set on the first FILL rather than the first dispatch. The entry
   still chases while flat and unfilled; only the re-entry after a completed round-trip is
   blocked. One umbrella and one child per run, so "which id did the venue amend" has a
   single answer.
2. **Max-2-candles flatten removed.** In l2b it closed 13 of 39 episodes at two bars. Here
   it would pre-empt exactly the cases worth measuring.
3. **Backstop widened to `-3 x max(tpPct, slPct)`.** A fixed -0.3% would fire before the
   0.5% leg in both planned runs.

Legs are `input.float`, so one file serves both runs:

| run | tpPct | slPct | expectation |
|---|---|---|---|
| A, stop-favoured | 0.5 | 0.05 | the SL triggers; the TP rests far away and should be cancelled by the venue |
| B, TP-favoured | 0.05 | 0.5 | the TP triggers; the SL rests far away and should be cancelled by the venue |

The near leg is close enough to execute within a bar or two; the far leg is parked so its
venue-side cancellation is observable. The file defaults (tp 0.2 / sl 0.05) are neither
run — set the inputs explicitly.

## Run A — pre-existing umbrella on the account, EXCLUDE from grading

The account already carries an umbrella from this morning's F13 run. It is SPENT, not
armed, and it cannot be cleared further:

    umbrella  damadq2vfqkc7397o0tg   [OCO] status=Activated price=1988.8 stop=1980.8
    child     39356                  [NORMAL] status=Canceled

Four pieces of evidence, none of them inference: the operator's app shows that row (09:50,
1,988.8) as **Đã hủy**, cancelled; the venue refuses a cancel with "order is done"; the
child is Canceled; and `Activated` is TERMINAL for a conditional — it made its normal order
and can do nothing else (operator ruling, 2026-09-18, commit `f88c7e43`, CLAUDE.md:244).
The API simply keeps the conditional row's terminal value, which is why it still reads
Activated.

**Disambiguation rule for grading:** the run's own umbrella is whichever OCO id has a
`createdDate` AFTER the launch instant. Exclude `damadq2vfqkc7397o0tg` and `39356` from
every row of the grading list below. Do not match on status, price or symbol — two
umbrellas on one contract will agree on all three, and run A's TP at 0.5% off a trigger
near this morning's levels can land close to the spent umbrella's 1988.8.

**COMPARE AS TIMEZONE-AWARE DATETIMES, NEVER AS STRINGS.** `createdDate` is UTC and carries
a `Z`; the launch time is quoted in ICT; they are 7 hours apart. The naive comparison does
not merely fail, it fails in the WRONG DIRECTION for an afternoon run and excludes the very
umbrella being measured:

| | |
|---|---|
| launch | 13:00 ICT = 06:00 UTC |
| run umbrella created | 13:05 ICT, venue reports `…T06:05…Z` |
| naive `"06:05" > "13:00"` | **False** — the run's umbrella is discarded |
| timezone-aware | **True** — correct |

A grader who then reported "run A placed no umbrella" would be reporting a bug that does
not exist. What makes this dangerous is that the MORNING umbrella is excluded correctly
under both forms, so the rule looks like it works when tested against the case already in
hand.

Verified on the real record, not assumed: the spent umbrella's `createdDate` is
`2026-09-18T02:50:16.977302Z`, which is 09:50:16.977 ICT, and the F13 log's wall-clock
prefix for the exit dispatch that created it reads `1789699817.235` = 09:50:17.235 ICT. The
two agree to within 0.26 s, and the umbrella is created just BEFORE our log line is written.

So: parse `createdDate` as UTC, convert the launch time from ICT to UTC, compare instants.
Exclude any OCO row at or before the launch instant; the run's umbrella is the one strictly
after. **If NO row is after the launch instant, that is a finding — "the run placed no
umbrella" — and is reported as such, never resolved by falling back to a price match.**

## Grading list, per run

Worker3 grades from the venue record, not the run log.

1. **Umbrella**: id, `stopPrice`, `stopOrderPrice`, status at placement and after the
   trigger.
2. **Child**: id at placement.
3. **On trigger, amend or spawn?** Did the venue keep the SAME child id and change its
   price, or mint a NEW id? This is the central question, and it needs THREE reads, not
   one:
   - the child row BEFORE the trigger,
   - the child row AFTER, read by the SAME id,
   - a FULL NORMAL-book listing after the trigger.

   The third is what makes the answer sound. A single after-read of the known id returns
   "unchanged" identically whether the venue amended nothing or minted a different order
   nobody looked for, so without the listing a spawn is indistinguishable from a no-op. If
   the venue spawns, the OLD child's terminal state is evidence too, which is why the
   before-row is captured rather than inferred.
4. **The other leg**: was it cancelled by the venue, and at what timestamp relative to the
   triggering leg? A gap here is a window where both legs could fill.
5. **The triggered stop-limit**: did it FILL, or rest unfilled? DNSE has no stop-market, so
   a triggered stop posts a limit, and a gap through it can leave it resting.
6. **Every WS frame for the child ids**, with timestamps.
7. **Position read after**, and whether the account returned flat without manual action.

## Grading traps, all measured today — read before grading

These cost three sessions time this morning. Each one makes a wrong grade look right.

- **The `FIRST LIVE FRAME` grep trap.** The #131 internal-error warning ends with the words
  "treat the absence of a FIRST LIVE FRAME as the real signal", so a naive grep for
  `FIRST LIVE FRAME` matches the warning that says the frame is MISSING. Measured on run 1:
  naive 2, anchored 0. **Only `WS ORDER SOURCE FIRST LIVE FRAME` is valid.** It caught a
  wait loop, a monitor and a manual count, in three different sessions.
- **WS frame ids are MASKED to a four-character suffix** (`id=*9336`). Any frame-to-id rule
  must be a suffix match anchored at or after the entry dispatch, or a coincidental suffix
  becomes a finding. `f13_grade.py` already does this; do not eyeball it.
- **`venue.py status` and `flat` do NOT scan the OCO book.** The umbrella is invisible to
  both, so neither is evidence about the stop leg in either direction. `venue.py order <id>`
  DOES read the OCO book and works fine. The gap is DISCOVERY, not reading.
- **The engine does not log the umbrella id.** `dispatched EXIT ... -> ['39356']` names the
  child only. The `#128-OBS child=... conditionOrderId=...` line exists at `broker.py:2961`
  but returns early at 2950-2952 when the child metadata has no `condition` field, which is
  what happened in run 1 — a conditional log line that does not fire is indistinguishable
  from a feature never built, and two sessions concluded the latter independently.
- **Field split across three reads**: the OCO listing carries `stopPrice`/`stopOrderPrice`
  but no `externalOrderId`; the order DETAIL carries `externalOrderId`; the child's
  `conditionOrderId` lives in metadata that neither the listing nor the toolkit surfaces.
  "Find the umbrella whose externalOrderId is the child" cannot be done from any single
  read.
- **A cancelled CONDITIONAL reports `filled=None` (field absent); a cancelled NORMAL order
  reports a real `filled=0`.** Reading the `None` as a zero invents a measurement. No-fill
  evidence for a conditional is `child=0` plus a flat position read.
- **`Activated` never means "triggered".** An OCO umbrella is Activated from birth —
  measured today at 13 ms after the child. Do not read it as execution.

## Authoring trap found building this vehicle — pine2pyne drops parentheses

Measured 2026-09-18 while setting the backstop. The transpiler dropped the parentheses in
this expression, twice, in two different spellings:

| written in Pine | generated Python | value at sl 0.05 |
|---|---|---|
| `-(slPct + 0.25)` | `-slPct + 0.25` | **+0.20** |
| `0.0 - (slPct + 0.25)` | `0.0 - slPct + 0.25` | **+0.20** |
| `-0.25 - slPct` | `-0.25 - slPct` | -0.30 (correct) |

Intended was -0.30. A POSITIVE threshold inverts the backstop: the test is
`openPnlPct < DOUBLE_CHECK_PCT`, so at +0.20 it fires on the first bar of every position,
cancels the bracket and flattens — destroying the measurement it exists to protect, and
placing cancel and close orders nobody asked for.

Two lessons, the second more important than the first:

1. **Prefer expressions with no parentheses to drop.** `-0.25 - slPct` reads identically
   left-to-right in both languages.
2. **Verify against the GENERATED `.py`, never the `.pine`.** I caught the first spelling
   by reading the generated line, then "fixed" it and checked my work by evaluating the
   PINE intent in Python — which of course gave the right answer while the generated code
   was still wrong. The check has to read the same object the runtime will. This is the
   `getsource` versus stale `.pyc` trap in a different costume, and it survived one round
   of me knowing about it.

No other vehicle uses a unary minus on a parenthesised expression (`grep -rnE "=\s*-\("`
over the live_test `.pine` files returns only this file), so the blast radius is limited to
what was built today.

## STANDING RESTRICTION until #164 lands

**No live bracket vehicle may re-enter after a stop within the same run until #164 lands.
#162 is necessary, NOT sufficient — measured in vitro.**

Measured live 2026-09-18: when the venue amends an OCO child in place and fills it, the
engine's forced cancel on that now-terminal id is refused permanently, the park never
clears, and every later exit dispatch is deferred behind it. A re-entry after a stop
therefore opened a position the engine never bracketed — naked for 66 seconds, ended only
by a manual flatten.

**#162 (committed `1cb440b8`) fixes the park and does NOT close this.** Reproduced on a
real `OrderSyncEngine` by two reviewers independently: with the plugin reporting
`ALREADY_FILLED` — the #162 path, already in the tree — the park clears, no deferral
occurs, **and the position is still never armed**. The cause is #164:
`_armed_protective_venue_qty` is a write-only ledger, so the engine computes zero
protection deficit and declines to arm, believing itself protected throughout. A
discriminating control confirms ownership — clearing only that stale belief, with the park
still live and the cancel still failing, makes the arm path dispatch.

So the restriction is lifted by **#164**, not by #162. This applies to ANY bracket vehicle,
not just l2c; the enter-once latch is a vehicle-level mitigation, not the fix.

### SINGLE ROUND TRIPS ARE PERMITTED — and this is compliance, not an override

Operator decision, conveyed by DNSEPlugin, 2026-09-18 ~12:20. Recorded with its
provenance rather than as a direct quote, because I did not receive it firsthand.

**Read the restriction literally: it prohibits RE-ENTRY after a stop.** A run that cannot
re-enter does not engage it. The enter-once latch makes re-entry impossible, so a
single-round-trip run COMPLIES with the rule as written rather than being excused from it.
That is a stronger footing than an exception, and it is why these runs may proceed while
#164 is still open.

**The mitigation is verified in the built artifact, not assumed.** From `HEAD`'s generated
`l2c_oco_execution.py`:

```
21:  hasTraded: PersistentSeries[bool] = False
25:  if strategy.opentrades > 0 or strategy.closedtrades > 0:
26:      hasTraded = True
28:  if barstate.isrealtime and strategy.opentrades == 0 and (not hasTraded):
```

`closedtrades` increments on the close and persists, which is what survives a round trip
completing inside one bar — the exact case that defeated the first latch and produced the
naked position. Pinned by `plugins/dnse/tests/test_l2c_latch.py`, which reads the GENERATED
file and evaluates the guard against that state rather than grepping for a field name.

**Why #164 cannot bite a single round trip.** The write-only ledger only refuses to RE-arm;
the FIRST arm writes an empty key and works, which run A demonstrated live. A naked window
needs a SECOND position, and the latch forbids one. If a run somehow produces a second
entry, the latch has failed and the run is aborted immediately — that, not #164, is the
condition to watch for.

**What stays forbidden**: any vehicle without a verified enter-once latch, and any run
permitting more than one round trip, until #164 lands.

## Preconditions and the abort rule

L0 green, `venue.py flat` exit 0 before launch, operator watching, DNSE app open. Size 1,
one round-trip per run. Any id that will not cancel, or a position not bracketed within one
bar, stops the run and is reported with the ids immediately.

**A RUN ENDS WHEN ITS PROCESS IS STOPPED, NOT WHEN ITS TRADE IS GRADED.** Stop the engine
explicitly the moment the round trip completes, and confirm no `pyne` process for the
vehicle remains before launching the next one.

Measured 2026-09-18: run B's engine was still alive when run A2 launched, so two l2c
engines ran concurrently on a netting account for seven minutes — the exact state this plan
forbids, and the reason it forbids it is that two vehicles net against each other and BOTH
results become meaningless. No harm resulted, verified rather than assumed: the only new
venue row after the A2 cutoff was A2's own entry, and run B dispatched zero entries after
its fill because the `closedtrades` latch held.

The cause was an assumption, not an oversight in the rules: "the trade is graded" was taken
for "the run is over". **The vehicle does not self-terminate** — the enter-once latch stops
it RE-ENTERING, it does not stop the engine, which keeps evaluating bars until it is killed
or times out. Run A this morning only ended because it was killed. One-at-a-time therefore
needs an explicit stop, not the assumption that a finished measurement means a finished
process.

It was found by someone checking something else entirely: `ps -o lstart` on the first
matching process returned the earlier run's start time. A check that prints what it
actually saw, rather than only answering the question it was asked, catches what nobody
thought to look for.

**READ THE TOOL'S EXIT STATUS, NEVER A PIPE'S — and this is the third time today one check
answered about a different object than the claim it was used to support.**

At the A2 abort the cleanup check was written as:

```
.venv/bin/python plugins/dnse/tools/venue.py flat 2>&1 | tail -1
echo "FLAT EXIT: $?"          # <- this is TAIL's status, not venue.py's
```

It printed `FLAT EXIT: 0` while the very output above it listed a LIVE order. Re-run with
the status captured to a variable (`cmd > f 2>&1; rc=$?`) it returned **exit 1, NOT CLEAN,
1 LIVE order** — an unfilled entry conditional that had to be cancelled. Trusting the first
line would have left a resting order on the account overnight.

Its two siblings from the same day: a backstop verified by evaluating the PINE expression
while the GENERATED `.py` still said something different, and a watcher whose silence was
read as a quiet market when the watcher itself had died. All three have one shape — the
check and the claim were about different objects — and all three looked like success.

So, for every gate in this plan: capture the status to a variable, read the summary LINE as
well as the code, and before trusting any check name one broken state it would CATCH and one
it would MISS.

**Flatten with `plugins/dnse/testing/live_test/flatten_api.py`** — it has an entry point,
argument parsing and `--dry-run`. `plugins/dnse/tools/flatten.py` is a LIBRARY MODULE: it
has no `__main__` block and no argument parsing, so running it as a script does nothing and
exits 0. Measured today: `flatten.py --help` returns exit 0 and zero bytes. A no-op that
exits 0 is indistinguishable from a tool that ran and found nothing to do, and this is the
command reserved for when something has already gone wrong.

## Status

Vehicle authored and transpiled clean. Offline proof (a plain `dnse:` backtest must place
ZERO trades under the `barstate.isrealtime` gate, and the latch must hold to exactly one
round-trip) is pending the review-manifest entry, because the backtest executes a new file.
Whether this runs today, and whether it replaces the 13:00 poll arm, is the operator's
decision — the poll arm's question (WS versus poll) was answered by run 1 this morning.
