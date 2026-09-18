# Fill-tier retier by risk control — move table and applied drafts (2026-09-18)

Operator decision (fixed): nothing merges; L0, L1, L4 unchanged. The fill tier splits by
RISK CONTROL: **L2** = the protective stop's distance from the REAL fill price is known and
bounded before the run AND size is exactly 1; **L3** = stop distance uncertain (reactive
exits, trailing stops, brackets armed a bar late, crossed-at-placement, prune/adoption races)
and/or size can reach 2 or more. Ids are never renamed; a moved row keeps its id and gets
"retiered from L<old> 2026-09-18".

Worktree `/home/mike/workspace/github/pynecore-worker2`, docs only, uncommitted. Applied edits:
`plugins/dnse/testing/live_test/README.md` (tier section, two registry rows, planned-additions
notes) and `CLAUDE.md` (DNSE testing section). Evidence per row cites the vehicle file, a
registry row, a log or a card.

## 1. Move table

| Case id | Current | Proposed | Reason (stop-distance certainty / max size) | Evidence |
|---|---|---|---|---|
| Live-L2-SingleFill | L2 | **L3** (retiered) | No protective stop at all; exposure bounded only by the ±7 % band and the next-evaluation `strategy.close("E")`; size 1 | `l2_fill_flatten.pine` header ("price-less entry → marketable LO at the band edge"; no `strategy.exit`); registry ✅ 08-12; `logs/l2_evidence.txt` |
| Live-L2-BracketFill | L2 | **L2** (stays) | SL −0.2 % / TP +0.2 % fixed off the entry TRIGGER and pre-placed on the entry bar; the stop-entry fill is capped by the stop child's LO price; `pyramiding=0`, `default_qty_value=1`, 2-bar cap, −0.3 % double-check | `l2b_fill_protect_flatten.pine` lines 24–41, 58–59, 77–87; README venue fact "PRE-PLACED … arms on the SAME bar" (l2b live 2026-09-15, #121); `test_stop_fill_price.py` (README offline table) |
| Live-L3-F01-LongMarket / F02-ShortMarket | L3 | **L3** | Market fill at the band-edge LO; protection `P` rests at `low[1]` / `high[1]` set at PLACEMENT, so its distance from the fill is the prior bar's range plus slippage — not bounded before the run; size 1 | `live_staged_fill.pine` lines 98–104, 156; registry F01 ✅ 09-07, F02 PARTIAL 09-07; `logs/fill_ladder_evidence.txt` |
| Live-L3-F03-LongStop / F04-ShortStop | L3 | **L3** | Stop entry at `high[1]`/`low[1]`, protection at `low[1]`/`high[1]`: distance = prior bar range, uncertain; the fill arrives on a NORMAL child after `Activated` (#39) | `live_staged_fill.pine` lines 105–111; registry ✅ 09-07; `logs/fill_ladder2_evidence.txt` |
| Live-L3-F05-LongStopLimit / F06-ShortStopLimit | L3 | **L3** | Same protection shape; the exit is withheld until the fill (#82b) so it arms after the entry — distance uncertain; size 1 | `live_staged_fill.pine` lines 112–124; registry F05 09-08 (#82 fixed, #83 parks F06); `logs/fill_regrade2_evidence.txt` |
| Live-L3-F07-OcaLongBreak / F08-OcaShortBreak | L3 | **L3** | Two entries in one OCA group (near + far, opposite sides); protection at a prior-bar level; if the cascade fails the far leg is a live opposite-direction entry (a flip) — uncertain protection and a two-order book | `live_staged_fill.pine` lines 125–141; registry ❌ NOT RUN |
| F9-BracketTrailingSL (planned, `live_staged_fill` state 8) | L3 | **L3** | Trailing SL re-issued on wait bars 1–2 (`f9SL + waitBars * close * 0.001`); trailing is L3 by definition | `live_staged_fill.pine` lines 142–157, 197–206; README planned list (#93) |
| Live-L3-F10-CrossedStopAtPlacement | L3 | **L3** | Crossed-at-placement entry; no `strategy.exit` in the vehicle (only `close_all`); L3 by definition | `live_crossed_stop.pine` (only line 55 entry, line 74 `close_all`); registry ✅ 08-18; `logs/t15_t16_t17_evidence_20260818.txt` |
| Live-L3-F11-OcaEntryGroupCancel | L3 | **L3** | Protection `P` at `close * 0.997` set at placement while the entry is a stop at `high[1]` — close-derived brackets measured inverted on 50.7 % of bars (l2b header); the FAR leg is deliberately left working (#42-B) | `live_oca_entry_group.pine` lines 83–91; registry ✅ 08-19; `l2b_fill_protect_flatten.pine` lines 25–29 |
| F12-PruneAdoptionRace (planned) | L3 | **L3** | A cancel racing the trigger (F3/F4 shape) — race, L3 by definition | README planned list (#95) |
| F10-FcRejectCapture (planned) | L3 | **UNSURE** (id collision, not moved) | Crossed both-set near-market entry is L3-shaped, but the id "F10" is already `Live-L3-F10-CrossedStopAtPlacement`; no vehicle file exists. Decides: assign a free F-number and write the vehicle | README planned list (#87 live item) |
| F13 (`run_f13_latency.sh`, #146; no registry row) | — | **UNSURE** (not moved) | The `l2b` vehicle is L2-shaped (BracketFill reasoning); the `l2` fallback is L3-shaped (SingleFill reasoning). Decides: the vehicle named in the runner's grading table for the run that is registered | #146 body ("`--vehicle l2b\|l2`, default l2b … `--fallback` on") |
| `l3_fill_fixed_sl_tp` (vehicle, no registry row) | L3 (file header) | **L3** | SL = `avg_fill × (1 − 0.20 %)` is a known distance, but it is derived from `strategy.position_avg_price` AFTER the fill — a reactive exit, which the README venue fact measures as arming ONE BAR AFTER the fill; "armed a bar late" is L3 by definition | `l3_fill_fixed_sl_tp.pine` lines 19–29, 69, 96; README venue fact (l2b_entry_update 2026-09-15) |
| `l2b_entry_update` (vehicle, no registry row) | — (named "2b") | **L3-shaped** | Exit gated on `strategy.position_size > 0` (line 45) — reactive, arms a bar late; the name "2b" should not be read as L2 | `l2b_entry_update.pine` lines 42–49; README venue fact |
| Gap report T44 TwoReadPositionDiscipline | rider | **rides its host** | No orders of its own | `docs/plan/l1-suite-gap-review-2026-09-18.md` (a) |
| Gap report T45 UmbrellaVisibilityGate | rider on l2b | **L2 rider** | Rides BracketFill; its only write is a child cancel immediately followed by the flatten | gap review (b) |
| Gap report T49 OpposingRunsNetZero | proposed L3 | **L3** | Two engines; brackets ±1 % derived from close on market fills (uncertain distance); two contracts on the account | gap review (c) |
| Gap report T55 "F14" StockPartialFill | proposed L3 | **L3** | 300 shares, no stop, partial slices | gap review (e) |
| Gap report T60 "F15" AppFlattenWithBracket | proposed L3 | **L2** (rename to `Live-L2-F15-…` before registering; not yet an id) | l2b vehicle, pre-placed bracket, size 1; the external flatten does not change the stop distance | gap review (g); `l2b_fill_protect_flatten.pine` |

Not fill cases, unchanged: Live-L4-WsDual, Live-L4-AtcWatchdogBoundary (passive), the B3
expiry-week set (operator-gated, order-level).

## 2. README draft — applied text

### 2a. Inserted after the "FILL-tier cases are parked, not skipped" paragraph (tier section)

```
### Three groups, five levels (operator decision 2026-09-18)

Read the levels as three groups. Nothing merges; L0, L1 and L4 are unchanged.

| Group | Levels | Places orders | Takes a position |
|---|---|---|---|
| No-fill ORDER tests | L0 (venue-semantics gate), L1 (staged no-fill probes + direct probes) | yes, all ≥4.5 % away or cancelled before they can fill | never |
| No-fill DATA tests | L4 (bar parity, latency, ATC delivery, tick delivery) | no | never |
| FILL tests | L2, L3 | yes | yes — FILL tier preconditions above |

**The L2 / L3 split is by RISK CONTROL, not by history.**
- **L2** = a fill case with very tight risk control: the protective stop's distance from the
  REAL fill price is known and bounded BEFORE the run (a bracket pre-placed on the entry bar,
  its level fixed relative to the entry trigger or the LO cap — see `l2b_fill_protect_flatten.pine`
  header), AND the position size is exactly 1 (`default_qty_value=1`, `pyramiding=0`, an
  enter-once latch).
- **L3** = a fill case where the stop's distance from the real filled price is uncertain
  (no stop at all; a stop set from a prior bar's low/high or from `close` rather than the fill;
  a reactive exit placed only after `position_size > 0`, which arms a bar late — README venue
  fact below; a trailing stop; a crossed-at-placement entry; a prune/adoption race)
  AND/OR the position size can reach 2 or more (frozen flip quantity, pyramiding, dual-set
  entries, two engines).

Ids are never renamed: a row that changes level keeps its `Live-L<n>-…` id and carries the
note "retiered from L<old> 2026-09-18". Under this criterion the only L2 vehicle today is
`l2b_fill_protect_flatten` (row `Live-L2-BracketFill`); `Live-L2-SingleFill` is retiered to L3
(no protective stop at all — see its row). The move table with per-case reasons is
`docs/plan/test-levels-retier-2026-09-18.md`.
```

### 2b. Registry rows (BracketFill moved above SingleFill; ids kept)

```
| **Live-L2-BracketFill** | fill + TP/SL bracket → flatten (`l2b_fill_protect_flatten`): stop entry at `higherHigh2`, native OCO bracket TP +0.2 % / SL −0.2 % off the TRIGGER, pre-placed on the entry bar, 1 lot, `pyramiding=0`, 2-bar cap. **L2 under the 2026-09-18 criterion**: stop distance fixed before the run relative to the trigger (the fill is capped by the stop child's LO price — `test_stop_fill_price.py`), size exactly 1 | ⚠️ 08-12 partial (`logs/l2b*_evidence.txt`); README venue fact: l2b live 2026-09-15 armed same bar (#121) — no tracked log, row not upgraded |
| **Live-L2-SingleFill** | one market fill → flatten (`l2_fill_flatten`): band-edge marketable LO, NO protective exit, `strategy.close("E")` on the next evaluation. **Retiered from L2 to L3 2026-09-18** (id kept): no stop at all, so the stop-distance condition is not met — exposure is bounded only by the ±7 % band and the next-bar close | ✅ 08-12 (`logs/l2_evidence.txt`) |
```

### 2c. Planned fill-tier additions — appended notes

```
- **F12-PruneAdoptionRace** … L3 under the 2026-09-18 criterion (a race; the protective level comes from a prior bar).
- **Retier notes (2026-09-18, no ids renamed):** F9 (trailing SL), F10-FcRejectCapture (crossed near-market both-set) and F12 are L3 by definition. **UNSURE, not moved:** **F13** (`run_f13_latency.sh`, #146 — no registry row yet) is L2-shaped when the `l2b` vehicle runs and L3-shaped on the `l2` fallback; decide per run from the vehicle named in the runner's grading table. **Id collision:** "F10-FcRejectCapture" above reuses the id of `Live-L3-F10-CrossedStopAtPlacement` in the registry table; it needs its own number before it is registered. Every F01–F11 row below stays L3: their protection exit `P` rests at `low[1]`/`high[1]` (`live_staged_fill.pine` line 98) or at `close * 0.997` (`live_oca_entry_group.pine` line 91), never at a fill-derived distance, and F10 has no protective exit at all.
```

## 3. CLAUDE.md draft — applied text (inserted in "DNSE testing", before "Live-run session mechanics")

```
### The five test levels (operator decision 2026-09-18; the README above is the source)

| Level | What it is | Where it runs | Tier | Precondition | Purpose |
|---|---|---|---|---|---|
| L0 | venue-semantics gate (`level0_venue_semantics/l0_order_semantics.py`) | direct client, no engine | no-fill | GOOD token; any hour except ATC (pre-open needs `--allow-conditionals-when-closed`) | proves auth, both books, place/rest/cancel today |
| L1 | staged no-fill probes T01–T33 + direct probes + runners | `pyne run … --broker` or direct | no-fill | open session + GOOD token; safe with an open user position | engine→plugin→venue order paths, ≥4.5 % away |
| L4 | data parity / latency / ATC / tick delivery (`level4_data_parity/`) | passive recorder | no-fill, no orders | no token | bar feed correctness |
| L2 | fill with TIGHT risk control (`l2b_fill_protect_flatten`) | `--broker`, FLAT account, supervised | FILL | flat account or sub-account, L0 + no-fill smoke green today | the protected fill→bracket→flatten chain |
| L3 | fill with UNCERTAIN stop distance and/or size ≥2 (`live_staged_fill` F01–F12, F10/F11 vehicles) | `--broker`, FLAT account, supervised | FILL | as L2, plus operator-in-the-loop close protocol | order-type fill semantics, cascades, races |

L2/L3 split criterion: L2 means the protective stop's distance from the REAL fill price is
known and bounded before the run AND the position size is exactly 1. L3 means the stop's
distance from the real fill is uncertain (reactive exits, trailing stops, brackets armed a
bar late, crossed-at-placement, prune/adoption races) and/or the size can reach 2 or more.
Three groups: no-fill ORDER tests (L0, L1), no-fill DATA tests (L4), FILL tests (L2, L3).
Mandatory run order every session: L0 (exit 0) → no-fill (L1 smoke T01–T03, L4) → fill LAST.
```

## 4. Sources read for this task

`plugins/dnse/testing/live_test/`: `l2_fill_flatten.pine/.toml`, `l2b_fill_protect_flatten.pine/.toml`,
`l2b_entry_update.pine`, `l3_fill_fixed_sl_tp.pine/.toml`, `live_staged_fill.pine/.toml`,
`live_crossed_stop.pine/.toml`, `live_oca_entry_group.pine/.toml`, `README.md` (tier table,
registry, planned additions, venue facts); `CLAUDE.md` DNSE testing section; cards #39, #82,
#87, #93, #95, #121, #146 (bodies read in the gap review); `docs/plan/l1-suite-gap-review-2026-09-18.md`.

# Evaluation of the 0.36 % fixed-stop proposal (2026-09-18)

Proposal (operator, verbatim): "If we place only 1 contract position size and add a fixed
stop-loss at 0.36%, then all the needed test cases can move from L3/L4 to L2." Evaluated per
case. Read-only; worktree at 773d8a2b. Line numbers refer to the vehicle files under
`plugins/dnse/testing/live_test/`.

## Part 1 — L4 places no orders; the proposal does not apply

`level4_data_parity/l4_bar_parity.py` header (lines 1–2): "PASSIVE: no orders". A grep of that
file for `_place`, `place_order`, `strategy.` and `cancel` returns nothing; it instruments
`client.get_ohlc` (line ~55) and records `watch_ohlcv` output. Registry rows Live-L4-T01, T02,
T03 (08-17) describe a recorder; the planned rows Live-L4-T04-TickDelivery and
Live-L4-T05-SynthParity describe delivery grading and a recorder; the README states "Live-L4 is
PASSIVE (no orders, no trading token)". No L4 row places orders, so none is misfiled and none
can "move to L2": L4 is already no-fill. The gap-report proposals I filed as L4 (T56, T57, T58)
are passive as well.

## Part 2 — L3 cases, one by one

Reading of the proposal's re-authoring: `default_qty_value=1`, `pyramiding=0`, an enter-once
latch, no opposite-direction entry, and a `strategy.exit` PRE-PLACED on the entry bar with
`stop = trigger × (1 ∓ 0.0036)`.

**"From the trigger" versus "from the fill".** For a stop entry the venue fills the NORMAL
child at or through the trigger, capped by the child's LO price (README offline table,
`test_stop_fill_price.py`, "stop→LO pricing (slippage floor)"). A stop fixed from the TRIGGER
is therefore known before the run and its distance from the real fill is 0.36 % minus the
fill-through amount, which is bounded by that LO cap. A stop fixed from the FILL can only be
computed after the fill (`strategy.position_avg_price`, `l3_fill_fixed_sl_tp.pine` lines
92–95), so it is a reactive exit and arms one bar late (README venue fact, l2b_entry_update
2026-09-15). The l2b header measured the difference: trigger-derived SL exactly 0.200 %,
close-derived SL 0.403 % median and inverted on 50.7 % of bars
(`l2b_fill_protect_flatten.pine` lines 24–29; the bracket itself lines 32–35, levels set at
lines 75–76 and pre-placed at lines 77–82). So the operator's 0.36 % must mean FROM THE
TRIGGER for a stop entry; for a MARKET entry there is no trigger, the pre-placed level can
only be close-derived, and the distance from the real fill is uncertain by construction.
That single fact decides most rows below.

| Case | (a) measures (registry) | (b) survives re-author? | (c) lost | (d) verdict |
|---|---|---|---|---|
| Live-L2-SingleFill (retiered) | "one market fill → flatten" | Partly: a market entry has no trigger; a pre-placed SL is close-derived (uncertain from the fill) | the no-exit fill→flatten chain (`l2_fill_flatten.pine` lines 37–60: entry, then `cancel_all` before `close("E")`) | MOVABLE WITH LOSS — only if re-authored as a stop entry (then it duplicates BracketFill); no L1 carrier (needs a fill) |
| Live-L3-F01 / F02 market | "market fills" | No: market entry, protection at `low[1]`/`high[1]` (`live_staged_fill.pine` line 98, 156); a 0.36 % SL would be close-derived | the market-fill path itself (band-edge LO, `live_staged_fill.pine` line 101) | STAYS L3 — the market fill IS the uncertainty |
| Live-L3-F03 / F04 stop | "stop-entry fills … conditional → Activated → NORMAL child" | Yes: stop entries at `high[1]`/`low[1]` (lines 106, 110); replace the `low[1]` protection (line 156) with `stop = trigger × 0.9964` / `× 1.0036` pre-placed | nothing; #39 attribution survives (the SL is a separate conditional, unrelated to the child) | MOVABLE — change: protLvl := trigger-relative 0.36 % for states 2–3 |
| Live-L3-F05 / F06 stop-limit | "stop-limit fills (#14 evidence)" | Yes: entry limit at `high[1] × 1.0002` (line 113) bounds the fill even tighter; SL trigger-relative; #82b withholds the exit until the fill and #121 arms it same bar (README venue fact) | nothing; the one-conditional-stop-limit check survives | MOVABLE — same change for states 4–5 |
| Live-L3-F07 / F08 OCA | "OCA sibling-cancel on a real fill" | No: needs two opposite entries in one group (lines 126–129, 135–138); "no flip" removes the subject | the fill-time cascade (#33) | STAYS L3 — no L1 can carry a fill cascade (T11 08-17 covers member CANCEL, not fill) |
| F9-BracketTrailingSL (planned, state 8) | "#93 post-fix grade … trail the SL two bars" | No: the trail (lines 197–206) is the measurement | the #93 loud-park on a live bracket modify | STAYS L3 — exits need a position (#82b), so no L1 carrier |
| Live-L3-F10-CrossedStopAtPlacement | "buy-stop trigger already BELOW market → must fill immediately" | No: the trigger is `close × (1 − crossPct)` (line 54), BELOW the fill by ~1 %; a trigger-relative SL is 1.36 % from the fill; a fill-relative one is reactive | the #34 crossed-detection path | STAYS L3 |
| Live-L3-F11-OcaEntryGroupCancel | "oca.cancel ENTRY pair: near fills → far leg must be CANCELLED" | No: NEAR long + FAR short (lines 83–86), FAR deliberately left working (#42-B, lines 102–104) | the engine cascade under a real fill | STAYS L3 |
| F12-PruneAdoptionRace (planned) | "conditional entry cancel racing the trigger" | No vehicle exists (UNVERIFIED); the race is the subject | the #95 adoption join | STAYS L3 |
| `l3_fill_fixed_sl_tp` (vehicle) | header: "measure fill → real-STOP placement latency … #118 GTD on the SL's record" | Partly: market entry (line 85); SL from `position_avg_price` after the fill (lines 92–96) is the point | the reactive-arm latency measurement (#107/#121) | MOVABLE WITH LOSS — pre-placing a close-derived SL turns it into a worse F01; no L1 carrier |
| `l2b_entry_update` (vehicle) | header: "conditional entry cancel+replace (#85) moves the stopPrice each candle" | Yes: chase (lines 41–42) keeps the trigger; pre-place `exit` with `stop = entryStop × 0.9964` each re-issue instead of the reactive bracket (lines 45–51) | nothing for the #85 measurement; its no-fill variant (padPct=5) is already L1 T19 (09-08) | MOVABLE — change: pre-place the bracket on the entry bar |
| Gap T45 UmbrellaVisibilityGate | rider on l2b | Yes (l2b already 0.2 % trigger-relative) | nothing | already L2 rider |
| Gap T49 OpposingRunsNetZero | two engines net to zero | No: the subject is two positions on one account | the netting measurement | STAYS L3 |
| Gap T55 StockPartialFill | 300-share partial slices | No: size is 3 lots by design; no stop | partial-fill slices | STAYS L3 |
| Gap T60 AppFlattenWithBracket | operator app-flatten under a bracket | Yes (l2b vehicle) | nothing | already L2-shaped |
| F13 (`run_f13_latency.sh`, #146; UNSURE) | WS-vs-poll fill→protection latency, l2b primary, l2 fallback | Yes on the l2b vehicle (trigger-relative 0.2 % bracket, lines 75–82); the l2 fallback is a market entry with no exit | the fallback's transport-only sample | MOVABLE — drop the `l2` fallback (or grade it as an L3 sample); then the UNSURE clears to L2 |

Paragraphs.

*SingleFill.* `l2_fill_flatten.pine` enters once on the first realtime bar (lines 37–39) and
closes on the next evaluation (lines 56–60) with no `strategy.exit`. Its remaining value is
the flatten ordering lesson at lines 49–55, already pinned in the comment. Re-authored with a
pre-placed SL it becomes BracketFill with a market entry — and a market entry cannot satisfy
"bounded before the run" (see the trigger/fill paragraph). Verdict MOVABLE WITH LOSS, and the
honest reading is: retire it in favour of BracketFill rather than move it.

*F01/F02.* The row exists to prove the marketable-LO fill path (`live_staged_fill.pine` line
101 log text "expect marketable LO at the band edge"). Any pre-placed stop for a market entry is
set from `close` on the placement bar; the l2b header (lines 25–29) measured that shape at a
0.403 % median distance with inversions. The proposal cannot make that distance known. STAYS.

*F03–F06.* These are stop and stop-limit entries at `high[1]`/`low[1]` (lines 106–124). The
current protection `P` is `low[1]`/`high[1]` (line 98) — a prior-bar level, which is why they
are L3 today. Replacing that single assignment with a trigger-relative level gives exactly the
BracketFill shape; the venue-side chain the rows measure (conditional → Activated → child fill
attributed to the Pine id, #39; ONE stop-limit conditional, #14) is unaffected, because the SL
is a separate conditional placed after the fill (#82b/#121). MOVABLE.

*F07/F08, F11.* The subject is a two-entry OCA group and what happens to the far leg on the
near fill (`live_staged_fill.pine` lines 125–141; `live_oca_entry_group.pine` lines 80–97 and
the deliberate non-cancel at 102–104). "No flip" removes the far leg, which removes the
measurement. STAYS L3. T11 (08-17) shows a member CANCEL leaves siblings; only a fill shows the
cascade.

*F9, F10, F12.* Trailing (lines 197–206), crossed-at-placement (line 54), and a cancel racing a
trigger are the uncertainty itself. STAYS L3.

*`l3_fill_fixed_sl_tp`.* Its header (lines 4–11) says it exists to measure fill → real-STOP
placement latency with a fill-derived SL; the stop is reactive by design (line 92 gate on
`position_size > 0`). Pre-placing it destroys the measurement. MOVABLE WITH LOSS; keep it as
the L3 reactive-arm vehicle.

*`l2b_entry_update`.* The #85 chase (lines 41–42) is independent of the bracket; the bracket
today is reactive (lines 45–51) only because it was authored that way. Pre-placing the exit on
each re-issue is the change the README venue fact already recommends. MOVABLE.

## Part 3 — the 0.36 % figure

**Provenance of the numbers.** The 1m and 15m statistics below have TWO independent readers
(the review agent's script in this section and a separate computation in the Worker4Isolated
session over the same tracked files; they agree to the last printed digit on median, p95 and
the fraction of bars at or above 0.36 %; the 15m five-bar-window figure differs by two points,
73 % vs 75 %, from a window-boundary convention). The 5m statistics are SINGLE-SOURCED from the
agent's run only and have not been independently re-read.

Current l2b bracket: TP `higherHigh2 × 1.002`, SL `higherHigh2 × 0.998` — 0.2 % off the
trigger (`l2b_fill_protect_flatten.pine` lines 75–76; rationale lines 24–35), double-check at
−0.3 % (line 94), 2-bar cap (lines 112–114). The proposal's 0.36 % is 1.8× wider than the SL
the suite runs today and sits OUTSIDE l2b's −0.3 % double-check, so adopting it on l2b would
also require moving the double-check.

Tick size: `workdir/data/dnse_VN30F1M_1.toml` `mintick = 0.10000000`, `pricescale = 10`,
`minmove = 1`, `pointvalue = 100000`; provider code `plugins/dnse/pynecore_dnse/provider.py`
line 508 `mintick, minmove, pricescale = 0.1, 1, 10` for derivatives. At the last tracked
close (1968.1) 0.36 % = 7.09 points = 71 ticks; 0.2 % = 3.94 points. With `pointvalue =
100000` VND, the bounded loss per contract is ~709,000 VND at 0.36 % versus ~394,000 VND at
0.2 %.

**Data caveat.** The tracked 1m file holds only 3 sessions (699 bars, 2026-09-15..17): it is
a provider-mode warmup snapshot (CLAUDE.md `--from`/#17: the run rewrites the shared
`.ohlcv`). "Last 20 sessions at 1m" is therefore not available in the tree. I ran the same
computation on the tracked 5m file (2,007 sessions, last 20 = 2026-08-04..09-03) and the 15m
file (17 sessions), and report all three. A 5m bar approximates a 5-bar 1m window.

Command (read-only, worktree venv; script text is in the scratchpad and reproduced here):

```
cd /home/mike/workspace/github/pynecore-worker2
for tf in 1 5 15; do .venv/bin/python range_stats_036.py workdir/data/dnse_VN30F1M_$tf.ohlcv; done
```

```python
# range_stats_036.py — last 20 ICT sessions of the given .ohlcv via pynecore.core.ohlcv.OHLCVReader
# per bar: (high-low)/close %, median / p95 / share >= 0.36 %; bar-one adverse move from the bar's
# own open (open->low for a long, open->high for a short) >= 0.36 %; non-overlapping 5-bar windows
# whose excursion from the window open reaches 0.36 % either way; 0.36 % in points at the last close.
```

Output:

```
=== dnse_VN30F1M_1.ohlcv
file bars=699 sessions=3 first=2026-09-15 last=2026-09-17
per-bar range %: median=0.0609 p95=0.1372 max=0.4131
fraction of bars with range >= 0.36%: 0.1431%
bar-one stop hit (open->low >= 0.36%, long): 0.0000%
bar-one stop hit (open->high >= 0.36%, short): 0.1431%
5-bar windows: n=139 excursion>=0.36% either way=2.1583% (down=0.7194%, up=1.4388%)
last close=1968.1  0.36% = 7.09 points = 70.9 ticks of 0.1
=== dnse_VN30F1M_5.ohlcv
file bars=101594 sessions=2007 first=2018-08-13 last=2026-09-03
last-20 sessions: 2026-08-04 .. 2026-09-03  bars=980
per-bar range %: median=0.1770 p95=0.4035 max=1.0839
fraction of bars with range >= 0.36%: 8.6735%
bar-one stop hit (open->low >= 0.36%, long): 1.7347%
bar-one stop hit (open->high >= 0.36%, short): 2.9592%
5-bar windows: n=180 excursion>=0.36% either way=39.4444% (down=19.4444%, up=21.6667%)
=== dnse_VN30F1M_15.ohlcv
file bars=275 sessions=17 first=2026-08-14 last=2026-09-10
per-bar range %: median=0.2953 p95=0.6540 max=1.0944
fraction of bars with range >= 0.36%: 35.6364%
bar-one stop hit (open->low >= 0.36%, long): 10.1818%
bar-one stop hit (open->high >= 0.36%, short): 10.9091%
5-bar windows: n=48 excursion>=0.36% either way=72.9167% (down=37.5000%, up=43.7500%)
```

Reading. At 1m a 0.36 % stop is almost never hit on the fill bar (0–0.14 %) and is reached
within five 1m bars in about 2 % of windows (3 sessions only). At 5m — the timeframe #146
fixes for F13 ("Timeframe stays 5m") — the fill bar itself hits it in 1.7 % (long) to 3.0 %
(short) of bars, and a 25-minute window reaches it in 39 % of cases. At 15m the fill bar hits
it in ~10 % of bars. So a 0.36 % stop is a rare bar-one event at 1m, a several-percent event
at 5m, and a one-in-ten event at 15m; the current 0.2 % is hit correspondingly more often
(l2b header lines 36–38: over 275 15m bars, 26 of 39 episodes closed by the bracket inside one
bar).

Consequence. Whenever the stop (or the TP, which sits at the same distance in l2b) fires
before the flatten, the case becomes a TWO-FILL case: the venue record carries an entry fill
and an exit fill the case did not set out to measure; the flat-by arithmetic changes (two
fills to book, the bracket's other leg to confirm cancelled — the OCO child, CLAUDE.md
`CO-ORD-013` — and the question whether protection re-arms after the exit fill); and the
grader must separate "flattened by protocol" from "stopped out". This is not a reason to
reject 0.36 % — it is the same event l2b already accepts at 0.2 % — but it means the L2
grading rule must name both outcomes, and the flat-by time must include the exit fill's
confirmation.

## Part 4 — proposed lists and docs edits under the proposal

New L2 list (after re-authoring as named in Part 2):
Live-L2-BracketFill (unchanged); Live-L3-F03-LongStop and F04-ShortStop (trigger-relative
protection); Live-L3-F05-LongStopLimit and F06-ShortStopLimit (same); `l2b_entry_update`
(pre-placed bracket per re-issue; register with an id); F13 on the l2b vehicle only; T45 and
T60 as L2 riders/cases. Ids of the F-rows stay `Live-L3-…` with the note "retiered from L3
2026-09-18 (trigger-relative protection)".

Residual L3 list: Live-L2-SingleFill (or retire it), F01, F02, F07, F08, F9, F10, F11, F12,
`l3_fill_fixed_sl_tp`, T49, T55.

Docs edits needed if the operator adopts the proposal:
- README registry rows F03–F06: append "retiered from L3 2026-09-18 (trigger-relative
  protection)" and name the changed line (`live_staged_fill.pine` line 98 becomes a
  per-state trigger-relative level for states 2–5). Row `Live-L2-SingleFill`: either keep
  the L3 note or mark it retired in favour of BracketFill. Planned list: F13 loses its UNSURE
  once the `l2` fallback is dropped.
- Move table (section 1 above): F03–F06 → L2 with the same reason; F13 → L2; `l2b_entry_update`
  → L2 after re-authoring.
- CLAUDE.md criterion line: **do not write 0.36 % into the criterion.** The Part 3 numbers
  show the hit rate depends on the timeframe (0.14 % at 1m, ~2–3 % at 5m, ~10 % at 15m per fill
  bar), so a single percentage is not a level definition; the level is defined structurally
  (pre-placed, trigger-relative, size exactly 1). State the per-vehicle distance in the vehicle
  header and the registry row, as l2b does today at 0.2 %. If the operator still wants a
  suite-wide number, it must also move l2b's −0.3 % double-check outward and be re-measured
  on l2b's 275-bar sample (header lines 24–29) before being written anywhere.

Sources read for this part: the vehicle files named above with the line numbers cited;
`level4_data_parity/l4_bar_parity.py`; `workdir/data/dnse_VN30F1M_{1,5,15}.ohlcv` and
`dnse_VN30F1M_1.toml`; `plugins/dnse/pynecore_dnse/provider.py` lines 506–522;
`src/pynecore/core/ohlcv.py` (`OHLCVReader`); README registry rows L2/L3/L4 and the planned
fill-tier list; card #146.
