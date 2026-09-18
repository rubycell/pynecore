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
