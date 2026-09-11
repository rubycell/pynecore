# flip / exit probes — evidence for the "NOT a bug" claims

These back the CLAUDE.md section **"Pine behaviours that LOOK like bugs and are NOT"**
and the characterization test
`tests/t01_lib/t30_strategy/test_130_flip_does_not_cancel_an_armed_exit.py`.

They are kept because that CLAUDE.md section tells future sessions **not** to
investigate — a claim of that shape suppresses re-testing, so it needs a stronger
evidence anchor than average, and it must stay re-runnable across upstream rebases.

Run any of them with the local transpiler, then `pyne run` (never `pyne compile`):

```bash
cd ../../../                                   # repo root
cd /home/mike/workspace/github/pine2pyne && .venv/bin/python -m pine2pyne \
    <abs path>/<probe>.pine -o <abs path>/<probe>.py
.venv/bin/pyne run <probe>.py dnse_VN30F1M_15
```

| Probe | Question | Measured 2026-09-10 |
|---|---|---|
| `flip_exit_probe.pine` | does a flip cancel the old side's armed exit? | **No** — short -1 goes to **+2** in one bar |
| `flip_exit_entry_first.pine` | **control**: is the +2 caused by the ORPHANED EXIT opening a position? | **No — refuted.** Swap the two levels so the ENTRY fires first and the exit is the orphan: position ends at **+1** and the orphan opens nothing. The +2 comes from the entry's qty-2 flip, frozen at placement (#105) |
| `tv_flip_exit_probe.pine` | what does **TradingView** do with the same setup? | **Identical** (-1 -> +2), VN301! 15m. Anchored on `last_bar_index` so it runs on any symbol/timeframe; `overlay=false` + `bgcolor` so the violation is visible on the chart |
| `pyramiding_market_entries.pine` | is `pyramiding` enforced for market entries? | **Yes** — 4 entries, distinct ids, position stays 1 |
| `pyramiding_stop_entries.pine` | …and for resting stop entries? | **Yes** — position stays 1 |
| `reversal_sizing.pine` | does a reversal mis-size? | **No** — short 1 then long qty=1 -> **+1** |
| `flip_idiom_no_exit.pine` | is the plain two-entry flip idiom (no `strategy.exit`) safe? | **Yes** — 50 trades, `Max contracts held = 1`, zero violations, including bars where both conditions fire |

**Mechanism (corrected 2026-09-11, #105).** The +2 is real and TradingView-matching, but
the first explanation recorded here — an orphaned exit firing as an opening order — is
REFUTED by `flip_exit_entry_first.pine`. What actually happens: a price-based entry
freezes its reversal augmentation at PLACEMENT, so a long entry stop placed while short 1
carries qty **2**; when a protective exit closes the short first, that frozen 2 opens from
flat. The outcome is therefore ORDER-DEPENDENT — exit level first gives 2, entry level
first gives 1.

`strategy.risk.max_position_size` bounds this in BACKTEST (cap applied at fill time) but
not on the LIVE path (cap gates the order at submit time only) — see #105 before relying
on it as a ceiling.

The TradingView run used the dedicated visual-test layout; see the
`tradingview-visual-test` skill in the pinescript repo for that procedure.

Note: `pyramiding_*` and `reversal_sizing` need a large `initial_capital` — at the
default, VN30F1M contracts (~196M VND each) are unaffordable and the entries simply
never fill, which reads as a false "position stayed at 0".
