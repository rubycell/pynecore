# PyneCore Verification Plan

## Purpose

Before making optimization or runtime changes to pynecore, verify against ground truth baselines to ensure no regressions.

## Quick Start

```bash
cd pynecore && source .venv/bin/activate
./verify_ground_truth.sh
```

## What Gets Verified

### 1. Sample Test Suites

| Suite | Script | Expected |
|-------|--------|----------|
| Samples | `test_all_samples.py` | 324/324 transpile+runtime, 88 skipped |
| PDS Samples | `test_all_sample_pds.py` | 236/236 transpile, 231/236 runtime, 94 skipped |

**Checks**: Pass/fail counts and failed file lists must match ground truth exactly.

### 2. Strategy Runs

| Strategy | Data Files | Output Files |
|----------|-----------|--------------|
| `bigtest.py` | bybit, demo, VN30F1M_15m_3regimes | .csv, _strat.csv, _trade.csv, _plot_meta.json |
| `bigtest2.py` | bybit, demo, VN30F1M_15m_3regimes | .csv, _strat.csv, _trade.csv, _plot_meta.json |

**Checks** (in order of criticality):
1. `_trade.csv` — Trade entries/exits must match exactly (dates, prices, quantities, profits)
2. `_strat.csv` — Strategy statistics must match exactly (P&L, drawdown, win rate)
3. `.csv` — Plot data should match (indicator values)
4. `_plot_meta.json` — Metadata should match

## Verification Modes

```bash
./verify_ground_truth.sh              # Full (both suites + strategies)
./verify_ground_truth.sh --quick      # Strategies only (~2 min)
./verify_ground_truth.sh --samples    # Sample suites only (~30 sec)
./verify_ground_truth.sh --strategies # Same as --quick
```

## Workflow: Making Runtime Changes

1. **Before changes**: Run `./verify_ground_truth.sh` — confirm all PASS
2. **Make changes** to runtime/transpiler code
3. **After changes**: Run `./verify_ground_truth.sh` — any FAIL = regression
4. **If FAIL**: Revert and investigate
5. **If intentional behavior change**: Update ground truth (see `ground_truth/README.md`)

## Ground Truth Location

```
pynecore/ground_truth/
├── README.md
├── samples/
│   ├── test_results.json
│   └── output/          (1150 files)
├── samples_pds/
│   ├── test_results_pds.json
│   └── output/          (859 files)
└── strategies/
    ├── bigtest/{bybit,demo,VN30F1M_15m_3regimes}/
    └── bigtest2/{bybit,demo,VN30F1M_15m_3regimes}/
```

## Known Limitations

- **Sample.py** and **comprehensive_feature_test.py** have pre-existing bugs and are excluded
- 5 PDS samples fail due to missing API implementations (known, recorded)
- The verification script does exact byte-level diffs — floating point changes will fail
- The `workdir/output/` directory is shared; strategy runs overwrite previous outputs
