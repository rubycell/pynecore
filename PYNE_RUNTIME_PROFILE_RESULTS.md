# PyneCore Runtime Profile Results

**Date:** 2026-02-23
**Script:** Superuptrend Long Strategy.py
**Data:** VN30F1M_5m.ohlcv (523,078 bars total, 63,349 after session filter)
**Actual CLI runtime:** 1m45s (105s)
**Profiled subset:** 10,001 bars in 4.69s (0.469 ms/bar)
**Extrapolated:** 63,349 bars × 0.469 ms = ~30s pure computation + overhead

---

## Raw Profile Data (sorted by tottime)

```
13,871,203 function calls in 4.659 seconds

ncalls    tottime  cumtime  filename:lineno(function)
--------- -------  -------  -----------------------------------------
45550     0.294    0.435    inspect.py:3148(_bind)
197070    0.282    0.520    series.py:179(__getitem__)
328169    0.256    0.335    function_isolation.py:21(isolate_function)
1230      0.209    4.606    SuperuptrendLongStrategy.py:54(main)
2161803   0.163    0.165    {isinstance}
490859    0.143    0.209    typing.py:426(inner)
49220     0.135    1.595    overload.py:132(dispatcher)
45530     0.130    0.179    inspect.py:2962(apply_defaults)
273360    0.092    0.164    overload.py:147(<genexpr>)
4920      0.080    0.105    rich/text.py:593(highlight_regex)
27060     0.070    0.244    ta.py:763(lowest)
77490     0.055    0.095    ta.py:507(ema)
18450     0.054    0.275    ta.py:566(highest)
740249    0.054    0.056    {len}
56256     0.054    0.077    safe_convert.py:4(safe_div)
23370     0.053    0.141    rich/text.py:719(render)
14760     0.051    1.375    SuperuptrendLongStrategy.py:245(getSMI)
140220    0.050    0.074    rich/cells.py:98(cell_len)
526084    0.050    0.050    {list.append}
40590     0.046    0.086    plot.py:42(plot)
94970     0.027    0.027    series.py:128(add)
198005    0.021    0.021    na.py:18(__new__)
27455     0.017    0.018    color.py:71(t)
27455     0.016    0.038    color.py:67(new)
```

## Raw Profile Data (sorted by cumtime)

```
ncalls    tottime  cumtime  filename:lineno(function)
--------- -------  -------  -----------------------------------------
1         0.001    4.686    script_runner.py:532(run)
1231      0.014    4.685    script_runner.py:242(run_iter)
1230      0.209    4.606    SuperuptrendLongStrategy.py:54(main)
49220     0.135    1.595    overload.py:132(dispatcher)
1230      0.002    1.517    log.py:169(info)
1230      0.002    1.515    logging/__init__.py:1510(info)
1230      0.005    1.481    rich/logging.py:132(emit)
14760     0.051    1.375    SuperuptrendLongStrategy.py:245(getSMI)
1230      0.009    1.206    rich/console.py:1648(print)
78720     0.044    1.038    rich/console.py:1300(render)
25830     0.010    1.023    rich/table.py:475(__rich_console__)
25830     0.039    0.812    rich/table.py:755(_render)
25830     0.044    0.657    rich/segment.py:309(split_and_crop_lines)
9840      0.024    0.642    rich/console.py:1351(render_lines)
29520     0.016    0.550    rich/padding.py:79(__rich_console__)
197070    0.282    0.520    series.py:179(__getitem__)
45550     0.019    0.453    inspect.py:3290(bind)
45550     0.294    0.435    inspect.py:3148(_bind)
23370     0.022    0.427    rich/text.py:689(__rich_console__)
328169    0.256    0.335    function_isolation.py:21(isolate_function)
18450     0.054    0.275    ta.py:566(highest)
27060     0.070    0.244    ta.py:763(lowest)
49220     0.039    0.211    {all}
490859    0.143    0.209    typing.py:426(inner)
4920      0.027    0.204    rich/text.py:1201(wrap)
```

---

## Bottleneck Analysis

### 1. Rich Log Rendering — 32% of runtime (1.49s / 4.66s)

The strategy has `log.info()` on every bar. Rich library renders a full table with
text wrapping, regex highlighting, and ANSI formatting for each log call.

**Call chain:** `log.info()` → `logging.info()` → `rich.logging.emit()` →
`rich.console.print()` → `rich.table._render()` → text wrapping, segment splitting

**Key numbers:**
- 1,230 log calls → 25,830 table renders → 78,720 console renders
- `rich/text.py` alone: highlight_regex (0.08s) + render (0.05s) + wrap (0.03s)

### 2. Overload Dispatch — 25% of runtime (cumtime 1.60s)

`overload.py:dispatcher` uses `inspect.Signature.bind()` on every ta.* function call
to match arguments to the correct overloaded implementation.

**Call chain:** `overload.dispatcher()` → `inspect.bind()` → `inspect._bind()` →
`inspect.apply_defaults()` → `typing.inner()` for Union type checks

**Key numbers:**
- 49,220 dispatch calls per 10K bars
- `inspect._bind` alone: 0.29s tottime
- `typing.inner` (Union checks): 0.14s for 490K calls
- `isinstance`: 2.16M calls, 0.16s

### 3. Function Isolation — 7% of runtime (0.34s)

328K calls to `function_isolation.isolate_function` — wraps every ta.* call to
manage per-bar state (save/restore context).

### 4. Series.__getitem__ — 6% of runtime (0.52s cumtime)

197K calls for `[N]` history access on circular buffer. Bounds checking and
modular arithmetic on each access.

### 5. ta.lowest / ta.highest — 5% of runtime (0.52s combined)

45K calls doing linear scans over the Series buffer. Fixed lookback period
means a sliding-window approach (monotonic deque) could reduce to O(1) amortized.

### 6. Plot writes — 2% of runtime (0.09s)

40K calls to `plot.plot()` (33 plots × 1230 bars). String formatting + I/O per call.

---

## Recommendations

### Quick Wins (low effort, high impact)

| # | Change | Impact | Effort |
|---|--------|--------|--------|
| 1 | **Disable Rich log handler** during `pyne run` — use plain StreamHandler or no-op | **32%** | Trivial |
| 2 | **Use `--from` flag** to limit backtest range during development | **Variable** | Zero (already exists) |
| 3 | **Remove `log.info()` calls** from strategy when not debugging | **32%** | Trivial (user-side) |

### Medium Effort (runtime code changes)

| # | Change | Impact | Effort |
|---|--------|--------|--------|
| 4 | **Cache overload dispatch** — pre-compute dispatch table keyed by `(num_args, type_tuple)`, first call resolves via inspect, subsequent calls are dict lookups | **25%** | Medium |
| 5 | **Monomorphic fast-path** for common ta.* calls — hardcode `(float, int)` signature to skip inspect entirely | **25%** | Medium |
| 6 | **Simplify function_isolation** — use `__slots__`-based save/restore instead of full dict operations | **7%** | Medium |
| 7 | **Sliding-window min/max** for ta.highest/ta.lowest — monotonic deque, O(1) amortized vs O(period) | **5%** | Medium |

### Larger Refactors (high effort)

| # | Change | Impact | Effort |
|---|--------|--------|--------|
| 8 | **C extension for Series** — circular buffer `__getitem__` in C | **6%** | Hard |
| 9 | **Batch plot CSV writes** — buffer in memory, flush periodically | **2%** | Medium |
| 10 | **NA type optimization** — 198K `NA.__new__` calls; use singleton pattern or sentinel value | **~1%** | Medium |

### Combined Impact Estimate

- Quick wins (1-3): **~32% faster** → 105s → ~71s
- + Overload cache (4-5): **~25% of remaining** → ~53s
- + Function isolation (6): **~7%** → ~50s
- + ta optimizations (7): **~5%** → ~47s
- **Total potential: 105s → ~47s (55% reduction)**

---

## Profiling Methodology

```bash
# Profiling command (from pynecore/workdir/)
source ../.venv/bin/activate
python3 -c "
import cProfile, pstats, io, time
from pathlib import Path
from pynecore.core.script_runner import ScriptRunner
from pynecore.core.ohlcv_file import OHLCVReader
from pynecore.core.syminfo import SymInfo

script_path = Path('scripts/Superuptrend Long Strategy.py')
ohlcv_path = Path('data/VN30F1M_5m.ohlcv')
syminfo = SymInfo.load_toml(Path('data/VN30F1M_5m.toml'))

with OHLCVReader(ohlcv_path) as reader:
    end_ts = reader.end_timestamp
    start_ts = end_ts - 3000000  # ~10K bars
    ohlcv_iter = reader.read_from(start_ts, end_ts)
    size = reader.get_size(start_ts, end_ts)
    runner = ScriptRunner(
        script_path=script_path,
        ohlcv_iter=ohlcv_iter,
        syminfo=syminfo,
        last_bar_index=size - 1
    )
    pr = cProfile.Profile()
    pr.enable()
    runner.run()
    pr.disable()

ps = pstats.Stats(pr)
ps.sort_stats('tottime').print_stats(60)
" > /dev/null  # redirect stdout to suppress log.info output
```

## Runtime Source Locations

- Main loop: `.venv/lib/python3.13/site-packages/pynecore/core/script_runner.py:242` (run_iter)
- Overload dispatch: `.venv/lib/python3.13/site-packages/pynecore/core/overload.py:132` (dispatcher)
- Function isolation: `.venv/lib/python3.13/site-packages/pynecore/core/function_isolation.py:21`
- Series buffer: `.venv/lib/python3.13/site-packages/pynecore/core/series.py:179` (__getitem__)
- ta functions: `.venv/lib/python3.13/site-packages/pynecore/lib/ta.py`
- Log module: `.venv/lib/python3.13/site-packages/pynecore/lib/log.py`
- Plot module: `.venv/lib/python3.13/site-packages/pynecore/lib/plot.py`
- Strategy engine: `.venv/lib/python3.13/site-packages/pynecore/lib/strategy/__init__.py`

---

## Benchmark: --from Flag Quick Win

```
Full run (all data):           1m46s (106s) — 63,349 bars after session filter
With --from 2024-12-01:        0m26s  (26s) — ~8,700 bars (1 month warmup + trade window)
Speedup:                       75% faster
```

Trade window starts 2025-01-01. Using `-f 2024-12-01` gives ~5000 bars of indicator
warmup (enough for EMA, SMI, etc.) and skips 4+ years of unnecessary bars.

**Dev command:** `pyne run "Superuptrend Long Strategy.py" VN30F1M_5m.ohlcv -f 2024-12-01`

---

## Python Acceleration Techniques Assessment

Reference: viblo.asia article "5 thủ thuật tăng tốc Python"

### Nature of the Bottleneck

The pynecore runtime bottleneck is **NOT numeric computation** — it's **Python overhead**:
- Type dispatch via `inspect.Signature.bind()` — 0.29s tottime
- Function isolation dict save/restore — 0.26s tottime
- `isinstance()` calls — 2.16M calls, 0.16s
- `typing` module Union checks — 0.14s for 490K calls
- Rich library formatting — 1.5s cumtime

This means techniques optimizing numeric loops (Numba, NumPy vectorization) have
limited impact. Techniques reducing Python interpreter overhead (PyPy, Cython) are
more promising.

### Technique 1: Numba JIT — Limited (5% potential)

**What it helps:** `ta.highest`, `ta.lowest`, `ta.ema` — numeric loops scanning Series.
**What it can't help:** inspect._bind, function_isolation, overload dispatch, Rich logging.
**Blocker:** Series uses Python objects (circular buffer of floats with NA support).
Numba requires NumPy arrays or primitive types — would need Series internals refactored
to NumPy-backed storage first.
**Verdict:** Low ROI for the effort. Fix overload dispatch first.

### Technique 2: Multi-Threading & Multiprocessing — Not for single runs

**Bar loop is strictly sequential:** Bar N depends on bar N-1 (series values, positions,
indicators, persistent variables). Cannot parallelize.
**Python GIL:** Threads don't help CPU-bound work.
**Where it DOES help:**
- `pyne optimize` — multiple parameter combos across processes (likely already implemented)
- Running multiple strategies on same data — one process per strategy
- Running same strategy on multiple data files — one process per data file
**Verdict:** Not applicable for single strategy run optimization.

### Technique 3: Cython & PyPy — Most Promising (15-50% potential)

**PyPy (30-50% potential):**
- JIT-compiles ALL Python overhead automatically
- Would speed up isinstance, dict ops, attribute access, inspect — exactly our bottlenecks
- Risk: C extension compatibility (numpy, mmap, struct). Must verify.
- Action: Test `pypy3 -m pynecore.cli run ...` to check compatibility

**Cython (15-25% potential):**
- Compile the 3 hottest files to C extensions:
  1. `overload.py` — dispatch with C-level type checking
  2. `function_isolation.py` — dict save/restore in C
  3. `series.py` — circular buffer `__getitem__` in C
- More targeted than PyPy, works with CPython ecosystem
- Risk: Maintenance burden of .pyx files

**Verdict:** PyPy is highest-ROI if compatible. Cython is the surgical alternative.

### Technique 4: Optimized Data Structures — Moderate (10% potential)

**Series circular buffer:** Currently Python list of floats. NumPy array backing would:
- Enable vectorized `ta.highest/lowest` (numpy.max over slice vs Python loop)
- Reduce memory overhead (float64 array vs list of Python float objects)
- Enable Numba JIT on ta.* functions once backing is NumPy

**NA handling:** 198K `NA.__new__` calls. Could use `float('nan')` sentinel instead
of custom NA objects to eliminate object creation overhead.

**Verdict:** Medium effort, enables other optimizations (Numba, vectorized ta.*).

### Technique 5: Profiling — Already Done

cProfile results are in this document. For deeper analysis:
- `line_profiler` on `overload.py:dispatcher` to find exact hot lines
- `py-spy` for sampling profiler (lower overhead, can profile full 63K bar run)
- `scalene` for memory + CPU combined profiling

### PYNE_OPTIMIZE_MODE: Skip Visual Operations (Quick Win)

**Status:** `PYNE_OPTIMIZE_MODE=1` is already SET by optimize command (line 221, 522)
but **never read** by the runtime. Visual functions still execute during optimize.

**Superuptrend Long Strategy has 75 visual calls per bar:**
- 33 `plot()` calls (TEMA bands, SMI levels, indicators)
- `label.new()`, `line.delete()`, `line.new()` for position tracking
- `fill()`, `bgcolor()`, `color.new()` calls

**Profile cost (per 10K bars):**
- `plot.plot()` — 40,590 calls, 0.086s cumtime
- `color.new()` — 27,455 calls, 0.038s
- `color.t()` — 27,455 calls, 0.018s
- Plus: label/line/box constructor overhead, string formatting for comments
- Estimated total: **~0.15-0.25s per 10K bars (3-5% of runtime)**

**Implementation:** When `PYNE_OPTIMIZE_MODE=1`, make these functions no-op:
- `plot()`, `plotshape()`, `plotchar()`, `plotarrow()`, `plotcandle()`, `plotbar()`
- `bgcolor()`, `barcolor()`, `fill()`
- `label.new()`, `label.set_*()`, `label.delete()`
- `line.new()`, `line.set_*()`, `line.delete()`
- `box.new()`, `box.set_*()`, `box.delete()`
- `table.new()`, `table.cell()`, `table.delete()`
- `log.info()`, `log.warning()`, `log.error()`

**Also skip in script_runner.py during optimize:**
- Plot CSV writing (already None via `plot_path=None`)
- `lib._plot_data.update(res)` — skip dict merge
- `_plot_meta` collection

**Files to modify:**
- `lib/plot.py` — early return when optimize mode
- `lib/log.py` — early return when optimize mode
- `lib/label.py`, `lib/line.py`, `lib/box.py`, `lib/table.py` — early return
- `core/script_runner.py` — skip plot data collection

**Impact for optimize:** ~5% per run × hundreds of combinations = significant total savings.
For complex strategies with many drawing objects, could be 10-15%.

### Verified Quick Wins

| # | Change | Measured Result | Status |
|---|--------|-----------------|--------|
| 1 | **Disable Rich log handler** (flip to opt-in) | 25.9s → 16.4s (**36% faster**) | DONE |
| 2 | **Use --from flag** (`-f 2024-12-01`) | 106s → 26s (**75% faster**) | Available |
| 3 | **Skip visuals in optimize mode** | Est. 5-15% per run | TODO |

### Cache Static Indicators Across Optimize Runs (Big Win)

**Problem:** During `pyne optimize`, many indicator calculations are identical across
all parameter combinations because they only depend on OHLC data, not on optimized inputs.

**Example — Superuptrend Long Strategy:**
```
getSMI(open, close, high, low, 5, 3)     # Same result in ALL optimize runs
getSMI(open, close, high, low, 13, 3)    # Same result in ALL optimize runs
... (12 SMI calls total)
getSMI(open, close, high, low, 1597, 3)  # Same result in ALL optimize runs
```
Also static: TEMA calculation, fib bands, supertrend — anything not depending on
optimized `input.*()` values.

**Profile cost:** `getSMI` alone = 1.375s cumtime per 10K bars (29% of non-log runtime).
With 500 optimize combinations: 499 × 1.375s = **687s wasted** on identical SMI computation.

**Possible implementations (from simple to complex):**

1. **User-side: split script into static + parameterized phases** (manual)
   - Compute all static indicators in a "precompute" library
   - Import cached results in the strategy. Requires script restructuring.

2. **Runtime: Series snapshot/restore between optimize runs** (medium)
   - After first run, snapshot all Series that weren't affected by input changes
   - On subsequent runs, restore snapshots instead of recomputing
   - Requires tracking which Series depend on which inputs (dependency graph)

3. **Runtime: mark functions as `@cacheable` in transpiler** (medium)
   - Transpiler detects functions whose inputs are only OHLC/constants
   - Wraps them with caching decorator that stores bar-by-bar results
   - On optimize re-run, replays cached values instead of executing

4. **Runtime: two-phase execution model** (hard, biggest payoff)
   - Phase 1: Run script once, record all Series values, mark which depend on inputs
   - Phase 2: For each optimize combo, only re-execute functions that depend on
     changed inputs, replay cached Series for everything else
   - Similar to TradingView's internal optimization — they likely do this

**Impact estimate:**
- Superuptrend: ~60-70% of per-bar computation is static indicators
- With 500 combos: total time drops from 500×16s = 8000s to 500×5s + 1×16s ≈ 2516s
- **~3x speedup for optimize runs** (strategy-dependent)

**Key insight:** The more complex the indicator logic vs the entry/exit logic,
the bigger the win. Strategies with heavy indicator computation (SMI, TEMA, etc.)
and simple parameterized entry logic benefit the most.

### Distributed Optimize Across Multiple Machines

**Setup:** 4 Linux (Ubuntu) machines on home network.
`pyne optimize` is embarrassingly parallel — each parameter combo is independent.

**Approach 1: Manual grid split (zero code changes)**

Split `optimize.json` into N subsets, one per machine. Example for 4 machines:

```bash
# machine1: optimize_part1.json — param "fast_length": {"min": 5, "max": 8, "step": 1}
# machine2: optimize_part2.json — param "fast_length": {"min": 9, "max": 12, "step": 1}
# machine3: optimize_part3.json — param "fast_length": {"min": 13, "max": 16, "step": 1}
# machine4: optimize_part4.json — param "fast_length": {"min": 17, "max": 20, "step": 1}
```

Then on each machine:
```bash
pyne optimize script.py data.ohlcv optimize_partN.json -o results_partN.csv -w 0
```

Merge results:
```bash
# On any machine, combine all CSVs (skip header from parts 2-4)
head -1 results_part1.csv > combined.csv
tail -n +2 -q results_part*.csv >> combined.csv
sort -t, -k<metric_col> -rn combined.csv > final_results.csv
```

**Approach 2: Orchestrator script (medium effort)**

A shell script that:
1. Reads optimize.json, counts total combos
2. Splits into N chunks (one per machine)
3. `scp` script + data + chunk to each machine (or assumes shared NFS/syncthing)
4. `ssh user@machineN "cd workdir && pyne optimize ... -o /tmp/results.csv"` in parallel
5. `scp` results back, merge, display top results

**Approach 3: `pyne optimize --distributed` flag (code change)**

Add to optimize command:
- `--chunk N/M` — run chunk N of M (e.g., `--chunk 1/4` on machine 1)
- Internally skips to the Nth slice of the parameter grid
- No external tooling needed, just run 4 SSH commands

Example:
```bash
ssh m1 "pyne optimize script.py data.ohlcv params.json --chunk 1/4 -o r1.csv" &
ssh m2 "pyne optimize script.py data.ohlcv params.json --chunk 2/4 -o r2.csv" &
ssh m3 "pyne optimize script.py data.ohlcv params.json --chunk 3/4 -o r3.csv" &
ssh m4 "pyne optimize script.py data.ohlcv params.json --chunk 4/4 -o r4.csv" &
wait
```

**Prerequisites:**
- pynecore installed on all machines (same version)
- OHLCV data + strategy .py file on all machines (rsync/syncthing/NFS)
- SSH key auth between machines (no password prompts)

**Impact:** Linear scaling with number of machines.
- 1 machine: 500 combos × 16s = 8000s (~2.2h)
- 4 machines: 125 combos × 16s = 2000s each (~33min wall time)
- 4 machines + all cores (e.g., 4×8 = 32 cores total): minutes

**Recommendation:** Start with Approach 1 (manual split) to validate. Then implement
Approach 3 (`--chunk N/M`) for a clean, reusable solution.

### Approach 4: GNU Parallel (best for home network)

[GNU Parallel](https://www.gnu.org/software/parallel/) distributes jobs across SSH-reachable
machines automatically — handles file transfer, result collection, load balancing, and retries.

**One-time setup:**
```bash
sudo apt install parallel
# SSH key auth to all machines
ssh-copy-id mike@machine2 && ssh-copy-id mike@machine3 && ssh-copy-id mike@machine4
# Register machines
cat > ~/.parallel/sshloginfile << 'EOF'
localhost
mike@machine2
mike@machine3
mike@machine4
EOF
```

**Usage with `--chunk N/M`** (requires pynecore code change):
```bash
parallel -S .. \
  'cd ~/pynecore/workdir && pyne optimize "Superuptrend Long Strategy.py" VN30F1M_5m.ohlcv optimize.json --chunk {}/4 -o /tmp/results_{}.csv' \
  ::: 1 2 3 4
```

**Usage with per-combo granularity** (perfect load balancing):
```bash
# 500 combos across 4 machines, 8 workers each = 32 parallel jobs
seq 0 499 | parallel -S .. -j 8 \
  'cd ~/pynecore/workdir && pyne optimize script.py data.ohlcv --combo {} -o /tmp/result_{}.csv'
```

**Key advantages over manual split:**
- `--transfer` auto-syncs files, `--return` auto-collects results, `--cleanup` removes temps
- Dynamic load balancing: fast machines automatically get more jobs
- Fault tolerance: retries failed jobs
- Auto-detects cores per machine

**Prerequisites:** pynecore + data files on all machines (rsync/syncthing/NFS), SSH key auth.

### Approach 5: Google Cloud HPC with Slurm (burst to cloud)

For large optimization runs (thousands of combos), burst to GCP using
[Cluster Toolkit](https://docs.cloud.google.com/cluster-toolkit/docs/overview) + Slurm.
Auto-scaling: nodes spin up on demand, destroy after 60s idle. Pay only for compute time.

**Architecture:**
- Controller: `c2-standard-4` (always on, manages queue)
- Compute nodes: Spot VMs, auto-scaled by Slurm (0 → N nodes on demand)
- Storage: Filestore NFS (shared across all nodes)

**Spot VM pricing (us-central1, approx):**

| Instance | vCPUs | RAM | Spot $/hr | On-demand $/hr |
|----------|-------|-----|-----------|----------------|
| e2-standard-4 | 4 | 16 GB | $0.062 | $0.134 |
| e2-highcpu-8 | 8 | 8 GB | $0.091 | $0.198 |
| c2-standard-8 | 8 | 32 GB | $0.072 | $0.363 |

**Cost estimate for 500-combo optimize:**
- Each combo: ~16s on 1 vCPU
- Total compute: 500 × 16s = 8,000 vCPU-seconds = 2.2 vCPU-hours
- With 32 Spot vCPUs (4× c2-standard-8): ~4 min wall time, **~$0.02 total**
- With 128 Spot vCPUs (16× c2-standard-8): ~1 min wall time, **~$0.02 total**
- Controller overhead: ~$0.36/hr (on while cluster exists)
- **Realistic total: $0.50–$2.00 per optimize session** (including setup/teardown)

**Setup steps:**
```bash
# 1. Install Cluster Toolkit
git clone https://github.com/GoogleCloudPlatform/cluster-toolkit.git
cd cluster-toolkit && make

# 2. Create cluster from blueprint (modify examples/hpc-slurm.yaml)
./gcluster create examples/hpc-slurm.yaml \
    -l ERROR --vars project_id=MY_PROJECT

# 3. Deploy (~5 min)
./gcluster deploy hpc-slurm

# 4. SSH to login node, submit jobs
gcloud compute ssh hpc-slurm-login0 --zone=us-central1-a

# 5. On login node: submit Slurm array job
sbatch --array=1-500 optimize_job.sh

# 6. When done, destroy cluster
./gcluster destroy hpc-slurm --auto-approve
```

**Slurm job script (`optimize_job.sh`):**
```bash
#!/bin/bash
#SBATCH --job-name=pyne-opt
#SBATCH --output=results/combo_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /shared/pynecore/workdir
source /shared/pynecore/.venv/bin/activate
pyne optimize "Superuptrend Long Strategy.py" VN30F1M_5m.ohlcv optimize.json \
    --combo $SLURM_ARRAY_TASK_ID -o /shared/results/result_${SLURM_ARRAY_TASK_ID}.csv
```

**When to use GCP vs home machines:**
- **Home (GNU Parallel):** ≤500 combos, regular usage, free compute
- **GCP (Slurm):** 1000+ combos, one-off large sweeps, need results fast, ~$1-5/run

**Prerequisites:** GCP account with billing, `gcloud` CLI, Terraform.

### Priority Order for Remaining Optimizations

1. **Skip visual ops in optimize mode** → 5-15% per optimize run (trivial)
2. **Cache static indicators across optimize runs** → up to 3x for optimize (hard)
3. **Distributed optimize (`--chunk N/M`)** → linear scaling with machines (medium)
4. **Cache overload dispatch** → 25% (medium, pure Python change)
5. **Test PyPy compatibility** → 30-50% if it works (low effort to test)
6. **Cython for hot files** → 15-25% (medium, if PyPy doesn't work)
7. **NumPy-backed Series** → 10% + unlocks Numba (larger refactor)
8. **Simplify function_isolation** → 7% (medium)
9. **Sliding-window ta.highest/lowest** → 5% (medium)
