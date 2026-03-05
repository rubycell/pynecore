# PyneCore Full Code Review — Step 1

**Date:** 2026-03-05
**Scope:** Full project review — core, transformers, strategy, CLI, utilities
**Reviewer:** Claude Opus 4.6

---

## Files Reviewed

### Core
- `src/pynecore/core/series.py`
- `src/pynecore/core/script_runner.py`
- `src/pynecore/core/import_hook.py`
- `src/pynecore/core/function_isolation.py`
- `src/pynecore/core/safe_convert.py`
- `src/pynecore/core/_var_cache.py`
- `src/pynecore/core/ohlcv_file.py`
- `src/pynecore/core/resampler.py`
- `src/pynecore/core/strategy_stats.py`

### Transformers
- All 13 transformers in `src/pynecore/transformers/`

### Strategy
- `src/pynecore/lib/strategy/__init__.py`
- `src/pynecore/lib/strategy/opentrades.py`
- `src/pynecore/lib/strategy/closedtrades.py`
- `src/pynecore/lib/strategy/commission.py`
- `src/pynecore/lib/strategy/risk.py`

### CLI
- `src/pynecore/cli/commands/optimize.py`
- `src/pynecore/cli/commands/run.py`

### Other
- `src/pynecore/lib/ta.py`

---

## Findings

---

### Finding 1: NA identity check in safe_convert.py is ineffective

**File:** `src/pynecore/core/safe_convert.py:15-16`
**Category:** LOGIC FLOW
**Severity:** MEDIUM
**Confidence:** HIGH

```python
def safe_div(a: float | NA[float], b: float | NA[float]):
    if b == 0 or b == 0.0:
        return NA(float)
    if a is NA() or b is NA():      # <-- BUG
        return NA(float)
```

`NA()` with no args defaults to `NA[int]` (type=`int`). NA uses `_type_cache` to return cached singletons per type. So `NA()` returns `NA[int]`, but `a` and `b` are `NA[float]` when NA. Since `NA[int]` and `NA[float]` are different cached objects, `a is NA()` is always `False` for float NA inputs.

**Why it doesn't crash:** The `try/except (ZeroDivisionError, TypeError)` on lines 18-20 catches the `TypeError` when NA participates in division. So the function still returns `NA(float)` — but the explicit check on line 15-16 is dead code for its intended purpose.

**Fix:** Change to `if isinstance(a, NA) or isinstance(b, NA):` or use `a is NA(float)`.

---

### Finding 2: Inconsistent exception handling in opentrades.profit()

**File:** `src/pynecore/lib/strategy/opentrades.py:312`
**Category:** LOGIC FLOW
**Severity:** MEDIUM
**Confidence:** HIGH

```python
# Line 312 — only catches IndexError
except IndexError:
    return 0.0

# All other 12+ methods catch both:
except (IndexError, AssertionError):
    return 0.0
```

The method has `assert lib._script is not None` and `assert lib._script.position is not None` on lines 309-310. If either assertion fails, `AssertionError` escapes the function and propagates up to the caller. Every other method in the file catches both.

**Fix:** Change line 312 to `except (IndexError, AssertionError):`.

---

### Finding 3: sys.path.pop(0) is unsafe in import_script()

**File:** `src/pynecore/core/script_runner.py:56-62`
**Category:** STATE MANAGEMENT
**Severity:** MEDIUM
**Confidence:** HIGH

```python
sys.path.insert(0, str(script_path.parent))
try:
    module = import_module(script_path.stem)
finally:
    sys.path.pop(0)   # <-- assumes [0] is what we inserted
```

If any code during `import_module` inserts into `sys.path[0]`, then `pop(0)` removes the wrong entry. The optimize.py code uses `sys.path.remove(str(script.parent))` (line 759) which is safer. This pattern also appears in `optimize.py:754` where both `insert(0,...)` and `remove(...)` are used.

**Fix:** Use `sys.path.remove(str(script_path.parent))` in the finally block, or store the value and remove it specifically.

---

### Finding 4: Resampler has dual caching

**File:** `src/pynecore/core/resampler.py:36-47`
**Category:** DUPLICATE CODE
**Severity:** LOW
**Confidence:** HIGH

```python
@classmethod
@lru_cache(maxsize=128)
def get_resampler(cls, timeframe: str) -> 'Resampler':
    if timeframe not in cls._resamplers:           # Manual cache
        cls._resamplers[timeframe] = cls(timeframe)
    return cls._resamplers[timeframe]
```

`@lru_cache` on the classmethod AND a manual `_resamplers` dict — double caching. The `lru_cache` means the manual dict lookup is never reached after the first call for each timeframe.

**Fix:** Remove one caching layer — either `lru_cache` or `_resamplers` dict.

---

### Finding 5: Resampler uses local timezone instead of UTC

**File:** `src/pynecore/core/resampler.py:58`
**Category:** LOGIC FLOW
**Severity:** MEDIUM
**Confidence:** HIGH

```python
current_dt = datetime.fromtimestamp(current_time_sec)  # Uses local tz!
```

`datetime.fromtimestamp()` without a timezone argument uses the system's local timezone. This produces different bar boundaries depending on where the code runs. Every other datetime construction in the codebase uses UTC explicitly (e.g., `script_runner.py:103` uses `datetime.fromtimestamp(ohlcv.timestamp, UTC)`).

**Fix:** `datetime.fromtimestamp(current_time_sec, UTC)` or use `datetime.utcfromtimestamp()`.

---

### Finding 6: Equity curve grows unbounded

**File:** `src/pynecore/core/script_runner.py:384`
**Category:** MEMORY LEAK
**Severity:** LOW
**Confidence:** HIGH

```python
_equity_curve.append(current_equity)
```

`equity_curve` is a plain `list[float]` that grows by one entry per bar. For a strategy run over 10 years of 1-minute data (~2.6M bars), this consumes ~20MB. Not critical, but for very long datasets it accumulates.

The curve is used only for Sharpe/Sortino calculation in `strategy_stats.py`. These ratios could be computed incrementally (running mean/variance) without storing the full curve.

**Impact:** LOW for typical use. Could matter in optimize mode with many sequential runs if the list isn't freed between runs (though `ScriptRunner` is recreated each time, so it is freed).

---

### Finding 7: Duplicate boilerplate in opentrades.py and closedtrades.py

**File:** `src/pynecore/lib/strategy/opentrades.py`, `src/pynecore/lib/strategy/closedtrades.py`
**Category:** DUPLICATE CODE
**Severity:** LOW
**Confidence:** HIGH

Both files contain 20+ methods with identical structure:

```python
@staticmethod
def method_name(trade_num: int) -> float | NA:
    if trade_num < 0:
        return NA(float)
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return lib._script.position.{open|closed}_trades[trade_num].ATTRIBUTE
    except (IndexError, AssertionError):
        return DEFAULT_VALUE
```

This pattern is repeated ~40 times across both files. A single helper like `_get_trade_attr(trade_list, trade_num, attr, default)` would eliminate the duplication and prevent inconsistencies like Finding 2.

---

### Finding 8: strategy_stats.py does not separate long/short avg winning/losing trades

**File:** `src/pynecore/core/strategy_stats.py:599-608`
**Category:** LOGIC FLOW
**Severity:** LOW
**Confidence:** HIGH

In `write_strategy_statistics_csv()`, the "Avg winning trade" and "Avg losing trade" rows write the **all-trades** average for both the Long and Short columns:

```python
csv_writer.write("Avg winning trade",
                 stats.avg_winning_trade, stats.avg_winning_trade_percent,
                 stats.avg_winning_trade, stats.avg_winning_trade_percent,  # same for Long!
                 0, ""
                 )
```

The Long column shows the overall average instead of the long-only average. Similarly for Short column (hardcoded to `0`). The `StrategyStatistics` dataclass doesn't have per-side avg winning/losing trade fields, so this is an incomplete feature.

---

### Finding 9: optimize.py lib_dir_str inserted into sys.path multiple times

**File:** `src/pynecore/cli/commands/optimize.py:731,855`
**Category:** STATE MANAGEMENT
**Severity:** LOW
**Confidence:** HIGH

```python
# Line 731 (parallel path)
if lib_dir_str:
    sys.path.insert(0, lib_dir_str)

# Line 855 (sequential path — within the same if/else block!)
if lib_dir_str:
    sys.path.insert(0, lib_dir_str)
```

In the parallel path, `lib_dir_str` is inserted at line 731 before the probe run, then each worker gets its own process with `sys.path.insert(0, lib_dir_str)` in `_worker_init`. The main process also inserts `script.parent` at line 754. The cleanup at line 899 only removes one occurrence: `sys.path.remove(lib_dir_str)`.

**Impact:** Extra path entries in `sys.path` for the main process. Harmless but messy.

---

### Finding 10: function_isolation.py — del without existence check

**File:** `src/pynecore/core/function_isolation.py:66`
**Category:** LOGIC FLOW
**Severity:** LOW
**Confidence:** MEDIUM

```python
if is_overloaded:
    del _function_cache[call_id_key]
```

If `call_id_key` is not in `_function_cache`, this raises `KeyError`. In normal use, the AST transformer ensures the overloaded dispatcher is registered first before the implementation replaces it. But an edge case (e.g., `reset()` called between the dispatcher and implementation calls) could trigger this.

**Fix:** Use `_function_cache.pop(call_id_key, None)` for safety.

---

## Categories Checked

| Category | Checked | Findings |
|----------|---------|----------|
| Dead code | Yes | Finding 1 (dead NA check) |
| Duplicate code | Yes | Findings 4, 7 |
| Memory leaks | Yes | Finding 6 |
| Logic flow | Yes | Findings 1, 2, 5, 8, 10 |
| Logical gates | Yes | No issues found |
| State management | Yes | Findings 3, 9 |

## Modules Skipped

- `src/pynecore/lib/ta.py` — Very large (3000+ lines), calculation-heavy with Pine Script compatibility tests. Would need dedicated review.
- `src/pynecore/providers/` — External data provider integrations (ccxt, capitalcom). Lower priority.
- `src/pynecore/types/` — Mostly pure data structures with minimal logic. Spot-checked `na.py` (confirmed singleton pattern).
- `src/pynecore/transformers/` — Each transformer was checked at high level for structural issues. Deep AST logic review would need separate pass.

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 0 |
| HIGH | 0 |
| MEDIUM | 4 (Findings 1, 2, 3, 5) |
| LOW | 6 (Findings 4, 6, 7, 8, 9, 10) |

**Top priority fixes:**
1. **Finding 2** — `opentrades.profit()` missing `AssertionError` catch — easy one-line fix
2. **Finding 1** — `safe_div()` NA check is dead code — replace `is NA()` with `isinstance`
3. **Finding 5** — Resampler local timezone — could cause wrong bar boundaries
4. **Finding 3** — `sys.path.pop(0)` race — use `remove()` instead
