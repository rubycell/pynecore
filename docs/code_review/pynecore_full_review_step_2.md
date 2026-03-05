# PyneCore Full Code Review - Step 2: Detailed Fix Planning

**Date:** 2026-03-05
**Input:** pynecore_full_review_step_1.md
**Scope:** Detailed fix plans for all 10 findings from step 1

---

## Fix 1: NA identity check in safe_convert.py

### Side-by-side comparison

**Before** (src/pynecore/core/safe_convert.py:13-20):

```python
def safe_div(a: float | NA[float], b: float | NA[float]):
    if b == 0 or b == 0.0:
        return NA(float)
    if a is NA() or b is NA():          # BUG: compares with NA[int] singleton
        return NA(float)
    try:
        return a / b
    except (ZeroDivisionError, TypeError):
        return NA(float)
```

**After**:

```python
def safe_div(a: float | NA[float], b: float | NA[float]):
    if isinstance(a, NA) or isinstance(b, NA):
        return NA(float)
    if b == 0 or b == 0.0:
        return NA(float)
    try:
        return a / b
    except (ZeroDivisionError, TypeError):
        return NA(float)
```

**Reasoning:**
- `isinstance(a, NA)` catches all NA types regardless of type parameter.
- Moved NA check before zero check: if b is NA, `b == 0` invokes `NA.__eq__` which returns True, so zero check would incorrectly trigger instead of NA check. Checking NA first gives correct precedence.
- The try/except fallback is kept as a safety net.

### Existing tests

tests/t00_pynecore/ast/test_070_safe_convert_transformer.py only verifies AST transformation output -- not safe_div() runtime behavior.

### New unit tests needed

```python
# tests/t00_pynecore/core/test_safe_convert.py
from pynecore.core.safe_convert import safe_div, safe_float, safe_int
from pynecore.types.na import NA

def __test_safe_div_normal__():
    """safe_div returns correct result for normal division"""
    assert safe_div(10.0, 2.0) == 5.0
    assert safe_div(0.0, 5.0) == 0.0

def __test_safe_div_zero__():
    """safe_div returns NA for division by zero"""
    assert isinstance(safe_div(10.0, 0), NA)
    assert isinstance(safe_div(10.0, 0.0), NA)

def __test_safe_div_na_numerator__():
    """safe_div returns NA when numerator is NA"""
    assert isinstance(safe_div(NA(float), 5.0), NA)

def __test_safe_div_na_denominator__():
    """safe_div returns NA when denominator is NA"""
    assert isinstance(safe_div(10.0, NA(float)), NA)

def __test_safe_div_na_int_type__():
    """safe_div returns NA for NA[int] (original bug edge case)"""
    assert isinstance(safe_div(NA(int), 5.0), NA)
    assert isinstance(safe_div(10.0, NA(int)), NA)
```

### Integration test

Run `python -m pytest` -- safe_div is called by safe_division_transformer.py for all Pyne division ops.

---

## Fix 2: Inconsistent exception handling in opentrades.profit()

### Side-by-side comparison

**Before** (src/pynecore/lib/strategy/opentrades.py:312):

```python
        except IndexError:
            return 0.0
```

**After**:

```python
        except (IndexError, AssertionError):
            return 0.0
```

**Reasoning:** Every other method (12+) catches both. Assert on lines 309-310 raises AssertionError when lib._script or position is None.

### Integration test

Run `python -m pytest tests/t01_lib/t30_strategy/`

---

## Fix 3: sys.path.pop(0) is unsafe in import_script()

### Side-by-side comparison

**Before** (src/pynecore/core/script_runner.py:55-62):

```python
    sys.path.insert(0, str(script_path.parent))
    try:
        module = import_module(script_path.stem)
    finally:
        sys.path.pop(0)
```

**After**:

```python
    script_dir = str(script_path.parent)
    sys.path.insert(0, script_dir)
    try:
        module = import_module(script_path.stem)
    finally:
        if script_dir in sys.path:
            sys.path.remove(script_dir)
```

**Reasoning:** sys.path.remove() removes exact value regardless of position shifts during import. Matches pattern in optimize.py:758-759.

### Integration test

Run `python -m pytest` -- any sys.path corruption shows as import failures.

---

## Fix 4: Resampler dual caching (LOW - defer)

Remove lru_cache, keep manual dict. Cosmetic, harmless. **Deferred.**

---

## Fix 5: Resampler uses local timezone instead of UTC

### Side-by-side comparison

**Before** (src/pynecore/core/resampler.py:1, 58):

```python
from datetime import datetime, timedelta
...
current_dt = datetime.fromtimestamp(current_time_sec)
```

**After**:

```python
from datetime import datetime, timedelta, UTC
...
current_dt = datetime.fromtimestamp(current_time_sec, UTC)
```

Also update epoch references (lines 81, 97):

```python
epoch = datetime(1970, 1, 1, tzinfo=UTC)
epoch_monday = datetime(1970, 1, 5, tzinfo=UTC)
```

**Reasoning:** All other datetime code uses UTC. Bar boundaries must be consistent across environments.

### Integration test

Run `python -m pytest` -- multi-timeframe scripts rely on Resampler.

---

## Fix 6: Equity curve unbounded (LOW - defer)

~20MB for 2.6M bars. ScriptRunner is short-lived. **Deferred.**

---

## Fix 7: Duplicate boilerplate opentrades/closedtrades (LOW - defer)

Large refactor touching Pine Script API surface. **Deferred.** Finding 2 addresses the one actionable item.

---

## Fix 8: strategy_stats long/short columns (LOW - defer)

Display-only. Would need 8 new dataclass fields. **Deferred.**

---

## Fix 9: optimize.py sys.path duplication (LOW - defer)

Harmless in CLI process. **Deferred.**

---

## Fix 10: function_isolation.py del without existence check

### Side-by-side comparison

**Before** (src/pynecore/core/function_isolation.py:64-66):

```python
    if is_overloaded:
        del _function_cache[call_id_key]
```

**After**:

```python
    if is_overloaded:
        _function_cache.pop(call_id_key, None)
```

**Reasoning:** pop(key, None) is a no-op if key missing, preventing KeyError.

### Integration test

Run `python -m pytest tests/t00_pynecore/ast/test_051_function_isolation_main_inner.py`

---

## Summary: Action Plan

### Fixes to implement now (MEDIUM severity):

| # | Finding | File | Change |
|---|---------|------|--------|
| 1 | NA identity check | core/safe_convert.py | Replace is NA() with isinstance, reorder checks |
| 2 | Missing AssertionError | lib/strategy/opentrades.py:312 | Add AssertionError to except |
| 3 | sys.path.pop(0) | core/script_runner.py:55-62 | Use sys.path.remove() |
| 5 | Local timezone | core/resampler.py:1,58,81,97 | Add UTC to datetime calls |
| 10 | del without check | core/function_isolation.py:66 | Use pop(key, None) |

### Deferred (LOW severity):

| # | Finding | Reason |
|---|---------|--------|
| 4 | Dual caching | Cosmetic |
| 6 | Unbounded equity | Low impact, short-lived |
| 7 | Duplicate boilerplate | Large refactor risk |
| 8 | Long/short columns | Display-only |
| 9 | sys.path duplication | Harmless |

### Test plan:

1. Create tests/t00_pynecore/core/test_safe_convert.py with unit tests
2. Run full test suite: `python -m pytest`
3. Verify strategy tests: `python -m pytest tests/t01_lib/t30_strategy/`
4. Verify AST tests: `python -m pytest tests/t00_pynecore/ast/`
