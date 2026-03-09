# PyneCore Source Code Changes

This document tracks modifications made to PyneCore's source code to fix runtime issues and improve compatibility.

**Latest Update**: 2026-02-16 - 1-bar Series offset fix + same-bar stoploss fix
**PyneCore Version**: Installed in `.venv/lib/python3.13/site-packages/pynecore/`

---

## Table of Contents

1. [1-Bar Series Offset + Same-Bar Stoploss (2026-02-16)](#1-bar-series-offset--same-bar-stoploss-2026-02-16)
2. [Transpiler and Runtime Fixes (2026-02-15)](#transpiler-and-runtime-fixes-2026-02-15)
3. [pine2pyne Transpiler Refactoring (2026-02-15)](#pine2pyne-transpiler-refactoring-2026-02-15)
4. [Priority 2: Array Statistics NA Handling](#priority-2-array-statistics-na-handling)
5. [ex_239: ClosedTradesModule Arithmetic](#ex_239-closedtradesmodule-arithmetic)
6. [ex_119: TOML Multiline String Support](#ex_119-toml-multiline-string-support)

---

## 1-Bar Series Offset + Same-Bar Stoploss (2026-02-16)

Two critical timing bugs fixed: series computation was 1 bar behind TradingView, and stoploss exits were delayed 1 bar when entry and stoploss trigger on the same candle.

### Transpiler: 1-Bar Series Offset Fix

#### Files: `pine2pyne/transformer.py`, `pine2pyne/import_resolver.py`

**Problem**: Scalar `var` declarations (e.g., `var float x = 0.0`) were transpiled to:

```python
x: Series[float] = nz(x[1], 0.0)
```

This self-referencing pattern has an off-by-one bug: `x[1]` is evaluated BEFORE `add()` advances the write pointer. Before `add()`, `[0]` = previous bar's value and `[1]` = 2 bars ago. But Pine Script expects `[1]` = previous bar's value. Result: all 39+ self-referencing Series variables in the strategy were 1 bar behind.

**Root Cause**: In `SeriesImpl.__getitem__`, the index is calculated as `_write_pos - 1 - key`. Before `add()` is called, `_write_pos` hasn't advanced yet, so `[1]` reads one position too far back.

**Fix** (`transformer.py`, `_transform_var_declaration`, scalar branch):

```python
# Before (wrong)
if converted_type in SCALAR_TYPES:
    type_hint = f"Series[{converted_type}]"
    value = FunctionCall(func='nz', args=[
        IndexAccess(object=Identifier(name=decl.name),
                    index=Literal(value=1, literal_type='int')),
        value
    ], kwargs={})
    self.import_resolver.lib_modules.add('nz')
    self.import_resolver.uses_series = True

# After (correct)
if converted_type in SCALAR_TYPES:
    type_hint = f"PersistentSeries[{converted_type}]"
    self.import_resolver.uses_persistent_series = True
```

The `PersistentSeriesTransformer` (runtime transformer, step 3) splits `PersistentSeries[T] = initial` into `Persistent[T] = initial` + `Series[T] = persistent_var`, avoiding the self-referencing `[1]` issue entirely.

**Fix** (`import_resolver.py`):

```python
# Added to __init__:
self.uses_persistent_series = False

# Added to generate_imports():
if self.uses_persistent_series:
    types_import.append('PersistentSeries')
```

**Verification**: First entry date shifted from 2025-01-03 (bar 1745) to 2025-01-02 (bar 1744), matching TradingView.

---

### Runtime: Same-Bar Stoploss Fix

#### File: `.venv/.../pynecore/lib/strategy/__init__.py`

**Problem**: When an entry stop order fills and the stoploss is also hit on the same bar, TradingView closes the trade on that bar. PyneCore delayed the exit to the next bar.

**Example**: Trade #3 on bigtest — entry at 88,284 (stop entry) on 2025-04-02. Bar OHLC: O=85130.5, H=88560.1, L=82281.9. Stoploss at 85,695.3. TradingView: both entry and exit on Apr 2. PyneCore (before fix): exit delayed to Apr 3.

**Root Cause** (3 interacting issues):

1. **Exit orders cleared before entries fill**: `process_orders()` cleared all exit orders when `self.open_trades` was empty, BEFORE entry stop/limit orders had a chance to fill in the orderbook processing phase.

2. **Exit orders consumed by `_fill_order` with no position**: When the orderbook processing found the exit stop in the low-phase price range, `_check_low_stop` called `fill_order` → `_fill_order`. With `self.size == 0`, neither the "close trade" branch nor the "new trade" branch executed, but the cleanup at line 780 (`if not self.open_trades: cancel all exits`) removed the exit order permanently.

3. **No re-check after entry fills**: After the entry stop filled during the high-phase, no mechanism existed to re-check exit orders that had been skipped or consumed.

**Execution trace (ohlc=false → open→low→high→close)**:

```text
Low phase:  Exit stop at 85695.3 in range [85130.5→82281.9]
            → _check_low_stop fires → fill_order → _fill_order
            → self.size=0, no position → exit consumed, removed
High phase: Entry stop at 88284 in range [85130.5→88560.1]
            → Entry fills → trade opens
            → But exit order is GONE
```

**Fix (4 changes)**:

**1. Guard `_fill_order`** — Skip exit orders when no position (they stay in orderbook):

```python
def _fill_order(self, order, price, h, l):
    # Skip exit orders when there's no position to close.
    if order.order_type == _order_type_close and self.size == 0.0:
        return
    # ... rest of method
```

**2. Guard `_check_already_filled`** — Don't convert exit orders to market when no trades:

```python
def _check_already_filled(self, order):
    # Skip exit orders when no position exists
    if order.order_type == _order_type_close and not self.open_trades:
        return False
    # ... rest of method
```

**3. Preserve exits with pending entries** — Only clear stale exit orders, keep ones with matching pending entries:

```python
# Before: clear ALL exits when flat
if not self.open_trades:
    for order in exit_orders:
        self._remove_order(order)

# After: preserve exits that have matching pending entry orders
if not self.open_trades:
    pending_entry_ids = set(self.entry_orders.keys())
    for order in exit_orders:
        if order.order_id not in pending_entry_ids:
            self._remove_order(order)
    exit_orders = [o for o in exit_orders if not o.cancelled]
```

**4. Same-bar exit re-check** — After all orderbook processing, check if exit orders should trigger for entries that just filled:

```python
# Added after orderbook processing, before P&L calculation:
if self.open_trades and self.exit_orders:
    bar_idx = int(lib.bar_index)
    for exit_order in list(self.exit_orders.values()):
        if exit_order.cancelled:
            continue
        # Find matching trade opened this bar
        matching_trade = None
        for trade in self.open_trades:
            if trade.entry_id == exit_order.order_id and trade.entry_bar_index == bar_idx:
                matching_trade = trade
                break
        if matching_trade is None:
            continue
        # Check stop/limit trigger against bar's full range
        if exit_order.stop is not None:
            if matching_trade.sign > 0 and self.l <= exit_order.stop:
                self.fill_order(exit_order, exit_order.stop, self.h, exit_order.stop)
                continue
            elif matching_trade.sign < 0 and self.h >= exit_order.stop:
                self.fill_order(exit_order, exit_order.stop, exit_order.stop, self.l)
                continue
        if exit_order.limit is not None:
            if matching_trade.sign > 0 and self.h >= exit_order.limit:
                self.fill_order(exit_order, exit_order.limit, exit_order.limit, self.l)
                continue
            elif matching_trade.sign < 0 and self.l <= exit_order.limit:
                self.fill_order(exit_order, exit_order.limit, self.h, exit_order.limit)
                continue
```

**Also reordered `process_orders()`**: Moved the gap-fill check and market order processing BEFORE exit order cleanup, so market entries fill before exit orders are cleared.

**Verification**: Trade #4 (formerly #3) — entry and stoploss both execute on 2025-04-02. Trade #2 also gained same-bar exit (Jan 13 "Safe Entry Stop").

---

### Test Results (2026-02-16)

| Metric | Before | After |
|--------|--------|-------|
| Transpile | 321/327 (98.2%) | 321/327 (98.2%) |
| Runtime | 312/327 (95.4%) | 312/327 (95.4%) |
| Regressions | — | **Zero** |
| 1st entry date | 2025-01-03 (bar 1745) | 2025-01-02 (bar 1744) |
| Trade #3 stoploss | 2025-04-03 (1 bar late) | 2025-04-02 (same bar) |

---

## Transpiler and Runtime Fixes (2026-02-15)

Multiple critical fixes to the transpiler codegen and PyneCore runtime. Test results improved from 311/327 to 312/327 runtime passes.

### Transpiler: Operator Precedence Fixes

#### File: `pine2pyne/codegen.py`

**1. Ternary operator missing parentheses**

Python's `if/else` ternary has lower precedence than `and`/`or`, causing semantic bugs when nested in logical expressions.

```python
# Before (wrong): added_condition and expr if cond else True
# Python parses as: (added_condition and expr) if cond else True
# After (correct): added_condition and (expr if cond else True)
```

**Fix** (`generate_ternary_op`): Wrap all ternary output in parentheses.

```python
# Before
return f'{true_expr} if {condition} else {false_expr}'
# After
return f'({true_expr} if {condition} else {false_expr})'
```

**2. Binary operator left-operand precedence**

Arithmetic expressions like `(a - b) / c` lost parentheses, generating `a - b / c`.

**Fix** (`generate_binary_op`): For `*`, `/`, `%` operators, wrap left operand when it contains `+` or `-`.

```python
# (a - b) / c was generating: a - b / c  (wrong)
# Now generates: (a - b) / c  (correct)
```

**3. Logical operator left-operand precedence**

Expressions like `(x or y) and z` lost parentheses, generating `x or y and z`.

**Fix**: For `and` operator, wrap left operand when it's an `or` expression.

```python
# (False or X or Y) and time >= trade_window
# was generating: False or X or Y and time >= trade_window (wrong - timeCheck only on Y)
# now generates: (False or X or Y) and time >= trade_window (correct)
```

---

### Transpiler: nz Import for var Declarations

#### File: `pine2pyne/transformer.py`

**Problem**: The `var` → `Series[T] = nz(name[1], initial)` transformation generates `nz()` calls after the import analysis phase, so the import resolver never sees them. This caused `missing_import: nz` errors in 4 test files.

**Fix** (`_transform_var_declaration`): Manually add `nz` to imports when generating the nz pattern for scalar var declarations.

```python
# Added after generating nz() call:
self.import_resolver.lib_modules.add('nz')
self.import_resolver.uses_series = True
```

**Tests Fixed**: ex_212, ex_279, ex_320, ex_321

---

### Runtime: strategy.cancel() Bug Fix

#### File: `.venv/.../pynecore/lib/strategy/__init__.py`

**Problem**: `strategy.cancel(id)` was removing **exit** orders instead of **entry** orders. Exit orders are keyed by `from_entry` (= the entry trade ID), so `cancel(entryId)` found and removed the exit order first, never reaching the entry order.

In TradingView, `strategy.cancel()` only cancels entry orders, never exit orders.

**Before:**
```python
def cancel(id: str):
    position = lib._script.position
    position._remove_order_by_id(id)  # Checks exit_orders first!
```

**After:**
```python
def cancel(id: str):
    position = lib._script.position
    # Only cancel entry orders - TradingView behavior
    order = position.entry_orders.get(id)
    if order:
        position._remove_order(order)
```

**Tests Fixed**: ex_237_supertrend_strategy

---

### Runtime: strategy.default_entry_qty() Implementation

#### File: `.venv/.../pynecore/lib/strategy/__init__.py`

**Problem**: Pine Script v6 built-in `strategy.default_entry_qty(fill_price)` was not implemented.

**Fix**: Added function that calculates default position size based on strategy settings (`fixed`, `percent_of_equity`, `cash`), accounting for commission types (percent, cash_per_contract, cash_per_order).

---

### Runtime: syminfo.mincontract Implementation

#### Files:
- `.venv/.../pynecore/lib/syminfo.py` — added `mincontract: float = 1.0`
- `.venv/.../pynecore/core/script_runner.py` — initialization logic

**Problem**: Pine Script v6 built-in `syminfo.mincontract` was not implemented.

**Fix**: Added attribute with crypto-aware initialization: `1 / 10^decimals` for crypto (6 decimals for BTC, 4 for others), `1.0` for non-crypto.

---

### Test Results

| Metric | Before | After |
|--------|--------|-------|
| Transpile | 321/327 (98.2%) | 321/327 (98.2%) |
| Runtime | 311/327 (95.1%) | 312/327 (95.4%) |
| Improvement | — | +1 (ex_237) |

Note: The nz import fix restored 4 tests that had regressed (net effect already included in the 311 baseline).

---

## pine2pyne Transpiler Refactoring (2026-02-15)

### Files Modified

- `pine2pyne/pine_builtins.py` (+31 lines)
- `pine2pyne/transformer.py` (1702→1626 lines, -76 lines)
- `pine2pyne/codegen.py` (cleanup)
- `pine2pyne/import_resolver.py` (cleanup)
- `pine2pyne/parser.py` (exception fix)

### Changes Summary

**Purpose**: Code quality improvements — eliminate duplicate code, dead code, and god methods while preserving exact behavior.

### 1. Canonical Method Sets (pine_builtins.py)

**Problem**: Method sets (`label_methods`, `line_methods`, `box_methods`, `table_methods`) were defined **3 times** inline in `transformer.py` with inconsistencies:

- Location 1 (line ~1320): Missing `copy` from label/line/box
- Location 2 (line ~1408): Has `copy`, has `cell_set_text` in table
- Location 3 (line ~1507): Has `copy`, has `matrix_methods` only here

**Solution**: Created canonical sets as single source of truth in `pine_builtins.py`:

```python
LABEL_METHODS: set[str] = {'set_x', 'set_y', 'set_xy', ..., 'copy', 'delete'}
LINE_METHODS: set[str] = {'set_x1', 'set_y1', ..., 'copy', 'delete'}
BOX_METHODS: set[str] = {'set_left', 'set_right', ..., 'copy', 'delete'}
TABLE_METHODS: set[str] = {'cell', 'merge_cells', 'set_position', 'clear', 'cell_set_text'}
MAP_METHODS: set[str] = {'put', 'put_all', 'get', 'remove', 'clear', 'keys', 'values', 'size'}
MATRIX_METHODS: set[str] = {'add_row', 'add_col', 'get', 'set'}
```

**Impact**: Eliminated 24 duplicate set definitions, resolved inconsistencies.

---

### 2. Helper Method Extraction (transformer.py)

**Problem**: `_transform_function_call()` was a 207-line "god method" with cyclomatic complexity ~25, handling 5+ distinct responsibilities.

**Solution**: Split into focused helpers:

```python
# New helper: Centralized module resolution (40 lines)
def _resolve_module_for_method(self, method_name, type_str=None) -> Optional[str]:
    """Two modes: type-aware (uses type_str) or heuristic (uses canonical sets)"""

# New helper: Handle var.method() patterns (30 lines)
def _transform_dotted_string_method_call(self, call, func) -> Optional[FunctionCall]:

# New helper: Handle obj.field.method() patterns (30 lines)
def _transform_chained_dotted_method_call(self, call, func) -> Optional[FunctionCall]:

# New helper: Handle MemberAccess-based calls (20 lines)
def _transform_member_access_method_call(self, call, func) -> Optional[FunctionCall]:

# Refactored main method (50 lines, down from 207)
def _transform_function_call(self, call) -> FunctionCall:
    """Dispatcher that calls helpers"""
```

**Impact**:

- Reduced method complexity from ~25 to ~5
- Each helper is focused, testable, and reusable
- Main method is now a readable dispatcher

---

### 3. Visitor Pattern Composition

**Problem**: Three functions (`_collect_history_ref_vars`, `_collect_reassignment_vars`, `_find_series_variables_in_function`) duplicated identical AST traversal logic (~25 lines each).

**Solution**: Extracted common visitor pattern:

```python
def _visit_ast_nodes(self, nodes, visitor_fn) -> None:
    """Generic depth-first AST walker. Composable with any visitor function."""

# Refactored collectors (8 lines each, down from 25)
def _collect_history_ref_vars(self, script) -> set:
    history_vars = set()
    def visitor(node):
        if isinstance(node, IndexAccess) and isinstance(node.object, Identifier):
            history_vars.add(node.object.name)
    self._visit_ast_nodes(list(script.declarations) + list(script.body), visitor)
    return history_vars
```

**Impact**: Eliminated ~50 lines of duplicate traversal logic.

---

### 4. Cleanup Changes

**codegen.py**:
- Removed unreachable `elif not any(self.output)` block (lines 313-315)
- Deduplicated BUG FIX comment (replaced 3-line comment with cross-reference)

**import_resolver.py**:
- Removed dead `uses_series_impl` field (never set to True anywhere)
- Removed guarded import block that would never execute

**parser.py**:
- Fixed bare `except:` → `except Exception:` (prevents catching KeyboardInterrupt)

---

### Test Results

**Baseline (before refactoring)**: 302/327 (92.4%) passing
**After refactoring**: 302/327 (92.4%) passing
**Behavioral changes**: **ZERO**

Sample-by-sample verification:

- ex_001_bar_index.pine: ✅ identical output
- ex_125_labelnew.pine: ✅ pass
- ex_131_linenew.pine: ✅ pass
- ex_096_boxnew.pine: ✅ pass
- Full suite: ✅ all 302 tests identical

---

### Code Metrics

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| transformer.py lines | 1702 | 1626 | -76 |
| Longest method | 207 lines | 50 lines | -157 |
| Method set definitions | 24 inline | 6 canonical | -18 |
| Visitor implementations | 3 × 25 lines | 1 × 16 lines + 3 × 8 lines | -35 |

---

### Maintenance Benefits

1. **Single Source of Truth**: Method sets now defined once in `pine_builtins.py`
2. **Focused Helpers**: Each transformation pattern has its own method
3. **Testability**: Helpers can be unit tested independently
4. **Readability**: 50-line dispatcher vs 207-line monolith
5. **Composability**: `_visit_ast_nodes()` reusable for future collectors

---

---

## Priority 2: Array Statistics NA Handling

### File: `.venv/lib/python3.13/site-packages/pynecore/lib/array.py`

### Problem
Python's `statistics` module functions (like `statistics.mean()`) cannot handle PyneCore's `NA` type objects. When arrays contain NA values (from accessing historical data beyond available bars, e.g., `close[9]` on bar 0), statistical functions crash with `ValueError: too many values to unpack (expected 2)`.

See detailed explanation: [NA_VALUE_FLOW.md](NA_VALUE_FLOW.md)

### Affected Functions
- `covariance()` - Line 219
- `standardize()` - Line 1093
- `variance()` - Line 1182

---

### 1. `covariance()` - Line 219

**Before:**
```python
def covariance(id1: list[Number], id2: list[Number], biased: bool = True) -> float:
    """Covariance of two arrays"""
    assert len(id1) == len(id2)

    mean1 = statistics.mean(id1)  # ❌ Crashes with NA values
    mean2 = statistics.mean(id2)

    if biased:
        return statistics.mean([(v1 - mean1) * (v2 - mean2) for v1, v2 in zip(id1, id2)])
    else:
        n = len(id1)
        return sum([(v1 - mean1) * (v2 - mean2) for v1, v2 in zip(id1, id2)]) / (n - 1)
```

**After:**
```python
def covariance(id1: list[Number], id2: list[Number], biased: bool = True) -> float:
    """Covariance of two arrays"""
    assert len(id1) == len(id2)

    # Filter out NA values before statistical calculations
    valid_pairs = [(v1, v2) for v1, v2 in zip(id1, id2)
                   if not isinstance(v1, NA) and not isinstance(v2, NA)]

    if not valid_pairs:
        return 0.0

    valid1, valid2 = zip(*valid_pairs)
    mean1 = statistics.mean(valid1)  # ✅ Works with filtered values
    mean2 = statistics.mean(valid2)

    if biased:
        return statistics.mean([(v1 - mean1) * (v2 - mean2) for v1, v2 in zip(valid1, valid2)])
    else:
        n = len(valid1)
        return sum([(v1 - mean1) * (v2 - mean2) for v1, v2 in zip(valid1, valid2)]) / (n - 1)
```

**Tests Fixed:**
- `ex_052_arraycovariance_example.pine`

---

### 2. `standardize()` - Line 1093

**Before:**
```python
def standardize(id: list[Number]) -> list[float]:
    """Standardizes an array to z-scores"""
    n = len(id)
    mean = statistics.mean(id)  # ❌ Crashes with NA values
    stdev = math.sqrt(statistics.mean([(v - mean) ** 2 for v in id]))
    return [(v - mean) / stdev for v in id]
```

**After:**
```python
def standardize(id: list[Number]) -> list[float]:
    """Standardizes an array to z-scores"""
    n = len(id)

    # Filter out NA values before statistical calculations
    valid_values = [v for v in id if not isinstance(v, NA)]

    if not valid_values:
        return [0.0] * n

    mean = statistics.mean(valid_values)  # ✅ Works with filtered values
    stdev = math.sqrt(statistics.mean([(v - mean) ** 2 for v in valid_values]))

    # Handle zero standard deviation edge case
    if stdev == 0:
        return [0.0] * n

    # Return z-scores, converting NA to 0.0
    z_scores = [(v - mean) / stdev if not isinstance(v, NA) else 0.0 for v in id]
    return z_scores
```

**Key Improvements:**
1. Filters NA values before computing mean and stdev
2. Handles edge case where all valid values are the same (stdev = 0)
3. Preserves array length by converting NA positions to 0.0 in output

**Tests Fixed:**
- `ex_088_arraystandardize_example.pine`

---

### 3. `variance()` - Line 1182

**Before:**
```python
def variance(id: list[Number], biased: bool = True) -> float:
    """Variance of an array"""
    mean = statistics.mean(id)  # ❌ Crashes with NA values

    if biased:
        return statistics.mean([(v - mean) ** 2 for v in id])
    else:
        n = len(id)
        return sum([(v - mean) ** 2 for v in id]) / (n - 1)
```

**After:**
```python
def variance(id: list[Number], biased: bool = True) -> float:
    """Variance of an array"""
    # Filter out NA values before statistical calculations
    valid_values = [v for v in id if not isinstance(v, NA)]

    if not valid_values:
        return 0.0

    mean = statistics.mean(valid_values)  # ✅ Works with filtered values

    if biased:
        return statistics.mean([(v - mean) ** 2 for v in valid_values])
    else:
        n = len(valid_values)
        return sum([(v - mean) ** 2 for v in valid_values]) / (n - 1)
```

**Tests Fixed:**
- `ex_092_arrayvariance_example.pine`

---

## ex_239: ClosedTradesModule Arithmetic

### File: `.venv/lib/python3.13/site-packages/pynecore/lib/strategy/closedtrades.py`

### Problem
The expression `strategy.closedtrades + strategy.opentrades` failed because `ClosedTradesModule.__add__()` didn't support adding `OpenTradesModule` objects. This is needed for total trade count calculations.

### Function: `__add__()` - Line 74

**Before:**
```python
def __add__(self, other):
    """Allow: strategy.closedtrades + 1"""
    if isinstance(other, int):
        return self() + other
    elif isinstance(other, ClosedTradesModule):
        return self() + other()
    return NotImplemented
```

**After:**
```python
def __add__(self, other):
    """Allow: strategy.closedtrades + 1 or strategy.closedtrades + strategy.opentrades"""
    if isinstance(other, int):
        return self() + other
    elif isinstance(other, ClosedTradesModule):
        return self() + other()
    elif hasattr(other, '__class__') and other.__class__.__name__ == 'OpenTradesModule':
        # Handle OpenTradesModule without circular import
        return self() + other()
    return NotImplemented
```

**Rationale:**
- Used duck typing (`__class__.__name__`) to avoid circular import
- Maintains compatibility with both `int` and module arithmetic
- Follows Python's operator overloading best practices

**Tests Fixed:**
- `ex_239_strategyopentradescommission_e_part1.pine`
- `ex_239_strategyopentradescommission_e_part2.pine`

---

## ex_119: TOML Multiline String Support

### File: `.venv/lib/python3.13/site-packages/pynecore/core/script.py`

### Problem
When `input.text_area()` has a default value containing newlines (e.g., `"Hello \nWorld!"`), PyneCore generates invalid TOML config files. The TOML parser fails with:
```
TOMLDecodeError: Expected '=' after a key in a key/value pair (at line 44, column 6)
```

### Root Cause
The `_format_value()` helper function in `Script.save()` wraps all strings in double quotes (`"..."`), which is invalid TOML syntax for multiline strings. TOML requires triple quotes (`"""..."""`) for strings containing newlines.

### Function: `_format_value()` (nested in `Script.save()`) - Line 121

**Before:**
```python
def _format_value(value) -> str:
    """Format value according to its type"""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Color):
        value = str(value)
    if isinstance(value, str):
        return f'"{value}"'  # ❌ Invalid for multiline strings
    return str(value)
```

**After:**
```python
def _format_value(value) -> str:
    """Format value according to its type"""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Color):
        value = str(value)
    if isinstance(value, str):
        # Use triple quotes for multiline strings in TOML
        if '\n' in value or '\r' in value:
            return f'"""{value}"""'  # ✅ Valid TOML multiline syntax
        return f'"{value}"'
    return str(value)
```

**Example Output:**
```toml
[inputs.i_text]
# Input metadata, cannot be modified
# input_type: "text_area"
#     defval: """Hello
World!"""
#      title: "Message"
```

**Tests Fixed:**
- `ex_119_inputtext_area.pine`

---

## Impact Summary

| Change | Files Fixed | Test Score Impact |
|--------|-------------|-------------------|
| Array NA handling | 3 | 297 → 300 (+3) |
| ClosedTrades arithmetic | 2 | 300 → 301 (+1) |
| TOML multiline strings | 1 | 301 → 302 (+1) |
| Operator precedence (ternary, binary, logical) | — | correctness fix |
| nz import for var declarations | 4 | restored regression |
| strategy.cancel() entry-only fix | 1 | 311 → 312 (+1) |
| strategy.default_entry_qty() | — | new API |
| syminfo.mincontract | — | new API |
| 1-bar Series offset (PersistentSeries) | — | timing correctness |
| Same-bar stoploss (process_orders reorder) | — | timing correctness |
| **Total** | **11** | **297 → 312 (+15)** |

**Final Test Results (2026-02-16):**

- Transpile: 321/327 (98.2%)
- Runtime: 312/327 (95.4%)
- Not tested (no .py): 6/327 (1.8%)

---

## Best Practices Learned

### 1. NA Value Handling Pattern
When wrapping Python stdlib functions that don't understand PyneCore's `NA` type:

```python
# Pattern: Filter NA before, handle edge cases, preserve array structure
valid_values = [v for v in array if not isinstance(v, NA)]

if not valid_values:
    return appropriate_default  # 0.0, [], etc.

result = stdlib_function(valid_values)  # Now safe

# If returning array, convert NA positions
return [transform(v) if not isinstance(v, NA) else default for v in array]
```

### 2. Circular Import Avoidance
When modules need to reference each other's types in operators:

```python
# Instead of: from .other_module import OtherClass  # ❌ Circular import
# Use duck typing:
if hasattr(other, '__class__') and other.__class__.__name__ == 'OtherClass':
    return self() + other()  # ✅ Works without import
```

### 3. TOML String Formatting
Always check for special characters when generating TOML:

```python
if isinstance(value, str):
    if '\n' in value or '\r' in value:
        return f'"""{value}"""'  # Multiline
    return f'"{value}"'  # Single line
```

---

## Maintenance Notes

### When to Revert/Update

1. **If PyneCore officially supports NA in statistics:**
   - Revert array.py changes
   - Use built-in NA handling instead

2. **If OpenTradesModule imports are refactored:**
   - Update closedtrades.py to use proper import
   - Remove duck typing workaround

3. **If TOML library is upgraded:**
   - Test multiline string handling
   - May need to adjust escape sequences

### Related Documentation

- [NA_VALUE_FLOW.md](NA_VALUE_FLOW.md) - Detailed explanation of NA value propagation
- [test_results.json](test_results.json) - Complete test results with error classifications
- [README.md](workdir/README.md) - Script status and common fixes

### Runtime Backup & Patching Guide

All runtime changes live in the installed package directory, **not** tracked by git:

```bash
.venv/lib/python3.13/site-packages/pynecore/
```

**Backup**: `pynecore_runtime_backup_2026-02-16.zip` (258K, excludes `__pycache__`)

**Restore after fresh `pip install`:**

```bash
cd pynecore/.venv/lib/python3.13/site-packages/
unzip -o /path/to/pynecore_runtime_backup_2026-02-16.zip
```

### Modified Runtime Files

| File (relative to `site-packages/`) | Changes | Section |
| ------------------------------------ | ------- | ------- |
| `pynecore/lib/strategy/__init__.py` | `cancel()` entry-only fix, `default_entry_qty()` impl, same-bar stoploss (4 changes) | See sections below |
| `pynecore/lib/strategy/closedtrades.py` | `__add__` supports OpenTradesModule | [ClosedTrades arithmetic](#ex_239-closedtradesmodule-arithmetic) |
| `pynecore/lib/strategy/opentrades.py` | Related arithmetic support | [ClosedTrades arithmetic](#ex_239-closedtradesmodule-arithmetic) |
| `pynecore/lib/array.py` | NA filtering in `covariance()`, `standardize()`, `variance()` | [Array NA Handling](#priority-2-array-statistics-na-handling) |
| `pynecore/lib/syminfo.py` | `mincontract` attribute added | See syminfo section |
| `pynecore/lib/map.py` | Map method fixes | — |
| `pynecore/lib/plot.py` | Plot data export support | — |
| `pynecore/lib/__init__.py` | Lib init changes for new exports | — |
| `pynecore/core/script_runner.py` | `syminfo.mincontract` initialization logic | See syminfo section |
| `pynecore/core/script.py` | TOML multiline string formatting | [TOML multiline](#ex_119-toml-multiline-string-support) |

### Patching a Forked PyneCore Repo

When you clone and fork pynecore's git source, the source paths map to:

```text
# Installed package path              →  Git source path
pynecore/lib/strategy/__init__.py     →  src/pynecore/lib/strategy/__init__.py  (or similar)
pynecore/lib/array.py                 →  src/pynecore/lib/array.py
pynecore/core/script.py               →  src/pynecore/core/script.py
...etc
```

> **Note**: Verify exact source layout after cloning. The mapping above assumes standard Python package structure. Use `find . -name "__init__.py" -path "*/strategy/*"` to locate the correct paths.

**Workflow:**

1. Clone the pynecore repo
2. Use this document's **before/after code blocks** to apply each patch
3. `pip install -e .` to install your patched version in editable mode
4. Run `python test_all_samples.py` to verify (expect 312/327 pass)

---

*Generated: 2026-02-14*
*Last updated: 2026-02-16 — Added runtime backup & patching guide*
