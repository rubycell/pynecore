# Plan: Correct Coding Style Drift in Recent Commits

## Context

The last ~20 commits (by a different contributor) drifted from project conventions: no conventional commit prefixes, abbreviated variable names, duplicated code patterns, missing Sphinx-style docstrings. This plan corrects the code style violations and adds enforcement. **No functional changes** — pure refactoring. All bad commits are on `main` — we do NOT rewrite git history.

---

## Phase 1: Extract `_get_open_trade` helper in `opentrades.py`

**File:** `src/pynecore/lib/strategy/opentrades.py`

12 static methods repeat the same 8-line try/except/assertion guard. Extract to a module-level helper:

```python
def _get_open_trade(trade_num: int):
    """
    Retrieve an open trade by index.

    :param trade_num: Zero-based trade index.
    :return: The Trade object, or None if unavailable.
    """
    if trade_num < 0 or lib._script is None or lib._script.position is None:
        return None
    try:
        return lib._script.position.open_trades[trade_num]
    except IndexError:
        return None
```

Each method becomes: `trade = _get_open_trade(trade_num)` → return `trade.attr` if found, else default.

**CRITICAL BEHAVIORAL NOTE:** `profit()` (line 300) currently catches ONLY `IndexError`, while ALL other methods catch `(IndexError, AssertionError)`. The helper replaces `assert` with explicit `None` checks, so `profit()` behavior changes: it would now return `0.0` when `lib._script is None` instead of raising `AssertionError`. This is almost certainly a bug in the original code (inconsistent with all 11 other methods), but flag in commit message.

**Commit:** `refactor(strategy): extract _get_open_trade helper to deduplicate trade lookup`
**Verify:** `python -m pytest tests/t01_lib/t30_strategy/`

---

## Phase 2: Extract `_get_closed_trade` helper in `closedtrades.py`

**File:** `src/pynecore/lib/strategy/closedtrades.py`

18 static methods with same pattern. Extract `_get_closed_trade` (reads `position.closed_trades`).

Also add missing docstring to `size()` (line 411) — it's the only method without one.

No behavioral edge cases here — all methods consistently catch `(IndexError, AssertionError)`.

**Commit:** `refactor(strategy): extract _get_closed_trade helper to deduplicate trade lookup`
**Verify:** `python -m pytest tests/t01_lib/t30_strategy/`

---

## Phase 3: Rename abbreviated variables in `ta.py`

**File:** `src/pynecore/lib/ta.py`

All renames are function-local (zero cross-file impact):

In `pivot_point_levels()` (~line 1970):
- `h` → `high_val`, `l` → `low_val`, `c` → `close_val`, `o` → `open_val`
- `p` → `pivot` (~20 occurrences within function)
- `range_` → `price_range`
- `r3` → `resistance_3`, `s3` → `support_3`
- `x` → `dm_sum` (DM calculation branch, line 2018)

In `alma()` (~line 176):
- `m` → `mean_offset`
- `s` → `sigma_scale`

**NOTE:** The `h` in list comprehensions like `[h.lower() for h in headers]` in `ohlcv_file.py` is idiomatic Python — do NOT rename those.

**Commit:** `style(ta): rename abbreviated variables to self-explaining names`
**Verify:** `python -m pytest tests/t01_lib/t20_ta/`

---

## Phase 4: Add type hints in `log.py`

**File:** `src/pynecore/lib/log.py`

- Line 32: `render()` — add type hints: `record: logging.LogRecord`, `traceback: Any`, `message_renderable: Any` (Rich doesn't export a public Renderable protocol, so `Any` is correct)
- Line 101: `formatTime()` — change `datefmt: str = None` → `datefmt: str | None = None`

**Commit:** `style(log): add missing type hints and fix parameter annotations`
**Verify:** `python -m pytest` (quick — no logic changes)

---

## Phase 5: Deduplicate column detection in `ohlcv_file.py`

**File:** `src/pynecore/core/ohlcv_file.py`

Lines 971-1009 (CSV `load_from_csv`) and 1110-1148 (TXT `load_from_txt`) contain identical column detection logic. Extract to:

```python
def _detect_column_indices(
    self,
    headers: list[str],
    timestamp_column: str | None,
    date_column: str | None,
    time_column: str | None,
) -> tuple[int | None, int | None, int | None, int, int, int, int, int]:
    """
    Detect timestamp and OHLCV column indices from headers.

    :param headers: Lowercased header names.
    :param timestamp_column: User-specified timestamp column name, or None.
    :param date_column: User-specified date column name, or None.
    :param time_column: User-specified time column name, or None.
    :return: (timestamp_idx, date_idx, time_idx, open_idx, high_idx, low_idx, close_idx, volume_idx)
    :raises ValueError: If required columns are not found.
    """
```

Also rename abbreviated column indices: `o_idx` → `open_idx`, `h_idx` → `high_idx`, `l_idx` → `low_idx`, `c_idx` → `close_idx`, `v_idx` → `volume_idx`.

**Commit:** `refactor(ohlcv): extract _detect_column_indices to deduplicate CSV and TXT loading`
**Verify:** `python -m pytest tests/t00_pynecore/`

---

## Phase 6: Rename + extract in `strategy/__init__.py` (HIGHEST RISK)

**File:** `src/pynecore/lib/strategy/__init__.py`

All sub-parts in one commit to keep file internally consistent.

### 6A: Rename instance attributes

`self.o` → `self.bar_open`, `self.h` → `self.bar_high`, `self.l` → `self.bar_low`, `self.c` → `self.bar_close`

**Scope of references** (verified by grep — ALL within this one file):
- `__slots__` (line 372): `'h', 'l', 'c', 'o'`
- `__init__` (lines 395-398): initialization
- `self.o/h/l/c` — ~56 references in methods: `process_orders`, `_check_already_filled`, `_check_high_stop`, `_check_high`, `_check_high_trailing`, `_check_low_stop`, `_check_low`, `_check_low_trailing`, `_check_close`, `fill_order`, openprofit calculation
- `position.c/h/l` — 7 references in module-level functions: `close()` (line 1539), `close_all()` (line 1566), `entry()` (lines 1622, 1627, 1633, 1638, 1642)

**No references from tests or other files.** `random.py` has `self.c` on class `LCG` — completely unrelated.

### 6B: Rename local variables

- `_fill_order()` params: `h` → `high`, `l` → `low` (line 570)
- In `_check_high_stop`, `_check_high`, `_check_low_stop`, `_check_low`, `_check_close`: `p` → `fill_price`

### 6C: Extract `_apply_slippage` helper

Current pattern (6 occurrences):
```python
p = <value>
if self._slippage_ticks > 0:
    p += syminfo.mintick * self._slippage_ticks * order.sign
```

Appears in:
- `_check_high_stop` (line 1014) — uses `order`
- `_check_low_stop` (line 1053) — uses `order`
- `_check_close` (lines 1092, 1099) — uses `order`
- Same-bar exit (lines 1285, 1292) — uses `exit_order`

**NOT** in `_check_high`/`_check_low` (no slippage on limit orders).
**NOT** in market order processing (lines 1160-1167 — different structure, already has good naming).

Extract to:
```python
def _apply_slippage(self, price: float, order: Order) -> float:
    """
    Apply slippage to a fill price based on order direction.

    :param price: Base fill price before slippage.
    :param order: The order being filled (sign determines direction).
    :return: Price adjusted for slippage.
    """
    if self._slippage_ticks > 0:
        return price + syminfo.mintick * self._slippage_ticks * order.sign
    return price
```

### 6D: Add Sphinx docstrings

Add `:param:` / `:return:` to: `_check_high_stop`, `_check_high`, `_check_high_trailing`, `_check_low_stop`, `_check_low`, `_check_low_trailing`, `_check_close`.

**Commit:** `refactor(strategy): rename abbreviated variables and extract _apply_slippage helper`
**Verify:** `python -m pytest tests/t01_lib/t30_strategy/` — FULL strategy test suite

---

## Phase 7: Add commit-msg hook for enforcement

**New file:** `.githooks/commit-msg`

```bash
#!/usr/bin/env bash
commit_regex='^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\(.+\))?: .{1,72}'
if ! grep -qE "$commit_regex" "$1"; then
    echo "ERROR: Commit message must follow Conventional Commits format."
    echo "Expected: type(scope): description"
    echo "Types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert"
    exit 1
fi
```

Update `CLAUDE.md` Code Style section to document the hook setup: `git config core.hooksPath .githooks`

**Commit:** `build(hooks): add commit-msg hook to enforce conventional commit format`

---

## Execution Order & Risk

| Phase | File | Risk | Why |
|-------|------|------|-----|
| 1 | opentrades.py | Low | Self-contained static methods, no cross-file refs |
| 2 | closedtrades.py | Low | Same pattern as Phase 1 |
| 3 | ta.py | Low | Local variable renames only |
| 4 | log.py | Very Low | Type hints only, no logic |
| 5 | ohlcv_file.py | Medium | Shared helper extraction from two loaders |
| 6 | strategy/__init__.py | **High** | 63+ attribute renames + helper extraction in hot path |
| 7 | .githooks + CLAUDE.md | None | New files only |

**After EACH phase:** `python -m pytest` (or relevant subset). Full suite after Phase 6.

---

## What We Are NOT Changing

- **Git history** — all bad commits are on main, no rebase
- **Market order slippage** (lines 1160-1167) — already uses descriptive `fill_price` and `slippage_amount`
- **List comprehension variables** — `h` in `[h.lower() for h in headers]` is idiomatic Python
- **`random.py` `self.c`** — different class (`LCG`), unrelated
- **Functional behavior** — all changes are renames/extractions, no logic changes (except the `profit()` AssertionError edge case, flagged above)
