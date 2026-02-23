# Test Results — February 22, 2026

**Python:** 3.13.7
**Pytest:** 9.0.2
**Platform:** Linux
**Command:** `.venv/bin/python -m pytest`

## Summary

| Group | Tests | Passed | Failed | Skipped | Errors |
|-------|-------|--------|--------|---------|--------|
| AST Transformation | 17 | 15 | 1 | 0 | 1 |
| AST `data/*_ast_modified.py` | — | — | — | — | 3 |
| Core / Data Handling | 4 | 3 | 0 | 1 | 0 |
| Library / Base | 1 | 1 | 0 | 0 | 0 |
| **Total** | **22** | **19** | **1** | **1** | **4** |

## Passed Tests (19)

### AST Transformation
- Simple Persistent
- Persistent in nested function
- Persistent Assign
- Kahan summation for += on Persistent float variables
- Simple Series
- Advanced Series
- Argument Series
- Persistent Series
- Import normalizer - global alias
- Import normalizer - from import
- Import normalizer - wildcard (import *)
- Import normalizer - in function
- Function Isolation - inner function in main
- Function Isolation - module level functions
- Function Isolation - lib functions

### Core
- Persistent variable in annotated assignment
- Non-persistent annotated assignment

### Data Handling
- CCXT timeframe conversion
- CCXT provider path handling
- CCXT session hours

### Library
- Base Library - time

## Skipped Tests (1)

- **CCXT real data download** — requires live API connection

## Failed Tests (1)

### `test_061_lib_series.py` — Library series (AST mismatch)

The `lib_series` transformer now scopes `lib.high` used inside a nested function differently. The reference file expects the old behavior.

**Diff:**

```diff
  __series_main·__lib·close__ = SeriesImpl()
- __series_main·__lib·high__ = SeriesImpl()
  __series_main·__lib·low__ = SeriesImpl()
- __series_function_vars__ = {'main': ('__series_main·__lib·close__', '__series_main·__lib·high__', '__series_main·__lib·low__')}
+ __series_main·nested·__lib·high__ = SeriesImpl()
+ __series_function_vars__ = {'main': ('__series_main·__lib·close__', '__series_main·__lib·low__'), 'main.nested': ('__series_main·nested·__lib·high__',)}
  __scope_id__ = ''

  def main():
      global __scope_id__
      __call_counter·main·nested·0__ = 0
      __lib·close = __series_main·__lib·close__.add(lib.close)
-     __lib·high = __series_main·__lib·high__.add(lib.high)
      __lib·low = __series_main·__lib·low__.add(lib.low)
      a: float = __series_main·__lib·close__[10]
      print(a)

      def nested():
+         __lib·high = __series_main·nested·__lib·high__.add(lib.high)
-         b: float = __series_main·__lib·high__[1]
+         b: float = __series_main·nested·__lib·high__[1]
          return b
```

**Root cause:** The transformer was updated to keep lib series local to the function where they are used, but the reference file `data/test_061_lib_series_ast_modified.py` was not updated.

## Collection Errors (4)

### 1. `test_045_lib_import_normalizer_invalid_alias.py`

```
SyntaxError: 'lib' must be imported as itself, not as an alias
```

This test file intentionally uses an invalid alias (`from pynecore import lib as l`) to test error handling. The import hook raises a `SyntaxError` during collection, preventing pytest from loading it.

### 2–4. `data/*_ast_modified.py` (3 files)

These reference files contain transformed AST output with walrus operator on attributes, which is invalid Python syntax:

```
SyntaxError: cannot use assignment expressions with attribute
```

Affected files:
- `data/test_043_lib_import_normalizer_wildcard_ast_modified.py`
- `data/test_044_lib_import_normalizer_function_ast_modified.py`
- `data/test_053_function_isolation_lib_ast_modified.py`

These files are supposed to be ignored by `conftest_ignore_patterns` in `pytest.ini`, but pytest is still attempting to collect them via `--import-mode=importlib`.
