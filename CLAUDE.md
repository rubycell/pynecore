# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is PyneCore

PyneCore is a Python framework that brings TradingView's Pine Script capabilities to Python through AST transformations at import time. Files marked with the `@pyne` magic comment in their module docstring are intercepted by the import hook and transformed through a pipeline of AST transformers, enabling Pine Script's bar-by-bar execution model in native Python.

## Build & Development Commands

```bash
# Install for development (all optional deps + dev tools)
pip install -e ".[all,dev]"

# Run the full test suite
python -m pytest

# Run a specific test file
python -m pytest tests/t01_lib/t20_ta/test_sma.py

# Run a single test function (note: test functions use __test_*__ naming)
python -m pytest tests/t01_lib/t20_ta/test_sma.py -k "test_sma"

# Type checking (pyright configured in pyrightconfig.json)
pyright src/

# CLI entry point
pyne run <script.py> <data.ohlcv>
pyne data download ccxt --symbol "BYBIT:BTC/USDT:USDT"
pyne compile <script.pine> --api-key <key>
```

**Pytest configuration** (in `pytest.ini`): uses `--import-mode=importlib`, stops on first failure (`-x`), ignores `**/data/*modified.py` files. Test function pattern is `__test_*__` (double underscores), not the standard `test_*`.

## Architecture

### AST Transformation Pipeline

The core innovation. When a module with `"""@pyne"""` docstring is imported:

1. `PyneLoader` (in `src/pynecore/core/import_hook.py`) intercepts the import
2. The source is run through an ordered pipeline of transformers in `src/pynecore/transformers/`:
   - `import_lifter.py` → `import_normalizer.py` → `persistent_series.py` → `lib_series.py` → `module_property.py` → `closure_arguments.py` → `function_isolation.py` → `series.py` → `unused_series_detector.py` → `persistent.py` → `input_transformer.py` → `safe_convert_transformer.py` → `safe_division_transformer.py`
3. The transformed AST is compiled and executed

Key transformations:
- `Series[T]` type annotations → circular buffer (`SeriesImpl`) with `[n]` historical access
- `Persistent[T]` type annotations → module-level globals that persist across bars
- `PersistentSeries[T]` → combination of both
- Function calls get isolated persistent state (each call site gets its own state)
- Built-in series (`close`, `open`, `high`, `low`, `volume`) → reads from current OHLCV context

### Execution Model

`ScriptRunner` (`src/pynecore/core/script_runner.py`) drives bar-by-bar execution:
- Loads OHLCV data from an iterator
- Sets current bar data in the `lib` module
- Calls the script's `main()` function once per bar
- Collects plots, strategy trades, and other outputs
- Yields `(candle, plots, closed_trades)` tuples via `run_iter()`

### Source Layout

| Directory | Purpose |
|-----------|---------|
| `src/pynecore/core/` | Runtime engine: import hook, script runner, SeriesImpl, function isolation, OHLCV parsing |
| `src/pynecore/transformers/` | AST transformation pipeline (one transformer per file) |
| `src/pynecore/lib/` | Pine Script-compatible function library (ta, plot, math, strategy, etc.) |
| `src/pynecore/types/` | Type definitions: Series, Persistent, NA, OHLCV, Box, Label, Line, etc. |
| `src/pynecore/cli/` | Typer-based CLI (`pyne` command) |
| `src/pynecore/providers/` | Data providers (CCXT, Capital.com) |

### Key Types

- **`SeriesImpl[T]`** (`core/series.py`): Circular buffer with configurable `max_bars_back` (default 500, max 5000). Supports `[n]` subscript for historical access.
- **`NA`** (`types/na.py`): Pine Script's missing/invalid value. Type-aware wrapper, propagates through operations.
- **`OHLCV`** (`types/ohlcv.py`): Candle data with `extra_fields` dict for additional columns from CSV.

## Testing System

Tests are Pyne scripts themselves ("dogfooding"). Each test file:
1. Starts with `"""@pyne"""` annotation
2. Contains a `main()` function (the actual Pyne script)
3. Contains `__test_*__` functions (pytest discovers these)

### Test Types

| Type | Fixtures Used | What It Verifies |
|------|---------------|------------------|
| AST transformation | `ast_transformed_code`, `file_reader` | Transformed code matches `data/*_ast_modified.py` reference files |
| CSV data-based | `csv_reader`, `runner`, `dict_comparator` | Script output matches pre-calculated values (often from TradingView) |
| Log output | `log_comparator`, `runner`, `csv_reader` | Logged output matches expected log strings |
| Strategy | `csv_reader`, `runner`, `strat_equity_comparator` | Trade entries/exits match TradingView backtesting reference data |

### Key Fixtures (from `tests/conftest.py`)

- **`runner`**: Creates `ScriptRunner` for bar-by-bar execution. Accepts `syminfo_override` dict and `syminfo_path`.
- **`dict_comparator`**: Compares dicts with `math.isclose()` tolerance (`abs_tol=1e-8`, `rel_tol=1e-5`). Handles NA values.
- **`csv_reader`**: Reads CSV test data. Defaults to `<test_name>.csv` in same directory, supports `subdir` parameter.
- **`ast_transformed_code`**: Re-imports the test file with AST debug output to capture transformed code.

### Test directory naming

Directories use `t00_`, `t01_` prefixes to control execution order (dependencies tested first).

## Code Style

- Line length ~100 chars, hard limit 120
- Type hints on all function signatures
- Sphinx-style reStructuredText docstrings (`:param:`, `:return:`)
- Zero mandatory dependencies for core; CLI and providers are optional extras

## Common Pitfalls

- **Never delete `__pycache__` without understanding**: The import hook caches transformed bytecode. Tests disable bytecode writing (`sys.dont_write_bytecode = True`).
- **Transformer ordering matters**: The pipeline order in the import hook is critical. Adding/reordering transformers can break downstream transformations.
- **Test functions need double underscores**: `__test_foo__`, not `test_foo`. Configured in `pytest.ini`.
- **Library series are special**: `close`, `open`, `high`, `low`, `volume`, `hl2`, `bar_index` etc. from `pynecore.lib` are transformed by `lib_series.py` to read from the current bar context — they are not regular variables.
- **Pine Script compatibility is the target**: Library functions in `src/pynecore/lib/` must produce identical results to TradingView's Pine Script. Reference CSVs often contain values exported directly from TradingView.
