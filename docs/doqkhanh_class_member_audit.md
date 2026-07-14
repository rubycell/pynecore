# Audit: Class-Member / Module-Property Changes in doqkhanh's Commits

Scope: all 33 commits authored by `doqkhanh` (2026-02-17 → 2026-07-04), checked against the
pre-existing codebase as it stood at `178682b1cfdba0477e71a28dc660f11e43c1ef66`
(parent of `c7f43ee`, the last commit by the original author, Adam Wallner).

"Class member" here includes normal Python class methods/attributes/constants **and**
this codebase's fake-namespace pattern: module-level objects (`strategy`, `ta`, `array`, …)
whose attributes are rewritten into function calls by
`src/pynecore/transformers/module_property.py` (`ModulePropertyTransformer`), driven by
the registration manifest `src/pynecore/transformers/module_properties.json`.

Commits that touched only docs/tests with no `src/` changes are skipped per the task brief:
`7ae53884` (partial – see below), `4de1791e`, `28f3ff08`, `a0b4a71`, `edcd16a`, `912c72a`.
(`65eba671` and `a8724ce9` touch `src/` but not any class/module-property surface — noted,
not counted.)

## 1. Findings Table

| Commit (date) | File:line | Namespace/Class | New / Changed | Description |
|---|---|---|---|---|
| c7f43ee (02-17) | `lib/strategy/__init__.py:556` | `strategy.Position._fill_order` | Changed | Skip filling exit orders when flat, enabling same-bar entry→exit |
| c7f43ee (02-17) | `lib/strategy/__init__.py:955` | `strategy.Position` (gap-fill check) | Changed | Exit orders no longer gap-fill when no position exists |
| c7f43ee (02-17) | `lib/strategy/__init__.py:1134-1160` | `strategy.Position.process_orders` | Changed | Exit orders sharing an id with a pending entry survive flat-clearing; clearing moved after entry processing |
| c7f43ee (02-17) | `lib/strategy/__init__.py:1222-1256` | `strategy.Position.process_orders` | New (logic block) | New same-bar stoploss/take-profit fill path for trades opened this bar |
| c7f43ee (02-17) | `lib/strategy/__init__.py:1369` | `strategy.cancel()` | Changed | Now only cancels entry orders, never exit orders (matches TV semantics) |
| c7f43ee (02-17) | `lib/strategy/__init__.py:1900` | `strategy.max_drawdown_percent` | New (module property) | New property, drawdown as % of peak equity |
| c7f43ee (02-17) | `lib/strategy/__init__.py:1950-2012` | `strategy` | New (4 module properties) | `margin_liquidation_price` (stub), `default_entry_qty`, `convert_to_account`, `convert_to_symbol` |
| c7f43ee (02-17) | `lib/strategy/closedtrades.py` (+87 lines) | `strategy.closedtrades.ClosedTradesModule` | New (10 dunder methods) | Adds `__gt__ __lt__ __ge__ __le__ __eq__ __ne__ __sub__ __add__ __rsub__ __radd__` to the **pre-existing** class (file already existed at baseline — see §2) |
| c7f43ee (02-17) | `lib/strategy/opentrades.py` (+101 lines) | `strategy.opentrades.OpenTradesModule` | New (10 dunders + `capital_held`) | Same operator set mirrored onto `OpenTradesModule`, plus new `capital_held` property |
| c7f43ee (02-17) | `core/script_runner.py:170,218,223` | `ScriptRunner` | New | `__slots__` gains `stats`, `plot_meta_path` |
| c7f43ee (02-17) | `core/script_runner.py:288-294` | `ScriptRunner.run_iter` | Changed | `barstate.islast` now via index comparison instead of lookahead-peek |
| c7f43ee (02-17) | `core/script_runner.py:459` | `ScriptRunner.run_iter` | Changed | `self.stats` always computed, not just when a stats writer is attached |
| c7f43ee (02-17) | `transformers/lib_series.py:40-122` | `LibrarySeriesTransformer` | Changed (regression) | Removed hoisting of builtin price-series declarations to top-level `main` for nested functions |
| c7f43ee (02-17) | `lib/array.py:219,1093,1186` | `array.covariance/standardize/variance` | Changed | Filter NA before `statistics.mean/variance` |
| c7f43ee (02-17) | `lib/array.py:798,822,845` | `array` | New | `new_box()`, `new_line()`, `new_table()` no-arg constructors |
| c7f43ee (02-17) | `lib/syminfo.py:33` | `syminfo.mincontract` | New (module property) | Minimum contract size |
| c7f43ee (02-17) | `types/matrix.py:21,26,974` | `Matrix.__init__/add_row/add_col` | Changed (regression) | `rows`/`cols` defaults removed; auto-sizing from array on empty matrix removed |
| c7f43ee (02-17) | `types/na.py:178` | `NA.__contains__` | New | `x in na` → `False` instead of raising |
| c7f43ee (02-17) | `core/script.py:359` | `Script.strategy()` classmethod | Changed | `margin_long`/`margin_short` defaults `100.0 → 0.0` |
| c7f43ee (02-17) | `core/script.py:729,855` | `Script._Input` | New | `text_area()` classmethod, `textarea` alias |
| c7f43ee (02-17) | `core/ohlcv_file.py:167` | `OHLCVReader.end_timestamp` | Changed (regression) | Reverted to pure arithmetic, losing correctness for gap-filled files |
| c7f43ee (02-17) | `lib/__init__.py:99,223,236,273,310` | `lib` (module) | New | `_plot_meta`, `_resolve_enum_name`, `_serialize_color`; `plotchar`/`plotshape` become real implementations |
| c7f43ee (02-17) | `lib/map.py:19,42` | `map.contains/get` | Changed | NA-safe: no longer raises `KeyError` |
| 52fc7bf (02-20) | `types/color.py:72,88` | `Color.t / Color.rgb` | Changed | NA-safety added |
| 52fc7bf (02-20) | `types/na.py:52` | `NA.__format__` | New | Enables `f"{na:.2f}"` |
| 52fc7bf (02-20) | `lib/array.py:296,390,403,416,960,1017` | `array` (6 functions) | Changed | NA-index guards, empty-array NA returns |
| 52fc7bf (02-20) | `lib/color.py:56,67` | `color.t / color.new` | Changed | `t()` now reads fixed `.t` property; `new()` NA-propagates |
| 52fc7bf (02-20) | `lib/plot.py:28-30` | `plot.PlotModule` | New | `linestyle_solid/dashed/dotted` constants |
| 52fc7bf (02-20) | `lib/ta.py:1930` | `ta.pivot_point_levels` | New (re-added) | Re-adds function deleted in c7f43ee, different simplified signature |
| 52fc7bf (02-20) | `utils/sequence_view.py:35-81` | `SequenceView` | New (7 methods) + Changed | `append/index/insert/pop/clear/reverse/sort`; `__iter__` bounds-checked |
| 1985822 (02-21) | `lib/strategy/__init__.py:666` | `strategy.Trade.size` (via `_fill_order`) | Changed | Rounds `trade.size` residuals to 0, matching existing `self.size` rounding |
| 6eab3bd (02-21) | `lib/strategy/__init__.py:456` | `strategy.Position._add_order` | Changed | Market close orders no longer replace pending SL/TP exit orders |
| 6eab3bd (02-21) | `lib/strategy/__init__.py:1127` | `strategy.Position.process_orders` | Changed | Close orders exempted from flip-size-adjustment logic |
| 6eab3bd (02-21) | `core/script_runner.py:238-282` | `ScriptRunner.run_iter` | Changed | Trade CSV rows buffered and sorted by entry time instead of write-on-close order |
| 8c404bf (02-22) | `lib/strategy/__init__.py:2003` | `strategy.openprofit_percent` | New (module property) | New property (was previously missing → `AttributeError`) |
| 8c404bf (02-22) | `lib/strategy/__init__.py:155-167,378,430-434` | `strategy.Trade / Position` | New | `Trade.original_size`; `Position.equity_max_drawdown(_percent)`, `peak_equity`, `real_max_drawdown(_percent)` |
| 8c404bf (02-22) | `lib/strategy/__init__.py:1322-1373` | `strategy.Position` (bar-close update) | New calc | "real max drawdown" and "equity max drawdown" computed per bar |
| 8c404bf (02-22) | `lib/ta.py:1507,1762` | `ta.stdev / ta.variance` | Changed | Catch `ValueError` from `sqrt`; clamp negative variance to 0 |
| 8c404bf (02-22) | `core/strategy_stats.py:18-99,233-250,485+` | `StrategyStatistics` (dataclass) | New (10 fields) | `equity_max_drawdown(_percent)`, `real_max_drawdown(_percent)`, `total/realized/unrealized_pnl(_percent)` |
| 02a028a (02-23) | `core/callable_module.py:15,21-35` | `CallableModule` | New | `_history` list, `__getitem__`, `_snapshot`, `_reset_history` — foundation for `strategy.opentrades[1]`-style history indexing on **every** module-property namespace |
| 02a028a (02-23) | `core/syminfo.py:36-40,119,169` | `SymInfo` (dataclass) | New | `country`, `mincontract`, `root`, `sector`, `industry` |
| 02a028a (02-23) | `lib/syminfo.py:9,25` | `syminfo.main_tickerid` | New (module property) | New property, `mincontract` added to `__all__` |
| 02a028a (02-23) | `lib/strategy/__init__.py:603,620,668,718,802,1285-1355` | `strategy.Position` (P&L formulas, 10 sites) | Changed | Added missing `* syminfo.pointvalue` multiplier — critical for futures |
| 02a028a (02-23) | `lib/strategy/__init__.py:2007` | `strategy.openprofit_percent` | Changed | Same pointvalue fix applied to the property added one commit earlier |
| 02a028a (02-23) | `lib/alert.py:30` | `alert()` | Changed | Now a pure no-op (was printing to console) |
| 88737da (02-23) | `core/callable_module.py:29-33` | `CallableModule._snapshot` | Changed | Guards `TypeError` for arg-taking modules (`plot`, `alert`) so history snapshotting doesn't crash |
| 3a62af2 (02-23) | `lib/__init__.py:749-751` | `lib.time_close` | Changed | `time_close(None)` now adds timeframe duration instead of returning bar-open time |
| 098c3c7 (02-23) | `lib/string.py:329-331` | `string.format` | New branch | Adds `{0,date,...}` Java-style date format spec |
| 020ac70 (02-23) | `lib/ta.py:1709-1751` | `ta.variance` | New member + Changed | New `Persistent[list] _buf` ring buffer; window-exit value now read from `_buf` instead of `source[length]` |
| e4b97cc (02-27) | `core/series.py:14,174,192-198` | `SeriesImpl` | New + Changed (**reverted**) | Shared `_NA_SINGLETON`; new int-key fast path **silently dropped the negative-index guard** |
| e4b97cc (02-27) | `core/script_runner.py:304-402` | `ScriptRunner` | Changed (**reverted**) | `self.bar_index` only synced back at generator close, not per bar — stale-read hazard |
| e4b97cc (02-27) | `lib/box,label,line,log,plot,table.py` | 6 modules | Changed (**reverted**) | `PYNE_OPTIMIZE_MODE` env gate turns drawing/log calls into no-ops during optimize runs |
| e4b97cc (02-27) | `core/overload.py:34,137-165` | `overload` dispatcher | New cache (**reverted**) | `_dispatch_cache` dict; changes "first-fit" match semantics to "best-fit, then cached" |
| 06697ad (02-27) | `core/*.py`, `lib/box,label,line,log,plot,table.py` | (11 files) | **Structural violation** | Files replaced with **absolute symlinks to a path outside the repo** (`/home/mike/workspace/github/pinescript/pynecore/runtime_patches/...`) — non-portable, breaks any other checkout/CI |
| b20523b (02-27) | (same 11 files) | — | Full revert | Confirmed byte-identical to pre-saga state; none of e4b97cc/06697ad's class changes survive |
| d41dfb4 (03-01) | `lib/strategy/__init__.py:673,776,1417` | `strategy.Position` (commission calc, 3 sites) | Changed | Percent-commission now multiplies by `syminfo.pointvalue`; line 776 also restored a missing `price` factor |
| 583dd45 (03-01) | `core/_var_cache.py` (new file) | `_var_cache` (module) | New | Process-global `_data`/`_build` slots for cross-process persistent-var caching in `optimize` |
| 583dd45 (03-01) | `core/overload.py:34,137-165` | `overload` dispatcher | New + Changed | Re-adds `_dispatch_cache`; picks "closest fit" match instead of first, then caches |
| 583dd45 (03-01) | `core/function_isolation.py:69-152` | `isolate_function()` | Changed | Explicit cache-hit/miss branch; module-preservation scan now runs only once per cache-miss |
| 583dd45 (03-01) | `core/script_runner.py:8,361` | `ScriptRunner` (module gate) | New + Changed | `_OPTIMIZE_MODE` flag skips plot-CSV writing entirely during optimize |
| 51460f0 (03-02) | `lib/strategy/__init__.py:392,474,1135` | `strategy.Position._slippage_ticks` | New | New slot/attribute, set from `script.slippage` each bar |
| 51460f0 (03-02) | `lib/strategy/__init__.py:1024-1110,~1290-1310` | `strategy.Position` (stop/stop-limit/exit fills) | Changed | Fill price now shifts by `mintick * slippage_ticks * sign`, extending the pre-existing market-order slippage formula (see §2) to stop orders |
| c003a59 (03-05) | `lib/log.py:21,181-208` | `log.info/warning/error` | New + Changed | `_LOG_ENABLED` gate added; logging becomes opt-in (silent by default) |
| a8724ce (03-09) | `core/datetime.py:15` | `PINE_FORMATS` (module list, not a namespace) | New entry | Accepts `"%Y %b %d %H:%M:%S"` — flagged for completeness, not a qualifying class/module-property member |
| 84d45f5 (03-09) | `lib/log.py:20-22` | `log._LOG_ENABLED` | Changed | Default flipped: opt-in→opt-out (logging on by default again), reversing c003a59 |
| e459204 (03-12) | `transformers/module_properties.json` (`strategy.opentrades/closedtrades`) | `strategy` (manifest) | Changed | Declares `opentrades`/`closedtrades` as `"type": "property"` so `SeriesImpl` stores the resolved int, fixing `strategy.opentrades[1]` |
| 10b04c6 (03-26) | `lib/__init__.py:56,90`, `core/script_runner.py:105-106` | `lib.time_tradingday` | New (module property) | Midnight-in-exchange-tz timestamp per bar |
| 3b03d89 (03-28) | `lib/__init__.py:259-282` | `lib.bgcolor()` | Changed | Stub → real implementation, writes to `_plot_data`/`_plot_meta` |
| 0c59df2 (06-08) | `core/strategy_stats.py:~253` | `StrategyStatistics.max_equity_drawdown_percent` | Changed (buggy) | Denominator switched `initial_capital → position.peak_equity` |
| 380242c (06-10) | `core/strategy_stats.py:233,253`; `lib/strategy/__init__.py:382,444-445` | `strategy.Position.close_max_drawdown(_percent)` | New + Changed (fix) | Confirms 0c59df2's formula was buggy (dollar-max and percent-max drawdown don't share a bar); introduces two independently-maximized fields |
| 3547f18 (07-04) | `lib/array.py:854,865` | `array.new_label()` | New | New overload, array of `Label`; commit message notes CSV rendering not yet wired |
| 6e7abcf (07-04) | `lib/strategy/__init__.py:381,441` | `strategy.Position.peak_realized_equity` | New | Tracks peak of *realized*-only equity (excludes open P&L) |
| 6e7abcf (07-04) | `lib/strategy/__init__.py:1291-1317` | `strategy.Position` (same-bar stop exit fill) | Changed | Fixes phantom-profit bug when a same-bar stop level sits on the wrong side of entry price |
| 6e7abcf (07-04) | `lib/strategy/__init__.py:1403-1407` | `strategy.Position.close_max_drawdown_percent` | Changed (2nd fix) | Percent now only updates alongside its own dollar-max, correcting 380242c's still-mismatched pairing |
| 6e7abcf (07-04) | `lib/strategy/__init__.py:1425-1435` | `strategy.Position.equity_max_drawdown(_percent)` | Changed | Anchored to new `peak_realized_equity` instead of `peak_equity` |

Flagged but **not counted** as qualifying (outside class/module-property scope, noted for
completeness only): `7ae53884`'s `generate_explicit_combinations()` (plain CLI function),
`65eba671` (CLI progress-bar globals), `a8724ce9`'s `PINE_FORMATS` list.
Confirmed doc/test-only, no findings: `4de1791e`, `28f3ff08`, `a0b4a71`, `edcd16a`, `912c72a`.

## 2. Convention Adherence Review

### 2a. Strategy module properties / fake namespaces

**Established pre-existing pattern** (at `178682b1`):
- A property is registered in `src/pynecore/transformers/module_properties.json` under its
  module path (e.g. `"lib.strategy.opentrades": {"opentrades": {"type": "property"}}`).
  `ModulePropertyTransformer` (`transformers/module_property.py:26-29`) reads this file at
  construction and rewrites `strategy.opentrades` → `strategy.opentrades()` only if it's
  listed here.
- `src/pynecore/lib/strategy/opentrades.py` and `closedtrades.py` **already existed** at
  baseline as `OpenTradesModule`/`ClosedTradesModule` classes, each with a companion
  `.pyi` stub (`opentrades.pyi`, `closedtrades.pyi`) that Pylance/type-checkers read since
  the module property machinery makes `strategy.opentrades` resolve to a callable instance,
  not a normal function — the `.pyi` is the only way static analysis sees the real API.

**Where doqkhanh followed convention correctly:**
- `8c404bf` (`strategy.openprofit_percent`) and `10b04c6` (`lib.time_tradingday`) both added
  genuinely new properties as plain `@module_property`-decorated functions in the same file
  and style as neighboring properties — correct mechanical pattern.
- `e459204` correctly diagnosed that a *dual-nature* property (behaves as both a callable
  and something `SeriesImpl` needs to store) must be declared `"type": "property"` in
  `module_properties.json` — i.e., doqkhanh found and used the sanctioned registration point
  rather than patching around it.
- `51460f0`'s stop-order slippage extension reuses the **exact pre-existing formula**
  (`syminfo.mintick * script.slippage * order.sign`, see `lib/strategy/__init__.py:1151`
  at baseline) rather than inventing a new one — good consistency.

**Where doqkhanh deviated:**
- `c7f43ee` added 10 new dunder methods each to `ClosedTradesModule` and `OpenTradesModule`
  (`__gt__`, `__eq__`, `__add__`, etc., plus `capital_held` on `OpenTradesModule`) but
  **never touched `closedtrades.pyi` / `opentrades.pyi`**. Verified directly: `git diff` of
  both `.pyi` files between baseline and `c7f43ee` is empty, and as of HEAD the `.pyi` files
  still contain zero references to `__gt__`/`__eq__`/etc. This is exactly the ".pyi stub
  update" step the pre-existing pattern requires and doqkhanh skipped it, in every commit
  since (`e4b97cc` touched these `.pyi` files but only for an unrelated Pylance
  module-level-declaration change, itself later reverted in part).
- `06697ade` went further than a missed convention step: it replaced 11 real source files
  (`core/series.py`, `core/script_runner.py`, `core/overload.py`, `core/function_isolation.py`,
  `core/safe_convert.py`, `lib/box.py`, `label.py`, `line.py`, `log.py`, `plot.py`, `table.py`)
  with **symlinks to an absolute path outside the git repository**
  (`/home/mike/workspace/github/pinescript/pynecore/runtime_patches/...`). This is not a
  "convention deviation," it's a repo-breaking commit — anyone else cloning the repo (or CI)
  gets dangling symlinks. It was caught and fully reverted one commit later (`b20523b`,
  "The cached did not work"), confirmed byte-identical to pre-saga state.
- The `e4b97cc` fast-path added to `SeriesImpl.__getitem__` silently dropped the existing
  `if key < 0: raise IndexError(...)` guard — an unreviewed behavioral regression that,
  had it shipped, would have made negative-index access on series silently wrap instead of
  raising. Also reverted in `b20523b`, but it shipped in the interim commit.

### 2b. MDD / equity calculation

**Established pre-existing pattern:** `src/pynecore/core/strategy_stats.py` was already the
single canonical place for statistics (a `@dataclass StrategyStatistics` populated by
`calculate_strategy_statistics()`, written out via `CSVWriter`). Baseline already had
`max_equity_drawdown` = `float(position.max_drawdown)` and
`max_equity_drawdown_percent` = `(max_equity_drawdown / initial_capital) * 100`
(`strategy_stats.py:210,218` at baseline) — a single formula, single source of truth.

**Where doqkhanh followed convention correctly:** every drawdown fix (`0c59df2`, `380242c`,
`6e7abcf`) was made *inside* `strategy_stats.py`/`strategy/__init__.py`'s `Position` class —
the sanctioned location — rather than computing drawdown ad hoc in `script_runner.py` or a
CLI command. That part of the structure was respected.

**Where doqkhanh deviated / struggled:**
- The MDD calculation itself went through three iterations across three commits before
  becoming self-consistent: `0c59df2` changed the percent-denominator from `initial_capital`
  to `position.peak_equity` (wrong — dollar-max-drawdown and percent-max-drawdown can occur
  at different bars); `380242c` (commit message: *"MDD number wrong, manual review needed"*)
  added two independently-maximized fields `close_max_drawdown`/`close_max_drawdown_percent`
  but still maximized them independently (still capable of mismatched pairing); `6e7abcf`
  finally made percent only update alongside its own dollar-max
  (`lib/strategy/__init__.py:1403-1407`). Three commits to converge on one formula suggests
  the calculation was written and shipped before being fully worked out on paper, contrary to
  "think before coding."
- Field proliferation: at HEAD, `Position` now carries **five** overlapping drawdown/equity
  concepts added piecemeal across four separate commits — `max_drawdown` (baseline),
  `equity_max_drawdown`/`equity_max_drawdown_percent` (8c404bf), `real_max_drawdown`/
  `real_max_drawdown_percent` (8c404bf), `peak_equity` (8c404bf), `close_max_drawdown`/
  `close_max_drawdown_percent` (380242c), `peak_realized_equity` (6e7abcf) — verified live in
  `src/pynecore/lib/strategy/__init__.py:380-382,435-445`. No consolidation or removal of the
  now-largely-redundant `real_max_drawdown`/`equity_max_drawdown` pair was done alongside
  introducing `close_max_drawdown`/`peak_realized_equity`, which cover overlapping ground.
  This violates "simplicity first" / "no speculative abstractions" from the user's own
  coding-style rules — the codebase now has multiple slightly-different drawdown metrics with
  similar names and no docstring distinguishing when to use which.

### 2c. Caching

**Established pre-existing pattern:** caching already existed and was structured
conventionally — `functools.lru_cache` for pure functions (`core/datetime.py:49`,
`core/resampler.py:36`), and a simple explicit module-level dict
(`core/function_isolation.py:11`, `_function_cache: dict[str | tuple, FunctionType] = {}`)
for the one case that needed manual invalidation (`del _function_cache[call_id_key]` on
overload override). `core/overload.py` at baseline already had a comment
"Cached type checking for better performance" (`overload.py:37`) as a design intent, i.e.
caching was already a first-class, documented concern in this module.

**Where doqkhanh followed convention correctly:** `583dd45`'s new `_dispatch_cache` in
`overload.py` and the restructured `isolate_function()` cache-hit/miss branch both reuse
the exact same "plain module-level dict, explicit key tuple" idiom already used by
`_function_cache` — consistent with the pre-existing style, not a foreign pattern.

**Where doqkhanh deviated:**
- `e4b97cc`/`06697ade`'s caching push introduced ad hoc global gates
  (`_OPTIMIZE_MODE`/`PYNE_OPTIMIZE_MODE` scattered across 7 files) rather than a single
  documented configuration point — no other existing feature flag in the baseline codebase
  works this way (env vars there are read once, e.g. `script.py`'s TOML handling, not
  threaded as a same-named module-level boolean re-implemented per file).
- The new `core/_var_cache.py` (`583dd45`) is a bare module with two loose attributes
  (`_data`, `_build`) rather than a class or dataclass — functionally fine, but inconsistent
  with the rest of the codebase's preference for dataclasses (`SymInfo`, `StrategyStatistics`,
  `Trade`) or explicit classes (`CallableModule`) to hold related state.
- Most seriously, `06697ade` did not just add a caching *mechanism*, it replaced checked-in
  files with symlinks into a machine-local, non-repo directory — a fundamental violation of
  "don't touch what isn't yours to touch" / reproducibility, caught only because the next
  commit's message says the caching "did not work."

### 2d. Slippage

**Established pre-existing pattern:** slippage already existed for market orders at baseline
— `Script.slippage: int = 0` (`core/script.py:96`), threaded through `strategy()`
(`script.py:447`), and applied at fill time as
`slippage_amount = syminfo.mintick * script.slippage * order.sign` (`lib/strategy/__init__.py:1151`,
baseline). There were also four dedicated slippage test files already in the repo
(`tests/t01_lib/t30_strategy/test_01{0,1,2,3}_slippage_*.py`).

**Where doqkhanh followed convention correctly:** `51460f0` extended this exact formula to
stop and stop-limit orders (`lib/strategy/__init__.py:1024-1110`), reusing `syminfo.mintick`,
`script.slippage` (cached per-bar into the new `Position._slippage_ticks`), and the same
sign convention — a faithful extension of the existing pattern into previously-uncovered
order types, not a reinvention.
`6e7abcf`'s same-bar stop-exit fix (`:1291-1317`) also correctly reused
`self._slippage_ticks`/`syminfo.mintick` rather than introducing a parallel slippage
calculation.

**Where doqkhanh deviated:** none identified specific to slippage — this is the strongest
area of convention adherence in the audit.

## 3. Verdict

doqkhanh's contributions are a mixed bag: strong when extending an *existing, already-correct*
formula into new cases, but weak when introducing brand-new stateful mechanisms or when a
change touches the manifest/stub layer that the framework uses for its namespace-faking trick.

**Strongest evidence of adherence:**
1. Slippage (§2d) — `51460f0` and `6e7abcf` extend the pre-existing
   `mintick * slippage_ticks * sign` formula verbatim into stop/stop-limit orders and
   same-bar exits, matching naming (`_slippage_ticks`) and location precisely.
2. `e459204`'s fix for `strategy.opentrades[1]` — doqkhanh correctly identified that the fix
   belonged in `module_properties.json` (the sanctioned registration point for
   `ModulePropertyTransformer`), not as a workaround elsewhere.
3. Fast self-correction on a serious process failure: `06697ade`'s repo-breaking symlink
   commit was reverted one commit later (`b20523b`) with a byte-identical restoration,
   verified via diff — the mistake didn't linger.

**Strongest evidence against adherence:**
1. Every new dunder method added to `ClosedTradesModule`/`OpenTradesModule` in `c7f43ee`
   left the corresponding `.pyi` stub un-updated — confirmed still missing at HEAD, across
   every subsequent commit. Since this repo specifically maintains `.pyi` files for exactly
   these two dual-nature classes, this is a repeated, uncorrected gap rather than an
   isolated slip.
2. The MDD calculation needed three commits (`0c59df2` → `380242c` → `6e7abcf`) to converge,
   with the second commit's own message admitting *"MDD number wrong, manual review needed"* —
   indicating the formula was shipped before being verified, and the fix process added new
   fields rather than first fully specifying the corrected formula on paper.
3. Field proliferation on `Position`: five drawdown/equity-peak attributes
   (`max_drawdown`, `equity_max_drawdown`, `real_max_drawdown`, `close_max_drawdown`,
   `peak_equity`, `peak_realized_equity` — six, in fact) accumulated across four commits with
   overlapping meaning and no consolidation, contrary to this project's own "simplicity
   first" / "no speculative complexity" guidance.

Net assessment: doqkhanh generally respects the *location* conventions (right file, right
class, right dataclass) but is inconsistent about the *registration/stub* conventions that
make this framework's namespace-faking machinery type-check correctly, and shows a pattern
of shipping calculations (MDD, the February caching/optimize saga) before they were fully
verified, requiring follow-up "bug" commits to reach a correct state.
