# Fix Plan — Code Review Step 1

**Date:** 2026-07-12
**Scope:** Divergent implementations of same-kind class/module members (array constructors,
drawing-object registries, strategy drawdown tracking)
**Source review:** ad-hoc pattern-consistency scan of `src/pynecore/lib/array.py`,
`src/pynecore/lib/{box,label,line,polyline,linefill,table}.py`,
`src/pynecore/lib/strategy/__init__.py`, `src/pynecore/core/strategy_stats.py`

---

## Priority 1 — `array.new_box` / `new_line` / `new_label` overloads are broken (runtime-verified)

**Files:** `src/pynecore/lib/array.py:805-873`

The project's `@overload` decorator (`core/overload.py`) is a runtime dispatcher: every
variant must be decorated and have a real body. The size-based constructors instead use the
stdlib `typing.overload` idiom (decorated stub with `...` body, then an **undecorated** plain
`def` that rebinds the name). Verified live:

- `array.new_box` resolves to the plain `def` at line 817 — the dispatcher and the
  two documented `(top_left, bottom_right, ...)` / `(left, top, right, bottom, ...)`
  overloads at lines 441 and 494 (plus their line/label twins) are unreachable.
- `array.new_box(2, initial_value)` — valid Pine — raises
  `TypeError: takes from 0 to 1 positional arguments but 2 were given`.
- `new_label` (added in `3547f18`) copy-pasted the same broken pattern.

**Fix steps:**
1. Decide the intended dispatch: keep the coordinate-based `new_box`/`new_line`/`new_label`
   overloads reachable, and make the size-based variant a real additional overload
   (decorated with the project's `@overload`, no stdlib `typing.overload` stub).
2. Remove the stray `...`-body stub defs at lines 805-806, 828-829, 852-853 — they are dead
   weight once real dispatch is restored, and if left decorated they'd register as an
   implementation that returns `None`.
3. Add `initial_value` support to `new_box`/`new_line`/`new_label` size constructors to match
   the primitive family (see Priority 2) — Pine's `array.new_box(size, initial_value)` takes it.
4. Add a regression test that calls `array.new_box(2)`, `array.new_line(2)`, `array.new_label(2)`
   *and* the coordinate-based overloads, to catch shadowing regressions in this file going forward.
5. Clean up the stray double blank line at `array.py:874-876`.

---

## Priority 2 — Unify the `array.new_*` construction pattern

**File:** `src/pynecore/lib/array.py`

Three incompatible patterns exist for what should be one family:

| Pattern | Members | Behavior |
|---|---|---|
| `(size, initial_value)` + `isinstance` assert | `new_bool`, `new_color`, `new_float`, `new_int`, `new_string` | matches Pine |
| `(size)` only | `new_box`, `new_line`, `new_label` | missing `initial_value` param |
| ad hoc | `new_table` (fills `None`, untyped `list`), `new_linefill` (no size ctor at all) | breaks `isinstance(x, NA)` checks; inconsistent surface |

**Fix steps:**
1. Extend `new_box`/`new_line`/`new_label` to accept `initial_value` (default `NA(Box)` /
   `NA(Line)` / `NA(Label)`), consistent with the primitive constructors.
2. Change `new_table` to fill with `NA(Table)` instead of `None`, and type its return as
   `list[Table | NA[Table]]` once a `Table` type exists in `types/`.
3. Either add a `new_linefill`-style size constructor for parity, or document why linefill is
   intentionally exempt (it may not be constructible without two lines).

---

## Priority 3 — Drawing-object registries: `delete`/`copy`/`all` diverge across box/label/line vs polyline/linefill/table

**Files:** `src/pynecore/lib/box.py`, `label.py`, `line.py`, `polyline.py`, `linefill.py`, `table.py`

| Member | box / label / line | polyline / linefill / table |
|---|---|---|
| `delete()` | bare `_registry.remove(id)` — raises `ValueError` on double-delete | guarded `if id in _registry: _registry.remove(id)` |
| `all()` | returns the live `_registry` list (caller mutation corrupts state) | `polyline.all()` returns `_copy(_registry)`; table/linefill still live |
| `copy()` | `_copy(id)` without appending to `_registry` — copied object never appears in `all()`, and deleting it raises | n/a (polyline/linefill/table have no `copy()`) |

Pine's contract (stated explicitly in `linefill.delete`'s own docstring) is "deleting an
already-deleted / nonexistent object has no effect." box/label/line violate this.

**Fix steps:**
1. Guard `box.delete`, `label.delete`, `line.delete` with the same
   `if id in _registry: _registry.remove(id)` pattern already used by polyline/linefill/table.
2. Make `box.copy`/`label.copy`/`line.copy` append the copy to `_registry` (matching `new()`),
   so copied objects are deletable and show up in `all()`.
3. Make `all()` consistently return either a live reference or a defensive copy across all six
   modules — pick one policy (recommend: defensive copy, matching `polyline.all()`, since Pine
   arrays returned to user code shouldn't alias internal registry state).
4. Add regression tests: create → copy → delete original → delete copy (should not raise) →
   `all()` reflects expected membership.

---

## Priority 4 — Strategy drawdown/runup `X` vs `X_percent` tracking is inconsistent

**Files:** `src/pynecore/lib/strategy/__init__.py`, `src/pynecore/core/strategy_stats.py`

Three different patterns compute the "percent" twin of a dollar drawdown/runup figure:

1. **Peak-at-trough, gated update** — `close_max_drawdown_percent`
   (`strategy/__init__.py:1403-1407`, changed in `6e7abcf`): percent is recomputed only when
   the dollar max updates in the same bar.
2. **Independently maxed** — `equity_max_drawdown_percent` (`:1435`),
   `real_max_drawdown_percent` (`:1384`), `trade.max_drawdown_percent` (`:1357`): dollar and
   percent are each `max()`'d separately and can come from different bars.
3. **Recomputed from a reconstructed pseudo-peak** —
   `strategy.max_drawdown_percent()` module property (`:2035-2042`): ignores both tracked
   fields and derives `initial + netprofit + openprofit + max_drawdown` on the fly.

Additional inconsistencies:
- `close_max_drawdown` anchors to `peak_equity` (includes unrealized profit); the new
  `equity_max_drawdown` (from `6e7abcf`) anchors to `peak_realized_equity` (excludes it) — two
  different "peak" definitions for sibling drawdown metrics.
- `max_equity_runup_percent` (`strategy_stats.py:254`) is `runup / initial_capital` (flat
  denominator), while its drawdown twin `max_equity_drawdown_percent` is peak-at-trough.
- `strategy_stats.py:238` copies `equity_max_drawdown_percent` unconditionally, but its twin
  `max_equity_drawdown_percent` (`:253`) is only set `if initial_capital > 0` — with
  non-positive capital one field stays populated and the other silently stays `0.0`.

**Fix steps:**
1. Pick one percent-tracking pattern (recommend peak-at-trough, gated update — it's the
   TradingView-parity behavior already adopted for `close_max_drawdown_percent`) and apply it
   uniformly to `equity_max_drawdown_percent`, `real_max_drawdown_percent`, and
   `trade.max_drawdown_percent`.
2. Extract a small helper, e.g. `_update_peak_to_trough(dollar_attr, percent_attr, peak, current)`,
   used by all four call sites instead of hand-rolled `if _dd > self.X: ...` blocks.
3. Reconcile `strategy.max_drawdown_percent()` to read the tracked field instead of
   reconstructing a peak inline, or document explicitly why it must differ.
4. Decide whether `close_max_drawdown`/`max_equity_runup` should share one peak definition
   (realized vs close-inclusive) and align `peak_equity` vs `peak_realized_equity` usage.
5. Fix the `initial_capital > 0` guard asymmetry in `strategy_stats.py:238` vs `:253` — both
   fields should follow the same guard.
6. Add a shared `_sign(size)` helper for the sign expression
   `0.0 if size == 0.0 else 1.0 if size > 0.0 else -1.0`, inlined at
   `strategy/__init__.py:117, 170, 700, 809`.

---

## Priority 5 — Minor cleanup

**File:** `src/pynecore/lib/strategy/__init__.py:1290-1322`

The long/short same-bar stop-exit blocks (added in `6e7abcf`) are copy-pasted mirror images;
the market-fallback line
(`matching_trade.entry_price - syminfo.mintick * self._slippage_ticks * matching_trade.sign`)
is byte-identical in both branches and only produces the right sign because
`matching_trade.sign` flips. Low risk today, but any future edit to one branch without the
other will silently diverge. Consider parameterizing by `matching_trade.sign` /
comparison direction once Priority 4 lands and this code is touched again.

---

## Suggested order of work

1. Priority 1 (broken overloads — real bug, blocks correct Pine-script array usage)
2. Priority 3 (`delete`/`copy` crash-on-double-delete — easy, high-value correctness fix)
3. Priority 4 (drawdown/runup percent consistency — affects backtest report accuracy)
4. Priority 2 (array constructor parity — API completeness)
5. Priority 5 (opportunistic cleanup, bundle with next touch of that code)
