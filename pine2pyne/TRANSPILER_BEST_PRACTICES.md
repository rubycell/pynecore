# Transpiler Best Practices — Lessons from Industry

Research notes for the `pine2pyne` (Pine Script → PyneCore Python) transpiler, compiled from analysis of Babel, TypeScript, CoffeeScript, and compiler engineering literature.

---

## What pine2pyne Already Does Right

| Practice | Status | Notes |
|----------|--------|-------|
| Multi-stage pipeline (Lexer → Parser → Transformer → CodeGen) | Done | Matches Babel's 3-stage and TypeScript's 5-stage models |
| AST-based transformation (not string manipulation) | Done | 50+ dataclass AST nodes in `ast_nodes.py` |
| Symbol table + type inference | Done | `symbol_table.py` + `type_inference.py` |
| Visitor-like transformation patterns | Done | `_transform_expression()`, `_transform_statement()` dispatch |
| Behavioral end-to-end testing | Done | `test_all_samples.py` runs 100+ samples |

---

## 1. Architecture Patterns

### Multi-Stage Pipeline (Babel, TypeScript)

> "A critical best practice is avoiding the mistake of trying to generate directly the code of the target language from the AST of the original language."
> — [How to write a transpiler — Strumenta](https://tomassetti.me/how-to-write-a-transpiler/)

**Babel's 3 stages:** Parsing → Transformation → Code Generation
**TypeScript's 5 stages:** Scanner → Parser → Binder → Type Checker → Emitter

pine2pyne follows: `Lexer → Parser → Transformer → CodeGenerator`

**Future consideration:** An intermediate representation (IR) layer between Pine AST and PyneCore AST would decouple source and target, enable optimization passes, and simplify multi-target support.

### Visitor Pattern with Path Context (Babel)

> "Path is an abstraction above node. It provides the link between nodes (the parent), as well as information such as scope and context."
> — [Manipulating AST with JavaScript — Tan Li Hau](https://lihautan.com/manipulating-ast-with-javascript)

Babel wraps each AST node in a `Path` object carrying parent references, scope info, and transformation metadata. This makes rules composable and independently testable.

**pine2pyne currently:** Uses method-dispatch (`_transform_*`) which works but lacks parent/scope context at each node. Adding a lightweight context object would clean up the `current_function_params` pattern.

### Plugin Architecture (Babel)

> "Babel's visitor pattern allows adding functionality without altering main code, effectively separating concerns."
> — [Understanding ASTs by Building Your Own Babel Plugin — SitePoint](https://www.sitepoint.com/understanding-asts-building-babel-plugin/)

Each transformation rule becomes a pluggable pass. Benefits: community contributions, custom optimizations (constant folding, dead code elimination), core stays clean.

---

## 2. Testing Strategies

### Four-Layer Testing

> "Testing should focus on four main aspects: (1) parsing source code and getting source AST, (2) converting source AST into intermediate AST, (3) converting intermediate AST into target AST, (4) converting target AST into target code."
> — [How to write a transpiler — Strumenta](https://tomassetti.me/how-to-write-a-transpiler/)

| Layer | What to Test | pine2pyne Status |
|-------|-------------|-----------------|
| Lexer | Token stream correctness | Implicit only |
| Parser | AST structure from source | Implicit only |
| Transformer | Each transformation rule independently | Implicit only |
| CodeGen | Python output formatting from mock AST | Implicit only |

Currently all testing is end-to-end (`test_all_samples.py`). Layer-by-layer unit tests would catch bugs earlier in the pipeline.

### Snapshot Testing (Babel, Jest)

> "A typical snapshot test case renders a component, takes a snapshot, then compares it to a reference snapshot file."
> — [Snapshot Testing — Jest](https://jestjs.io/docs/snapshot-testing)

Captures the entire transpiled output as a reference. Any code change that alters output must be explicitly approved. With 100+ sample files already transpiling, this is essentially a free regression safety net.

### Behavioral Equivalence Testing

> "Top-level functional equivalence requires that, for any possible set of inputs x, the two pieces of code produce the same output."
> — [LLM-Based Code Translation — UC Berkeley](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-174.pdf)

Store "known good" outputs from TradingView for key indicators. Compare transpiled PyneCore output numerically (with tolerance) against TradingView reference data.

---

## 3. Error Handling

### Don't Stop at First Error

> "Common mistakes include vague error messages, inadequate recovery mechanisms, and failure to address edge cases."
> — [Error Handling in Compiler Design — GeeksforGeeks](https://www.geeksforgeeks.org/error-handling-compiler-design/)

Collect all errors, report together. Users fix everything in one pass instead of one-at-a-time.

### Panic-Mode Recovery

When encountering an error, skip tokens until reaching a "synchronization point" (newline, keyword). Prevents one syntax error from causing 100 cascading errors.

### Helpful Error Messages with Context

```
Error at line 42, column 15:
    myVar = ta.sma(close, period
                                ^
Expected closing parenthesis ')' but found end of line.
```

---

## 4. Code Generation

### Source Maps (CoffeeScript, Babel)

> "Source maps are JSON files that contain information on how to map your transpiled source code back to its original source."
> — [What are source maps? — web.dev](https://web.dev/articles/source-maps)

Map generated Python line numbers back to original Pine Script lines. When a runtime error occurs in the generated `.py`, show the corresponding `.pine` source location.

### Delegate to Existing Tools (CoffeeScript 2)

> "With transpilers like Babel around, there's no need for the CoffeeScript compiler to duplicate functionality."
> — [Announcing CoffeeScript 2](http://coffeescript.org/announcing-coffeescript-2/)

Keep the transpiler focused on syntax transformation. Let PyneCore handle runtime semantics (Series, Persistent, etc.). Don't reimplement algorithms — just map function names.

---

## 5. Anti-Patterns to Avoid

| Anti-Pattern | Why It's Bad | Alternative |
|--------------|-------------|-------------|
| Direct source-to-target generation | Complex, fragile, hard to test | Multi-stage pipeline with AST |
| No intermediate representation | Couples source and target tightly | Add IR layer between ASTs |
| Cascading errors | One error causes 100+ spurious errors | Panic-mode recovery |
| Vague error messages | Users don't know how to fix | Add context, hints, source snippets |
| No regression tests | Changes break existing code silently | Snapshot testing |
| Monolithic transformer | Hard to maintain | Visitor pattern + composable passes |
| Stopping at first error | Slow iteration cycle | Collect all errors |

---

## Sources

- [How to write a transpiler — Strumenta](https://tomassetti.me/how-to-write-a-transpiler/)
- [Babel Plugin Handbook](https://github.com/jamiebuilds/babel-handbook/blob/master/translations/en/plugin-handbook.md)
- [Understanding ASTs by Building Babel Plugin — SitePoint](https://www.sitepoint.com/understanding-asts-building-babel-plugin/)
- [TypeScript Compiler Architecture — Satellytes](https://www.satellytes.com/blog/post/typescript-ast-type-checker/)
- [Manipulating AST with JavaScript — Tan Li Hau](https://lihautan.com/manipulating-ast-with-javascript)
- [Snapshot Testing — Jest](https://jestjs.io/docs/snapshot-testing)
- [Error Handling in Compiler Design — GeeksforGeeks](https://www.geeksforgeeks.org/error-handling-compiler-design/)
- [What are source maps? — web.dev](https://web.dev/articles/source-maps)
- [Announcing CoffeeScript 2](http://coffeescript.org/announcing-coffeescript-2/)
- [Intermediate Representation — Wikipedia](https://en.wikipedia.org/wiki/Intermediate_representation)

---

## Top 3 Actionable Next Steps — Implementation Details

### 1. Extract Duplicated Method Sets to `pine_builtins.py`

**Problem:** `label_methods`, `line_methods`, `box_methods`, `table_methods` sets are defined 3 times in `transformer.py` (lines ~1320, ~1408, ~1507) with **inconsistencies** between them (e.g., `copy` missing from Location 1, `cell_set_text` only in Location 2, `matrix_methods` only in Location 3).

**Implementation:**

```python
# pine_builtins.py — add after PYNECORE_METHOD_TRANSFORMS (line 243)

LABEL_METHODS: set[str] = {
    'set_x', 'set_y', 'set_xy', 'set_text', 'set_color', 'set_textcolor',
    'set_size', 'set_style', 'set_textalign', 'set_tooltip',
    'get_x', 'get_y', 'get_text', 'copy', 'delete',
}
LINE_METHODS: set[str] = {
    'set_x1', 'set_y1', 'set_x2', 'set_y2', 'set_xy1', 'set_xy2',
    'set_extend', 'set_width', 'set_xloc',
    'get_x1', 'get_y1', 'get_x2', 'get_y2', 'get_price', 'copy', 'delete',
}
BOX_METHODS: set[str] = {
    'set_left', 'set_right', 'set_top', 'set_bottom',
    'set_lefttop', 'set_rightbottom',
    'set_border_color', 'set_border_width', 'set_border_style',
    'set_bgcolor', 'set_text', 'set_text_color', 'set_text_size',
    'get_left', 'get_right', 'get_top', 'get_bottom', 'copy', 'delete',
}
TABLE_METHODS: set[str] = {'cell', 'merge_cells', 'set_position', 'clear', 'cell_set_text'}
MAP_METHODS: set[str] = {'put', 'put_all', 'get', 'remove', 'clear', 'keys', 'values', 'size'}
MATRIX_METHODS: set[str] = {'add_row', 'add_col', 'get', 'set'}
```

Then in `transformer.py`:

```python
# Update import (around line 11)
from .pine_builtins import (
    ...,
    LABEL_METHODS, LINE_METHODS, BOX_METHODS, TABLE_METHODS,
    MAP_METHODS, MATRIX_METHODS,
)
```

Delete all 3 inline set definitions. Replace references with the imported constants.

**Behavioral note on `copy`:** Location 1 (chained dotted calls, line ~1320) intentionally excludes `copy` from sets — it has a separate `elif method_name == 'copy'` that routes to `udt_copy()`. With `copy` now in the canonical sets, this `elif` would become unreachable. Fix: check `copy` BEFORE the set lookup in the chained dotted handler:

```python
# In _transform_chained_dotted_method_call:
if method_name == 'copy':
    # Always udt_copy for chained access (obj.field.copy())
    return FunctionCall(func='udt_copy', ...)

module_name = _resolve_module_for_method(method_name)  # Now safe — copy handled above
```

**Effort:** Small. ~30 min. Low risk.

---

### 2. Split `_transform_function_call()` into Focused Methods

**Problem:** 207-line god method with 5+ responsibilities, cyclomatic complexity ~25, 8+ nesting levels.

**Implementation — extract 3 helpers + 1 utility:**

#### Helper A: `_resolve_module_for_method(method_name, type_str=None) -> Optional[str]`

Centralizes the repeated "method name → module" resolution. Two modes:

```python
def _resolve_module_for_method(self, method_name: str, type_str: str = None) -> Optional[str]:
    """Resolve PyneCore module name from method name and/or object type."""
    if type_str:
        # Unwrap Persistent[X] / Series[X]
        if type_str.startswith(('Persistent[', 'Series[')):
            type_str = type_str[type_str.index('[') + 1 : type_str.rindex(']')]
        # Map type → module
        if type_str.startswith('list'):     return 'array'
        if type_str.startswith('dict'):     return 'map'
        if type_str in ('line', 'label', 'box', 'table', 'linefill',
                         'Line', 'Label', 'Box', 'Table'):
            return type_str.lower()
        if type_str.startswith(('matrix', 'Matrix')): return 'matrix'
        if '<' in type_str: return type_str[:type_str.index('<')]
        return None

    # Heuristic: infer from method name alone
    if method_name in LABEL_METHODS:  return 'label'
    if method_name in LINE_METHODS:   return 'line'
    if method_name in BOX_METHODS:    return 'box'
    if method_name in TABLE_METHODS:  return 'table'
    if method_name in MAP_METHODS:    return 'map'
    if method_name in MATRIX_METHODS: return 'matrix'
    return None
```

This replaces 3 copies of the same if/elif chain + 3 copies of the type unwrapping logic.

#### Helper B: `_transform_dotted_string_method_call(call, func) -> Optional[FunctionCall]`

Extracted from lines 1262-1307. Handles `var.method()` patterns:
1. Check @method user functions
2. Look up var type from `current_function_params` / symbol table
3. Call `_resolve_module_for_method(method, type_str)` for module resolution
4. Return `module.method(var, args...)` or None

#### Helper C: `_transform_chained_dotted_method_call(call, func) -> Optional[FunctionCall]`

Extracted from lines 1309-1351. Handles `obj.field.method()` patterns:
1. Guard: skip if first component is a known module or uppercase
2. Check `copy` special case → `udt_copy()` (BEFORE set matching)
3. Call `_resolve_module_for_method(method)` for heuristic resolution
4. Return `module.method(obj.field, args...)` or None

#### Helper D: `_transform_member_access_method_call(call, func) -> Optional[FunctionCall]`

Extracted from lines 1405-1451. Handles `MemberAccess` node-based calls:
1. Call `_resolve_module_for_method(method)` for module resolution
2. Return `module.method(obj, args...)` or None

#### Rewritten `_transform_function_call`

Becomes a ~50-line dispatcher:

```python
def _transform_function_call(self, call):
    func = call.func
    if isinstance(func, str):
        if '.' in func and '<' not in func and '.' not in func.split('.')[0]:
            result = self._transform_dotted_string_method_call(call, func)
            if result: return result

            if '.' in func:
                first = func.split('.')[0]
                if first not in KNOWN_MODULES and not first[0].isupper():
                    result = self._transform_chained_dotted_method_call(call, func)
                    if result: return result

        # Special cases (timeframe.in_seconds, array.max/min) — kept inline, ~20 lines
        # Generics, UDT.new, int/float casts, renames — kept inline, ~20 lines

    elif isinstance(func, MemberAccess):
        result = self._transform_member_access_method_call(call, func)
        if result: return result
        # Generics in MemberAccess — kept inline, ~8 lines

    # Transform args/kwargs and return
    ...
```

**Also apply to `_transform_method_call`:** Replace its inline sets (lines 1507-1511) with `_resolve_module_for_method()`. Keep the existing `copy` + MemberAccess guard (line 1485) before set-based resolution.

**Effort:** Medium. ~2 hours. Medium risk — requires careful control flow preservation. Verify with full test suite after each helper extraction.

---

### 3. Add Snapshot Tests

**Problem:** Currently `test_all_samples.py` tests "does it run without error" but doesn't catch output regressions. A refactoring that silently changes generated code would go undetected.

**Implementation:**

#### Option A: Using `pytest-snapshot` (recommended)

```bash
pip install pytest-snapshot
```

```python
# tests/test_transpile_snapshots.py

import pytest
from pathlib import Path
from pine2pyne import transpile

SAMPLE_DIR = Path(__file__).parent.parent / 'sample' / 'pinescript'
PINE_FILES = sorted(SAMPLE_DIR.glob('*.pine'))


@pytest.fixture
def snapshot_dir():
    return Path(__file__).parent / 'snapshots'


@pytest.mark.parametrize('pine_file', PINE_FILES, ids=lambda f: f.stem)
def test_transpile_snapshot(pine_file, snapshot):
    """Verify transpiled output matches stored snapshot."""
    source = pine_file.read_text()
    result = transpile(source)
    snapshot.assert_match(result, f'{pine_file.stem}.py')
```

**First run:** `pytest --snapshot-update` — creates `tests/snapshots/*.py` reference files.
**Subsequent runs:** `pytest` — fails if any output differs from snapshot.
**Intentional changes:** `pytest --snapshot-update` — review diff, accept.

#### Option B: Manual snapshot directory (no extra dependency)

```python
# tests/test_transpile_snapshots.py

import pytest
from pathlib import Path
from pine2pyne import transpile

SAMPLE_DIR = Path(__file__).parent.parent / 'sample' / 'pinescript'
SNAPSHOT_DIR = Path(__file__).parent / 'snapshots'
PINE_FILES = sorted(SAMPLE_DIR.glob('*.pine'))


@pytest.mark.parametrize('pine_file', PINE_FILES, ids=lambda f: f.stem)
def test_transpile_snapshot(pine_file):
    """Verify transpiled output matches stored snapshot."""
    source = pine_file.read_text()
    result = transpile(source)

    snapshot_path = SNAPSHOT_DIR / f'{pine_file.stem}.py'

    if not snapshot_path.exists():
        # First run: create snapshot
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(result)
        pytest.skip(f'Snapshot created: {snapshot_path.name}')

    expected = snapshot_path.read_text()
    assert result == expected, (
        f'Transpiled output differs from snapshot.\n'
        f'Run with --update-snapshots to accept changes.\n'
        f'Snapshot: {snapshot_path}'
    )
```

Add a `conftest.py` fixture to handle `--update-snapshots` flag:

```python
# tests/conftest.py

def pytest_addoption(parser):
    parser.addoption('--update-snapshots', action='store_true', default=False)

@pytest.fixture(autouse=True)
def update_snapshots_if_requested(request, pine_file=None):
    yield
    if request.config.getoption('--update-snapshots'):
        # Re-run and overwrite snapshot after test
        ...
```

#### Workflow

```bash
# Generate initial snapshots (one-time)
pytest tests/test_transpile_snapshots.py --snapshot-update

# Normal development — catch regressions
pytest tests/test_transpile_snapshots.py

# After intentional changes — review + accept
pytest tests/test_transpile_snapshots.py --snapshot-update
git diff tests/snapshots/  # Review what changed
```

#### What this catches

- Method set changes that alter which module a method resolves to
- Transformer refactoring that changes code generation order
- Import resolver changes that add/remove/reorder imports
- Any "pure refactoring" that accidentally changes output

**Effort:** Small. ~1 hour. Zero risk — additive only, no existing code changes.
