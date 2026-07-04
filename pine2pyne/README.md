# pine2pyne: Pine Script v6 → PyneCore Python Transpiler

A complete transpiler that converts TradingView Pine Script v6 code to PyneCore Python for local backtesting and analysis.

## Features

- **Full Pine Script v6 Support**: Lexer, parser, and transformer for Pine Script v6 syntax
- **Comprehensive Transformations**: Implements all 54 transformation rules from spec
- **Smart Type Inference**: Automatically determines Series vs Persistent types
- **Import Resolution**: Generates optimal import statements for PyneCore
- **CLI Interface**: Command-line tool for batch conversion

## Architecture

```
pine2pyne/
├── lexer.py              # Tokenizer with indentation tracking
├── tokens.py             # Token type definitions
├── parser.py             # Recursive descent parser
├── ast_nodes.py          # AST node definitions (50+ node types)
├── transformer.py        # Pine → PyneCore transformation (1626 lines)
├── codegen.py            # Python code generator
├── symbol_table.py       # Variable scope tracking
├── type_inference.py     # Series/Persistent type inference
├── import_resolver.py    # Import statement generation
├── pine_builtins.py      # Pine Script built-ins mapping + canonical method sets
├── errors.py             # Error types with source locations
├── cli.py                # Command-line interface
├── __init__.py          # Package initialization
└── __main__.py          # Module entry point
```

### Transpilation Pipeline (4 stages)

```
Source Code (.pine)
        ↓
    [Lexer]        → Tokens with INDENT/DEDENT
        ↓
    [Parser]       → Pine Script AST (50+ node types)
        ↓
  [Transformer]    → PyneCore-compatible AST
        ↓
  [Code Generator] → Python source code (.py)
```

**Key Components:**

1. **Lexer** (`lexer.py`) - Converts Pine Script source into tokens with indentation tracking
2. **Parser** (`parser.py`) - Recursive descent parser building an AST from tokens
3. **Transformer** (`transformer.py`) - Applies 54 transformation rules to convert Pine AST to PyneCore AST
   - Visitor pattern for AST traversal (`_visit_ast_nodes()`)
   - Centralized module resolution (`_resolve_module_for_method()`)
   - Focused method extraction for maintainability
4. **Code Generator** (`codegen.py`) - Emits Python source code from transformed AST
5. **Symbol Table** (`symbol_table.py`) - Tracks variable scopes and declarations
6. **Type Inference** (`type_inference.py`) - Determines Series vs Persistent types
7. **Import Resolver** (`import_resolver.py`) - Generates optimal PyneCore imports
8. **Pine Builtins** (`pine_builtins.py`) - Canonical mappings for functions, modules, types, and method sets

## Usage

### As a Python Module

```python
from pine2pyne import transpile

pine_script = """
//@version=6
indicator("Simple MA", overlay=true)

length = input.int(14, "Length")
ma = ta.sma(close, length)
plot(ma, "MA", color=color.blue)
"""

python_code = transpile(pine_script)
print(python_code)
```

### Command Line

```bash
# RECOMMENDED: Use as Python module
python -m pine2pyne input.pine                    # Output to stdout
python -m pine2pyne input.pine -o output.py       # Output to file
python -m pine2pyne scripts/*.pine -o output/     # Batch convert

# Alternative: Direct CLI script
python pynecore/pine2pyne/cli.py strategy.pine
python pynecore/pine2pyne/cli.py strategy.pine -o strategy.py

# Validate syntax only
python -m pine2pyne strategy.pine --validate
```

**Command Line Usage (Detailed):**

```bash
# From pynecore/ directory:
cd pynecore

# Convert single file to stdout
python -m pine2pyne other_transpiler/Sample.pine

# Convert and save output
python -m pine2pyne other_transpiler/Sample.pine -o output/Sample.py

# Batch convert all .pine files in a directory
python -m pine2pyne examples/*.pine -o output/

# Validate without converting
python -m pine2pyne strategy.pine --validate
```

## Transformation Rules

Implements all 54 rules including:

- **Structure**: `//@version=6` → `@pyne` docstring, `indicator()`/`strategy()` → decorators
- **Variables**: `var` → `Persistent[T]`, global variables → `Series[T]`
- **Inputs**: Extracted to `main()` parameters with type hints
- **Control Flow**: `? :` → `if/else`, `:=` → `=`, `for/to/by` → `range()`
- **Literals**: `true`/`false` → `True`/`False`, `na` → `NA()`
- **Functions**: Implicit returns → explicit `return`, `=>` → `def`
- **Modules**: `str` → `string`, `array.from` → `array.from_items`
- **Types**: Drawing types (`line`→`Line`), basic types (`string`→`str`)
- **Constants**: Plot styles, strategy constants, colors preserved

## Example Transpilation

**Input (Pine Script):**
```pinescript
//@version=6
indicator("Simple Moving Average", overlay=true)

length = input.int(14, "Length", minval=1)
source = input.source(close, "Source")

ma = ta.sma(source, length)

plot(ma, "MA", color=color.blue)
```

**Output (PyneCore Python):**
```python
"""
@pyne
"""
from pynecore import Series
from pynecore.lib import input, script
from pynecore.lib import close

@script.indicator("Simple Moving Average", overlay=True)
def main(
    length: int = input.int(14, minval=1, title='Length'),
    source: Series[float] = input.source(close, title='Source')
):
    ma: Series[float] = ta.sma(source, length)
    plot(ma, 'MA', color=color.blue)
```

## Current Status

**Production Ready**: samples 324/324 pass (100%), PDS 231/236 pass (97.9%).

Remaining PDS failures: label.new/box.new argument pattern mismatches (3), array.new_label not implemented (2).

```bash
# Run tests
cd pynecore && python test_all_samples.py && python test_all_sample_pds.py
```

## Known Limitations

- Import statements (`import user/lib/version`) converted to comments
- Negative indices (`close[-1]`) preserved (will error in PyneCore runtime)
- 5 PDS edge cases: `label.new`/`box.new` argument patterns, `array.new_label` not implemented
