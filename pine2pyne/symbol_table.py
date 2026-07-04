"""
Symbol table for tracking variable declarations and scopes.

Used by the transformer to determine variable types and manage scope.
"""
from typing import Dict, List, Optional, Set
from enum import Enum


class VariableKind(Enum):
    """Classification of variable kinds in Pine Script."""
    VAR = "var"  # Persistent variable (var keyword)
    VARIP = "varip"  # Persistent intrabar variable
    INPUT = "input"  # Input parameter
    SERIES = "series"  # Series variable (default for globals)
    SIMPLE = "simple"  # Simple (non-series) variable
    FUNCTION = "function"  # Function name
    PARAMETER = "parameter"  # Function parameter
    TYPE = "type"  # User-defined type (UDT) declaration


class Symbol:
    """Represents a symbol (variable, function) in the symbol table."""

    def __init__(
        self,
        name: str,
        kind: VariableKind,
        type_hint: Optional[str] = None,
        is_global: bool = False,
        line: int = 0,
        column: int = 0,
    ):
        self.name = name
        self.kind = kind
        self.type_hint = type_hint
        self.is_global = is_global
        self.line = line
        self.column = column
        self.is_series = False  # Will be set by type inference
        self.is_indexed = False  # True if variable is indexed with [n]
        self.is_from_ta_function = False  # True if assigned from ta.* function


class Scope:
    """Represents a lexical scope."""

    def __init__(self, parent: Optional['Scope'] = None):
        self.parent = parent
        self.symbols: Dict[str, Symbol] = {}

    def define(self, symbol: Symbol) -> None:
        """Define a symbol in this scope."""
        self.symbols[symbol.name] = symbol

    def lookup(self, name: str) -> Optional[Symbol]:
        """Look up a symbol in this scope or parent scopes."""
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

    def lookup_local(self, name: str) -> Optional[Symbol]:
        """Look up a symbol only in this scope (not parent)."""
        return self.symbols.get(name)


class SymbolTable:
    """Manages symbol definitions and scopes."""

    def __init__(self):
        self.global_scope = Scope()
        self.current_scope = self.global_scope
        self.builtin_symbols = self._init_builtins()

    def _init_builtins(self) -> Set[str]:
        """Initialize built-in Pine Script variables."""
        return {
            # Price data
            'open', 'high', 'low', 'close', 'volume',
            'hl2', 'hlc3', 'ohlc4', 'hlcc4',
            # Bar data
            'bar_index', 'time', 'timenow',
            # Bar state
            'barstate.isfirst', 'barstate.islast', 'barstate.ishistory',
            'barstate.isrealtime', 'barstate.isnew', 'barstate.isconfirmed',
            # Constants
            'na', 'true', 'false',
            # Color constants
            'color.red', 'color.green', 'color.blue', 'color.white', 'color.black',
            'color.yellow', 'color.orange', 'color.purple', 'color.gray',
            # Plot styles
            'plot.style_line', 'plot.style_stepline', 'plot.style_histogram',
            'plot.style_cross', 'plot.style_area', 'plot.style_columns',
            'plot.style_circles', 'plot.style_linebr', 'plot.style_steplinebr',
            # Strategy constants
            'strategy.long', 'strategy.short',
            'strategy.commission.percent', 'strategy.commission.cash_per_contract',
            'strategy.commission.cash_per_order',
            'strategy.percent', 'strategy.fixed', 'strategy.cash',
            'strategy.percent_of_equity',
        }

    def enter_scope(self) -> None:
        """Enter a new scope."""
        self.current_scope = Scope(parent=self.current_scope)

    def exit_scope(self) -> None:
        """Exit the current scope."""
        if self.current_scope.parent:
            self.current_scope = self.current_scope.parent

    def define(self, symbol: Symbol) -> None:
        """Define a symbol in the current scope."""
        self.current_scope.define(symbol)

    def lookup(self, name: str) -> Optional[Symbol]:
        """Look up a symbol in current or parent scopes."""
        # Check builtins first
        if name in self.builtin_symbols:
            return Symbol(name, VariableKind.SERIES, is_global=True)

        return self.current_scope.lookup(name)

    def lookup_global(self, name: str) -> Optional[Symbol]:
        """Look up a symbol only in global scope."""
        return self.global_scope.lookup_local(name)

    def is_builtin(self, name: str) -> bool:
        """Check if name is a built-in symbol."""
        return name in self.builtin_symbols

    def get_all_globals(self) -> List[Symbol]:
        """Get all symbols defined in global scope."""
        return list(self.global_scope.symbols.values())

    def mark_as_indexed(self, name: str) -> None:
        """Mark a variable as being indexed (e.g., close[1])."""
        symbol = self.lookup(name)
        if symbol:
            symbol.is_indexed = True
            symbol.is_series = True  # Indexed variables are always Series

    def mark_as_series(self, name: str) -> None:
        """Mark a variable as a Series type."""
        symbol = self.lookup(name)
        if symbol:
            symbol.is_series = True

    def mark_from_ta_function(self, name: str) -> None:
        """Mark a variable as assigned from a ta.* function."""
        symbol = self.lookup(name)
        if symbol:
            symbol.is_from_ta_function = True
            symbol.is_series = True  # ta.* functions return Series
