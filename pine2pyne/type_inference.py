"""
Type inference for Pine Script variables.

Determines whether variables should be Series[T] or Persistent[T] in PyneCore.
"""
from typing import Set
from .ast_nodes import *
from .symbol_table import SymbolTable, Symbol, VariableKind
from .pine_builtins import get_type_name


class TypeInference:
    """Analyzes AST to infer variable types (Series vs Persistent vs simple)."""

    def __init__(self, symbol_table: SymbolTable):
        self.symbol_table = symbol_table
        self.ta_functions = self._init_ta_functions()

    def _init_ta_functions(self) -> Set[str]:
        """Initialize complete set of ta.* functions/variables (Pine v5/v6)."""
        return {
            # Moving averages
            'ta.sma', 'ta.ema', 'ta.wma', 'ta.vwma', 'ta.swma', 'ta.alma',
            'ta.rma', 'ta.hma', 'ta.linreg',
            # Oscillators & indicators
            'ta.rsi', 'ta.macd', 'ta.stoch', 'ta.cci', 'ta.cmo', 'ta.mfi',
            'ta.mom', 'ta.roc', 'ta.tsi', 'ta.wpr',
            # Volatility
            'ta.atr', 'ta.tr', 'ta.bb', 'ta.bbw', 'ta.kc', 'ta.kcw',
            # Trend
            'ta.dmi', 'ta.supertrend', 'ta.sar', 'ta.cog',
            # Volume-based (functions and built-in variables)
            'ta.obv', 'ta.nvi', 'ta.pvi', 'ta.pvt', 'ta.vwap',
            'ta.accdist', 'ta.iii', 'ta.wad', 'ta.wvad',
            # Statistical
            'ta.dev', 'ta.stdev', 'ta.variance', 'ta.correlation',
            'ta.median', 'ta.mode', 'ta.range', 'ta.cum',
            'ta.percentile_linear_interpolation', 'ta.percentile_nearest_rank',
            'ta.percentrank',
            # Lookback
            'ta.highest', 'ta.lowest', 'ta.highestbars', 'ta.lowestbars',
            'ta.valuewhen', 'ta.barssince', 'ta.change',
            # Pivot
            'ta.pivothigh', 'ta.pivotlow',
            # Boolean series
            'ta.rising', 'ta.falling', 'ta.cross', 'ta.crossover', 'ta.crossunder',
        }

    def infer_types(self, script: Script) -> None:
        """Infer types for all variables in the script."""
        # First pass: mark variables with var/varip as Persistent
        self._mark_persistent_variables(script)

        # Second pass: mark variables from ta.* functions as Series
        self._mark_ta_series_variables(script)

        # Third pass: mark indexed variables as Series
        self._mark_indexed_variables(script)

        # Fourth pass: mark global non-var variables as Series (PRIMARY RULE)
        self._mark_global_series_variables(script)

        # Fifth pass: propagate Series type from assignments
        self._propagate_series_types(script)

    def _mark_persistent_variables(self, script: Script) -> None:
        """Mark all var/varip variables as Persistent."""
        for decl in script.declarations:
            if isinstance(decl, (VarDecl, VaripDecl)):
                symbol = self.symbol_table.lookup(decl.name)
                if symbol:
                    # These are already marked as VAR/VARIP in symbol table
                    symbol.is_series = False  # Not Series, will be Persistent

    def _mark_ta_series_variables(self, script: Script) -> None:
        """Mark variables assigned from ta.* functions as Series."""
        # Check global assignments
        for stmt in script.body:
            if isinstance(stmt, Assignment):
                if self._is_ta_function_call(stmt.value):
                    if isinstance(stmt.target, str):
                        self.symbol_table.mark_from_ta_function(stmt.target)

    def _mark_indexed_variables(self, script: Script) -> None:
        """Mark variables that are indexed (e.g., close[1]) as Series."""
        self.indexed_var_names: set[str] = set()
        self._find_indexed_in_node(script)

    def _find_indexed_in_node(self, node: Any) -> None:
        """Recursively find indexed access patterns."""
        if isinstance(node, IndexAccess):
            if isinstance(node.object, Identifier):
                self.indexed_var_names.add(node.object.name)
                self.symbol_table.mark_as_indexed(node.object.name)

        # Recursively check all child nodes
        if isinstance(node, (list, tuple)):
            for item in node:
                self._find_indexed_in_node(item)
        elif isinstance(node, dict):
            for item in node.values():
                self._find_indexed_in_node(item)
        elif hasattr(node, '__dict__'):
            for attr_value in node.__dict__.values():
                if isinstance(attr_value, (ASTNode, list, tuple, dict)):
                    self._find_indexed_in_node(attr_value)

    def _mark_global_series_variables(self, script: Script) -> None:
        """Mark all global non-var, non-input variables as Series (PRIMARY RULE)."""
        for symbol in self.symbol_table.get_all_globals():
            # Skip var/varip (they're Persistent)
            if symbol.kind in (VariableKind.VAR, VariableKind.VARIP):
                continue

            # Skip inputs (they become function parameters)
            if symbol.kind == VariableKind.INPUT:
                continue

            # Skip functions
            if symbol.kind == VariableKind.FUNCTION:
                continue

            # Everything else is Series by default
            if symbol.is_global:
                symbol.is_series = True

    def _propagate_series_types(self, script: Script) -> None:
        """Propagate Series type from one variable to another in assignments."""
        for stmt in script.body:
            if isinstance(stmt, Assignment):
                if isinstance(stmt.target, str):
                    # Check if value is a Series variable
                    if isinstance(stmt.value, Identifier):
                        source_symbol = self.symbol_table.lookup(stmt.value.name)
                        if source_symbol and source_symbol.is_series:
                            self.symbol_table.mark_as_series(stmt.target)

    def _is_ta_function_call(self, expr: Expression) -> bool:
        """Check if expression is a ta.* function call."""
        if isinstance(expr, FunctionCall):
            if isinstance(expr.func, str):
                return expr.func in self.ta_functions
            elif isinstance(expr.func, MemberAccess):
                func_name = f"{expr.func.object}.{expr.func.member}"
                return func_name in self.ta_functions
        return False

    def infer_type_hint(self, value: Expression) -> str:
        """Infer the Python type hint for an expression."""
        if isinstance(value, Literal):
            type_map = {
                'int': 'int',
                'float': 'float',
                'string': 'str',
                'bool': 'bool',
                'color': 'Color',
            }
            return type_map.get(value.literal_type, 'Any')

        if isinstance(value, NaLiteral):
            return 'Any'  # NA can be any type

        if isinstance(value, UnaryOp):
            if value.op == 'not':
                return 'bool'
            # Unary +/- preserves operand type
            return self.infer_type_hint(value.operand)

        if isinstance(value, BinaryOp):
            # Boolean operations always return bool
            if value.op in ('and', 'or', '==', '!=', '>', '<', '>=', '<='):
                return 'bool'

            # Infer from operands
            left_type = self.infer_type_hint(value.left)
            right_type = self.infer_type_hint(value.right)

            # If either is float, result is float
            if left_type == 'float' or right_type == 'float':
                return 'float'

            # If both are int, result is int (except for division)
            if left_type == 'int' and right_type == 'int':
                if value.op == '/':
                    return 'float'
                return 'int'

            return 'Any'

        if isinstance(value, TernaryOp):
            # Type is union of true and false branches
            true_type = self.infer_type_hint(value.true_expr)
            false_type = self.infer_type_hint(value.false_expr)
            if true_type == false_type:
                return true_type
            # If mixed int/float, return float
            if {true_type, false_type} == {'int', 'float'}:
                return 'float'
            return 'Any'

        if isinstance(value, FunctionCall):
            # Check if it's a ta.* function
            func_name = None
            if isinstance(value.func, str):
                func_name = value.func
            elif isinstance(value.func, MemberAccess):
                func_name = f"{value.func.object}.{value.func.member}"

            if func_name and func_name in self.ta_functions:
                return 'float'  # Most ta.* functions return float series

            # input.* functions
            if func_name and func_name.startswith('input.'):
                input_type_map = {
                    'input.int': 'int',
                    'input.float': 'float',
                    'input.bool': 'bool',
                    'input.string': 'str',
                    'input.color': 'Color',
                    'input.source': 'float',  # Sources are float series
                }
                return input_type_map.get(func_name, 'Any')

            # PyneCore collection constructors
            if func_name:
                # matrix.new<float>() -> Matrix[float]
                if func_name.startswith('matrix.new'):
                    # Check for generic type in func_name: matrix.new<float>
                    if '<' in func_name and '>' in func_name:
                        generic = func_name[func_name.index('<')+1:func_name.index('>')]
                        return f'Matrix[{generic}]'
                    return 'Matrix[float]'  # Default to float

                # array.new_*() -> list[type]
                if func_name.startswith('array.new_'):
                    array_type_map = {
                        'array.new_int': 'list[int]',
                        'array.new_float': 'list[float]',
                        'array.new_bool': 'list[bool]',
                        'array.new_string': 'list[str]',
                        'array.new_color': 'list[Color]',
                        'array.new_line': 'list[Line]',
                        'array.new_label': 'list[Label]',
                        'array.new_box': 'list[Box]',
                        'array.new_table': 'list[Table]',
                    }
                    return array_type_map.get(func_name, 'list[Any]')

                # map.new<K,V>() -> dict[K, V]
                if func_name.startswith('map.new'):
                    if '<' in func_name and '>' in func_name:
                        generic = func_name[func_name.index('<')+1:func_name.index('>')]
                        parts = [p.strip() for p in generic.split(',')]
                        if len(parts) == 2:
                            k = get_type_name(parts[0])
                            v = get_type_name(parts[1])
                            return f'dict[{k}, {v}]'
                    return 'dict[str, float]'  # Default generic

                # map.get -> float (most common map value type in Pine strategies)
                if func_name == 'map.get':
                    return 'float'

                # map.copy(m) / map.keys(m) / map.values(m) -> same type
                if func_name == 'map.copy':
                    return 'dict[str, float]'  # Copy returns a map
                if func_name in ('map.keys', 'array.copy'):
                    return 'list[Any]'
                if func_name == 'map.values':
                    return 'list[Any]'

                # label.new(), line.new(), box.new(), table.new()
                if func_name in ('label.new', 'line.new', 'box.new', 'table.new'):
                    type_name = func_name.split('.')[0].capitalize()
                    return type_name

            # Check if it's a UDT constructor call (e.g., TradeStats())
            if isinstance(value.func, str):
                # Check if the function name is a known UDT type
                symbol = self.symbol_table.lookup(value.func)
                if symbol and symbol.kind == VariableKind.TYPE:
                    return value.func  # Return the UDT type name
                # Fallback: if function name starts with uppercase, assume it's a UDT
                elif value.func and value.func[0].isupper():
                    return value.func

        # Handle MethodCall (e.g., TradeStats.new() before transformation)
        if isinstance(value, MethodCall):
            # Check if it's UDT.new() pattern
            if value.method == 'new' and isinstance(value.object, Identifier):
                # Check if the object is a UDT type
                if value.object.name[0].isupper():
                    return value.object.name  # Return just the UDT name without .new suffix

        # Handle FunctionCall for UDT constructors (after transformation: TradeStats())
        if isinstance(value, FunctionCall):
            if isinstance(value.func, str) and value.func and value.func[0].isupper():
                # Already a constructor call - return the type name directly
                return value.func

        if isinstance(value, Identifier):
            symbol = self.symbol_table.lookup(value.name)
            if symbol and symbol.type_hint:
                return symbol.type_hint

        return 'Any'
