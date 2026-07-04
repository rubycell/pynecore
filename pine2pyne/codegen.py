"""
Code generator: Emit Python source code from transformed AST.

Generates PEP 8 compliant Python code compatible with PyneCore.
"""
from typing import List, Union
from .ast_nodes import *
from .transformer import PyneTransformedScript
from .errors import CodeGenError


class CodeGenerator:
    """Generates Python source code from transformed AST."""

    def __init__(self, indent_size: int = 4):
        self.indent_size = indent_size
        self.indent_level = 0
        self.output: List[str] = []

    @staticmethod
    def _esc(name: str) -> str:
        """Escape Python reserved words used as identifiers."""
        from .pine_builtins import sanitize_identifier
        return sanitize_identifier(name)

    def _get_all_type_hints(self, script) -> List[str]:
        """Extract all type hints from script for Any detection."""
        hints = []

        # Check main parameters
        for param in script.main_params:
            if param.type_hint:
                hints.append(param.type_hint)

        # Check main body statements
        for stmt in script.main_body:
            if hasattr(stmt, 'type_hint') and stmt.type_hint:
                hints.append(stmt.type_hint)

        # Check global vars
        for var in script.global_vars:
            if hasattr(var, 'type_hint') and var.type_hint:
                hints.append(var.type_hint)

        # Check helper functions
        for func in script.helper_functions:
            if func.return_type:
                hints.append(func.return_type)
            for param in func.params:
                if param.type_hint:
                    hints.append(param.type_hint)
            # Check function body (handle both list and single statement)
            body = func.body if isinstance(func.body, list) else [func.body]
            for stmt in body:
                if hasattr(stmt, 'type_hint') and stmt.type_hint:
                    hints.append(stmt.type_hint)

        return hints

    def generate(self, script: PyneTransformedScript) -> str:
        """Generate complete Python source code."""
        self.output = []
        self._varname_to_slot = {name: (slot, deps) for slot, (name, deps) in script.cache_slots.items()}
        self._num_cache_slots = script.num_cache_slots

        # 1. @pyne magic docstring with attribution
        self.emit('"""')
        self.emit('@pyne')
        self.emit('')
        self.emit('This code was compiled by Pine2Pyne — the Pine Script to PyneCore\'s Python compiler.')        
        self.emit('"""')
        self.emit('')

        # 2. Import statements
        # Check if Any is used in type hints
        uses_any = any('[Any]' in stmt or ': Any' in stmt for stmt in self._get_all_type_hints(script))

        if uses_any:
            self.emit('from typing import Any')

        for import_stmt in script.imports:
            self.emit(import_stmt)
        if script.imports or uses_any:
            self.emit('')

        # 2b. Variable cache support for pyne optimize
        if self._num_cache_slots > 0:
            self.emit('try:')
            self.emit('    from pynecore.core import _var_cache as _vcm')
            self.emit('except ImportError:')
            self.emit('    _vcm = None')
            self.emit('')
            # Emit __var_deps__ dict
            deps_parts = []
            for slot in sorted(script.cache_slots):
                _, deps = script.cache_slots[slot]
                if deps:
                    items = ', '.join(f"'{d}'" for d in sorted(deps))
                    deps_parts.append(f'{slot}: frozenset({{{items}}})')
                else:
                    deps_parts.append(f'{slot}: frozenset()')
            self.emit(f'__var_deps__ = {{{", ".join(deps_parts)}}}')
            self.emit(f'__num_cache_slots__ = {self._num_cache_slots}')
            self.emit('')

        # 3. Type declarations (custom types with @udt)
        for type_decl in script.type_declarations:
            self.generate_type_declaration(type_decl)
            self.emit('')

        # 3b. Enum declarations
        for enum_decl in script.enum_declarations:
            self.generate_enum_declaration(enum_decl)
            self.emit('')

        # 3c. Module-level constants (e.g., const string group labels, pre-main literals)
        # These must be at module level because they are referenced by main()
        # parameter defaults which Python evaluates at function definition time.
        if script.module_constants:
            for const_stmt in script.module_constants:
                self.generate_statement(const_stmt)
            self.emit('')

        # 4. Main function with decorator (helper functions now inside main)
        if script.main_decorator:
            decorator = self.generate_decorator(script.main_decorator)
            self.emit(decorator)

        # Generate main function signature
        self.emit('def main(')
        self.indent_level += 1

        # Generate parameters
        if script.main_params:
            for i, param in enumerate(script.main_params):
                param_str = self.generate_parameter(param)
                if i < len(script.main_params) - 1:
                    param_str += ','
                self.emit(param_str)
        else:
            # No parameters - but Python requires at least empty ()
            pass

        self.indent_level -= 1

        # Close parameter list and add colon
        if script.main_params:
            self.emit('):')
        else:
            self.output[-1] += '):'  # Add closing paren and colon to def line

        self.indent_level += 1

        # Emit variable cache locals at top of main body
        if self._num_cache_slots > 0:
            self.emit('_vc = _vcm._data if _vcm else None')
            self.emit('_vb = _vcm._build if _vcm else None')

        # Generate global var declarations (Persistent)
        for var_decl in script.global_vars:
            self.generate_statement(var_decl)

        # Generate main body with blank lines between statements
        # Rule: add blank line after function defs and between different statement kinds
        prev_kind = None  # 'persistent', 'func', 'body'
        prev_source_line = 0
        for i, stmt in enumerate(script.main_body):
            source_line = getattr(stmt, 'line', 0)

            # Determine statement kind
            if isinstance(stmt, Assignment) and isinstance(stmt.type_hint, str) and 'Persistent' in stmt.type_hint:
                kind = 'persistent'
            elif isinstance(stmt, FuncDecl):
                kind = 'func'
            else:
                kind = 'body'

            # Add blank lines based on context
            if i > 0:
                gap = source_line - prev_source_line if source_line > 0 and prev_source_line > 0 else 0

                if prev_kind == 'func':
                    # Always blank line after function definition
                    self.emit('')
                elif prev_kind != kind:
                    # Transition between statement types
                    if gap > 5:
                        # Large gap (section change) - add extra blank lines
                        blanks = min((gap - 1) // 10 + 1, 3)
                        for _ in range(blanks):
                            self.emit('')
                    else:
                        self.emit('')
                elif kind == 'persistent' and gap >= 3:
                    # Gap between Persistent groups (e.g., maps → arrays)
                    self.emit('')
                elif kind == 'body' and gap >= 3:
                    # Gap between body statements (comment/blank line in source)
                    self.emit('')

            self.generate_statement(stmt)
            prev_kind = kind
            prev_source_line = source_line

        self.indent_level -= 1

        return '\n'.join(self.output)

    # ========================================================================
    # Emission helpers
    # ========================================================================

    def emit(self, line: str = '') -> None:
        """Emit a line of code with current indentation."""
        if line:
            indent = ' ' * (self.indent_level * self.indent_size)
            self.output.append(indent + line)
        else:
            self.output.append('')

    def emit_inline(self, text: str) -> None:
        """Emit text without newline (append to last line)."""
        if self.output:
            self.output[-1] += text
        else:
            self.output.append(text)

    # ========================================================================
    # Top-level code generation
    # ========================================================================

    def generate_decorator(self, decl: Union[IndicatorDecl, StrategyDecl]) -> str:
        """Generate @script.indicator or @script.strategy decorator."""
        # Escape single quotes in title to avoid breaking the string literal
        title = decl.title.replace("'", "\\'") if decl.title else ''
        if isinstance(decl, IndicatorDecl):
            decorator = f"@script.indicator('{title}'"
        else:
            decorator = f"@script.strategy('{title}'"

        # Add kwargs
        if decl.kwargs:
            for key, value in decl.kwargs.items():
                decorator += f', {key}={self.generate_decorator_value(value)}'

        decorator += ')'
        return decorator

    def generate_decorator_value(self, value: Any) -> str:
        """Generate decorator kwarg value."""
        if isinstance(value, bool):
            return 'True' if value else 'False'
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, Literal):
            return self.generate_literal(value)
        elif isinstance(value, MemberAccess):
            return self.generate_member_access(value)
        elif isinstance(value, Identifier):
            return value.name
        elif isinstance(value, Expression):
            # Handle any other expression type
            return self.generate_expression(value)
        else:
            return str(value)

    def generate_parameter(self, param: Parameter) -> str:
        """Generate function parameter."""
        param_str = self._esc(param.name)

        # Add type hint
        if param.type_hint:
            param_str += f': {param.type_hint}'

        # Add default value
        if param.default:
            default_str = self.generate_expression(param.default)
            # Fix NA() to na for function default parameters
            if default_str == 'NA()':
                default_str = 'na'
            # PEP 8: spaces around = in defaults when type annotation present
            if param.type_hint:
                param_str += f' = {default_str}'
            else:
                param_str += f'={default_str}'

        return param_str

    # ========================================================================
    # Type declaration generation
    # ========================================================================

    def generate_type_declaration(self, type_decl: TypeDecl) -> None:
        """Generate custom type definition with @udt decorator."""
        from .transformer import convert_pine_generic_to_python
        from .pine_builtins import get_type_name

        # Emit @udt decorator
        self.emit('@udt')
        self.emit(f'class {self._esc(type_decl.name)}:')
        self.indent_level += 1

        # Generate fields
        for field_name, field_type, default_value in type_decl.fields:
            # Convert Pine type to Python type (handles generics like array<T> → list[T])
            py_type = convert_pine_generic_to_python(field_type)
            fname = self._esc(field_name)

            # For NA() calls, use base type only (NA(list) not NA(list[X]))
            na_type = py_type.split('[')[0] if '[' in py_type else py_type

            # Generate field with type annotation and default
            if default_value:
                if isinstance(default_value, NaLiteral):
                    # na default becomes NA(type)
                    self.emit(f'{fname}: {py_type} = NA({na_type})')
                else:
                    default_str = self.generate_expression(default_value)
                    self.emit(f'{fname}: {py_type} = {default_str}')
            else:
                # No default - use type's zero value or na()
                if py_type == 'int':
                    self.emit(f'{fname}: {py_type} = 0')
                elif py_type == 'float':
                    self.emit(f'{fname}: {py_type} = 0.0')
                elif py_type == 'str':
                    # str type should use NA(str) not empty string
                    self.emit(f'{fname}: {py_type} = NA({py_type})')
                elif py_type == 'bool':
                    self.emit(f'{fname}: {py_type} = False')
                else:
                    self.emit(f'{fname}: {py_type} = NA({na_type})')

        self.indent_level -= 1

    def generate_enum_declaration(self, enum_decl: EnumDecl) -> None:
        """Generate enum as a plain class with string constants."""
        self.emit(f'class {self._esc(enum_decl.name)}:')
        self.indent_level += 1

        for member in enum_decl.members:
            if isinstance(member, tuple):
                name, value = member
                value_str = self.generate_expression(value)
                self.emit(f'{self._esc(name)} = {value_str}')
            else:
                # No explicit value — use member name as string value
                self.emit(f'{self._esc(member)} = "{member}"')

        self.indent_level -= 1

    # ========================================================================
    # Function generation
    # ========================================================================

    def generate_function(self, func: FuncDecl) -> None:
        """Generate function definition."""
        # Emit @method decorator if this is a method
        if hasattr(func, 'is_method') and func.is_method:
            self.emit('@method')

        # Function signature
        params = ', '.join(self.generate_parameter(p) for p in func.params)
        self.emit(f'def {self._esc(func.name)}({params}):')

        self.indent_level += 1

        # Track if we emitted any body content
        body_was_empty = False

        # Function body
        if isinstance(func.body, (list, tuple)):
            for i, stmt in enumerate(func.body):
                is_last = (i == len(func.body) - 1)
                # Last expression in body is the return value
                if is_last and isinstance(stmt, Expression):
                    self.emit(f'return {self.generate_expression(stmt)}')
                else:
                    self.generate_statement(stmt)
            # If body list was empty or all statements were None, add pass
            if not func.body or all(s is None for s in func.body):
                self.emit('pass')
                body_was_empty = True
        elif isinstance(func.body, Expression):
            # Single expression - add return
            self.emit(f'return {self.generate_expression(func.body)}')
        elif isinstance(func.body, ReturnStatement):
            self.generate_statement(func.body)
        else:
            # Empty or None body - add pass
            self.emit('pass')
            body_was_empty = True

        self.indent_level -= 1

    # ========================================================================
    # Statement generation
    # ========================================================================

    def generate_statement(self, stmt: Statement) -> None:
        """Generate a statement."""
        # Flatten lists of statements (from transforms that expand one stmt into multiple)
        if isinstance(stmt, list):
            for s in stmt:
                self.generate_statement(s)
            return
        if isinstance(stmt, FuncDecl):
            # Functions can now appear inside main()
            self.generate_function(stmt)
            self.emit('')  # Add blank line after function
        elif isinstance(stmt, Assignment):
            self.generate_assignment(stmt)
        elif isinstance(stmt, IfStatement):
            self.generate_if_statement(stmt)
        elif isinstance(stmt, ForLoop):
            self.generate_for_loop(stmt)
        elif isinstance(stmt, ForInLoop):
            self.generate_for_in_loop(stmt)
        elif isinstance(stmt, WhileLoop):
            self.generate_while_loop(stmt)
        elif isinstance(stmt, BreakStatement):
            self.emit('break')
        elif isinstance(stmt, ContinueStatement):
            self.emit('continue')
        elif isinstance(stmt, ReturnStatement):
            self.generate_return_statement(stmt)
        elif isinstance(stmt, ExpressionStatement):
            self.emit(self.generate_expression(stmt.expr))
        elif isinstance(stmt, RawCode):
            for line in stmt.code.split('\n'):
                self.emit(line)
        else:
            self.emit(f'# TODO: Unsupported statement type: {type(stmt).__name__}')

    def generate_assignment(self, stmt: Assignment) -> None:
        """Generate assignment statement."""
        from .ast_nodes import BinaryOp, Identifier, MemberAccess, TupleDestructure

        # Handle tuple destructuring: [a, b, c] = expr
        if isinstance(stmt.target, TupleDestructure):
            names = ', '.join(self._esc(n) for n in stmt.target.names)
            value = self.generate_expression(stmt.value)
            type_hint = f': {stmt.type_hint}' if stmt.type_hint else ''
            self.emit(f'{names}{type_hint} = {value}')
            return

        # Check for variable cache slot (pyne optimize support)
        if isinstance(stmt.target, str) and stmt.target in self._varname_to_slot:
            slot, _ = self._varname_to_slot[stmt.target]
            target = self._esc(stmt.target)
            value_expr = self.generate_expression(stmt.value)
            cache_expr = f'_vc[{slot}][int(bar_index)] if _vc is not None and _vc[{slot}] is not None else {value_expr}'
            if stmt.type_hint:
                self.emit(f'{target}: {stmt.type_hint} = {cache_expr}')
            else:
                self.emit(f'{target} = {cache_expr}')
            self.emit(f'if _vb is not None and _vb[{slot}] is not None: _vb[{slot}].append({target})')
            return

        target = self._esc(stmt.target if isinstance(stmt.target, str) else str(stmt.target))

        # Check if this is a compound assignment (x = x + y) that should be (x += y)
        if isinstance(stmt.value, BinaryOp) and not stmt.type_hint:
            left_matches = False

            # Check if the left side of the BinaryOp matches the target
            if isinstance(stmt.value.left, Identifier):
                left_matches = (stmt.value.left.name == target)
            elif isinstance(stmt.value.left, MemberAccess):
                # For member access like this.wins, reconstruct the full name
                left_name = f"{stmt.value.left.object}.{stmt.value.left.member}"
                left_matches = (left_name == target)

            # If it's a compound assignment pattern, generate +=, -=, etc.
            if left_matches and stmt.value.op in ['+', '-', '*', '/', '%']:
                compound_op = f'{stmt.value.op}='
                right_value = self.generate_expression(stmt.value.right)
                self.emit(f'{target} {compound_op} {right_value}')
                return

        # Regular assignment
        value = self.generate_expression(stmt.value)

        # Wrap string literals assigned to Color type as Color('...')
        if stmt.type_hint == 'Color' and value.startswith("'") and value.endswith("'"):
            value = f'Color({value})'

        # Add type hint if present
        if stmt.type_hint:
            self.emit(f'{target}: {stmt.type_hint} = {value}')
        else:
            self.emit(f'{target} = {value}')

    def generate_if_statement(self, stmt: IfStatement) -> None:
        """Generate if statement."""
        condition = self.generate_expression(stmt.condition)
        self.emit(f'if {condition}:')

        self.indent_level += 1
        for s in stmt.body:
            self.generate_statement(s)
        self.indent_level -= 1

        # Generate elif clauses
        for elif_cond, elif_body in stmt.elseifs:
            elif_condition = self.generate_expression(elif_cond)
            self.emit(f'elif {elif_condition}:')
            self.indent_level += 1
            for s in elif_body:
                self.generate_statement(s)
            self.indent_level -= 1

        # Generate else clause
        if stmt.else_body:
            self.emit('else:')
            self.indent_level += 1
            for s in stmt.else_body:
                self.generate_statement(s)
            self.indent_level -= 1

    def generate_for_loop(self, stmt: ForLoop) -> None:
        """Generate for loop with pine_range()."""
        # Convert Pine 'for i = from to to_val [by step]' to Python pine_range()
        from_val = self.generate_expression(stmt.from_val)
        to_val = self.generate_expression(stmt.to_val)

        # Pine's 'to' is inclusive, pine_range handles this correctly
        # No need to add 1 like we did with range()
        var = self._esc(stmt.var)
        if stmt.step:
            step = self.generate_expression(stmt.step)
            self.emit(f'for {var} in pine_range({from_val}, {to_val}, {step}):')
        else:
            self.emit(f'for {var} in pine_range({from_val}, {to_val}):')

        self.indent_level += 1
        for s in stmt.body:
            self.generate_statement(s)
        self.indent_level -= 1

    def generate_for_in_loop(self, stmt: ForInLoop) -> None:
        """Generate for-in loop."""
        vars_str = ', '.join(self._esc(v) for v in stmt.vars)
        iterable = self.generate_expression(stmt.iterable)

        # If multiple vars, it's tuple unpacking
        if len(stmt.vars) > 1:
            self.emit(f'for {vars_str} in enumerate({iterable}):')
        else:
            self.emit(f'for {vars_str} in {iterable}:')

        self.indent_level += 1
        for s in stmt.body:
            self.generate_statement(s)
        self.indent_level -= 1

    def generate_while_loop(self, stmt: WhileLoop) -> None:
        """Generate while loop."""
        condition = self.generate_expression(stmt.condition)
        self.emit(f'while {condition}:')

        self.indent_level += 1
        for s in stmt.body:
            self.generate_statement(s)
        self.indent_level -= 1

    def generate_return_statement(self, stmt: ReturnStatement) -> None:
        """Generate return statement."""
        if stmt.expr:
            self.emit(f'return {self.generate_expression(stmt.expr)}')
        else:
            self.emit('return')

    # ========================================================================
    # Expression generation
    # ========================================================================

    def generate_expression(self, expr: Expression) -> str:
        """Generate an expression."""
        if isinstance(expr, BinaryOp):
            return self.generate_binary_op(expr)
        elif isinstance(expr, UnaryOp):
            return self.generate_unary_op(expr)
        elif isinstance(expr, TernaryOp):
            return self.generate_ternary_op(expr)
        elif isinstance(expr, FunctionCall):
            return self.generate_function_call(expr)
        elif isinstance(expr, MethodCall):
            return self.generate_method_call(expr)
        elif isinstance(expr, MemberAccess):
            return self.generate_member_access(expr)
        elif isinstance(expr, IndexAccess):
            return self.generate_index_access(expr)
        elif isinstance(expr, ArrayLiteral):
            return self.generate_array_literal(expr)
        elif isinstance(expr, Identifier):
            return self.generate_identifier(expr)
        elif isinstance(expr, Literal):
            return self.generate_literal(expr)
        elif isinstance(expr, NaLiteral):
            return 'na'
        else:
            return f'# TODO: {type(expr).__name__}'

    def generate_identifier(self, expr: Identifier) -> str:
        """Generate identifier."""
        from .pine_builtins import TA_VARIABLE_FUNCTIONS

        name = self._esc(expr.name)

        # Pine Script "variable-functions": used without () in Pine Script
        # but implemented as functions in PyneCore (e.g., ta.vwap -> ta.vwap())
        if name in TA_VARIABLE_FUNCTIONS:
            # Some functions need default args (e.g., ta.vwap defaults to hlc3)
            defaults = TA_VARIABLE_FUNCTIONS.get(name, '')
            return f'{name}({defaults})'

        return name

    def generate_binary_op(self, expr: BinaryOp) -> str:
        """Generate binary operation."""
        left = self.generate_expression(expr.left)
        right = self.generate_expression(expr.right)

        # Convert Pine operators to Python
        op_map = {
            'and': 'and',
            'or': 'or',
            'not': 'not',
        }
        op = op_map.get(expr.op, expr.op)

        # Add parentheses based on operator precedence rules
        # Precedence: * / % (high) > + - (low)
        LOWER_THAN_MUL = ('+', '-')

        # Multiplication: wrap lower-precedence operands
        if expr.op == '*':
            if isinstance(expr.left, BinaryOp) and expr.left.op in LOWER_THAN_MUL:
                left = f'({left})'
            if isinstance(expr.right, BinaryOp) and expr.right.op in ('+', '-', '/'):
                right = f'({right})'
            return f'{left} * {right}'

        # Division: wrap lower-precedence left operand, and any right operand
        if expr.op == '/':
            if isinstance(expr.left, BinaryOp) and expr.left.op in LOWER_THAN_MUL:
                left = f'({left})'
            if isinstance(expr.right, BinaryOp) and expr.right.op in ('+', '-', '/', '*'):
                right = f'({right})'
            return f'{left} / {right}'

        # Modulo: wrap lower-precedence operands
        if expr.op == '%':
            if isinstance(expr.left, BinaryOp) and expr.left.op in LOWER_THAN_MUL:
                left = f'({left})'
            if isinstance(expr.right, BinaryOp) and expr.right.op in ('+', '-', '/', '*', '%'):
                right = f'({right})'
            return f'{left} % {right}'

        # Operator precedence handling for logical operators
        if expr.op in ('and', 'or'):
            # Wrap 'not' in parens when used as operand of 'and'/'or'
            if isinstance(expr.right, UnaryOp) and expr.right.op == 'not':
                right = f'({right})'
            if isinstance(expr.left, UnaryOp) and expr.left.op == 'not':
                left = f'({left})'
            # Wrap 'or' in parens when used as operand of 'and' (lower precedence)
            if expr.op == 'and':
                if isinstance(expr.left, BinaryOp) and expr.left.op == 'or':
                    left = f'({left})'
                if isinstance(expr.right, BinaryOp) and expr.right.op == 'or':
                    right = f'({right})'

        # Don't wrap in parentheses for other ops - Python operator precedence handles this
        return f'{left} {op} {right}'

    def generate_unary_op(self, expr: UnaryOp) -> str:
        """Generate unary operation."""
        operand = self.generate_expression(expr.operand)

        # Convert Pine operators to Python
        op_map = {
            'not': 'not ',
        }
        op = op_map.get(expr.op, expr.op)

        # In Python, 'not' has lower precedence than comparison operators but
        # higher than 'and'/'or'. Pine's not(A or B) must keep parentheses,
        # otherwise Python parses 'not A or B' as '(not A) or B'.
        if expr.op == 'not' and isinstance(expr.operand, BinaryOp) and expr.operand.op in ('and', 'or'):
            return f'{op}({operand})'

        return f'{op}{operand}'

    def generate_ternary_op(self, expr: TernaryOp) -> str:
        """Generate ternary operation (Python if/else expression)."""
        condition = self.generate_expression(expr.condition)
        true_expr = self.generate_expression(expr.true_expr)
        false_expr = self.generate_expression(expr.false_expr) if expr.false_expr else 'None'

        return f'({true_expr} if {condition} else {false_expr})'

    def generate_function_call(self, call: FunctionCall) -> str:
        """Generate function call."""
        # Generate function name
        if isinstance(call.func, str):
            func = call.func
        elif isinstance(call.func, MemberAccess):
            func = self.generate_member_access(call.func)
        elif isinstance(call.func, NaLiteral):
            # Special case: na(x) is a function call, not NA() literal
            func = 'na'
            # If NA() with no args, just return 'na' (for default params)
            if not call.args and not call.kwargs:
                return 'na'
        else:
            func = self.generate_expression(call.func)

        # Generate arguments (positional - use single quotes for strings)
        args = [self.generate_expression(arg) for arg in call.args]

        # Generate keyword arguments (use double quotes for string literals)
        kwargs = []
        for key, value in call.kwargs.items():
            # Special case: options parameter should use tuple () instead of list []
            if key == 'options' and isinstance(value, ArrayLiteral):
                elements = [self._generate_kwarg_value(e) for e in value.elements]
                kwargs.append(f'{key}=({", ".join(elements)})')
            else:
                kwargs.append(f'{key}={self._generate_kwarg_value(value)}')

        all_args = args + kwargs
        return f'{func}({", ".join(all_args)})'

    def _generate_kwarg_value(self, value: Expression) -> str:
        """Generate kwarg value with double quotes for strings."""
        if isinstance(value, Literal) and value.literal_type == 'string':
            return f'"{self._escape_string(value.value, chr(34))}"'
        elif isinstance(value, Literal) and value.literal_type in ('int', 'float', 'bool'):
            # For non-string literals, use normal generation
            return self.generate_literal(value)
        else:
            return self.generate_expression(value)

    def generate_method_call(self, call: MethodCall) -> str:
        """Generate method call."""
        from .pine_builtins import PYNECORE_METHOD_TRANSFORMS
        from .ast_nodes import BinaryOp, UnaryOp, TernaryOp, IfExpression

        obj = self.generate_expression(call.object)
        args = [self.generate_expression(arg) for arg in call.args]
        kwargs = [f'{key}={self.generate_expression(value)}' for key, value in call.kwargs.items()]

        # Check if this method should be transformed to a module function call
        # (e.g., m.add_row(...) → matrix.add_row(m, ...))
        if call.method in PYNECORE_METHOD_TRANSFORMS:
            module = PYNECORE_METHOD_TRANSFORMS[call.method]
            # Transform to module.function(obj, args...)
            all_args = [obj] + args + kwargs
            return f'{module}.{call.method}({", ".join(all_args)})'

        # FIX: Wrap complex expressions in parentheses to preserve precedence
        # This ensures (expression).method() keeps correct semantics
        if isinstance(call.object, (BinaryOp, UnaryOp, TernaryOp, IfExpression)):
            obj = f'({obj})'

        # Otherwise, keep as method call
        all_args = args + kwargs
        return f'{obj}.{call.method}({", ".join(all_args)})'

    def generate_member_access(self, expr: MemberAccess) -> str:
        """Generate member access."""
        from .pine_builtins import TA_VARIABLE_FUNCTIONS

        if isinstance(expr.object, str):
            obj = expr.object
        elif isinstance(expr.object, Expression):
            obj = self.generate_expression(expr.object)
        else:
            obj = str(expr.object)

        result = f'{obj}.{expr.member}'

        # Pine Script "variable-functions": used without () in Pine Script
        # but implemented as functions in PyneCore (e.g., ta.vwap -> ta.vwap())
        if result in TA_VARIABLE_FUNCTIONS:
            defaults = TA_VARIABLE_FUNCTIONS.get(result, '')
            result = f'{result}({defaults})'

        return result

    def generate_index_access(self, expr: IndexAccess) -> str:
        """Generate index access."""
        obj = self.generate_expression(expr.object)
        index = self.generate_expression(expr.index)
        return f'{obj}[{index}]'

    def generate_array_literal(self, expr: ArrayLiteral) -> str:
        """Generate array literal."""
        elements = [self.generate_expression(e) for e in expr.elements]
        return f'[{", ".join(elements)}]'

    def _escape_string(self, value: str, quote: str) -> str:
        """Escape a string value for Python output."""
        escaped = value.replace('\\', '\\\\')
        escaped = escaped.replace('\n', '\\n')
        escaped = escaped.replace('\t', '\\t')
        escaped = escaped.replace('\r', '\\r')
        escaped = escaped.replace(quote, f'\\{quote}')
        return escaped

    def generate_literal(self, expr: Literal) -> str:
        """Generate literal value."""
        if expr.literal_type == 'string':
            if expr.is_double_quoted:
                return f'"{self._escape_string(expr.value, chr(34))}"'
            else:
                return f"'{self._escape_string(expr.value, chr(39))}'"
        elif expr.literal_type == 'bool':
            return 'True' if expr.value else 'False'
        elif expr.literal_type == 'color':
            if expr.is_double_quoted:
                return f'"{self._escape_string(expr.value, chr(34))}"'
            else:
                return f"'{self._escape_string(expr.value, chr(39))}'"
        else:
            return str(expr.value)


def generate_code(script: PyneTransformedScript) -> str:
    """Convenience function to generate Python code from transformed script."""
    codegen = CodeGenerator()
    return codegen.generate(script)
