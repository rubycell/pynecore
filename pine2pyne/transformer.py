"""
AST Transformer: Pine Script AST → PyneCore-compatible AST.

Implements all 54 transformation rules from the transpiler specification.
"""
from typing import List, Dict, Any, Union, Optional
from .ast_nodes import *
from .symbol_table import SymbolTable, Symbol, VariableKind
from .type_inference import TypeInference
from .import_resolver import ImportResolver
from .pine_builtins import (
    get_function_name, get_module_name, get_type_name,
    get_plot_style_remap, needs_plot_style_remap,
    is_input_function, TYPE_RENAMES, PYNECORE_LIB_MODULES,
    LABEL_METHODS, LINE_METHODS, BOX_METHODS, TABLE_METHODS,
    MAP_METHODS, MATRIX_METHODS, ARRAY_METHODS, MAP_UNIQUE_METHODS,
    SHARED_COLLECTION_METHODS,
)

# Known module names - used to guard against transforming module.function() calls
# as if they were chained member access (e.g. array.size != obj.size)
KNOWN_MODULES = PYNECORE_LIB_MODULES | {
    'bar_index', 'close', 'open', 'high', 'low', 'volume', 'time',
    'hl2', 'hlc3', 'ohlc4', 'hlcc4',
    'dayofmonth', 'dayofweek', 'hour', 'minute', 'month', 'second',
    'weekofyear', 'year', 'timenow',
    'adjustment', 'earnings', 'dividends', 'splits',
    'order', 'str',
}
from .errors import TransformerError, UnsupportedFeatureError


class PyneTransformedScript:
    """Holds the transformed script ready for code generation."""

    def __init__(self):
        self.imports: List[str] = []
        self.type_declarations: List[TypeDecl] = []  # Custom type definitions
        self.enum_declarations: List[EnumDecl] = []  # Enum definitions
        self.helper_functions: List[FuncDecl] = []
        self.main_decorator: Optional[Union[IndicatorDecl, StrategyDecl]] = None
        self.main_params: List[Parameter] = []
        self.main_body: List[Statement] = []
        self.global_vars: List[Union[VarDecl, Assignment]] = []
        self.module_constants: List[Assignment] = []  # Pre-main literal constants
        self.cache_slots: dict[int, tuple[str, frozenset[str]]] = {}  # slot → (varname, deps)
        self.num_cache_slots: int = 0


def convert_pine_generic_to_python(pine_type: str) -> str:
    """
    Convert Pine Script generic types to Python type hints.

    Examples:
        map<string, float> -> dict[str, float]
        array<int> -> list[int]
    """
    # Check if it contains generic syntax
    if '<' not in pine_type or '>' not in pine_type:
        return get_type_name(pine_type)

    # Extract base type and parameters
    if '<' in pine_type:
        base_type = pine_type[:pine_type.index('<')]
        params_str = pine_type[pine_type.index('<')+1:pine_type.rindex('>')]
        params = [p.strip() for p in params_str.split(',')]

        # Convert base type
        if base_type == 'map':
            base_type = 'dict'
        elif base_type == 'array':
            base_type = 'list'
        else:
            # For other types like matrix, line, etc., use type name mapping
            base_type = get_type_name(base_type)

        # Convert parameter types
        converted_params = [get_type_name(p) for p in params]

        return f"{base_type}[{', '.join(converted_params)}]"

    return get_type_name(pine_type)


class Transformer:
    """Transforms Pine Script AST to PyneCore-compatible form."""

    def __init__(self):
        self.symbol_table = SymbolTable()
        self.type_inference = TypeInference(self.symbol_table)
        self.import_resolver = ImportResolver()
        self.input_declarations: List[InputDecl] = []
        self.current_function_params: Dict[str, str] = {}  # Track current function parameters
        self.method_functions: set = set()  # Track @method decorated function names
        self._renamed_functions: Dict[str, str] = {}  # Pine name -> sanitized Python name

        # PRIORITY 1 FIX: Track Series variables in user-defined functions
        self.in_user_function = False  # Are we inside a user-defined function?
        self.series_variables: set = set()  # Variables that need to be Series objects
        self.function_local_vars: set = set()  # All local variables in current function

    def _get_module_references_in_expr(self, expr: Optional[Expression]) -> set[str]:
        """
        Recursively find all module references in an expression.
        Returns set of module names that are referenced (e.g., 'position' in position.middle_center).
        """
        if expr is None:
            return set()

        modules = set()

        # Handle Identifier with dotted names (e.g., 'position.middle_center')
        if isinstance(expr, Identifier):
            if '.' in expr.name:
                module = expr.name.split('.')[0]
                if module in KNOWN_MODULES or module in self.import_resolver.lib_modules:
                    modules.add(module)

        # Handle MemberAccess nodes
        if isinstance(expr, MemberAccess):
            # Handle module.member access
            if isinstance(expr.object, str):
                if expr.object in KNOWN_MODULES or expr.object in self.import_resolver.lib_modules:
                    modules.add(expr.object)
            elif isinstance(expr.object, Identifier):
                if expr.object.name in KNOWN_MODULES or expr.object.name in self.import_resolver.lib_modules:
                    modules.add(expr.object.name)

        # Recursively check nested expressions
        if hasattr(expr, '__dict__'):
            for attr_value in expr.__dict__.values():
                if isinstance(attr_value, (ASTNode, Expression)):
                    modules.update(self._get_module_references_in_expr(attr_value))
                elif isinstance(attr_value, list):
                    for item in attr_value:
                        if isinstance(item, (ASTNode, Expression)):
                            modules.update(self._get_module_references_in_expr(item))

        return modules

    def _rewrite_module_references(self, expr: Optional[Expression]) -> Optional[Expression]:
        """
        Rewrite an expression to use module aliases where needed.
        Replaces module references with their aliased versions (e.g., position -> _position_module).
        """
        if expr is None:
            return None

        # Handle Identifier with dotted names (e.g., 'position.middle_center' -> '_position_module.middle_center')
        if isinstance(expr, Identifier):
            if '.' in expr.name:
                module, rest = expr.name.split('.', 1)
                if module in self.import_resolver.aliased_modules:
                    alias = self.import_resolver.aliased_modules[module]
                    return Identifier(name=f"{alias}.{rest}")

        # Handle MemberAccess nodes
        if isinstance(expr, MemberAccess):
            # Rewrite module.member to alias.member if module is aliased
            if isinstance(expr.object, str):
                if expr.object in self.import_resolver.aliased_modules:
                    return MemberAccess(
                        object=self.import_resolver.aliased_modules[expr.object],
                        member=expr.member
                    )
            elif isinstance(expr.object, Identifier):
                if expr.object.name in self.import_resolver.aliased_modules:
                    return MemberAccess(
                        object=self.import_resolver.aliased_modules[expr.object.name],
                        member=expr.member
                    )

        # Recursively rewrite nested expressions
        if hasattr(expr, '__dict__'):
            new_dict = {}
            for attr_name, attr_value in expr.__dict__.items():
                if isinstance(attr_value, (ASTNode, Expression)):
                    new_dict[attr_name] = self._rewrite_module_references(attr_value)
                elif isinstance(attr_value, list):
                    new_dict[attr_name] = [
                        self._rewrite_module_references(item) if isinstance(item, (ASTNode, Expression)) else item
                        for item in attr_value
                    ]
                else:
                    new_dict[attr_name] = attr_value

            # Create new instance with rewritten attributes
            return type(expr)(**new_dict)

        return expr

    def _visit_ast_nodes(self, nodes: list, visitor_fn) -> None:
        """Walk all AST nodes depth-first, calling visitor_fn on each."""
        def visit(node):
            visitor_fn(node)
            if not hasattr(node, '__dataclass_fields__'):
                return
            for field_name in node.__dataclass_fields__:
                val = getattr(node, field_name)
                if hasattr(val, '__dataclass_fields__'):
                    visit(val)
                elif isinstance(val, list):
                    for item in val:
                        if hasattr(item, '__dataclass_fields__'):
                            visit(item)
                elif isinstance(val, dict):
                    for v in val.values():
                        if hasattr(v, '__dataclass_fields__'):
                            visit(v)

        for node in nodes:
            visit(node)

    def _collect_reassignment_vars(self, script: Script) -> set:
        """Collect variable names that are reassigned with := operator."""
        reassign_vars = set()

        def visitor(node):
            if isinstance(node, Reassignment):
                reassign_vars.add(node.target)

        self._visit_ast_nodes(list(script.declarations) + list(script.body), visitor)
        return reassign_vars

    def _find_series_variables_in_function(self, func_body: List[Statement]) -> set:
        """Find variables in a function that need to be Series objects.
        A variable needs to be Series if it's reassigned with := or indexed with [n].
        """
        series_vars = set()

        def visitor(node):
            if isinstance(node, Reassignment):
                series_vars.add(node.target)
            if isinstance(node, IndexAccess):
                if isinstance(node.object, Identifier):
                    series_vars.add(node.object.name)

        self._visit_ast_nodes(func_body, visitor)
        return series_vars

    def transform(self, script: Script) -> PyneTransformedScript:
        """Transform entire script."""
        output = PyneTransformedScript()

        # Build symbol table
        self._build_symbol_table(script)

        # Infer types
        self.type_inference.infer_types(script)

        # (Removed: _history_ref_vars replaced by symbol.is_indexed from type inference)

        # Collect variables that use := (reassignment)
        self._reassignment_vars = self._collect_reassignment_vars(script)

        # Analyze imports
        self.import_resolver.analyze(script)

        # Transform script declaration (indicator/strategy)
        if script.script_decl:
            output.main_decorator = self._transform_script_declaration(script.script_decl)

        # Extract input declarations to main() parameters
        self._extract_input_declarations(script)
        output.main_params = self._create_main_parameters()

        # Transform type and enum declarations - stay at module level
        for decl in script.declarations:
            if isinstance(decl, TypeDecl):
                output.type_declarations.append(decl)  # Keep as-is for codegen
            elif isinstance(decl, EnumDecl):
                output.enum_declarations.append(decl)  # Keep as-is for codegen

        # Identify body assignments that must be module-level constants.
        # In Pine Script, declarations before the first function call (like
        # `const string G_ENTRY = '...'` or `DEFAULT_PYRAMIDING = 6`) can be
        # referenced by input.* group= params and strategy() kwargs, which
        # become main() parameter defaults evaluated at function definition time.
        # These must be emitted at module level, not inside main().
        module_const_names = self._collect_module_constant_names(script)

        # Merge all non-type, non-input declarations with body statements
        # sorted by source line number to preserve original Pine Script ordering
        all_items: List[tuple] = []
        for decl in script.declarations:
            if isinstance(decl, (VarDecl, VaripDecl)):
                all_items.append(('var', decl))
            elif isinstance(decl, FuncDecl):
                all_items.append(('func', decl))
            # TypeDecl and InputDecl are handled separately above

        for stmt in script.body:
            all_items.append(('body', stmt))

        # Sort by line number to preserve source order
        all_items.sort(key=lambda x: x[1].line)

        # Process all items in source order into main_body
        # Propagate source line numbers for blank line detection in codegen
        for item_type, item in all_items:
            source_line = item.line
            if item_type == 'var':
                # Check if this var declaration should be a module-level constant
                if (isinstance(item, (VarDecl, VaripDecl)) and
                    item.name in module_const_names and
                    isinstance(item.value, Literal)):
                    # Emit as simple module-level assignment (not Persistent)
                    transformed = self._transform_statement(
                        Assignment(target=item.name, value=item.value,
                                   type_hint=item.type_hint))
                    if transformed and not isinstance(transformed, list):
                        transformed.line = source_line
                        output.module_constants.append(transformed)
                    continue
                result = self._transform_var_declaration(item)
                result.line = source_line
                output.main_body.append(result)
            elif item_type == 'func':
                result = self._transform_function(item)
                result.line = source_line
                output.main_body.append(result)
            else:  # body
                # Check if this is a literal assignment that should be module-level
                if (isinstance(item, Assignment) and
                    isinstance(item.target, str) and
                    item.target in module_const_names and
                    isinstance(item.value, Literal)):
                    # Emit as module-level constant instead of inside main()
                    transformed = self._transform_statement(item)
                    if transformed and not isinstance(transformed, list):
                        transformed.line = source_line
                        output.module_constants.append(transformed)
                    continue

                transformed = self._transform_statement(item)
                if transformed:
                    if isinstance(transformed, list):
                        for t in transformed:
                            t.line = source_line
                        output.main_body.extend(transformed)
                    else:
                        transformed.line = source_line
                        output.main_body.append(transformed)

        # Insert Series wrapper assignments for input.source parameters at body start
        # These allow [n] subscript access on source inputs
        if hasattr(self, '_source_input_wrappers'):
            for orig_name, src_param_name in reversed(self._source_input_wrappers):
                wrapper = Assignment(
                    target=orig_name,
                    value=Identifier(name=src_param_name),
                    type_hint='Series[float]'
                )
                wrapper.line = 0  # Place at beginning
                output.main_body.insert(0, wrapper)

        # Resolve function overloading (multiple defs with same name)
        self._resolve_overloaded_functions(output.main_body)

        # Analyze variable cache dependencies for optimize mode
        self._analyze_cache_deps(output)

        # Analyze transformed AST for type names in type hints (Matrix, Line, etc.)
        self._analyze_transformed_types(output)

        # If variable cache is used, bar_index must be imported (used in _vc[N][int(bar_index)])
        if output.num_cache_slots > 0:
            self.import_resolver.builtin_variables.add('bar_index')

        # Generate imports
        output.imports = self.import_resolver.generate_imports()

        return output

    # ------------------------------------------------------------------
    # Variable cache dependency analysis (for pyne optimize)
    # ------------------------------------------------------------------

    # OHLCV builtins — always available, no input dependency
    _OHLCV_BUILTINS = frozenset({
        'close', 'open', 'high', 'low', 'volume',
        'hl2', 'hlc3', 'ohlc4', 'hlcc4',
        'time', 'time_close', 'bar_index',
    })

    # Safe module references — accessing their members introduces no dependency
    _SAFE_MODULES = frozenset({
        'ta', 'math', 'color', 'str', 'array', 'map', 'nz', 'na',
        'timeframe', 'syminfo', 'chart',
    })

    # Modules whose state depends on trading — always unknown
    _TRADING_STATE_MODULES = frozenset({'strategy', 'position'})

    def _analyze_cache_deps(self, output: 'PyneTransformedScript') -> None:
        """Analyze which main_body assignments can be cached in optimize mode.

        For each top-level assignment of a function call result, compute its
        transitive input-parameter dependencies.  Assignments whose deps are
        a subset of non-optimized params (or empty) can be cached across
        optimizer runs.
        """
        # 1. Collect input param names
        input_params: set[str] = set()
        for p in output.main_params:
            input_params.add(p.name)

        # 2. Build initial deps_map: OHLCV builtins → empty, params → {name}
        deps_map: dict[str, frozenset[str]] = {}
        for name in self._OHLCV_BUILTINS:
            deps_map[name] = frozenset()
        for name in input_params:
            deps_map[name] = frozenset({name})

        # 3. Pre-scan: find targets assigned more than once or inside control flow
        reassigned: set[str] = set()
        control_assigned: set[str] = set()
        assign_count: dict[str, int] = {}

        def _count_assigns(stmts):
            for s in stmts:
                if isinstance(s, Assignment) and isinstance(s.target, str):
                    assign_count[s.target] = assign_count.get(s.target, 0) + 1
                elif isinstance(s, Reassignment) and isinstance(s.target, str):
                    reassigned.add(s.target)

        def _scan_control_flow(stmts):
            for s in stmts:
                if isinstance(s, (IfStatement, ForLoop, ForInLoop, WhileLoop)):
                    bodies = []
                    if isinstance(s, IfStatement):
                        bodies.append(s.body)
                        for _, elif_body in s.elseifs:
                            bodies.append(elif_body)
                        if s.else_body:
                            bodies.append(s.else_body)
                    elif isinstance(s, (ForLoop, ForInLoop, WhileLoop)):
                        bodies.append(s.body)
                    for body in bodies:
                        for bs in body:
                            if isinstance(bs, Assignment) and isinstance(bs.target, str):
                                control_assigned.add(bs.target)
                            elif isinstance(bs, Reassignment) and isinstance(bs.target, str):
                                control_assigned.add(bs.target)
                        _scan_control_flow(body)

        _count_assigns(output.main_body)
        _scan_control_flow(output.main_body)
        multi_assigned = {k for k, v in assign_count.items() if v > 1}
        skip_targets = multi_assigned | reassigned | control_assigned

        # 4. Walk top-level assignments sequentially
        slot_idx = 0
        for stmt in output.main_body:
            if not isinstance(stmt, Assignment):
                continue
            if isinstance(stmt.target, TupleDestructure):
                continue
            target = stmt.target
            if not isinstance(target, str):
                continue

            # Skip Persistent/PersistentSeries
            if stmt.type_hint and 'Persistent' in stmt.type_hint:
                continue

            # Skip reassigned / control-flow assigned
            if target in skip_targets:
                continue

            # Only cache function call results
            if not isinstance(stmt.value, (FunctionCall, MethodCall)):
                # Still track deps for downstream use if it's a simple identifier
                if isinstance(stmt.value, Identifier):
                    ref_deps = deps_map.get(stmt.value.name)
                    if ref_deps is not None:
                        deps_map[target] = ref_deps
                continue

            # Skip input.*() calls from caching, but track in deps_map
            # so downstream variables can correctly resolve their deps
            if isinstance(stmt.value, FunctionCall):
                func = stmt.value.func
                is_input = False
                if isinstance(func, str) and func == 'input':
                    is_input = True
                elif isinstance(func, MemberAccess):
                    if (isinstance(func.object, str) and func.object == 'input'):
                        is_input = True
                    elif (isinstance(func.object, Identifier) and func.object.name == 'input'):
                        is_input = True
                if is_input:
                    deps_map[target] = frozenset({target})
                    continue

            # Compute deps
            deps = self._expr_deps(stmt.value, deps_map)
            if '__UNKNOWN__' not in deps:
                output.cache_slots[slot_idx] = (target, deps)
                slot_idx += 1
                deps_map[target] = deps
            else:
                # Track as unknown for downstream
                deps_map[target] = frozenset({'__UNKNOWN__'})

        output.num_cache_slots = slot_idx

    def _expr_deps(
        self,
        expr,
        deps_map: dict[str, frozenset[str]],
    ) -> frozenset[str]:
        """Recursively compute input-parameter dependencies of an expression."""
        if expr is None:
            return frozenset()

        if isinstance(expr, (Literal, NaLiteral)):
            return frozenset()

        if isinstance(expr, Identifier):
            return deps_map.get(expr.name, frozenset({'__UNKNOWN__'}))

        if isinstance(expr, FunctionCall):
            result: set[str] = set()
            for arg in expr.args:
                result |= self._expr_deps(arg, deps_map)
            for val in expr.kwargs.values():
                result |= self._expr_deps(val, deps_map)
            return frozenset(result)

        if isinstance(expr, MethodCall):
            result = set(self._expr_deps(expr.object, deps_map))
            for arg in expr.args:
                result |= self._expr_deps(arg, deps_map)
            for val in expr.kwargs.values():
                result |= self._expr_deps(val, deps_map)
            return frozenset(result)

        if isinstance(expr, BinaryOp):
            return self._expr_deps(expr.left, deps_map) | self._expr_deps(expr.right, deps_map)

        if isinstance(expr, UnaryOp):
            return self._expr_deps(expr.operand, deps_map)

        if isinstance(expr, TernaryOp):
            return (self._expr_deps(expr.condition, deps_map) |
                    self._expr_deps(expr.true_expr, deps_map) |
                    self._expr_deps(expr.false_expr, deps_map))

        if isinstance(expr, MemberAccess):
            obj = expr.object
            if isinstance(obj, str):
                if obj in self._SAFE_MODULES:
                    return frozenset()
                if obj in self._TRADING_STATE_MODULES:
                    return frozenset({'__UNKNOWN__'})
                if obj in self._OHLCV_BUILTINS:
                    return frozenset()
                return deps_map.get(obj, frozenset({'__UNKNOWN__'}))
            if isinstance(obj, Identifier):
                if obj.name in self._SAFE_MODULES:
                    return frozenset()
                if obj.name in self._TRADING_STATE_MODULES:
                    return frozenset({'__UNKNOWN__'})
                if obj.name in self._OHLCV_BUILTINS:
                    return frozenset()
                return deps_map.get(obj.name, frozenset({'__UNKNOWN__'}))
            return self._expr_deps(obj, deps_map)

        if isinstance(expr, IndexAccess):
            return self._expr_deps(expr.object, deps_map) | self._expr_deps(expr.index, deps_map)

        if isinstance(expr, ArrayLiteral):
            result = set()
            for elem in expr.elements:
                result |= self._expr_deps(elem, deps_map)
            return frozenset(result)

        # Unknown node type → mark as unknown
        return frozenset({'__UNKNOWN__'})

    def _get_func_name_from_call(self, call: FunctionCall) -> Optional[str]:
        """Extract function name string from a FunctionCall node."""
        if isinstance(call.func, str):
            return call.func
        elif isinstance(call.func, MemberAccess):
            if isinstance(call.func.object, str):
                return f"{call.func.object}.{call.func.member}"
            elif isinstance(call.func.object, Identifier):
                return f"{call.func.object.name}.{call.func.member}"
        return None

    def _analyze_transformed_types(self, output: PyneTransformedScript) -> None:
        """Analyze transformed AST to extract type names from type hints."""
        import re

        # Track if Series is actually used
        series_used = False
        persistent_used = False

        def extract_types_from_hint(type_hint: str) -> None:
            """Extract type names like Line, Matrix from type hints."""
            nonlocal series_used, persistent_used

            if not type_hint:
                return

            # Check for Series and Persistent usage
            if 'Series[' in type_hint:
                series_used = True
            if 'Persistent[' in type_hint:
                persistent_used = True

            # Match capital letter type names that might be drawing/structural types
            for match in re.finditer(r'\b([A-Z][a-zA-Z]*)\b', type_hint):
                type_name = match.group(1)
                # Add to import resolver if it's a known type
                self.import_resolver.add_drawing_type(type_name)

        def scan_statement(stmt: Statement) -> None:
            """Recursively scan a statement for type hints."""
            # Handle lists (flattened statement lists from hoisting)
            if isinstance(stmt, list):
                for s in stmt:
                    scan_statement(s)
                return
            if stmt is None:
                return

            # Check if statement has type hint
            if hasattr(stmt, 'type_hint') and stmt.type_hint:
                extract_types_from_hint(stmt.type_hint)

            # Recursively scan nested structures
            if isinstance(stmt, FuncDecl) and hasattr(stmt, 'params'):
                for param in stmt.params:
                    if hasattr(param, 'type_hint') and param.type_hint:
                        extract_types_from_hint(param.type_hint)
                if hasattr(stmt, 'body') and isinstance(stmt.body, list):
                    for body_stmt in stmt.body:
                        scan_statement(body_stmt)

            # Scan all compound statement bodies
            elif isinstance(stmt, IfStatement):
                for body_stmt in (stmt.body or []):
                    scan_statement(body_stmt)
                for _, elif_body in (stmt.elseifs or []):
                    for body_stmt in elif_body:
                        scan_statement(body_stmt)
                for body_stmt in (stmt.else_body or []):
                    scan_statement(body_stmt)
            elif isinstance(stmt, (ForLoop, ForInLoop, WhileLoop)):
                for body_stmt in (stmt.body or []):
                    scan_statement(body_stmt)

        # Scan global vars for type hints (var/varip declarations)
        for var in output.global_vars:
            if hasattr(var, 'type_hint') and var.type_hint:
                extract_types_from_hint(var.type_hint)

        # Scan main body for type hints
        for stmt in output.main_body:
            scan_statement(stmt)

        # Scan main parameters for type hints
        for param in output.main_params:
            if hasattr(param, 'type_hint') and param.type_hint:
                extract_types_from_hint(param.type_hint)

        # Update import resolver based on actual usage in transformed code
        self.import_resolver.uses_series = series_used
        self.import_resolver.uses_persistent = persistent_used

    # ========================================================================
    # Symbol table building
    # ========================================================================

    def _build_symbol_table(self, script: Script) -> None:
        """Build symbol table from script declarations."""
        # Register input declarations
        for decl in script.declarations:
            if isinstance(decl, InputDecl):
                symbol = Symbol(
                    decl.name,
                    VariableKind.INPUT,
                    is_global=True
                )
                self.symbol_table.define(symbol)

        # Register var/varip declarations
        for decl in script.declarations:
            if isinstance(decl, VarDecl):
                type_hint = self._infer_type_from_declaration(decl)
                symbol = Symbol(
                    decl.name,
                    VariableKind.VAR,
                    type_hint=type_hint,
                    is_global=True
                )
                self.symbol_table.define(symbol)
            elif isinstance(decl, VaripDecl):
                type_hint = self._infer_type_from_declaration(decl)
                symbol = Symbol(
                    decl.name,
                    VariableKind.VARIP,
                    type_hint=type_hint,
                    is_global=True
                )
                self.symbol_table.define(symbol)

        # Register type declarations (UDTs)
        for decl in script.declarations:
            if isinstance(decl, TypeDecl):
                symbol = Symbol(
                    decl.name,
                    VariableKind.TYPE,
                    is_global=True
                )
                self.symbol_table.define(symbol)

        # Register function declarations
        for decl in script.declarations:
            if isinstance(decl, FuncDecl):
                symbol = Symbol(
                    decl.name,
                    VariableKind.FUNCTION,
                    is_global=True
                )
                self.symbol_table.define(symbol)

        # Register assignments as variables (including nested scopes)
        self._register_assignments_recursive(script.body)

    def _register_assignments_recursive(self, stmts: List[Statement]) -> None:
        """Recursively register assignments in the symbol table, including nested scopes."""
        for stmt in stmts:
            if isinstance(stmt, Assignment) and isinstance(stmt.target, str):
                if not self.symbol_table.lookup_global(stmt.target):
                    if hasattr(stmt, 'type_hint') and stmt.type_hint:
                        type_hint = convert_pine_generic_to_python(stmt.type_hint)
                    else:
                        type_hint = self.type_inference.infer_type_hint(stmt.value)
                    symbol = Symbol(
                        stmt.target,
                        VariableKind.SERIES,
                        type_hint=type_hint,
                        is_global=True
                    )
                    symbol.is_series = True
                    self.symbol_table.define(symbol)
            # Recurse into nested scopes
            if isinstance(stmt, IfStatement):
                self._register_assignments_recursive(stmt.body)
                for _, elif_body in stmt.elseifs:
                    self._register_assignments_recursive(elif_body)
                if stmt.else_body:
                    self._register_assignments_recursive(stmt.else_body)
            elif isinstance(stmt, (ForLoop, ForInLoop, WhileLoop)):
                if isinstance(stmt.body, list):
                    self._register_assignments_recursive(stmt.body)

    def _infer_type_from_declaration(self, decl: Union[VarDecl, VaripDecl]) -> str:
        """Infer type from var/varip declaration."""
        if decl.type_hint:
            return convert_pine_generic_to_python(decl.type_hint)
        return self.type_inference.infer_type_hint(decl.value)

    # ========================================================================
    # Module-level constant detection
    # ========================================================================

    def _collect_identifier_names(self, node: Any) -> set:
        """Recursively collect all Identifier names referenced in an AST node."""
        names = set()
        if isinstance(node, Identifier):
            names.add(node.name)
        elif isinstance(node, (FunctionCall, MethodCall)):
            for arg in getattr(node, 'args', []):
                names.update(self._collect_identifier_names(arg))
            for val in getattr(node, 'kwargs', {}).values():
                names.update(self._collect_identifier_names(val))
        elif isinstance(node, (BinaryOp,)):
            names.update(self._collect_identifier_names(node.left))
            names.update(self._collect_identifier_names(node.right))
        elif isinstance(node, (UnaryOp,)):
            names.update(self._collect_identifier_names(node.operand))
        elif isinstance(node, (TernaryOp,)):
            names.update(self._collect_identifier_names(node.condition))
            names.update(self._collect_identifier_names(node.true_expr))
            names.update(self._collect_identifier_names(node.false_expr))
        elif isinstance(node, MemberAccess):
            names.update(self._collect_identifier_names(node.object))
        elif isinstance(node, IndexAccess):
            names.update(self._collect_identifier_names(node.object))
            names.update(self._collect_identifier_names(node.index))
        return names

    def _collect_module_constant_names(self, script: Script) -> set:
        """Collect names of body assignments that must be module-level constants.

        In Pine Script, literal assignments before the first function call
        (e.g., `const string G_ENTRY = '...'`) can be referenced by input.*()
        group= params and strategy/indicator kwargs. Since these become main()
        parameter defaults (evaluated at definition time), the referenced
        variables must exist at module level.
        """
        # Collect all identifier names referenced in input declarations
        referenced_names = set()
        for decl in script.declarations:
            if isinstance(decl, InputDecl):
                for arg in decl.args:
                    referenced_names.update(self._collect_identifier_names(arg))
                for val in decl.kwargs.values():
                    referenced_names.update(self._collect_identifier_names(val))

        # Also collect names referenced in strategy/indicator kwargs
        if script.script_decl:
            for val in script.script_decl.kwargs.values():
                referenced_names.update(self._collect_identifier_names(val))

        # Find body assignments that are literal values AND referenced by inputs/decorator
        module_const_names = set()
        for stmt in script.body:
            if (isinstance(stmt, Assignment) and
                isinstance(stmt.target, str) and
                isinstance(stmt.value, Literal) and
                stmt.target in referenced_names):
                module_const_names.add(stmt.target)

        # Also check var/varip declarations with literal values referenced by inputs/decorator
        # (e.g., `var string G_STRATEGY = 'Strategy Settings'` used in group= params)
        for decl in script.declarations:
            if (isinstance(decl, (VarDecl, VaripDecl)) and
                isinstance(decl.value, Literal) and
                decl.name in referenced_names):
                module_const_names.add(decl.name)

        return module_const_names

    # ========================================================================
    # Top-level transformations
    # ========================================================================

    def _transform_script_declaration(self, decl: Union[IndicatorDecl, StrategyDecl]) -> Union[IndicatorDecl, StrategyDecl]:
        """Transform indicator/strategy declaration (Rules 2-3, 17, 54)."""
        # Transform kwargs: convert true/false to True/False (Rule 17, 54)
        transformed_kwargs = {}
        for key, value in decl.kwargs.items():
            transformed_kwargs[key] = self._transform_decorator_value(value)

        if isinstance(decl, IndicatorDecl):
            return IndicatorDecl(title=decl.title, kwargs=transformed_kwargs)
        else:
            return StrategyDecl(title=decl.title, kwargs=transformed_kwargs)

    def _transform_decorator_value(self, value: Any) -> Any:
        """Transform decorator keyword argument values (Rule 17, 54)."""
        if isinstance(value, Literal):
            if value.literal_type == 'bool':
                # Convert true/false to True/False (Rule 17)
                return value.value  # Will be True or False in Python
            return value
        elif isinstance(value, Identifier):
            # Handle true/false identifiers
            if value.name == 'true':
                return True
            elif value.name == 'false':
                return False
        return value

    def _extract_input_declarations(self, script: Script) -> None:
        """Extract input declarations for main() parameters (Rules 8-9)."""
        for decl in script.declarations:
            if isinstance(decl, InputDecl):
                self.input_declarations.append(decl)

    def _create_main_parameters(self) -> List[Parameter]:
        """Create main() function parameters from input declarations (Rule 8)."""
        params = []
        self._source_input_wrappers = []  # Track input.source params needing Series wrappers

        for input_decl in self.input_declarations:
            args = list(input_decl.args)
            kwargs = dict(input_decl.kwargs)

            # Convert excess positional args to keyword args.
            # Pine Script allows input.int(defval, title, minval, maxval, ...)
            # but PyneCore requires minval=, maxval=, ... as keyword-only.
            from .pine_builtins import INPUT_POSITIONAL_PARAMS
            param_names = INPUT_POSITIONAL_PARAMS.get(input_decl.func)
            if param_names and len(args) > 2:
                excess_args = args[2:]
                args = args[:2]
                for i, val in enumerate(excess_args):
                    if i < len(param_names):
                        kwargs[param_names[i]] = val

            func_call = FunctionCall(
                func=input_decl.func,
                args=args,
                kwargs=kwargs
            )

            # input.source returns a float per bar — needs Series wrapping
            # for [n] subscript access to work
            if input_decl.func in ('input.source', 'input'):
                # Check if any arg is a source identifier (close, open, high, low, etc.)
                is_source = input_decl.func == 'input.source'
                if not is_source and args:
                    # input(close, ...) where first arg is a source
                    first_arg = args[0]
                    if isinstance(first_arg, Identifier) and first_arg.name in (
                        'close', 'open', 'high', 'low', 'volume', 'hl2', 'hlc3', 'ohlc4', 'hlcc4'
                    ):
                        is_source = True

                if is_source:
                    # Rename param and add Series wrapper assignment to body
                    src_param_name = f'_{input_decl.name}_src'
                    param = Parameter(name=src_param_name, type_hint=None, default=func_call)
                    self._source_input_wrappers.append((input_decl.name, src_param_name))
                    params.append(param)
                    continue

            param = Parameter(
                name=input_decl.name,
                type_hint=None,
                default=func_call
            )
            params.append(param)

        return params

    # ========================================================================
    # Function transformations
    # ========================================================================

    def _find_indexed_parameters(self, body: Union[Expression, List[Statement]], param_names: set) -> set:
        """Find which parameters are accessed with indexing (e.g., source[i])."""
        indexed_params = set()

        def scan_expr(expr):
            """Recursively scan expression for IndexAccess nodes."""
            if expr is None:
                return

            if isinstance(expr, IndexAccess):
                # Check if the indexed object is a parameter
                if isinstance(expr.object, Identifier) and expr.object.name in param_names:
                    indexed_params.add(expr.object.name)
                # Recursively scan the index expression
                scan_expr(expr.index)
            elif isinstance(expr, BinaryOp):
                scan_expr(expr.left)
                scan_expr(expr.right)
            elif isinstance(expr, UnaryOp):
                scan_expr(expr.operand)
            elif isinstance(expr, FunctionCall):
                for arg in expr.args:
                    scan_expr(arg)
            elif isinstance(expr, MemberAccess):
                scan_expr(expr.object)
            elif isinstance(expr, TernaryOp):
                scan_expr(expr.condition)
                scan_expr(expr.true_expr)
                scan_expr(expr.false_expr)

        def scan_stmt(stmt):
            """Recursively scan statement for IndexAccess nodes."""
            if stmt is None:
                return

            if isinstance(stmt, Assignment):
                scan_expr(stmt.value)
            elif isinstance(stmt, Reassignment):
                scan_expr(stmt.value)
            elif isinstance(stmt, ExpressionStatement):
                scan_expr(stmt.expr)
            elif isinstance(stmt, ReturnStatement):
                scan_expr(stmt.expr)
            elif isinstance(stmt, IfStatement):
                scan_expr(stmt.condition)
                for s in stmt.body:
                    scan_stmt(s)
                for cond, elseif_body in stmt.elseifs:
                    scan_expr(cond)
                    for s in elseif_body:
                        scan_stmt(s)
                if stmt.else_body:
                    for s in stmt.else_body:
                        scan_stmt(s)
            elif isinstance(stmt, ForLoop):
                scan_expr(stmt.from_val)
                scan_expr(stmt.to_val)
                if stmt.step:
                    scan_expr(stmt.step)
                for s in (stmt.body if isinstance(stmt.body, list) else [stmt.body]):
                    scan_stmt(s)
            elif isinstance(stmt, ForInLoop):
                scan_expr(stmt.iterable)
                for s in (stmt.body if isinstance(stmt.body, list) else [stmt.body]):
                    scan_stmt(s)
            elif isinstance(stmt, WhileLoop):
                scan_expr(stmt.condition)
                for s in (stmt.body if isinstance(stmt.body, list) else [stmt.body]):
                    scan_stmt(s)

        # Scan the body
        if isinstance(body, Expression):
            scan_expr(body)
        elif isinstance(body, list):
            for stmt in body:
                scan_stmt(stmt)
        else:
            scan_stmt(body)

        return indexed_params

    def _transform_function(self, func: FuncDecl) -> FuncDecl:
        """Transform function declaration (Rules 20-22, 46-48)."""
        # Save current function params context
        old_params = self.current_function_params
        self.current_function_params = {}

        # PRIORITY 1 FIX: Save and set function context
        old_in_function = self.in_user_function
        old_series_vars = self.series_variables
        old_local_vars = self.function_local_vars

        self.in_user_function = True
        self.series_variables = self._find_series_variables_in_function(
            func.body if isinstance(func.body, list) else [func.body]
        )
        self.function_local_vars = set()

        # Find which parameters are indexed in the function body
        param_names = {p.name for p in func.params}
        indexed_params = self._find_indexed_parameters(func.body, param_names)

        # BUG FIX: Detect parameter shadowing - when parameter name matches a module name
        # that is used in the default value. This is a Python limitation where default values
        # are evaluated at function definition time, creating ambiguity.
        # Solution: Generate aliased imports for shadowed modules (e.g., position as _position_module)
        for param in func.params:
            if param.default and param.name in KNOWN_MODULES:
                # Parameter name matches a known module - check if default value references that module
                module_refs = self._get_module_references_in_expr(param.default)
                if param.name in module_refs:
                    # Shadowing detected! Generate alias for this module
                    alias_name = f"_{param.name}_module"
                    self.import_resolver.aliased_modules[param.name] = alias_name
                    # Also ensure the module is in lib_modules so it gets imported
                    self.import_resolver.lib_modules.add(param.name)

        # Transform parameters
        # For methods, the first parameter has special syntax
        transformed_params = []
        for i, param in enumerate(func.params):
            # Convert Pine generic types to Python types
            if param.type_hint:
                type_hint = convert_pine_generic_to_python(param.type_hint)
                # Wrap in Series[T] if parameter is indexed with [n] history access.
                # The runtime SeriesTransformer will create SeriesImpl and register it.
                if param.name in indexed_params:
                    type_hint = f'Series[{type_hint}]'
            else:
                # If parameter is indexed and has no type hint, add bare Series
                type_hint = 'Series' if param.name in indexed_params else None

            # Track parameter types for method call transformation
            if type_hint:
                self.current_function_params[param.name] = type_hint

            # Rewrite default value to use module aliases if needed
            default_value = None
            if param.default:
                # Check if this default value needs module alias rewriting
                if param.name in self.import_resolver.aliased_modules:
                    default_value = self._rewrite_module_references(param.default)
                else:
                    default_value = param.default
                # Now transform the (possibly rewritten) default value
                default_value = self._transform_expression(default_value)

            transformed_params.append(Parameter(
                name=param.name,
                type_hint=type_hint,
                default=default_value
            ))

        # Transform body
        if isinstance(func.body, Expression):
            # Single-line arrow function (Rule 21)
            transformed_body = ReturnStatement(expr=self._transform_expression(func.body))
        else:
            # Multi-line function (Rule 22)
            transformed_body = [self._transform_statement(stmt) for stmt in func.body]
            # PRIORITY 1 FIX: Flatten nested lists (from Series variable initialization)
            # and remove None entries
            flattened = []
            for s in transformed_body:
                if s is None:
                    continue
                elif isinstance(s, list):
                    flattened.extend(s)
                else:
                    flattened.append(s)
            transformed_body = flattened

            if transformed_body:
                last = transformed_body[-1]
                if isinstance(last, ExpressionStatement):
                    # Add explicit return for last expression (Rule 20)
                    transformed_body[-1] = ReturnStatement(expr=last.expr)
                elif isinstance(last, Assignment):
                    # Last statement is an assignment - Pine Script returns the assigned value
                    # Both `=` and `:=` assignments should return the target
                    target_name = last.target if isinstance(last.target, str) else last.target.name
                    transformed_body.append(ReturnStatement(
                        expr=Identifier(name=target_name)
                    ))
                elif isinstance(last, IfStatement):
                    # If function ends with if/else block, apply __block_result__ pattern
                    transformed_body = self._apply_block_result_pattern(transformed_body)

        # Restore previous params context
        self.current_function_params = old_params

        # PRIORITY 1 FIX: Restore function context
        self.in_user_function = old_in_function
        self.series_variables = old_series_vars
        self.function_local_vars = old_local_vars

        # Track method functions for call transformation
        if func.is_method:
            self.method_functions.add(func.name)

        # Sanitize function name if it clashes with Python reserved words
        from .pine_builtins import sanitize_identifier
        sanitized_name = sanitize_identifier(func.name)
        if sanitized_name != func.name:
            self._renamed_functions[func.name] = sanitized_name

        return FuncDecl(
            name=sanitized_name,
            params=transformed_params,
            body=transformed_body,
            is_method=func.is_method,
            is_export=False,  # Remove export keyword (Rule 46)
            indexed_params=indexed_params,
        )

    def _resolve_overloaded_functions(self, main_body: List[Statement]) -> None:
        """Resolve function overloading by renaming overloads and generating dispatchers.

        Pine Script supports function overloading (same name, different param types/counts).
        Python does not, so we rename each overload and generate a dispatcher function
        that checks argument count and types at runtime.
        """
        # Collect function declarations by name
        func_groups: Dict[str, List[tuple]] = {}  # name -> [(index, FuncDecl)]
        for i, stmt in enumerate(main_body):
            if isinstance(stmt, FuncDecl):
                if stmt.name not in func_groups:
                    func_groups[stmt.name] = []
                func_groups[stmt.name].append((i, stmt))

        # Only process groups with multiple overloads
        overloaded = {name: group for name, group in func_groups.items() if len(group) > 1}
        if not overloaded:
            return

        # Process each overloaded group
        # Work backwards to preserve indices when inserting
        for name, group in sorted(overloaded.items(), key=lambda x: x[1][-1][0], reverse=True):
            # Rename each overload
            for j, (idx, func) in enumerate(group, 1):
                func.name = f'_{name}_{j}'

            # Generate dispatcher and insert after last overload
            dispatcher = self._generate_overload_dispatcher(name, [func for _, func in group])
            last_idx = group[-1][0]
            dispatcher.line = main_body[last_idx].line
            main_body.insert(last_idx + 1, dispatcher)

    def _generate_overload_dispatcher(self, name: str, overloads: List[FuncDecl]) -> RawCode:
        """Generate a dispatcher function for overloaded functions.

        Dispatches by argument count first, then by type of the first argument.
        """
        # Group overloads by parameter count
        by_count: Dict[int, List[FuncDecl]] = {}
        for func in overloads:
            count = len(func.params)
            if count not in by_count:
                by_count[count] = []
            by_count[count].append(func)

        lines = [f'def {name}(*_args):']

        # Generate dispatch logic
        counts = sorted(by_count.keys())
        for ci, count in enumerate(counts):
            funcs = by_count[count]
            prefix = 'if' if ci == 0 else 'elif'
            lines.append(f'    {prefix} len(_args) == {count}:')

            if len(funcs) == 1:
                # Single overload for this arg count — call directly
                args = ', '.join(f'_args[{i}]' for i in range(count))
                lines.append(f'        return {funcs[0].name}({args})')
            else:
                # Multiple overloads — dispatch by type of first argument
                for fi, func in enumerate(funcs):
                    type_hint = func.params[0].type_hint if func.params else None
                    check = self._type_check_expr('_args[0]', type_hint)
                    fi_prefix = 'if' if fi == 0 else 'elif'
                    if fi == len(funcs) - 1:
                        # Last overload is the fallback
                        args = ', '.join(f'_args[{i}]' for i in range(count))
                        lines.append(f'        else:')
                        lines.append(f'            return {func.name}({args})')
                    else:
                        args = ', '.join(f'_args[{i}]' for i in range(count))
                        lines.append(f'        {fi_prefix} {check}:')
                        lines.append(f'            return {func.name}({args})')

        return RawCode(code='\n'.join(lines))

    @staticmethod
    def _type_check_expr(var: str, type_hint: Optional[str]) -> str:
        """Generate isinstance check expression for a Pine type hint."""
        if not type_hint:
            return 'True'
        # Map Pine/Python type hints to isinstance checks
        # bool must come before int/float since bool is a subclass of int
        type_map = {
            'bool': f'isinstance({var}, bool)',
            'int': f'isinstance({var}, int) and not isinstance({var}, bool)',
            'float': f'isinstance({var}, (int, float)) and not isinstance({var}, bool)',
            'str': f'isinstance({var}, str)',
            'Color': f'not isinstance({var}, (bool, int, float, str))',
        }
        return type_map.get(type_hint, 'True')

    def _apply_block_result_pattern(self, body: List[Statement]) -> List[Statement]:
        """Apply __block_result__ pattern when function ends with an if statement.

        Only applies to the LAST statement in the body when it's an IfStatement.
        Inserts __block_result__ = na before it, converts last expression in each
        branch to __block_result__ = expr, and adds return __block_result__ after.
        """
        if not body:
            return body

        last = body[-1]
        if not isinstance(last, IfStatement):
            return body

        # Build new body: all statements before the last if, then the pattern
        new_body = list(body[:-1])

        # Insert __block_result__ = na before the if
        new_body.append(Assignment(
            target='__block_result__',
            value=Identifier(name='na'),
            type_hint=None
        ))
        # Ensure na is imported (this assignment is created after import analysis)
        self.import_resolver.lib_modules.add('na')

        # Transform the last if statement for __block_result__
        new_body.append(self._transform_if_for_block_result(last))

        # Add return __block_result__
        new_body.append(ReturnStatement(expr=Identifier(name='__block_result__')))

        return new_body

    def _transform_if_for_block_result(self, if_stmt: IfStatement) -> IfStatement:
        """Transform if statement to assign last expression to __block_result__."""
        # Transform if body
        new_if_body = list(if_stmt.body)
        if new_if_body:
            last_stmt = new_if_body[-1]

            # Determine what to assign to __block_result__
            if isinstance(last_stmt, Assignment):
                # After assignment, capture the assigned value
                # e.g., this.max_win = math.max(...) -> also assign this.max_win to __block_result__
                new_if_body.append(Assignment(
                    target='__block_result__',
                    value=Identifier(name=last_stmt.target) if isinstance(last_stmt.target, str) else last_stmt.target,
                    type_hint=None
                ))
            elif isinstance(last_stmt, ExpressionStatement):
                # Replace expression statement with __block_result__ assignment
                new_if_body[-1] = Assignment(
                    target='__block_result__',
                    value=last_stmt.expr,
                    type_hint=None
                )
            elif isinstance(last_stmt, ReturnStatement):
                # Replace return with __block_result__ assignment
                new_if_body[-1] = Assignment(
                    target='__block_result__',
                    value=last_stmt.expr,
                    type_hint=None
                )

        # Transform else body
        new_else_body = None
        if if_stmt.else_body:
            new_else_body = list(if_stmt.else_body)
            if new_else_body:
                last_stmt = new_else_body[-1]

                # Determine what to assign to __block_result__
                if isinstance(last_stmt, Assignment):
                    # After assignment, capture the assigned value
                    new_else_body.append(Assignment(
                        target='__block_result__',
                        value=Identifier(name=last_stmt.target) if isinstance(last_stmt.target, str) else last_stmt.target,
                        type_hint=None
                    ))
                elif isinstance(last_stmt, ExpressionStatement):
                    # Replace expression statement with __block_result__ assignment
                    new_else_body[-1] = Assignment(
                        target='__block_result__',
                        value=last_stmt.expr,
                        type_hint=None
                    )
                elif isinstance(last_stmt, ReturnStatement):
                    # Replace return with __block_result__ assignment
                    new_else_body[-1] = Assignment(
                        target='__block_result__',
                        value=last_stmt.expr,
                        type_hint=None
                    )

        return IfStatement(
            condition=if_stmt.condition,
            body=new_if_body,
            else_body=new_else_body
        )

    # ========================================================================
    # Statement transformations
    # ========================================================================

    def _transform_statement(self, stmt: Statement) -> Union[Statement, List[Statement], None]:
        """Transform statement."""
        # Clear hoisted statements before transforming
        self._hoisted_stmts = []

        if isinstance(stmt, Assignment):
            result = self._transform_assignment(stmt)
        elif isinstance(stmt, Reassignment):
            result = self._transform_reassignment(stmt)
        elif isinstance(stmt, (VarDecl, VaripDecl)):
            result = self._transform_var_declaration(stmt)
        elif isinstance(stmt, IfStatement):
            result = self._transform_if_statement(stmt)
        elif isinstance(stmt, ForLoop):
            result = self._transform_for_loop(stmt)
        elif isinstance(stmt, ForInLoop):
            result = self._transform_for_in_loop(stmt)
        elif isinstance(stmt, WhileLoop):
            result = self._transform_while_loop(stmt)
        elif isinstance(stmt, SwitchStatement):
            result = self._transform_switch_statement(stmt)
        elif isinstance(stmt, ExpressionStatement):
            # Skip stray declaration calls (indicator/strategy/library) in body
            if isinstance(stmt.expr, FunctionCall) and isinstance(stmt.expr.func, str):
                if stmt.expr.func in ('indicator', 'strategy', 'library'):
                    return None
            result = ExpressionStatement(expr=self._transform_expression(stmt.expr))
        elif isinstance(stmt, (BreakStatement, ContinueStatement)):
            result = stmt
        else:
            result = stmt

        # Prepend any hoisted statements (e.g. from FunctionCall[N] extraction)
        hoisted = self._hoisted_stmts
        self._hoisted_stmts = []
        if hoisted:
            if isinstance(result, list):
                return hoisted + result
            elif result is not None:
                return hoisted + [result]
            return hoisted
        return result

    def _transform_assignment(self, stmt: Assignment) -> Union[Assignment, List[Statement]]:
        """Transform assignment (Rules 5-7)."""
        # Handle for/for-in loop used as value expression:
        # x = for i in arr ... body → for i in arr: body; x = last_expr
        if isinstance(stmt.value, (ForInLoop, ForLoop)):
            return self._transform_for_as_expression(stmt)

        # Handle TupleDestructure where any target needs Series annotation.
        # Split: [a, b] = func() → _tmp = func(); a: Series[float] = _tmp[0]; b = _tmp[1]
        if isinstance(stmt.target, TupleDestructure):
            indexed_names = self.type_inference.indexed_var_names
            needs_split = any(n in indexed_names for n in stmt.target.names)
            if needs_split:
                transformed_value = self._transform_expression(stmt.value)
                temp_name = '_tuple_tmp'
                statements = [Assignment(target=temp_name, value=transformed_value)]
                for index, name in enumerate(stmt.target.names):
                    index_expr = IndexAccess(
                        object=Identifier(name=temp_name),
                        index=Literal(value=index, literal_type='int'),
                    )
                    if name in indexed_names:
                        sym = self.symbol_table.lookup(name)
                        base_type = (sym.type_hint if sym and sym.type_hint else
                                     self.type_inference.infer_type_hint(stmt.value))
                        if base_type in ('Any', None):
                            base_type = 'float'
                        series_type_hint = f"Series[{base_type}]"
                        self.import_resolver.uses_series = True
                    else:
                        series_type_hint = None
                    statements.append(Assignment(target=name, value=index_expr, type_hint=series_type_hint))
                return statements

        # PRIORITY 1 FIX: Track first assignment to Series variables in functions
        is_first_assignment_to_series = False
        if self.in_user_function and isinstance(stmt.target, str):
            if stmt.target in self.series_variables and stmt.target not in self.function_local_vars:
                is_first_assignment_to_series = True
                self.function_local_vars.add(stmt.target)

        # Determine type annotation first (we need it for na value transformation)
        type_hint = None
        converted_type = None
        if isinstance(stmt.target, str):
            symbol = self.symbol_table.lookup(stmt.target)
            # CRITICAL: In Pine Script, series is the default type qualifier
            if hasattr(stmt, 'type_hint') and stmt.type_hint:
                # Use the explicit type hint from parser (e.g., map<int, float> mapExample = na)
                converted_type = convert_pine_generic_to_python(stmt.type_hint)
                # If variable is used with historical reference [], wrap in Series[type]
                if symbol and symbol.is_indexed:
                    type_hint = f"Series[{converted_type}]"
                    self.import_resolver.uses_series = True
                else:
                    type_hint = converted_type
            elif symbol:
                # No explicit type in Pine source — only infer for na value transformation
                # and var/varip Persistent wrapping
                if symbol.kind in (VariableKind.VAR, VariableKind.VARIP):
                    base_type = symbol.type_hint or self.type_inference.infer_type_hint(stmt.value)
                    converted_type = base_type
                    type_hint = f"Persistent[{base_type}]"
                elif isinstance(stmt.value, NaLiteral):
                    # Need type for na() conversion even without explicit annotation
                    base_type = symbol.type_hint or self.type_inference.infer_type_hint(stmt.value)
                    converted_type = base_type
                elif isinstance(stmt.target, str) and stmt.target in getattr(self, '_reassignment_vars', set()) and symbol.is_indexed:
                    # Variable uses both := (reassignment) and [] (historical reference)
                    # WITHOUT 'var' keyword - this is a Series in Pine Script semantics
                    base_type = symbol.type_hint or self.type_inference.infer_type_hint(stmt.value)
                    if base_type == 'Any':
                        base_type = 'float'
                    converted_type = base_type
                    type_hint = f"Series[{base_type}]"
                    self.import_resolver.uses_series = True
                elif isinstance(stmt.target, str) and symbol.is_indexed:
                    # Variable is used with [] - in Pine Script, series is the default type qualifier
                    base_type = symbol.type_hint or self.type_inference.infer_type_hint(stmt.value)
                    if base_type == 'Any':
                        base_type = 'float'
                    converted_type = base_type
                    type_hint = f"Series[{base_type}]"
                    self.import_resolver.uses_series = True
                # Otherwise: no type annotation for body assignments without explicit Pine types

        # Transform value - special case for na with type hint
        if isinstance(stmt.value, NaLiteral) and converted_type:
            # Convert na to na(Type) where Type is the declared type
            value = FunctionCall(func='na', args=[Identifier(name=converted_type)], kwargs={})
        else:
            value = self._transform_expression(stmt.value)

        # Suppress 'Any' type annotations - omit annotation when type is unknown
        if type_hint == 'Any':
            type_hint = None

        # PRIORITY 1 FIX: For Series variables in functions, add Series type hint
        if is_first_assignment_to_series:
            self.import_resolver.uses_series = True
            # Get the base type for the Series (default to float — Pine's default series type)
            base_type = converted_type or self.type_inference.infer_type_hint(stmt.value)
            if base_type == 'Any':
                base_type = 'float'
            type_hint = f"Series[{base_type}]"

        return Assignment(target=stmt.target, value=value, type_hint=type_hint)

    def _transform_reassignment(self, stmt: Reassignment) -> Assignment:
        """
        Transform reassignment := to = (Rule 11).
        """
        # Default behavior: regular assignment
        # PRIORITY 1 FIX: Series type hint will be added during assignment transformation
        return Assignment(
            target=stmt.target,
            value=self._transform_expression(stmt.value)
        )

    def _transform_var_declaration(self, decl: Union[VarDecl, VaripDecl]) -> Assignment:
        """Transform var/varip declaration to Persistent or Series assignment (Rule 5-6).

        Scalar types (int, float, bool, str) use Series[T] = nz(name[1], initial)
        to support both persistence and [n] subscripting after reassignment.
        Complex types (dict, list, objects) use Persistent[T] since they're
        modified in-place and don't need reassignment.
        """
        base_type = decl.type_hint or self.type_inference.infer_type_hint(decl.value)
        # Convert generic types properly
        converted_type = convert_pine_generic_to_python(base_type)

        # Scalar types that support the Series + nz pattern for var emulation
        SCALAR_TYPES = {'int', 'float', 'bool', 'str'}

        # Custom UDT detection: uppercase name, not a built-in type, no generic brackets
        is_custom_udt = (
            converted_type and
            converted_type[0].isupper() and
            not converted_type.startswith(('Color', 'Line', 'Label', 'Box', 'Table', 'Matrix', 'Polyline', 'LineFill')) and
            '[' not in converted_type
        )

        # Special case: if value is na, convert to na(Type) function call
        value = decl.value
        if isinstance(value, NaLiteral):
            value = FunctionCall(func='na', args=[Identifier(name=converted_type)], kwargs={})
        else:
            value = self._transform_expression(value)

        # Scalar var declarations: use PersistentSeries[T] = initial
        # PersistentSeriesTransformer splits this into Persistent[T] + Series[T],
        # avoiding the self-referencing [1] off-by-one bug in the old nz(name[1], initial) pattern
        if converted_type in SCALAR_TYPES:
            type_hint = f"PersistentSeries[{converted_type}]"
            self.import_resolver.uses_persistent_series = True
        elif converted_type == 'Any':
            type_hint = 'Persistent'
        elif is_custom_udt:
            # Custom UDTs need Persistent to survive across bars (var semantics)
            type_hint = f"Persistent[{converted_type}]"
        else:
            type_hint = f"Persistent[{converted_type}]"

        return Assignment(
            target=decl.name,
            value=value,
            type_hint=type_hint
        )

    def _transform_if_statement(self, stmt: IfStatement) -> IfStatement:
        """Transform if statement (Rule 15)."""
        condition = self._transform_expression(stmt.condition)
        # Save hoisted stmts from condition before body transforms clear them
        condition_hoisted = list(self._hoisted_stmts)
        self._hoisted_stmts = []
        body = [self._transform_statement(s) for s in stmt.body]

        elseifs = []
        for elif_cond, elif_body in stmt.elseifs:
            transformed_cond = self._transform_expression(elif_cond)
            # Save hoisted stmts from elif condition
            elif_hoisted = list(self._hoisted_stmts)
            self._hoisted_stmts = []
            transformed_body = [self._transform_statement(s) for s in elif_body]
            condition_hoisted.extend(elif_hoisted)
            elseifs.append((transformed_cond, transformed_body))

        else_body = None
        if stmt.else_body:
            else_body = [self._transform_statement(s) for s in stmt.else_body]

        # Restore all condition hoisted stmts so they propagate to enclosing _transform_statement
        self._hoisted_stmts = condition_hoisted
        return IfStatement(
            condition=condition,
            body=body,
            elseifs=elseifs,
            else_body=else_body
        )

    def _transform_for_loop(self, stmt: ForLoop) -> ForLoop:
        """Transform for loop to range() (Rules 12-13)."""
        from_val = self._transform_expression(stmt.from_val)
        to_val = self._transform_expression(stmt.to_val)
        step = self._transform_expression(stmt.step) if stmt.step else None
        # Save hoisted stmts from expressions before body transforms clear them
        expr_hoisted = list(self._hoisted_stmts)
        self._hoisted_stmts = []

        body = [self._transform_statement(s) for s in stmt.body]

        # Restore expression hoisted stmts
        self._hoisted_stmts = expr_hoisted
        return ForLoop(
            var=stmt.var,
            from_val=from_val,
            to_val=to_val,
            step=step,
            body=body
        )

    def _transform_for_in_loop(self, stmt: ForInLoop) -> ForInLoop:
        """Transform for-in loop (Rule 14)."""
        iterable = self._transform_expression(stmt.iterable)
        # Save hoisted stmts from iterable before body transforms clear them
        expr_hoisted = list(self._hoisted_stmts)
        self._hoisted_stmts = []
        body = [self._transform_statement(s) for s in stmt.body]

        # Restore expression hoisted stmts
        self._hoisted_stmts = expr_hoisted
        return ForInLoop(
            vars=stmt.vars,
            iterable=iterable,
            body=body
        )

    def _transform_for_as_expression(self, stmt: Assignment) -> List[Statement]:
        """Transform for/for-in loop used as a value expression.
        x = for i in arr ... body → initialize x, then for loop with last expr assigning to x.
        """
        target = stmt.target if isinstance(stmt.target, str) else str(stmt.target)
        loop = stmt.value

        # Initialize variable with empty string or na
        type_hint = None
        if hasattr(stmt, 'type_hint') and stmt.type_hint:
            type_hint = convert_pine_generic_to_python(stmt.type_hint)
        init_value = Literal(value='', literal_type='string') if type_hint == 'str' else NaLiteral()
        init_stmt = Assignment(target=target, value=init_value, type_hint=type_hint)

        # Transform loop body — the last expression becomes an assignment to target
        body = [self._transform_statement(s) for s in loop.body if s is not None]
        if body:
            last = body[-1]
            if isinstance(last, ExpressionStatement):
                # Replace last ExpressionStatement with assignment to target
                body[-1] = Assignment(target=target, value=last.expr)
            elif isinstance(last, (Assignment, Reassignment)):
                # Add re-assignment of target after the last statement
                value = Identifier(name=last.target if isinstance(last.target, str) else str(last.target))
                body.append(Assignment(target=target, value=value))

        # Build the transformed loop
        if isinstance(loop, ForInLoop):
            transformed_loop = ForInLoop(
                vars=loop.vars,
                iterable=self._transform_expression(loop.iterable),
                body=body
            )
        else:
            transformed_loop = ForLoop(
                var=loop.var,
                from_val=self._transform_expression(loop.from_val),
                to_val=self._transform_expression(loop.to_val),
                step=self._transform_expression(loop.step) if loop.step else None,
                body=body
            )

        return [init_stmt, transformed_loop]

    def _transform_while_loop(self, stmt: WhileLoop) -> WhileLoop:
        """Transform while loop."""
        condition = self._transform_expression(stmt.condition)
        # Save hoisted stmts from condition before body transforms clear them
        condition_hoisted = list(self._hoisted_stmts)
        self._hoisted_stmts = []
        body = [self._transform_statement(s) for s in stmt.body]

        # Restore condition hoisted stmts
        self._hoisted_stmts = condition_hoisted
        return WhileLoop(condition=condition, body=body)

    def _transform_switch_statement(self, stmt: SwitchStatement) -> IfStatement:
        """Transform switch to if/elif/else chain (Rules 41-42)."""
        # Convert switch cases to if/elif/else
        if not stmt.cases:
            return None

        all_expr_hoisted = []

        # First case becomes if
        first_case_expr, first_case_body = stmt.cases[0]

        if stmt.expr:
            # switch expr => cases compare against expr
            first_condition = BinaryOp(
                left=self._transform_expression(stmt.expr),
                op='==',
                right=self._transform_expression(first_case_expr)
            )
        else:
            # switch without expr => cases are boolean conditions
            first_condition = self._transform_expression(first_case_expr)

        # Save hoisted stmts from first condition
        all_expr_hoisted.extend(self._hoisted_stmts)
        self._hoisted_stmts = []
        transformed_body = [self._transform_statement(s) for s in first_case_body]

        # Remaining cases become elif
        elseifs = []
        for case_expr, case_body in stmt.cases[1:]:
            if stmt.expr:
                elif_condition = BinaryOp(
                    left=self._transform_expression(stmt.expr),
                    op='==',
                    right=self._transform_expression(case_expr)
                )
            else:
                elif_condition = self._transform_expression(case_expr)

            # Save hoisted stmts from elif condition
            all_expr_hoisted.extend(self._hoisted_stmts)
            self._hoisted_stmts = []
            elif_body = [self._transform_statement(s) for s in case_body]
            elseifs.append((elif_condition, elif_body))

        # Default case becomes else
        else_body = None
        if stmt.default:
            else_body = [self._transform_statement(s) for s in stmt.default]

        # Restore all expression hoisted stmts
        self._hoisted_stmts = all_expr_hoisted
        return IfStatement(
            condition=first_condition,
            body=transformed_body,
            elseifs=elseifs,
            else_body=else_body
        )

    # ========================================================================
    # Expression transformations
    # ========================================================================

    def _transform_expression(self, expr: Expression) -> Expression:
        """Transform expression."""
        if isinstance(expr, BinaryOp):
            return BinaryOp(
                left=self._transform_expression(expr.left),
                op=expr.op,
                right=self._transform_expression(expr.right)
            )
        elif isinstance(expr, UnaryOp):
            return UnaryOp(
                op=expr.op,
                operand=self._transform_expression(expr.operand)
            )
        elif isinstance(expr, TernaryOp):
            return self._transform_ternary(expr)
        elif isinstance(expr, FunctionCall):
            return self._transform_function_call(expr)
        elif isinstance(expr, MethodCall):
            return self._transform_method_call(expr)
        elif isinstance(expr, MemberAccess):
            return self._transform_member_access(expr)
        elif isinstance(expr, IndexAccess):
            return self._transform_index_access(expr)
        elif isinstance(expr, Identifier):
            return self._transform_identifier(expr)
        elif isinstance(expr, Literal):
            return self._transform_literal(expr)
        elif isinstance(expr, NaLiteral):
            return self._transform_na_literal(expr)
        elif isinstance(expr, ArrayLiteral):
            return ArrayLiteral(elements=[self._transform_expression(e) for e in expr.elements])
        elif isinstance(expr, IfExpression):
            return self._transform_if_expression(expr)
        elif isinstance(expr, SwitchExpression):
            return self._transform_switch_expression(expr)
        else:
            return expr

    def _transform_ternary(self, expr: TernaryOp) -> TernaryOp:
        """Transform ternary operator (Rule 10): cond ? a : b => a if cond else b"""
        # Note: We keep it as TernaryOp, codegen will convert to Python syntax
        return TernaryOp(
            condition=self._transform_expression(expr.condition),
            true_expr=self._transform_expression(expr.true_expr),
            false_expr=self._transform_expression(expr.false_expr)
        )

    def _transform_generic_function_name(self, func_name: str) -> str:
        """Transform function name with generics."""
        if '<' not in func_name:
            return func_name

        # For map.new<K,V>() -> map.new() (just strip generics)
        if 'map.new<' in func_name or func_name.startswith('map.new<'):
            return 'map.new'

        # For array.new<T>() -> array.new_T()
        if 'array.new<' in func_name or func_name.startswith('array.new<'):
            # Extract the type from array.new<type>
            start = func_name.index('<') + 1
            end = func_name.index('>')
            type_param = func_name[start:end].strip()

            # Map Pine Script types to PyneCore array.new_* functions
            type_map = {
                'string': 'array.new_string',
                'int': 'array.new_int',
                'float': 'array.new_float',
                'bool': 'array.new_bool',
                'color': 'array.new_color',
                'line': 'array.new_line',
                'label': 'array.new_label',
                'box': 'array.new_box',
                'table': 'array.new_table',
            }

            return type_map.get(type_param, 'list')

        # For matrix.new<T>() -> matrix.new() (just strip generics)
        if 'matrix.new<' in func_name or func_name.startswith('matrix.new<'):
            return 'matrix.new'

        # For other generic functions, strip generics for now
        base_name = func_name[:func_name.index('<')]
        return base_name

    def _parse_dotted_path(self, path: str) -> Expression:
        """Parse a dotted path string like 'obj.lbl' into nested MemberAccess expressions."""
        parts = path.split('.')
        if len(parts) == 1:
            return Identifier(name=parts[0])

        # Build nested MemberAccess from left to right
        expr = Identifier(name=parts[0])
        for part in parts[1:]:
            expr = MemberAccess(object=expr, member=part)
        return expr

    def _resolve_module_for_method(self, method_name: str, type_str: str = None) -> Optional[str]:
        """Resolve PyneCore module name from method name and/or object type.

        Two modes:
        - Type-aware (type_str provided): derive module from type string.
        - Heuristic (no type_str): infer module from method name using canonical sets.
        """
        if type_str:
            # Unwrap Persistent[X] or Series[X]
            if type_str.startswith('Persistent[') or type_str.startswith('Series['):
                start = type_str.index('[') + 1
                end = type_str.rindex(']')
                type_str = type_str[start:end]

            if type_str.startswith('list'):
                return 'array'
            elif type_str.startswith('dict'):
                return 'map'
            elif type_str in ('line', 'label', 'box', 'table', 'linefill',
                              'Line', 'Label', 'Box', 'Table'):
                return type_str.lower()
            elif type_str.startswith('matrix') or type_str.startswith('Matrix'):
                return 'matrix'
            elif '<' in type_str:
                return type_str[:type_str.index('<')]
            return None

        # Heuristic: infer module from method name alone.
        # Priority: drawing-specific > map-unique > array-unique > shared (map default)
        if method_name in LABEL_METHODS:
            return 'label'
        elif method_name in LINE_METHODS:
            return 'line'
        elif method_name in BOX_METHODS:
            return 'box'
        elif method_name in TABLE_METHODS:
            return 'table'
        elif method_name in MAP_UNIQUE_METHODS:
            return 'map'
        elif method_name in ARRAY_METHODS:
            return 'array'
        elif method_name in SHARED_COLLECTION_METHODS:
            # Shared methods (get, set, size, etc.) — default to array since
            # it's the most common collection, but map/matrix may also use these.
            # Type-aware path should be preferred when possible.
            return 'array'
        elif method_name in MATRIX_METHODS:
            return 'matrix'
        return None

    def _transform_dotted_string_method_call(self, call: FunctionCall, func: str) -> Optional[FunctionCall]:
        """Handle var.method() → module.method(var, ...) for dotted string function names.

        Returns transformed FunctionCall or None if not a method call pattern.
        """
        parts = func.split('.', 1)  # Split only on first dot
        var_name = parts[0]
        method_name = parts[1] if len(parts) > 1 else None

        if not method_name or var_name[0].isupper():  # Not a variable method call
            return None

        # Check user-defined @method functions first
        if method_name in self.method_functions:
            transformed_args = [Identifier(name=var_name)] + [self._transform_expression(arg) for arg in call.args]
            transformed_kwargs = {k: self._transform_expression(v) for k, v in call.kwargs.items()}
            return FunctionCall(func=method_name, args=transformed_args, kwargs=transformed_kwargs)

        # Check if var_name has a known built-in type
        type_str = self.current_function_params.get(var_name)
        if not type_str:
            symbol = self.symbol_table.lookup(var_name)
            if symbol and symbol.type_hint:
                type_str = symbol.type_hint

        if type_str:
            module_name = self._resolve_module_for_method(method_name, type_str)
            if module_name:
                self.import_resolver.add_module(module_name)
                transformed_args = [Identifier(name=var_name)] + [self._transform_expression(arg) for arg in call.args]
                transformed_kwargs = {k: self._transform_expression(v) for k, v in call.kwargs.items()}
                return FunctionCall(func=f'{module_name}.{method_name}', args=transformed_args, kwargs=transformed_kwargs)

        return None

    def _transform_chained_dotted_method_call(self, call: FunctionCall, func: str) -> Optional[FunctionCall]:
        """Handle obj.field.method() → module.method(obj.field, ...) for chained access.

        Returns transformed FunctionCall or None if not a chained method pattern.
        """
        first_component = func.split('.')[0]
        if first_component in KNOWN_MODULES or first_component[0].isupper():
            return None

        parts = func.rsplit('.', 1)  # Split on LAST dot
        obj_path = parts[0]  # "obj.lbl"
        method_name = parts[1]  # "set_x"

        # Special case: .copy() on object fields -> udt_copy()
        # Must be checked BEFORE set matching (Location 1 behavior)
        if method_name == 'copy':
            obj_expr = self._parse_dotted_path(obj_path)
            transformed_args = [obj_expr] + [self._transform_expression(arg) for arg in call.args]
            transformed_kwargs = {k: self._transform_expression(v) for k, v in call.kwargs.items()}
            self.import_resolver.uses_udt_copy = True
            return FunctionCall(func='udt_copy', args=transformed_args, kwargs=transformed_kwargs)

        module_name = self._resolve_module_for_method(method_name)
        if module_name:
            self.import_resolver.add_module(module_name)
            obj_expr = self._parse_dotted_path(obj_path)
            transformed_args = [obj_expr] + [self._transform_expression(arg) for arg in call.args]
            transformed_kwargs = {k: self._transform_expression(v) for k, v in call.kwargs.items()}
            return FunctionCall(func=f'{module_name}.{method_name}', args=transformed_args, kwargs=transformed_kwargs)

        return None

    def _transform_member_access_method_call(self, call: FunctionCall, func: MemberAccess) -> Optional[FunctionCall]:
        """Handle MemberAccess-based method calls like obj.lbl.set_x(...).

        Returns transformed FunctionCall or None if not a recognized method.
        """
        method_name = func.member
        module_name = self._resolve_module_for_method(method_name)

        if module_name:
            self.import_resolver.add_module(module_name)
            obj_expr = self._transform_expression(func.object)
            transformed_args = [obj_expr] + [self._transform_expression(arg) for arg in call.args]
            transformed_kwargs = {k: self._transform_expression(v) for k, v in call.kwargs.items()}
            return FunctionCall(func=f'{module_name}.{method_name}', args=transformed_args, kwargs=transformed_kwargs)

        return None

    def _transform_function_call(self, call: FunctionCall) -> FunctionCall:
        """Transform function call (Rules 23-25, 49-51)."""
        func = call.func
        if isinstance(func, str):
            # Apply renamed function mapping (e.g., pass -> pass_)
            if func in self._renamed_functions:
                func = self._renamed_functions[func]
                call = FunctionCall(func=func, args=call.args, kwargs=call.kwargs)

            # Convert excess positional args to keyword args for input.* calls.
            # Pine Script allows input.int(defval, title, minval, maxval, ...)
            # but PyneCore requires minval=, maxval=, ... as keyword-only.
            from .pine_builtins import INPUT_POSITIONAL_PARAMS
            param_names = INPUT_POSITIONAL_PARAMS.get(func)
            if param_names and len(call.args) > 2:
                new_args = list(call.args[:2])
                new_kwargs = dict(call.kwargs)
                for i, val in enumerate(call.args[2:]):
                    if i < len(param_names):
                        new_kwargs[param_names[i]] = val
                call = FunctionCall(func=func, args=new_args, kwargs=new_kwargs)

            # Handle method calls disguised as dotted function calls
            if '.' in func and '<' not in func and '.' not in func.split('.')[0]:
                # Try var.method() → module.method(var, ...)
                result = self._transform_dotted_string_method_call(call, func)
                if result:
                    return result

                # Try chained member access: obj.lbl.set_x → label.set_x(obj.lbl, ...)
                if '.' in func:
                    result = self._transform_chained_dotted_method_call(call, func)
                    if result:
                        return result

            # SPECIAL CASE: timeframe.in_seconds() with no args
            if func == 'timeframe.in_seconds' and len(call.args) == 0 and not call.kwargs:
                return FunctionCall(
                    func='timeframe.in_seconds',
                    args=[MemberAccess(object=Identifier(name='timeframe'), member='period')],
                    kwargs={},
                )

            # SPECIAL CASE: array.max(a, nth) / array.min(a, nth) with 2 args
            if func in ('array.max', 'array.min') and len(call.args) == 2:
                self.import_resolver.add_module('order')
                arr_expr = self._transform_expression(call.args[0])
                nth_expr = self._transform_expression(call.args[1])
                sort_order = 'order.descending' if func == 'array.max' else 'order.ascending'
                inner = FunctionCall(
                    func='array.sort_indices',
                    args=[arr_expr, Identifier(name=sort_order)],
                    kwargs={},
                )
                idx = FunctionCall(func='array.get', args=[inner, nth_expr], kwargs={})
                return FunctionCall(func='array.get', args=[arr_expr, idx], kwargs={})

            # Handle generic syntax (map.new<K,V>, array.new<T>, etc.)
            if '<' in func:
                func = self._transform_generic_function_name(func)
            elif '.new' in func:
                base_name = func.replace('.new', '')
                if base_name and base_name[0].isupper():
                    func = base_name
            elif func == 'int':
                func = 'cast_int'
                self.import_resolver.uses_cast_int = True
            elif func == 'float':
                func = 'cast_float'
                self.import_resolver.uses_cast_float = True
            elif func in ('color', 'bool', 'string'):
                # Type cast: color(na) → na, color(x) → x
                # When arg is na, return plain na (Pine na is untyped at runtime)
                if len(call.args) == 1 and isinstance(call.args[0], NaLiteral):
                    self.import_resolver.add_module('na')
                    return NaLiteral()
                # For non-na args, pass through as identity (type casts are no-ops in Python)
                if len(call.args) == 1:
                    return self._transform_expression(call.args[0])
                func = func  # keep as-is for module usage
            else:
                func = get_function_name(func)
                if '.' in func:
                    parts = func.split('.')
                    parts[0] = get_module_name(parts[0])
                    func = '.'.join(parts)

        elif isinstance(func, MemberAccess):
            result = self._transform_member_access_method_call(call, func)
            if result:
                return result

            # Handle generics in member access
            if '<' in func.member:
                func.member = self._transform_generic_function_name(f"{func.object if isinstance(func.object, str) else 'obj'}.{func.member}")
                if func.member in ('dict', 'list'):
                    func = func.member
            else:
                func = self._transform_member_access(func)

        # Transform arguments
        transformed_args = [self._transform_expression(arg) for arg in call.args]
        transformed_kwargs = {k: self._transform_expression(v) for k, v in call.kwargs.items()}
        return FunctionCall(func=func, args=transformed_args, kwargs=transformed_kwargs)

    def _transform_method_call(self, call: MethodCall) -> Union[MethodCall, FunctionCall]:
        """Transform method call (Rule 48)."""
        # BUG FIX: Transform user-defined @method functions
        # Example: (expr).even_decimal() -> even_decimal(expr)
        # This must be first to handle methods on expressions, not just simple identifiers
        if call.method in self.method_functions:
            obj = self._transform_expression(call.object)
            args = [obj] + [self._transform_expression(arg) for arg in call.args]
            kwargs = {k: self._transform_expression(v) for k, v in call.kwargs.items()}
            return FunctionCall(func=call.method, args=args, kwargs=kwargs)

        # Special case: UDT.new() should become UDT() (remove .new())
        if call.method == 'new' and isinstance(call.object, Identifier):
            # Check if this is a UDT type (starts with uppercase)
            if call.object.name[0].isupper():
                # Convert to function call: Statistics.new() -> Statistics()
                args = [self._transform_expression(arg) for arg in call.args]
                kwargs = {k: self._transform_expression(v) for k, v in call.kwargs.items()}
                return FunctionCall(func=call.object.name, args=args, kwargs=kwargs)

        # Special case: .copy() on UDT object fields should become udt_copy()
        # Example: this.lbl.copy() -> udt_copy(this.lbl)
        if call.method == 'copy' and isinstance(call.object, MemberAccess):
            obj_expr = self._transform_expression(call.object)
            args = [obj_expr] + [self._transform_expression(arg) for arg in call.args]
            kwargs = {k: self._transform_expression(v) for k, v in call.kwargs.items()}
            self.import_resolver.uses_udt_copy = True
            return FunctionCall(func='udt_copy', args=args, kwargs=kwargs)

        # Transform method calls on built-in types to function calls
        # Example: id.push(value) -> array.push(id, value)
        type_str = None
        if isinstance(call.object, Identifier):
            # Look up the variable's type - check current function params first, then symbol table
            type_str = self.current_function_params.get(call.object.name)
            if not type_str:
                symbol = self.symbol_table.lookup(call.object.name)
                if symbol and symbol.type_hint:
                    type_str = symbol.type_hint
        elif isinstance(call.object, MemberAccess):
            # For member access like this.lbl, infer type from method name heuristically
            module_from_method = self._resolve_module_for_method(call.method)
            if module_from_method:
                type_str = {
                    'label': 'Label', 'line': 'Line', 'box': 'Box',
                    'table': 'Table', 'matrix': 'Matrix',
                }.get(module_from_method)

        # Resolve module: type-aware first, then heuristic fallback
        module_name = None
        if type_str:
            module_name = self._resolve_module_for_method(call.method, type_str)
        if not module_name:
            # Heuristic fallback when type is unknown
            module_name = self._resolve_module_for_method(call.method)

        if module_name and module_name in ('array', 'line', 'label', 'box', 'table', 'map', 'matrix', 'linefill'):
            # Ensure the resolved module is imported
            self.import_resolver.add_module(module_name)
            obj = self._transform_expression(call.object)
            args = [obj] + [self._transform_expression(arg) for arg in call.args]
            kwargs = {k: self._transform_expression(v) for k, v in call.kwargs.items()}
            func_name = f'{module_name}.{call.method}'
            return FunctionCall(func=func_name, args=args, kwargs=kwargs)

        obj = self._transform_expression(call.object)
        args = [self._transform_expression(arg) for arg in call.args]
        kwargs = {k: self._transform_expression(v) for k, v in call.kwargs.items()}

        return MethodCall(object=obj, method=call.method, args=args, kwargs=kwargs)

    def _transform_member_access(self, expr: MemberAccess) -> MemberAccess:
        """Transform member access (Rules 23-25)."""
        # Handle module renames
        if isinstance(expr.object, str):
            obj = get_module_name(expr.object)
        else:
            obj = self._transform_expression(expr.object)

        # Handle plot style remapping (Rule 25)
        if isinstance(expr.object, str):
            full_name = f"{expr.object}.{expr.member}"
            if needs_plot_style_remap(full_name):
                remapped = get_plot_style_remap(full_name)
                parts = remapped.split('.')
                return MemberAccess(object=parts[0], member=parts[1])

        return MemberAccess(object=obj, member=expr.member)

    def _transform_index_access(self, expr: IndexAccess) -> Expression:
        """Transform index access (Rule 52: warn about negative indices).

        When the object is a FunctionCall (e.g. ta.lowest(ohlc4, 50)[10]),
        hoist the call into a temp Series variable so [N] works at runtime.
        """
        obj = self._transform_expression(expr.object)
        index = self._transform_expression(expr.index)

        # Check for negative index (Rule 52)
        if isinstance(index, UnaryOp) and index.op == '-':
            pass

        # Hoist function calls used with history access [N]
        if isinstance(obj, FunctionCall):
            if not hasattr(self, '_hoisted_counter'):
                self._hoisted_counter = 0
            self._hoisted_counter += 1
            temp_name = f'_series_{self._hoisted_counter}'
            # Store hoisted assignment to be prepended by caller
            if not hasattr(self, '_hoisted_stmts'):
                self._hoisted_stmts = []
            self._hoisted_stmts.append(Assignment(
                target=temp_name, value=obj, type_hint='Series[float]'
            ))
            self.import_resolver.uses_series = True
            return IndexAccess(object=Identifier(name=temp_name), index=index)

        return IndexAccess(object=obj, index=index)

    def _transform_identifier(self, expr: Identifier) -> Identifier:
        """Transform identifier (Rule 23 for module prefixes)."""
        # Handle true/false conversion (Rule 17)
        if expr.name == 'true':
            return Literal(value=True, literal_type='bool')
        elif expr.name == 'false':
            return Literal(value=False, literal_type='bool')

        return Identifier(name=expr.name)

    def _transform_literal(self, expr: Literal) -> Literal:
        """Transform literal (Rule 17: true/false to True/False)."""
        if expr.literal_type == 'bool':
            # Ensure Python boolean values
            return Literal(value=bool(expr.value), literal_type='bool')
        return expr

    def _transform_na_literal(self, expr: NaLiteral) -> Identifier:
        """Transform na literal to na identifier (Rule 19)."""
        # In PyneCore, 'na' is used directly, not NA()
        return Identifier(name='na')

    def _extract_last_expression_from_block(self, block):
        """Extract the last expression from a block of statements.
        In Pine Script, the last evaluated expression in a block is the return value.
        """
        if isinstance(block, Expression):
            return self._transform_expression(block)
        if isinstance(block, list) and block:
            last = block[-1]
            # If the last item is an expression statement (ExpressionStatement or raw Expression)
            if isinstance(last, Expression):
                return self._transform_expression(last)
            # If it's an Assignment, the value becomes the return
            if hasattr(last, 'value') and isinstance(last.value, Expression):
                return self._transform_expression(last.value)
        return Literal(value=None, literal_type='none')

    def _transform_if_expression(self, expr: IfExpression) -> TernaryOp:
        """Transform if expression to Python inline if/else (Rule 16)."""
        # if cond\n    val1\nelse\n    val2  =>  val1 if cond else val2
        condition = self._transform_expression(expr.condition)

        # Handle single expression vs block
        true_expr = self._extract_last_expression_from_block(expr.true_expr)

        if expr.false_expr:
            false_expr = self._extract_last_expression_from_block(expr.false_expr)
        else:
            false_expr = None

        return TernaryOp(condition=condition, true_expr=true_expr, false_expr=false_expr)

    def _transform_switch_expression(self, expr: SwitchExpression) -> Expression:
        """Transform switch expression to nested ternary (if-elif-else chain)."""
        # Transform the switch expression to a chain of ternary operators
        # switch expr
        #     val1 => result1
        #     val2 => result2
        #     => default
        # Becomes: result1 if (expr == val1) else (result2 if (expr == val2) else default)

        # Or for switch without expr (condition-based):
        # switch
        #     cond1 => result1
        #     cond2 => result2
        #     => default
        # Becomes: result1 if cond1 else (result2 if cond2 else default)

        if not expr.cases:
            # No cases, just return default
            if expr.default:
                return self._extract_last_expression_from_block(expr.default)
            return Literal(value=None, literal_type='none')

        # Build nested ternary from right to left
        result = None
        if expr.default:
            result = self._extract_last_expression_from_block(expr.default)
        else:
            result = Literal(value=None, literal_type='none')

        # Process cases in reverse order to build nested ternary
        for case_cond, case_value in reversed(expr.cases):
            # Transform case value
            value_expr = self._extract_last_expression_from_block(case_value)

            # Build condition
            if expr.expr:
                # With expression: (expr == case_cond)
                condition = BinaryOp(
                    left=self._transform_expression(expr.expr),
                    op='==',
                    right=self._transform_expression(case_cond)
                )
            else:
                # Without expression: just use case_cond as boolean
                condition = self._transform_expression(case_cond)

            # Build ternary: value if condition else result
            result = TernaryOp(
                condition=condition,
                true_expr=value_expr,
                false_expr=result
            )

        return result
