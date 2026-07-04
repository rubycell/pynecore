"""
Import resolver for PyneCore transpiled code.

Collects all used symbols and generates appropriate import statements.
"""
from typing import Set, List
from .ast_nodes import *
from .pine_builtins import (
    PYNECORE_LIB_MODULES, DRAWING_TYPES, CHART_TYPES,
    SERIES_VARIABLES, is_series_variable, needs_plot_style_remap
)


class ImportResolver:
    """Collects used symbols and generates import statements."""

    def __init__(self):
        self.uses_series = False
        self.uses_persistent = False
        self.uses_persistent_series = False
        self.uses_na = False
        self.uses_any = False
        self.uses_udt = False  # Custom type definitions
        self.uses_udt_copy = False  # udt_copy for UDT field copying
        self.uses_method = False  # User-defined methods
        self.uses_method_call = False  # method_call for explicit method calling
        self.uses_pine_range = False  # pine_range for loops
        self.uses_cast_int = False  # cast_int for type casting
        self.uses_cast_float = False  # cast_float for type casting
        # Note: SeriesImpl for function params is handled by runtime SeriesTransformer
        self.lib_modules: Set[str] = set()
        self.builtin_variables: Set[str] = set()
        self.drawing_types: Set[str] = set()
        self.chart_types: Set[str] = set()
        self.uses_plot_style = False
        self.aliased_modules: dict[str, str] = {}  # Maps module name to alias (e.g., 'position' -> '_position_module')

    def analyze(self, script: Script) -> None:
        """Analyze script to determine what needs to be imported."""
        # Check for Series usage (var declarations, globals)
        for decl in script.declarations:
            if isinstance(decl, (VarDecl, VaripDecl)):
                self.uses_persistent = True
            elif isinstance(decl, Assignment):
                # Global assignments are Series
                self.uses_series = True

        # Check body for Series usage
        if script.body:
            self.uses_series = True

        # Scan entire AST for function calls and identifiers
        self._scan_node(script)

    def _scan_node(self, node: Any) -> None:
        """Recursively scan AST node for symbols."""
        if node is None:
            return

        # Handle NaLiteral
        if isinstance(node, NaLiteral):
            self.uses_na = True

        # Handle function calls
        if isinstance(node, FunctionCall):
            self._process_function_call(node)

        # Handle member access
        if isinstance(node, MemberAccess):
            self._process_member_access(node)

        # Handle identifiers
        if isinstance(node, Identifier):
            self._process_identifier(node.name)

        # Handle script declaration
        if isinstance(node, (IndicatorDecl, StrategyDecl, LibraryDecl)):
            self.lib_modules.add('script')
            # Scan decorator kwargs for module references (e.g., currency=currency.USD)
            if hasattr(node, 'kwargs') and node.kwargs:
                for value in node.kwargs.values():
                    self._scan_node(value)

        # Handle input declarations
        if isinstance(node, InputDecl):
            self.lib_modules.add('input')

        # Handle var/varip declarations (including inside blocks)
        if isinstance(node, (VarDecl, VaripDecl)):
            self.uses_persistent = True

        # Handle type declarations (custom types)
        if isinstance(node, TypeDecl):
            self.uses_udt = True
            # Scan field types for drawing type imports (e.g., lbl: Label)
            from .pine_builtins import get_type_name
            for field_info in node.fields:
                if len(field_info) >= 2:
                    pine_type = field_info[1]  # type_hint string (lowercase from Pine, e.g., "label")
                    default_val = field_info[2] if len(field_info) >= 3 else None
                    if isinstance(pine_type, str):
                        # Convert Pine type to Python type (e.g., "label" → "Label")
                        py_type = get_type_name(pine_type)
                        # Check if it's a drawing or chart type
                        if py_type in DRAWING_TYPES:
                            self.drawing_types.add(py_type)
                        elif py_type in CHART_TYPES:
                            self.chart_types.add(py_type)
                        # Codegen generates na(Type) for non-primitive fields without defaults
                        if default_val is None and py_type not in ('int', 'float', 'bool'):
                            self.uses_na = True
                        elif isinstance(default_val, NaLiteral):
                            self.uses_na = True

        # Handle method declarations
        if isinstance(node, FuncDecl):
            if hasattr(node, 'is_method') and node.is_method:
                self.uses_method = True

        # Handle for loops (need pine_range)
        if isinstance(node, ForLoop):
            self.uses_pine_range = True

        # Check for Any in type hints
        if isinstance(node, (Assignment, Reassignment, VarDecl, VaripDecl, Parameter)):
            if hasattr(node, 'type_hint') and node.type_hint:
                type_hint_str = str(node.type_hint)
                if 'Any' in type_hint_str:
                    self.uses_any = True

                # Extract drawing/structural types from type hints like Persistent[Line], Matrix[float], etc.
                import re
                # Match type names inside brackets: Persistent[Line], Matrix[float], dict[str, Line], etc.
                # Look for capital letter type names that might be drawing types
                for match in re.finditer(r'\b([A-Z][a-zA-Z]*)\b', type_hint_str):
                    type_name = match.group(1)
                    if type_name in DRAWING_TYPES:
                        self.drawing_types.add(type_name)
                    elif type_name in CHART_TYPES:
                        self.chart_types.add(type_name)

        # Recursively scan children
        if isinstance(node, (list, tuple)):
            for item in node:
                self._scan_node(item)
        elif hasattr(node, '__dict__'):
            for attr_value in node.__dict__.values():
                if isinstance(attr_value, (ASTNode, list, tuple, dict)):
                    self._scan_node(attr_value)

    def _process_function_call(self, call: FunctionCall) -> None:
        """Process a function call to extract module dependencies."""
        func_name = None

        if isinstance(call.func, str):
            func_name = call.func
            # Handle dotted function names like "box.new" -> extract "box"
            if '.' in func_name:
                module = func_name.split('.')[0]
                if module in PYNECORE_LIB_MODULES:
                    self.lib_modules.add(module)
                # Check for str.format -> import string
                elif module == 'str':
                    self.lib_modules.add('string')
            # Handle standalone function names like "plot", "bgcolor", "barcolor"
            elif func_name in PYNECORE_LIB_MODULES:
                self.lib_modules.add(func_name)
            # Handle series variables used as functions (e.g., time(), time_close())
            elif is_series_variable(func_name):
                self.builtin_variables.add(func_name)
                self.uses_series = True
        elif isinstance(call.func, NaLiteral):
            # Special case: na() used as a function call, not a literal
            # Import the na function from pynecore.lib
            self.lib_modules.add('na')
        elif isinstance(call.func, MemberAccess):
            # Extract module name from member access
            if isinstance(call.func.object, str):
                module = call.func.object
                if module in PYNECORE_LIB_MODULES:
                    self.lib_modules.add(module)
                # Check for str.format -> import string
                elif module == 'str':
                    self.lib_modules.add('string')
                func_name = f"{module}.{call.func.member}"
            elif isinstance(call.func.object, Identifier):
                module = call.func.object.name
                if module in PYNECORE_LIB_MODULES:
                    self.lib_modules.add(module)
                # Check for str.format -> import string
                elif module == 'str':
                    self.lib_modules.add('string')
                func_name = f"{module}.{call.func.member}"

        # Recursively scan all arguments and kwargs for module references
        for arg in call.args:
            self._scan_node(arg)
        for value in call.kwargs.values():
            self._scan_node(value)
            # Also check for plot style references in kwargs
            if isinstance(value, MemberAccess):
                # Handle both Identifier and str object types
                module_name = value.object.name if isinstance(value.object, Identifier) else str(value.object)
                full_name = f"{module_name}.{value.member}"
                if needs_plot_style_remap(full_name):
                    self.uses_plot_style = True

    def _process_member_access(self, member: MemberAccess) -> None:
        """Process member access to extract module dependencies."""
        if isinstance(member.object, str):
            module = member.object
            if module in PYNECORE_LIB_MODULES:
                self.lib_modules.add(module)
            # Also check for 'str' module (Pine uses str.format, etc.)
            elif module == 'str':
                self.lib_modules.add('string')

            # Check for plot style
            full_name = f"{module}.{member.member}"
            if needs_plot_style_remap(full_name):
                self.uses_plot_style = True

        elif isinstance(member.object, Identifier):
            module = member.object.name
            if module in PYNECORE_LIB_MODULES:
                self.lib_modules.add(module)
            # Also check for 'str' module (Pine uses str.format, etc.)
            elif module == 'str':
                self.lib_modules.add('string')

            # Check for plot style
            full_name = f"{module}.{member.member}"
            if needs_plot_style_remap(full_name):
                self.uses_plot_style = True

    def _process_identifier(self, name: str) -> None:
        """Process identifier to check for built-in variables and module constants."""
        # Check for dotted identifiers like "xloc.bar_time" or "line.style_dashed"
        if '.' in name:
            module = name.split('.')[0]
            if module in PYNECORE_LIB_MODULES:
                self.lib_modules.add(module)
        # Check for built-in Series variables
        elif is_series_variable(name):
            self.builtin_variables.add(name)
            self.uses_series = True

    def add_drawing_type(self, type_name: str) -> None:
        """Manually add a drawing type that was detected."""
        if type_name in DRAWING_TYPES:
            self.drawing_types.add(type_name)
        elif type_name in CHART_TYPES:
            self.chart_types.add(type_name)

    def add_module(self, module_name: str) -> None:
        """Manually add a module dependency."""
        if module_name in PYNECORE_LIB_MODULES:
            self.lib_modules.add(module_name)

    def generate_imports(self) -> List[str]:
        """Generate import statements in the correct order."""
        imports = []

        # 0. Typing imports (if used)
        if self.uses_any:
            imports.append("from typing import Any")

        # 1. Pine range (if used)
        if self.uses_pine_range:
            imports.append("from pynecore import pine_range")

        # 2. Pine cast (if used)
        if self.uses_cast_int or self.uses_cast_float:
            cast_funcs = []
            if self.uses_cast_float:
                cast_funcs.append('cast_float')
            if self.uses_cast_int:
                cast_funcs.append('cast_int')
            imports.append(f"from pynecore.core.pine_cast import {', '.join(sorted(cast_funcs))}")

        # 3. Pine method (only import what's used)
        if self.uses_method or self.uses_method_call:
            method_imports = []
            if self.uses_method:
                method_imports.append('method')
            if self.uses_method_call:
                method_imports.append('method_call')
            imports.append(f"from pynecore.core.pine_method import {', '.join(method_imports)}")

        # 4. Pine UDT (only import what's used)
        if self.uses_udt or self.uses_udt_copy:
            udt_imports = []
            if self.uses_udt:
                udt_imports.append('udt')
            if self.uses_udt_copy:
                udt_imports.append('udt_copy')
            imports.append(f"from pynecore.core.pine_udt import {', '.join(udt_imports)}")

        # 5. PyneCore core types - Series only (Persistent moved to pynecore.types)
        # Note: Series import is now handled in types import section with Persistent

        # 6. Main library imports + built-in variables (combined into multi-line statement)
        # Ensure na is imported when used as bare identifier (e.g., f = na)
        if self.uses_na:
            self.lib_modules.add('na')
            self.lib_modules.add('NA')

        all_lib_imports = []
        if self.lib_modules:
            all_lib_imports.extend(sorted(self.lib_modules))
        if self.builtin_variables:
            all_lib_imports.extend(sorted(self.builtin_variables))

        if all_lib_imports:
            # Sort and combine all lib imports
            all_lib_imports = sorted(set(all_lib_imports))

            # Separate regular imports from aliased imports
            regular_imports = []
            aliased_items = []
            for item in all_lib_imports:
                if item in self.aliased_modules:
                    # Generate alias like "position as _position_module"
                    aliased_items.append(f"{item} as {self.aliased_modules[item]}")
                else:
                    regular_imports.append(item)

            # Combine regular and aliased imports
            combined_imports = regular_imports + aliased_items

            # Generate multi-line import statement like ground truth
            # Format: from pynecore.lib import (\n    item1, item2, item3,\n    item4, item5\n)
            import_str = "from pynecore.lib import (\n    "
            import_str += ', '.join(combined_imports)
            import_str += "\n)"
            imports.append(import_str)

        # 8. Plot styles (if used)
        if self.uses_plot_style:
            imports.append("from pynecore.lib import plot_style")

        # 9. Types from pynecore.types (drawing types + Persistent + Series + Matrix if used)
        types_import = []
        if self.drawing_types:
            types_import.extend(sorted(self.drawing_types))
        if self.uses_persistent:
            types_import.append('Persistent')
        if self.uses_persistent_series:
            types_import.append('PersistentSeries')
        if self.uses_series:
            types_import.append('Series')
        # Matrix is always imported if we have drawing types or persistent (it's a common type)
        # Actually, only import if explicitly used

        if types_import:
            imports.append(f"from pynecore.types import {', '.join(sorted(set(types_import)))}")

        # 10. Chart types (if used)
        if self.chart_types:
            types = sorted(self.chart_types)
            imports.append(f"from pynecore.types.chart import {', '.join(types)}")

        return imports
