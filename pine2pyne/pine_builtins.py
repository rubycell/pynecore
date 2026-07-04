"""
Mapping of Pine Script built-in functions, variables, and constants to PyneCore equivalents.
"""
from typing import Dict, Set


# Functions that need to be renamed
FUNCTION_RENAMES: Dict[str, str] = {
    'array.from': 'array.from_items',  # 'from' is Python keyword
}

# Module renames
MODULE_RENAMES: Dict[str, str] = {
    'str': 'string',  # str is Python built-in
}

# Type renames (Pine Script -> Python/PyneCore)
TYPE_RENAMES: Dict[str, str] = {
    # Drawing types
    'line': 'Line',
    'label': 'Label',
    'box': 'Box',
    'table': 'Table',
    'linefill': 'LineFill',
    'polyline': 'Polyline',
    'color': 'Color',
    'chart.point': 'ChartPoint',

    # Data structure types
    'matrix': 'Matrix',

    # Basic types (in type hints)
    'string': 'str',
    'int': 'int',
    'float': 'float',
    'bool': 'bool',
}

# Built-in variables that are always Series
SERIES_VARIABLES: Set[str] = {
    'open', 'high', 'low', 'close', 'volume',
    'hl2', 'hlc3', 'ohlc4', 'hlcc4',
    'bar_index', 'last_bar_index', 'last_bar_time',
    'time', 'time_close', 'time_tradingday', 'timenow',
    'dayofmonth', 'dayofweek', 'hour', 'minute', 'second', 'month', 'weekofyear', 'year',
}

# Constants that should be preserved as-is
CONSTANTS: Set[str] = {
    # Strategy constants
    'strategy.long',
    'strategy.short',
    'strategy.commission.percent',
    'strategy.commission.cash_per_contract',
    'strategy.commission.cash_per_order',
    'strategy.percent',
    'strategy.fixed',
    'strategy.cash',
    'strategy.percent_of_equity',
    'strategy.closedtrades',
    'strategy.opentrades',

    # Color constants
    'color.red', 'color.green', 'color.blue', 'color.white', 'color.black',
    'color.yellow', 'color.orange', 'color.purple', 'color.gray',
    'color.silver', 'color.maroon', 'color.fuchsia', 'color.lime',
    'color.olive', 'color.navy', 'color.teal', 'color.aqua',

    # Plot styles (need remapping to plot_style module)
    'plot.style_line',
    'plot.style_stepline',
    'plot.style_histogram',
    'plot.style_cross',
    'plot.style_area',
    'plot.style_columns',
    'plot.style_circles',
    'plot.style_linebr',
    'plot.style_steplinebr',

    # Barstate
    'barstate.isfirst',
    'barstate.islast',
    'barstate.ishistory',
    'barstate.isrealtime',
    'barstate.isnew',
    'barstate.isconfirmed',
}

# Plot style constants need special handling
PLOT_STYLE_REMAP: Dict[str, str] = {
    'plot.style_line': 'plot_style.style_line',
    'plot.style_stepline': 'plot_style.style_stepline',
    'plot.style_histogram': 'plot_style.style_histogram',
    'plot.style_cross': 'plot_style.style_cross',
    'plot.style_area': 'plot_style.style_area',
    'plot.style_columns': 'plot_style.style_columns',
    'plot.style_circles': 'plot_style.style_circles',
    'plot.style_linebr': 'plot_style.style_linebr',
    'plot.style_steplinebr': 'plot_style.style_steplinebr',
}

# Modules that should be imported from pynecore.lib
# Synced with https://github.com/PyneSys/pynecore/tree/main/src/pynecore/lib
PYNECORE_LIB_MODULES: Set[str] = {
    # Core modules (submodules as .py files in pynecore/lib/)
    'script', 'input', 'plot', 'color', 'ta', 'math', 'strategy',
    'array', 'matrix', 'map', 'string',
    'timeframe', 'runtime', 'alert', 'label', 'line', 'box', 'table',
    'linefill', 'polyline', 'log', 'barstate', 'session', 'syminfo',
    'adjustment', 'barmerge', 'chart', 'dividends', 'earnings',
    # Standalone functions (defined in pynecore/lib/__init__.py)
    'currency', 'timestamp', 'max_bars_back',
    # Plotting and coloring functions
    'bgcolor', 'barcolor', 'alertcondition', 'plotarrow', 'plotbar', 'plotcandle',
    'plotchar', 'plotshape', 'fill',
    # Utility functions
    'na', 'nz', 'fixnan',
    # Constant namespaces
    'xloc', 'yloc', 'position', 'size', 'shape', 'location',
    'extend', 'font', 'text', 'display', 'scale', 'hline', 'format',
    'order',
}

# Drawing and structural types that should be imported from pynecore.types
DRAWING_TYPES: Set[str] = {
    'Line', 'Label', 'Box', 'Table', 'LineFill', 'Polyline', 'Color', 'Matrix',
}

# Chart types that should be imported from pynecore.types.chart
CHART_TYPES: Set[str] = {
    'ChartPoint',
}

# PyneCore object method transformations
# Maps method names to the module that should be used for the function call
# Format: method_name -> (module_name, requires_transformation)
PYNECORE_METHOD_TRANSFORMS: dict[str, str] = {
    # Matrix methods
    'add_row': 'matrix',
    'add_col': 'matrix',
    'columns': 'matrix',
    'rows': 'matrix',
    'fill': 'matrix',
    'avg': 'matrix',
    'max': 'matrix',
    'min': 'matrix',
    'median': 'matrix',
    'mode': 'matrix',
    'sum': 'matrix',
    'concat': 'matrix',
    'copy': 'matrix',  # Also used by labels/lines but context-dependent
    'det': 'matrix',
    'diff': 'matrix',
    'eigenvalues': 'matrix',
    'eigenvectors': 'matrix',
    'inv': 'matrix',
    'kron': 'matrix',
    'mult': 'matrix',
    'pinv': 'matrix',
    'pow': 'matrix',
    'rank': 'matrix',
    'remove_col': 'matrix',
    'remove_row': 'matrix',
    'reshape': 'matrix',
    'reverse': 'matrix',
    'col': 'matrix',
    'row': 'matrix',
    'set': 'matrix',  # Also used by labels/lines
    'sort': 'matrix',
    'submatrix': 'matrix',
    'swap_columns': 'matrix',
    'swap_rows': 'matrix',
    'trace': 'matrix',
    'transpose': 'matrix',

    # Map methods
    'put': 'map',
    'put_all': 'map',
    'get': 'map',  # Also used by arrays
    'contains': 'map',
    'remove': 'map',
    'clear': 'map',
    'keys': 'map',
    'values': 'map',
    'size': 'map',  # Also used by arrays

    # Array methods (handled differently - array methods stay as is mostly)

    # Label methods
    'set_x': 'label',
    'set_y': 'label',
    'set_xy': 'label',
    'set_text': 'label',
    'set_color': 'label',
    'set_textcolor': 'label',
    'set_size': 'label',
    'set_style': 'label',
    'set_textalign': 'label',
    'set_tooltip': 'label',
    'get_x': 'label',
    'get_y': 'label',
    'get_text': 'label',
    'delete': 'label',  # Also line, box

    # Line methods
    'set_x1': 'line',
    'set_y1': 'line',
    'set_x2': 'line',
    'set_y2': 'line',
    'set_xy1': 'line',
    'set_xy2': 'line',
    'set_extend': 'line',
    'set_width': 'line',
    'get_x1': 'line',
    'get_y1': 'line',
    'get_x2': 'line',
    'get_y2': 'line',
    'get_price': 'line',

    # Box methods
    'set_left': 'box',
    'set_right': 'box',
    'set_top': 'box',
    'set_bottom': 'box',
    'set_lefttop': 'box',
    'set_rightbottom': 'box',
    'set_border_color': 'box',
    'set_border_width': 'box',
    'set_border_style': 'box',
    'set_bgcolor': 'box',
    'set_text': 'box',
    'set_text_color': 'box',
    'set_text_size': 'box',
    'get_left': 'box',
    'get_right': 'box',
    'get_top': 'box',
    'get_bottom': 'box',

    # Table methods
    'cell': 'table',
    'merge_cells': 'table',
    'set_position': 'table',
}

# Canonical module-to-methods sets for heuristic method → module resolution.
# Used when the transformer cannot determine the type from symbol table and
# must infer the module from the method name alone.
# Built as the UNION of all inline set definitions in transformer.py.

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

TABLE_METHODS: set[str] = {'cell', 'merge_cells', 'set_position', 'cell_set_text'}

MAP_METHODS: set[str] = {'put', 'put_all', 'get', 'remove', 'clear', 'keys', 'values', 'size', 'contains'}

MATRIX_METHODS: set[str] = {'add_row', 'add_col', 'get', 'set'}

# Array-unique methods that unambiguously map to the array module.
# Does NOT include shared methods (get, set, size, etc.) to avoid
# misresolving map/matrix calls when type info is unavailable.
ARRAY_METHODS: set[str] = {
    'push', 'pop', 'shift', 'unshift', 'slice', 'insert', 'indexof', 'lastindexof',
    'includes', 'sort', 'sort_indices', 'reverse', 'concat', 'first', 'last',
    'join', 'percentile_linear_interpolation', 'percentile_nearest_rank', 'percentrank',
    'abs', 'binary_search', 'binary_search_leftmost', 'binary_search_rightmost',
    'every', 'some', 'range', 'standardize', 'stdev', 'variance', 'covariance',
    'fill',
}

# Map-unique methods (methods that ONLY belong to map, not array)
MAP_UNIQUE_METHODS: set[str] = {'put', 'put_all', 'keys', 'values', 'contains'}

# Methods shared between array, map, and matrix.
# These can only be resolved via type-aware path (symbol table lookup).
# In heuristic fallback, prefer array since it's the most common collection.
SHARED_COLLECTION_METHODS: set[str] = {
    'get', 'set', 'size', 'clear', 'remove', 'copy',
    'avg', 'min', 'max', 'sum', 'median', 'mode',
}


def get_function_name(pine_func: str) -> str:
    """Get the PyneCore function name for a Pine Script function."""
    return FUNCTION_RENAMES.get(pine_func, pine_func)


def get_module_name(pine_module: str) -> str:
    """Get the PyneCore module name for a Pine Script module."""
    return MODULE_RENAMES.get(pine_module, pine_module)


def get_type_name(pine_type: str) -> str:
    """Get the Python/PyneCore type name for a Pine Script type."""
    # Strip .new suffix from UDT types (e.g., TradeStats.new -> TradeStats)
    if '.new' in pine_type and pine_type and pine_type[0].isupper():
        pine_type = pine_type.replace('.new', '')
    return TYPE_RENAMES.get(pine_type, pine_type)


def is_series_variable(name: str) -> bool:
    """Check if a variable is a built-in Series variable."""
    return name in SERIES_VARIABLES


def is_constant(name: str) -> bool:
    """Check if a name is a built-in constant."""
    return name in CONSTANTS


def needs_plot_style_remap(name: str) -> bool:
    """Check if a name needs plot_style module remapping."""
    return name in PLOT_STYLE_REMAP


def get_plot_style_remap(name: str) -> str:
    """Get the remapped plot_style constant name."""
    return PLOT_STYLE_REMAP.get(name, name)


# Input functions that extract to main() parameters
INPUT_FUNCTIONS: Set[str] = {
    'input.int',
    'input.float',
    'input.bool',
    'input.string',
    'input.color',
    'input.source',
    'input.timeframe',
    'input.session',
    'input.symbol',
    'input.price',
    'input.time',
}


def is_input_function(func_name: str) -> bool:
    """Check if function is an input.* function."""
    return func_name in INPUT_FUNCTIONS


# Pine Script input.* positional parameter names (after defval, title).
# Pine Script allows these as positional; PyneCore requires keyword-only.
INPUT_POSITIONAL_PARAMS: Dict[str, list[str]] = {
    'input.int':       ['minval', 'maxval', 'step', 'tooltip', 'inline', 'group', 'confirm', 'display', 'options'],
    'input.float':     ['minval', 'maxval', 'step', 'tooltip', 'inline', 'group', 'confirm', 'display', 'options'],
    'input.bool':      ['tooltip', 'inline', 'group', 'confirm', 'display'],
    'input.string':    ['options', 'tooltip', 'inline', 'group', 'confirm', 'display'],
    'input.color':     ['tooltip', 'inline', 'group', 'confirm', 'display'],
    'input.timeframe': ['options', 'tooltip', 'inline', 'group', 'confirm', 'display'],
    'input.session':   ['options', 'tooltip', 'inline', 'group', 'confirm', 'display'],
    'input.symbol':    ['options', 'tooltip', 'inline', 'group', 'confirm', 'display'],
    'input.source':    ['tooltip', 'inline', 'group', 'confirm', 'display'],
    'input.price':     ['tooltip', 'inline', 'group', 'confirm', 'display'],
    'input.time':      ['tooltip', 'inline', 'group', 'confirm', 'display'],
}


# Python reserved words that cannot be used as function/variable names.
PYTHON_RESERVED_WORDS: Set[str] = {
    'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
    'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
    'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
    'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
    'try', 'while', 'with', 'yield',
}


def sanitize_identifier(name: str) -> str:
    """Rename Python reserved words used as identifiers by appending underscore."""
    if name in PYTHON_RESERVED_WORDS:
        return f'{name}_'
    return name


# TA functions that return Series
TA_FUNCTIONS: Set[str] = {
    'ta.sma', 'ta.ema', 'ta.wma', 'ta.vwma', 'ta.swma', 'ta.alma',
    'ta.rma', 'ta.hma', 'ta.linreg', 'ta.rsi', 'ta.macd',
    'ta.stoch', 'ta.cci', 'ta.mfi', 'ta.obv', 'ta.sar',
    'ta.atr', 'ta.tr', 'ta.bb', 'ta.bbw', 'ta.kc', 'ta.kcw',
    'ta.dmi', 'ta.adx', 'ta.supertrend', 'ta.highest', 'ta.lowest',
    'ta.valuewhen', 'ta.barssince', 'ta.change', 'ta.roc',
    'ta.mom', 'ta.cum', 'ta.dev', 'ta.stdev', 'ta.variance',
    'ta.correlation', 'ta.median', 'ta.mode', 'ta.range',
    'ta.cog', 'ta.percentile_linear_interpolation',
    'ta.percentile_nearest_rank', 'ta.percentrank', 'ta.pivothigh',
    'ta.pivotlow', 'ta.rising', 'ta.falling', 'ta.cross', 'ta.crossover',
    'ta.crossunder',
}


def is_ta_function(func_name: str) -> bool:
    """Check if function is a ta.* function."""
    return func_name in TA_FUNCTIONS


# Pine Script built-in "variable-functions": used without () in Pine Script
# but require () in PyneCore because they're implemented as functions.
# Values are the default arguments to pass when auto-calling.
TA_VARIABLE_FUNCTIONS: Dict[str, str] = {
    'ta.vwap': '(high + low + close) / 3',  # default source is hlc3
    'ta.tr': '',           # no args needed
    'ta.accdist': '',
    'ta.obv': '',
    'ta.iii': '',
    'ta.nvi': '',
    'ta.pvi': '',
    'ta.pvt': '',
    'ta.wad': '',
    'ta.wvad': '',
}
