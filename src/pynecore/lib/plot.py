from typing import Any
import os
import sys

from ..core.callable_module import CallableModule
from ..types.plot import PlotEnum, Plot

_OPTIMIZE_MODE = os.environ.get("PYNE_OPTIMIZE_MODE") == "1"


#
# Module object
#

class PlotModule(CallableModule):
    #
    # Constants
    #

    style_area = PlotEnum()
    style_areabr = PlotEnum()
    style_circles = PlotEnum()
    style_columns = PlotEnum()
    style_cross = PlotEnum()
    style_histogram = PlotEnum()
    style_line = PlotEnum()
    style_linebr = PlotEnum()
    style_stepline = PlotEnum()
    style_stepline_diamond = PlotEnum()

    linestyle_solid = PlotEnum()
    linestyle_dashed = PlotEnum()
    linestyle_dotted = PlotEnum()

    #
    # Functions
    #


#
# Callable module function
#

# noinspection PyProtectedMember
def plot(series: Any, title: str | None = None, color: Any = None,
         linewidth: int = 1, style: Any = None, trackprice: bool = False,
         histbase: float = 0.0, offset: int = 0, join: bool = False,
         editable: bool = True, show_last: int | None = None,
         display: Any = None, format: Any = None, precision: int | None = None,
         force_overlay: bool = None, **__):
    """
    Plot series, by default a CSV is generated, but this can be extended

    :param series: The value to plot in every bar
    :param title: The title of the plot, if multiple plots are created with the same title, a
                  number will be appended
    :param color: Color of the plot
    :param linewidth: Width of the plotted line (1-4)
    :param style: Plot style (plot.style_line, etc.)
    :param trackprice: Show horizontal price tracking line
    :param histbase: Base level for histogram style
    :param offset: Shift the plot left/right by this number of bars
    :param join: Join points with a line
    :param editable: If true, the plot style can be edited in the style dialog
    :param show_last: Number of bars to show from the last bar
    :param display: Controls where the plot is displayed
    :param format: Format string for the plot value
    :param precision: Number of decimal places
    :param force_overlay: If true, force display on the main chart pane
    :return: A Plot object, can be used to reference the plot in other functions
    """
    if _OPTIMIZE_MODE:
        return Plot()
    from .. import lib
    if lib._lib_semaphore:
        return Plot()

    if lib.bar_index == 0:  # Only check if it is the first bar for performance reasons
        # Check if it is called from the main function
        if sys._getframe(2).f_code.co_name not in ('main', 'plotchar', 'plotshape'):  # noqa
            raise RuntimeError("The plot function can only be called from the main function!")

    # Ensure unique title
    if title is None:
        title = 'Plot'
    # Handle duplicate titles
    c = 0
    t = title
    while t in lib._plot_data:
        t = title + ' ' + str(c)
        c += 1
    title = t

    # Capture visual metadata on first encounter (only for direct plot calls)
    if title not in lib._plot_meta:
        caller = sys._getframe(1).f_code.co_name
        if caller not in ('plotchar', 'plotshape'):
            meta = {"type": "plot"}
            color_hex = lib._serialize_color(color)
            if color_hex is not None:
                meta["color"] = color_hex
            if linewidth != 1:
                meta["linewidth"] = linewidth
            if style is not None:
                style_name = lib._resolve_enum_name(style, PlotModule)
                if style_name is not None:
                    meta["style"] = style_name
            if trackprice:
                meta["trackprice"] = True
            if histbase != 0.0:
                meta["histbase"] = histbase
            if offset != 0:
                meta["offset"] = offset
            if display is not None:
                from ..lib import display as display_module
                display_name = lib._resolve_enum_name(display, display_module)
                if display_name is not None:
                    meta["display"] = display_name
            lib._plot_meta[title] = meta

    # Store plot data
    lib._plot_data[title] = series

    return Plot()


#
# Module initialization
#

PlotModule(__name__)
