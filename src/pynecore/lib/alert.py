"""
Alert

This is a callable module, so the module itself is both a function and a namespace
"""
from __future__ import annotations
from ..core.callable_module import CallableModule

from ..types.alert import AlertEnum


#
# Module object
#

class AlertModule(CallableModule):
    #
    # Constants
    #

    freq_all = AlertEnum()
    freq_once_per_bar = AlertEnum()
    freq_once_per_bar_close = AlertEnum()


#
# Callable module function
#

def alert(
        message: str,
        freq: AlertEnum = AlertModule.freq_once_per_bar
) -> None:
    """
    Alert function - no-op in backtesting mode.

    :param message: Alert message to display
    :param freq: Alert frequency (currently ignored)
    """
    pass
    

#
# Module initialization
#

AlertModule(__name__)
