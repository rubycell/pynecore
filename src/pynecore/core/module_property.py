from __future__ import annotations
from typing import TypeVar, Callable

T = TypeVar('T')


def module_property(func: Callable[..., T]) -> T:
    """
    Decorator for Pine-style hybrid property/functions.
    At runtime, returns the function with a marker attribute.
    For type checking, pretends to return the function's return type
    so Pylance sees timeframe.multiplier as int, not a function.
    """
    setattr(func, '__module_property__', True)
    return func  # type: ignore[return-value]
