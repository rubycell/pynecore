import copy as _copy
from dataclasses import dataclass, replace as _dataclass_replace, is_dataclass
from typing import TypeVar, Type, Any, dataclass_transform

__all__ = ['udt', 'udt_copy']

T = TypeVar('T')


def udt_copy(obj: T, **changes: Any) -> T:
    """
    Copy a UDT or object value (used by the transpiler for Pine `.copy()`).

    A dataclass UDT is copied with ``dataclasses.replace`` (honouring optional
    field overrides). Non-dataclass objects — the drawing types (Label, Box,
    Line, …) are plain classes, not dataclasses — are deep-copied instead, so
    `.copy()` on e.g. a `map<string, label>` value works rather than raising
    "replace() should be called on dataclass instances".
    """
    if is_dataclass(obj):
        return _dataclass_replace(obj, **changes)
    return _copy.deepcopy(obj)


@dataclass_transform()
def udt(cls: Type[T]) -> Type[T]:
    """
    Custom dataclass decorator that adds a `copy` method to the class.

    This decorator applies the standard dataclass decorator and then adds
    a `copy` method that creates a copy of the instance using dataclass.replace().

    :param cls: The class to decorate
    :return: The decorated class with added copy method
    """
    # Apply the standard dataclass decorator with slots=True for better performance
    decorated_cls = dataclass(cls, slots=True)  # type: ignore

    def copy(self: T, **changes: Any) -> T:
        """
        Create a copy of this instance with optional field modifications.

        :param self: The instance to copy
        :param changes: Optional keyword arguments to override field values
        :return: A new instance with the specified changes
        """
        return udt_copy(self, **changes)

    # noinspection PyShadowingNames
    @classmethod  # noqa
    def new(cls, *args, **kwargs) -> T:
        """
        Pine Script-style constructor method.

        Creates a new instance of the class using the same arguments as __init__.
        This provides Pine Script compatibility where UDTs are created with .new().

        :param cls: The class to construct
        :param args: Positional arguments for the constructor
        :param kwargs: Keyword arguments for the constructor
        :return: A new instance of the class
        """
        return cls(*args, **kwargs)

    # Add the methods to the class
    decorated_cls.copy = copy  # type: ignore
    decorated_cls.new = new

    return decorated_cls
