from typing import cast, Callable
from types import ModuleType
import sys


class CallableModule:
    """
    Callable wrapper for a module.
    Supports Pine Script-style [N] history access (e.g. strategy.opentrades[1]).
    """

    def __init__(self, module_name: str):
        self._module: ModuleType = sys.modules[module_name]
        self._func: Callable = getattr(self._module, self._module.__name__.split(".")[-1])
        self._history: list = []
        sys.modules[module_name] = cast(ModuleType, self)

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def __getitem__(self, index: int):
        """Pine Script [N] history access. [0] = current bar, [1] = 1 bar ago, etc."""
        if index == 0:
            return self()
        if index < 0 or index > len(self._history):
            return 0
        return self._history[-index]

    def _snapshot(self):
        """Capture current value into history buffer. Called before main() each bar."""
        try:
            self._history.append(self())
        except TypeError:
            pass  # Skip modules that require arguments (e.g. PlotModule, AlertModule)

    def _reset_history(self):
        """Clear history buffer. Called at the start of each run."""
        self._history.clear()
