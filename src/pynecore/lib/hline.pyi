from ..core.callable_module import CallableModule
from ..types.hline import HLine, HLineEnum
from . import color as _color
from . import display as _display

class HLineModule(CallableModule):
    style_solid = HLineEnum()
    style_dotted = HLineEnum()
    style_dashed = HLineEnum()

    def hline(
            self,
            price: float,
            title: str = "",
            color: _color.Color = _color.blue,
            linestyle: HLineEnum = HLineModule.style_solid,
            linewidth: int = 1,
            editable: bool = True,
            display: _display.Display = _display.all
    ) -> HLine:
        ...


hline: HLineModule = HLineModule(__name__)
