"""#152 — the OCO book must be visible, and ARMED vs SPENT must come from the CHILD.

Measured live 2026-09-18. A long 1 @ 1984.8 was fully bracketed:

    NORMAL  39356                 New        price 1988.8  stopPrice None
    OCO     damadq2vfqkc7397o0tg  Activated  price 1988.8  stopPrice 1980.8

``broker.get_open_orders`` scans ``_CATEGORIES = ("NORMAL", "STOP")`` and never
lists the OCO book, so ``venue.py status``/``flat``, ``flatten.py``'s sweep and
``naked_watch.py`` were all blind to the umbrella carrying the stop. The operator's
app shows the same gap from a third direction. Over 24 hours this produced a wrong
call in BOTH directions — "FLATTEN NOW, the stoploss does not exist" on a protected
position, and an inability to show a spent umbrella was harmless on a flat account.

THE CONSTRAINT THIS FILE EXISTS TO PIN. A SPENT umbrella's row is IDENTICAL in
shape to an ARMED one — captured from the venue at 10:22 ICT after its child was
cancelled and with the account flat:

    orderStatus Activated   price 1988.8   stopPrice 1980.8   stopOrderPrice 1980.6
    modifiedDate 2026-09-18T02:59:27.004736Z   <- the child-cancel instant

Status, both prices and the stop all survive. So ARMED/SPENT cannot be read off the
umbrella row at ALL; it must be resolved through ``externalOrderId`` to the child.
A classifier that reads the umbrella's own status is the wrong implementation these
pins catch.

Widening ``_CATEGORIES`` is NOT the fix — broker.py:194-197 records it as rejected on
#43 (double-counts ``get_open_orders``, +50% Get-Orders budget forever). The read is
therefore toolkit-side and additive.
"""
import importlib.util
import pathlib
import sys

import pytest

_TOOLS = pathlib.Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    """Path-load a tool module, registering it in ``sys.modules`` FIRST.

    The registration is load-bearing, not tidiness: a ``@dataclass`` under
    ``from __future__ import annotations`` resolves its annotations through
    ``sys.modules[spec.name]`` during ``exec_module`` and raises without it. It
    also means a mutation test patches the SAME module object the shell imports,
    rather than a second copy that silently keeps the original behaviour.
    """
    path = _TOOLS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(_TOOLS))
venue = _load("venue")


# --- fixtures copied from the REAL venue records of 2026-09-18 -------------
# Verbatim field names and values; an invented fixture pins a shape the venue
# never produces and turns an untested path green.

_UMBRELLA = {
    "id": "damadq2vfqkc7397o0tg", "orderCategory": "OCO",
    "orderStatus": "Activated", "orderType": "LO", "side": "NS",
    "price": 1988.8, "stopPrice": 1980.8, "stopOrderPrice": 1980.6,
    "quantity": 1, "durationType": "DAY", "symbol": "41I1GA000",
    "createdDate": "2026-09-18T02:50:16.977302Z",
    "modifiedDate": "2026-09-18T02:50:17.089094Z",
}
_CHILD_LIVE = {
    "id": "39356", "orderCategory": "NORMAL", "orderStatus": "New",
    "side": "NS", "price": 1988.8, "stopPrice": None, "quantity": 1,
    "fillQuantity": 0, "symbol": "41I1GA000",
}
_CHILD_CANCELLED = dict(_CHILD_LIVE, orderStatus="Canceled")
#: The umbrella's DETAIL — the only read that carries ``externalOrderId``.
#: Measured 2026-09-18 on prod: the OCO BOOK LISTING does NOT have this field
#: (its keys are accountNo, createdDate, durationType, id, loanPackageId,
#: marketType, modifiedDate, orderCategory, orderStatus, orderType, price,
#: quantity, side, stopOrderPrice, stopPrice, symbol). So the child cannot be
#: joined from listing rows, and one detail read per umbrella is the floor.
_DETAIL = {"damadq2vfqkc7397o0tg": {
    "id": "damadq2vfqkc7397o0tg", "orderStatus": "Activated",
    "price": 1988.8, "stopPrice": 1980.8, "externalOrderId": 39356,
    "orderCategory": "OCO",
}}


class _StubClient:
    """The detail endpoint, which is the ONLY place ``externalOrderId`` lives.

    Measured 2026-09-18 on prod: an OCO BOOK row carries stopPrice,
    stopOrderPrice, durationType and symbol but NOT externalOrderId. The linkage
    to the child exists only on the order detail, so a child cannot be joined
    from listing rows and one detail read per umbrella is the floor.
    """

    def __init__(self, details):
        self._details = details
        self.detail_calls = []

    def get_order_detail(self, account, oid, market_type, order_category=None):
        self.detail_calls.append((str(oid), order_category))
        body = self._details.get(str(oid))
        return (200, body) if body is not None else (404, None)


class _StubBroker:
    """Only the seams the helper uses: a book read, and the detail endpoint."""

    account_id = "acct"
    market_type = "DERIVATIVE"

    def __init__(self, books, details=None):
        self._books = books
        self.client = _StubClient(details or {})
        self.reads = []

    def _read_book_rows_sync(self, category):
        self.reads.append(category)
        rows = self._books.get(category)
        if rows is None:                     # unreadable, NOT empty
            return None, None
        return list(rows), None


# --- the classifier -------------------------------------------------------

def __test_bracketed_position_reads_ARMED__():
    """An umbrella whose child is alive is ARMED — the position IS stopped."""
    broker = _StubBroker({"OCO": [_UMBRELLA], "NORMAL": [_CHILD_LIVE]}, _DETAIL)
    umbrellas = venue.oco_umbrellas(broker)
    assert umbrellas is not None
    assert len(umbrellas) == 1
    assert umbrellas[0].state is venue.UmbrellaState.ARMED
    assert umbrellas[0].stop_price == 1980.8
    assert umbrellas[0].child_id == "39356"


def __test_spent_umbrella_reads_SPENT_though_its_row_is_unchanged__():
    """THE discriminating pin. The umbrella row is byte-identical to the armed
    case — same Activated, same 1988.8, same stopPrice 1980.8. Only the CHILD
    differs. A classifier reading the umbrella's own status returns ARMED here
    and this pin catches it."""
    broker = _StubBroker({"OCO": [_UMBRELLA], "NORMAL": [_CHILD_CANCELLED]}, _DETAIL)
    umbrellas = venue.oco_umbrellas(broker)
    assert umbrellas is not None
    assert umbrellas[0].state is venue.UmbrellaState.SPENT, (
        "a cancelled child means the bracket is dead; the umbrella row still "
        "reads Activated with a live-looking stopPrice and must not fool us")


def __test_no_umbrella_is_an_empty_list_not_None__():
    """Absence and could-not-read must not share a value — None is reserved for
    the failed read, and a caller that conflates them reports a flat account as
    protected."""
    broker = _StubBroker({"OCO": [], "NORMAL": []})
    assert venue.oco_umbrellas(broker) == []


def __test_unreadable_OCO_book_is_None_never_empty__():
    """A failed read must never look like 'no protection here'. Callers turn
    this into exit 2 (could-not-determine), never exit 0."""
    broker = _StubBroker({"OCO": None, "NORMAL": []})
    assert venue.oco_umbrellas(broker) is None


def __test_child_missing_from_the_book_is_UNKNOWN_not_ARMED__():
    """If externalOrderId names a child we cannot see, we do not know whether the
    bracket is live. Defaulting to ARMED would be the permissive answer inside a
    function whose job is to refuse when it cannot corroborate."""
    broker = _StubBroker({"OCO": [_UMBRELLA], "NORMAL": []}, _DETAIL)
    umbrellas = venue.oco_umbrellas(broker)
    assert umbrellas is not None
    assert umbrellas[0].state is venue.UmbrellaState.UNKNOWN


def __test_one_detail_read_per_umbrella_on_the_OCO_book_only__():
    """Cost pin, CORRECTED against the venue (#43 budget; W0 polls every 15 s).

    The first version of this pin asserted the child was resolved WITHOUT a
    detail read, following a reviewer's suggestion to join ``externalOrderId``
    against the NORMAL rows already in hand. That is impossible: measured on
    prod 2026-09-18, the OCO book LISTING does not carry ``externalOrderId`` at
    all — its keys are accountNo, createdDate, durationType, id, loanPackageId,
    marketType, modifiedDate, orderCategory, orderStatus, orderType, price,
    quantity, side, stopOrderPrice, stopPrice, symbol. The linkage exists only
    on the order DETAIL, so ONE detail read per umbrella is the floor, not an
    avoidable cost.

    What this pin forbids is the cost that IS avoidable: asking the wrong book.
    ``_detail_today`` walks NORMAL, STOP and OCO in turn, which would triple it.
    """
    broker = _StubBroker({"OCO": [_UMBRELLA], "NORMAL": [_CHILD_LIVE]}, _DETAIL)
    venue.oco_umbrellas(broker)
    assert broker.reads.count("OCO") == 1
    assert len(broker.client.detail_calls) == 1
    assert broker.client.detail_calls[0][1] == "OCO", (
        "ask the OCO book directly; _detail_today walks three books per umbrella")


def __test_no_umbrellas_costs_no_detail_read__():
    """An empty OCO book short-circuits: no detail call, no NORMAL read."""
    broker = _StubBroker({"OCO": [], "NORMAL": [_CHILD_LIVE]}, _DETAIL)
    assert venue.oco_umbrellas(broker) == []
    assert broker.client.detail_calls == []


def __test_the_private_book_seam_still_exists__():
    """Guard pin. The helper calls ``broker._read_book_rows_sync``, a PRIVATE
    method — exactly the thing a refactor relocates silently. If it is renamed
    this reddens here rather than at 09:50 on a live position."""
    from pynecore_dnse.broker import DNSEBroker
    assert hasattr(DNSEBroker, "_read_book_rows_sync"), (
        "venue.oco_umbrellas depends on this seam; promote a public accessor "
        "and update the helper rather than deleting this pin")
