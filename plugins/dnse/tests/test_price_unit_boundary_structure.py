"""#119 structural guard — every WIRE price crosses the boundary in a funnel.

The #119 bug was not one line: it was a UNIT SEAM spread over a write path, an
amend diff and four silent readbacks. A behavioural test can only pin the sites
someone thought of; this one is mechanical — it walks the plugin's AST and
asserts that no venue price key (``price``, ``stopPrice``, ``stopOrderPrice``,
``averagePrice``, ``costPrice``, ``lastPrice``) is read or written OUTSIDE the
functions that own the conversion. Add a new price site anywhere else and this
test names it, before a 1000x fill does.

WHAT IT CANNOT SEE (read this before trusting it):

* **Dynamic access.** ``row.get(key)`` with a variable key, ``**payload``
  merges, ``json.loads`` results handed around, ``getattr`` — all invisible.
* **Values, not keys.** A price already extracted into a local (or arriving as
  a WS frame field, a dataclass attribute, a function argument) is invisible
  once it leaves the dict. The funnels are asserted by NAME, not by proof that
  the conversion inside them is right — the behavioural tests do that.
* **Other modules.** Only the files listed in ``_SCOPE`` are scanned;
  ``fill_slices`` is explicitly in scope and documented as wire-unit.
* **The unit itself.** It proves WHERE the boundary is crossed, never that the
  scale chosen there is correct.
"""
import ast
from pathlib import Path

import pynecore_dnse

#: Payload/row keys whose value is a VENUE-UNIT price (đồng for stocks).
#: Deliberately NOT here: ``matchPrice`` (market data — already feed units),
#: ``basicPrice`` / ``ceilingPrice`` / ``floorPrice`` (secdef — feed units).
_WIRE_PRICE_KEYS = frozenset({
    "price", "stopPrice", "stopOrderPrice", "averagePrice", "costPrice",
    "lastPrice",
})

#: module filename -> the functions allowed to touch a raw wire price.
_SCOPE = {
    "broker.py": frozenset({
        "_to_exchange_order",      # THE read funnel (~16 call sites feed it)
        "_place",                  # THE write funnel
        "_amend_normal",           # the amend PUT + its one-unit diff
        "_recover_half_applied_amend",   # the #86 half-applied re-check
        "_scan_row",               # averagePrice -> OrderEvent.fill_price
        "get_position",            # costPrice -> entry_price
    }),
    "fill_slices.py": frozenset({
        "parse_reports",           # wire-unit by contract (see its docstring)
    }),
    "provider.py": frozenset(),    # secdef/market data only — no wire prices
}

_PACKAGE = Path(pynecore_dnse.__file__).parent


def _enclosing_functions(tree: ast.AST) -> "dict[int, str]":
    """node id -> name of the innermost enclosing function definition."""
    owner: dict[int, str] = {}

    def walk(node, current):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            current = node.name
        owner[id(node)] = current
        for child in ast.iter_child_nodes(node):
            walk(child, current)

    walk(tree, "<module>")
    return owner


def _wire_price_sites(path: Path) -> "list[tuple[int, str, str]]":
    """``(lineno, key, enclosing function)`` for every raw wire-price access."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = _enclosing_functions(tree)
    sites: list[tuple[int, str, str]] = []

    def record(node, key):
        if key in _WIRE_PRICE_KEYS:
            sites.append((node.lineno, key, owner.get(id(node), "<module>")))

    for node in ast.walk(tree):
        # row.get("price") / row.pop("price")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("get", "pop") and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            record(node, node.args[0].value)
        # row["price"]
        elif (isinstance(node, ast.Subscript)
              and isinstance(node.slice, ast.Constant)
              and isinstance(node.slice.value, str)):
            record(node, node.slice.value)
        # {"price": ...} — payload construction
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    record(key, key.value)
    return sites


def __test_no_wire_price_crosses_the_boundary_outside_the_codec_funnels__():
    """Mechanical completeness: adding a price site outside a funnel fails."""
    offenders = []
    for filename, allowed in _SCOPE.items():
        for lineno, key, function in _wire_price_sites(_PACKAGE / filename):
            if function not in allowed:
                offenders.append(f"{filename}:{lineno} {key!r} in {function}()")

    assert not offenders, (
        "raw venue price(s) touched outside the #119 codec funnels — convert "
        "with `_wire_price`/`_from_wire` (or add the function to `_SCOPE` "
        "with a reason):\n  " + "\n  ".join(offenders))


def __test_the_scanner_actually_finds_the_known_funnel_sites__():
    """Red-first for the scanner itself: a detector that finds NOTHING would
    pass the test above vacuously. Pin the sites we know exist."""
    found = {(function, key)
             for _, key, function in _wire_price_sites(_PACKAGE / "broker.py")}

    for expected in (("_place", "price"), ("_place", "stopPrice"),
                     ("_place", "stopOrderPrice"),
                     ("_to_exchange_order", "price"),
                     ("_to_exchange_order", "averagePrice"),
                     ("_to_exchange_order", "stopPrice"),
                     ("_amend_normal", "price"),
                     ("_scan_row", "averagePrice"),
                     ("get_position", "costPrice")):
        assert expected in found, (
            f"the AST scanner missed a KNOWN wire-price site {expected} — it "
            f"is not proving anything")


def __test_market_data_and_secdef_keys_are_out_of_scope__():
    """The feed side must NOT be rescaled (TV parity + the tracked datasets),
    so its keys are deliberately absent from the wire-price set."""
    for feed_key in ("matchPrice", "basicPrice", "ceilingPrice", "floorPrice"):
        assert feed_key not in _WIRE_PRICE_KEYS
