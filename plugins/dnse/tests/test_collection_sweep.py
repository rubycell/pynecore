"""#157: every test function in the fake-venue suites must actually be COLLECTED.

This pin exists because the trap it catches already happened, on 2026-09-18. A test named
``__test_an_uncalled_endpoint_answers_404_as_production_does`` — missing the trailing
underscores — silently never ran. ``pytest.ini`` sets ``python_functions = __test_*__``, so a
name lacking the suffix is not a failing test, it is not a test at all. The suite reported
"6 passed" and every figure derived from it was wrong by one, invisibly.

Nothing about a pass count can reveal this. It surfaced only because a mutant that should have
been caught escaped, and the escape pointed at a pin that did not exist. This sweep makes the
same mistake fail directly and immediately instead.

The sweep is deliberately AST-based rather than a regex over the text: a regex would also match
the name inside a docstring or a comment, and a sweep that cries wolf gets deleted.
"""
import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "testing"))

#: The suites this pin guards. The fake-venue work is where the trap bit.
_GUARDED = (
    "test_fake_venue_conformance.py",
    "test_venue_day.py",
    "test_venue_http.py",
    "test_ws_source_endpoint.py",
    "test_collection_sweep.py",
    "test_162_park_replay.py",
    "test_trade_list_parity.py",
)


def _uncollectable_tests(path: Path) -> list[str]:
    """Function names that LOOK like tests but do not match ``__test_*__``.

    A name beginning ``__test`` is unambiguously intended as a test; if it does not also end
    with ``__`` pytest will never collect it under this project's configuration.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            if name.startswith("__test") and not name.endswith("__"):
                bad.append(name)
    return bad


def __test_every_intended_test_in_the_fake_venue_suites_is_collectable__():
    """The real sweep."""
    here = Path(__file__).resolve().parent
    offenders = {}
    for filename in _GUARDED:
        path = here / filename
        if not path.exists():
            continue
        found = _uncollectable_tests(path)
        if found:
            offenders[filename] = found

    assert not offenders, (
        "these functions look like tests but pytest will NEVER collect them, because "
        "pytest.ini requires __test_*__ (note the trailing underscores). They are not failing "
        f"tests, they do not exist: {offenders}")


def __test_the_sweep_detects_a_name_that_pytest_would_skip__(tmp_path):
    """The discriminating half, and the reason this pin is worth anything.

    A sweep that always returned an empty list would pass the test above forever. So feed it the
    exact shape that bit on 2026-09-18 and require that it is reported.
    """
    crafted = tmp_path / "test_crafted.py"
    crafted.write_text(
        "def __test_properly_named__():\n"
        "    pass\n"
        "\n"
        "def __test_missing_its_suffix():\n"
        "    pass\n"
        "\n"
        "def helper_not_a_test():\n"
        "    pass\n"
    )

    found = _uncollectable_tests(crafted)

    assert found == ["__test_missing_its_suffix"], (
        "the sweep must report the unsuffixed test and nothing else; flagging the correctly "
        "named one or the helper would make it noise that gets deleted")


def __test_the_sweep_ignores_a_matching_name_inside_a_docstring__():
    """A regex over the file text would flag this docstring, which mentions
    __test_not_a_real_function by name. The AST walk must not."""
    import textwrap
    source = textwrap.dedent('''
        """A module whose docstring names __test_not_a_real_function in prose."""

        def __test_real__():
            """Mentions __test_also_not_real in its own docstring."""
    ''') + "    pass\n"

    tree = ast.parse(source)
    names = [n.name for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name.startswith("__test")]

    assert names == ["__test_real__"]
