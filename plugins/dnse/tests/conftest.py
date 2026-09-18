"""Shared fixtures for DNSE plugin unit tests — no live network, no real filesystem.

Every DNSE call funnels through the client wrapper, so a single fake client injected
as ``broker._client`` / ``provider._client`` intercepts the whole REST surface. Set
per-method canned ``(status, body)`` replies; drive ``async`` methods with
``asyncio.run`` and monkeypatch ``time.sleep`` / ``asyncio.sleep`` so poll loops don't
wait. Test functions use the repo convention ``__test_*__`` (see ``pytest.ini``).
"""
import asyncio

import pytest
import pynecore.lib as lib

lib.bar_index = 0  # let the [BROKER] log formatter render during broker._emit()


class _FakeClient:
    """Configurable stand-in for the DNSE client wrapper.

    ``_FakeClient(post_order=(201, {"id": "1"}),
                  get_positions=lambda *a, **k: (200, {"positions": [...]}))``

    A response may be a canned ``(status, body)`` tuple or a callable receiving the
    call's args. Any unset method returns ``(200, {})``. Every call is recorded in
    ``.calls`` as ``(method_name, args, kwargs)``; ``.count(name)`` tallies one method.

    PER-CATEGORY ORDER BOOKS (#152). DNSE's books are DISJOINT — an order lives on
    exactly one of NORMAL / STOP / OCO, and ``_read_book_rows_sync`` asks for one
    category at a time via ``order_category=``. A single ``get_orders`` response
    served every category, so a STOP conditional also came back from the OCO
    listing and was mistaken for a bracket umbrella. To model the real venue, pass
    a DICT keyed by category::

        _FakeClient(get_orders={"STOP": (200, {"orders": [...], "totalPages": 1}),
                                "OCO":  (200, {"orders": [], "totalPages": 1})})

    A category absent from the dict answers empty, not the other categories' rows.
    A NON-dict response keeps the old behaviour and serves every category, so
    tests that do not care about the distinction are untouched.
    """

    def __init__(self, **responses):
        self._responses = responses
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def _call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            resp = self._responses.get(name)
            if isinstance(resp, dict) and "order_category" in kwargs:
                # Per-category book. An unregistered category is EMPTY rather
                # than absent-and-falling-back: falling back would put a STOP
                # order on the OCO book, which is the shape this exists to stop.
                resp = resp.get(str(kwargs["order_category"]),
                                (200, {"orders": [], "totalPages": 1}))
            if callable(resp):
                return resp(*args, **kwargs)
            return resp if resp is not None else (200, {})

        return _call

    def count(self, method):
        return sum(1 for call in self.calls if call[0] == method)


@pytest.fixture
def fake_client():
    """Factory for a :class:`_FakeClient`; inject as ``broker._client`` / ``provider._client``."""
    return lambda **responses: _FakeClient(**responses)


@pytest.fixture
def collect():
    """Collect up to ``n`` items from an async generator, then close it: ``collect(gen, n)``."""
    def _collect(agen, n, *, timeout=1.0):
        async def _run():
            out = []
            try:
                for _ in range(n):
                    out.append(await asyncio.wait_for(agen.__anext__(), timeout))
            except (StopAsyncIteration, asyncio.TimeoutError):
                pass
            finally:
                await agen.aclose()
            return out
        return asyncio.run(_run())
    return _collect
