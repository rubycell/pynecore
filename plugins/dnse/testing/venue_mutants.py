"""#157 stage E: discriminating controls for the venue conformance suite.

A pin that has only ever been run against a CORRECT venue has proven nothing. It might assert
something every implementation satisfies, or assert nothing at all. So every fact the fake
reproduces gets a MUTANT: a deliberately wrong venue that the pin must catch. A pin that stays
green against its mutant is not a pin, and this runner fails on it.

Two rules this harness follows, both learned expensively:

* **Mutants are RUNTIME PATCHES, never source edits.** Editing the source and restoring it later
  leaves the approved hashes disturbed and, worse, can leave a stale ``.pyc`` running the mutant
  after the source is restored — a failure that ``inspect.getsource`` cannot see, because it
  reads the ``.py`` while the code object came from the cache. Every run here sets
  ``PYTHONDONTWRITEBYTECODE=1``.
* **Colour is read from BEHAVIOUR, not from a count.** The runner reads pytest's own exit status
  AND its summary line, and it refuses to interpret a crash as a caught mutant: a harness whose
  fixture explodes marks every mutant "caught" and reports a perfect score. Near-uniform results
  are the signature of a broken harness, so each mutant must name the ONE test it targets and
  that test must fail for the RIGHT reason.

Usage:
    PYTHONDONTWRITEBYTECODE=1 python plugins/dnse/testing/venue_mutants.py          # run all
    MUTANT=oco_spawns pytest ... -p venue_mutants                                   # one, manually
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_TESTS = "plugins/dnse/tests/test_fake_venue_conformance.py"


# --------------------------------------------------------------------------- the mutants
# Each entry: the ONE test that must catch it, and a patch that makes the venue wrong.

def _mutant_oco_spawns(venue_core):
    """WRONG: the OCO stop leg spawns a second order instead of amending the child in place.

    This is the model the suite's first draft encoded, before #159 measured the truth. It is the
    single most important mutant here: it is the plausible wrong answer.
    """
    original = venue_core.FakeVenue._trigger_conditionals

    def patched(self, price):
        for order in list(self._orders.values()):
            if (order["book"] == "STOP_BOOK" and order["category"] == "OCO"
                    and order["stopPrice"] is not None and self._crossed(order, price)):
                child = self._orders.get(order["externalOrderId"])
                if child is not None and child["orderStatus"] not in venue_core._TERMINAL:
                    self._make(self._new_normal_id(), "NORMAL", order["side"],
                               order["quantity"], price, None, venue_core.NEW)
                return
        original(self, price)

    venue_core.FakeVenue._trigger_conditionals = patched


def _mutant_activated_fills(venue_core):
    """WRONG: an Activated conditional accumulates a fill, instead of being terminal."""
    original = venue_core.FakeVenue._match_resting

    def patched(self, price, volume):
        original(self, price, volume)
        for order in self._orders.values():
            if order["orderStatus"] == venue_core.ACTIVATED:
                order["fillQuantity"] = order["quantity"]

    venue_core.FakeVenue._match_resting = patched


def _mutant_activates_on_placement(venue_core):
    """WRONG: a conditional activates when placed, without waiting for a print."""
    venue_core.FakeVenue._crossed = staticmethod(lambda order, price: True)


def _mutant_uppercase_statuses(venue_core):
    """WRONG: the uppercase vocabulary testing/fake_dnse.py synthesised (FILLED, CANCELLED).

    The plugin's _STATUS_MAP absorbs the difference, so only a venue-level pin can catch it.
    """
    venue_core.FILLED = "FILLED"
    venue_core.CANCELED = "CANCELLED"


def _mutant_shell_disappears(venue_core):
    """WRONG: the Activated shell is dropped from the STOP book once triggered (#41 says it stays)."""
    original = venue_core.FakeVenue.orders

    def patched(self, *, book):
        return [o for o in original(self, book=book)
                if o["orderStatus"] != venue_core.ACTIVATED]

    venue_core.FakeVenue.orders = patched


def _mutant_terminal_cancel_succeeds(venue_core):
    """WRONG: cancelling a terminal order succeeds, so #162's permanent park could never occur."""
    original = venue_core.FakeVenue.cancel

    def patched(self, order_id):
        order = self._orders.get(order_id)
        if order is not None and order["orderStatus"] in venue_core._TERMINAL:
            return dict(order)
        return original(self, order_id)

    venue_core.FakeVenue.cancel = patched


def _mutant_qty1_partial_fills(venue_core):
    """WRONG: a qty-1 derivative emits a PartiallyFilled tick, a state the venue cannot produce."""
    original = venue_core.FakeVenue._match_resting

    def patched(self, price, volume):
        for order in list(self._orders.values()):
            if (order["book"] == "NORMAL" and order["quantity"] == 1
                    and order["orderStatus"] == venue_core.NEW
                    and order["price"] is not None and self._marketable(order, price)):
                order["fillQuantity"] = 0.5
                order["orderStatus"] = venue_core.PARTIALLY_FILLED
                self._record(order)
                return
        original(self, price, volume)

    venue_core.FakeVenue._match_resting = patched


def _mutant_nondeterministic_ids(venue_core):
    """WRONG: ids drift between replays, so no pin built on the record is reliable evidence."""
    counter = {"n": 0}
    original = venue_core.FakeVenue._new_normal_id

    def patched(self):
        counter["n"] += 1
        return str(700000 + counter["n"])

    venue_core.FakeVenue._new_normal_id = patched
    _ = original


def _mutant_umbrella_clears_stop_price(venue_core):
    """WRONG: the spent umbrella drops its stopPrice, making armed and spent distinguishable.

    Helpful, and false: #159 measured the spent row still carrying stopPrice on a flat account.
    A fake that cleans this up would hide the exact ambiguity #152 has to resolve.
    """
    original = venue_core.FakeVenue.order

    def patched(self, order_id):
        found = original(self, order_id)
        if found and found["orderStatus"] == venue_core.ACTIVATED and found["externalOrderId"]:
            child = self._orders.get(found["externalOrderId"])
            if child is not None and child["orderStatus"] == venue_core.FILLED:
                found["stopPrice"] = None
        return found

    venue_core.FakeVenue.order = patched


def _mutant_integer_conditional_ids(venue_core):
    """WRONG: the conditional book issues integer ids, so the two books become indistinguishable."""
    counter = {"n": 0}

    def patched(self):
        counter["n"] += 1
        return str(900000 + counter["n"])

    venue_core.FakeVenue._new_conditional_id = patched


def _mutant_day_label_defaults(venue_core):
    """WRONG: a day with no label is assumed RECORDED instead of refused.

    The most dangerous mutant in this file. It does not break anything visibly — it silently
    converts a synthesised day into a claimed measurement, which is how a fabricated result gets
    reported as evidence.
    """
    import venue_day
    original = venue_day.VenueDay.from_dict.__func__

    def patched(cls, raw):
        if "label" not in raw:
            raw = dict(raw, label="RECORDED")
        return original(cls, raw)

    venue_day.VenueDay.from_dict = classmethod(patched)
    _ = venue_core


def _mutant_day_closes_only(venue_core):
    """WRONG: the synthesiser emits only each bar's close, so no intrabar extreme is replayed
    and a stop the real session triggered never triggers."""
    import venue_day
    original = venue_day.synthesise_day

    def patched(bars, *, symbol, session_open_ts=None):
        day = original(bars, symbol=symbol, session_open_ts=session_open_ts)
        day.prints = [p for p in day.prints
                      if any(p["price"] == b["close"] and p["bar_ts"] == b["timestamp"]
                             for b in day.bars)]
        return day

    venue_day.synthesise_day = patched
    _ = venue_core


def _mutant_day_volume_invented(venue_core):
    """WRONG: print volume is invented rather than conserved from the bar, so fills differ."""
    import venue_day
    original = venue_day.synthesise_day

    def patched(bars, *, symbol, session_open_ts=None):
        day = original(bars, symbol=symbol, session_open_ts=session_open_ts)
        for tick in day.prints:
            tick["volume"] = 1.0
        return day

    venue_day.synthesise_day = patched
    _ = venue_core


def _mutant_day_always_partial(venue_core):
    """WRONG: every day is flagged partial, so the flag stops meaning anything."""
    import venue_day
    original = venue_day.synthesise_day

    def patched(bars, *, symbol, session_open_ts=None):
        day = original(bars, symbol=symbol, session_open_ts=session_open_ts)
        day.partial = True
        return day

    venue_day.synthesise_day = patched
    _ = venue_core


def _mutant_tick_source_always_sets_base_url(venue_core):
    """WRONG (#160): the tick source passes base_url ALWAYS, even when nothing is configured.

    This is the plausible over-fix. It satisfies "honour the configured endpoint" while quietly
    moving the definition of the production host out of the vendored constant and into the
    plugin, so a config change could repoint production. The control pin exists to catch exactly
    this, and it passed both before and after the real fix — which proves nothing until this
    mutant shows it can fail.
    """
    from pynecore_dnse import tick_source as ts
    original = ts.WSTickSource.__init__

    def patched(self, api_key, api_secret, wire_symbol, queue_max=20_000,
                ws_url=None, client_factory=None):
        return original(self, api_key, api_secret, wire_symbol, queue_max,
                        ws_url or "wss://ws-openapi.dnse.com.vn", client_factory)

    ts.WSTickSource.__init__ = patched
    _ = venue_core


def _control_noop(venue_core):
    """NOT a mutant: changes nothing. Its targeted test must still PASS.

    This is the harness's own discriminating control. Ten mutants all reporting CAUGHT is
    precisely the signature of a runner that fails everything — a crashing fixture, a bad path,
    a plugin that never loads would all produce a perfect score. If this control is ever
    reported CAUGHT, the result table is measuring the harness and not the venue, and every
    other row on it is worthless.
    """
    _ = venue_core


#: Controls that MUST escape. A caught control means the harness is broken.
CONTROLS: dict[str, tuple[str, object]] = {
    "noop_control": ("__test_the_oco_stop_leg_amends_the_existing_child_in_place_and_never_spawns__",
                     _control_noop),
}


MUTANTS: dict[str, tuple[str, object]] = {
    "oco_spawns": ("__test_the_oco_stop_leg_amends_the_existing_child_in_place_and_never_spawns__",
                   _mutant_oco_spawns),
    "activated_fills": ("__test_an_activated_conditional_never_fills_and_stays_terminal__",
                        _mutant_activated_fills),
    "activates_on_placement": ("__test_a_resting_stop_does_not_activate_before_a_print_crosses_its_trigger__",
                               _mutant_activates_on_placement),
    "uppercase_statuses": ("__test_a_quantity_one_derivative_never_partially_fills__",
                           _mutant_uppercase_statuses),
    "shell_disappears": ("__test_an_activated_shell_persists_on_the_stop_book_for_the_rest_of_the_day__",
                         _mutant_shell_disappears),
    "terminal_cancel_succeeds": ("__test_cancelling_a_venue_amended_terminal_child_is_refused_permanently__",
                                 _mutant_terminal_cancel_succeeds),
    "qty1_partial_fills": ("__test_a_quantity_one_derivative_never_partially_fills__",
                           _mutant_qty1_partial_fills),
    "nondeterministic_ids": ("__test_two_identical_replays_produce_identical_records__",
                             _mutant_nondeterministic_ids),
    "umbrella_clears_stop_price": ("__test_the_spent_umbrella_still_reads_activated_with_its_stop_price__",
                                   _mutant_umbrella_clears_stop_price),
    "integer_conditional_ids": ("__test_normal_order_gets_an_integer_id_and_a_conditional_gets_a_string_id__",
                                _mutant_integer_conditional_ids),
    # stage B: the venue day
    "day_label_defaults": ("test_venue_day.py::__test_a_day_without_a_label_is_REFUSED_rather_than_assumed__",
                           _mutant_day_label_defaults),
    "day_closes_only": ("test_venue_day.py::__test_a_synthetic_day_reconstructs_each_bar_from_its_prints__",
                        _mutant_day_closes_only),
    "day_volume_invented": ("test_venue_day.py::__test_a_synthetic_day_conserves_each_bar_volume__",
                            _mutant_day_volume_invented),
    "day_always_partial": ("test_venue_day.py::__test_a_day_covering_the_session_open_is_not_partial__",
                           _mutant_day_always_partial),
    # 160: the WS endpoint seam
    "tick_source_always_sets_base_url": (
        "test_ws_source_endpoint.py::__test_the_tick_source_omits_base_url_when_none_is_configured__",
        _mutant_tick_source_always_sets_base_url),
}


# --------------------------------------------------------------------------- pytest plugin

def pytest_configure(config):
    """Applied when pytest loads this file as a plugin with MUTANT set."""
    name = os.environ.get("MUTANT")
    if not name:
        return
    table = {**MUTANTS, **CONTROLS}
    if name not in table:
        # Raise rather than skip. A silently ignored mutant name makes every row read CAUGHT
        # or ESCAPED for reasons that have nothing to do with the patch.
        raise RuntimeError(f"unknown mutant {name!r}; known: {sorted(table)}")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import venue_core
    table[name][1](venue_core)


# --------------------------------------------------------------------------- runner

def _run_one(name: str, test_id: str, repo: Path) -> tuple[bool, str]:
    """Run ONE targeted test under ONE mutant. Returns (caught, evidence-line)."""
    env = dict(os.environ, MUTANT=name, PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(repo / "plugins" / "dnse" / "testing"))
    # A target may name its own file ("test_venue_day.py::...") or just a test in the default
    # conformance file. Being explicit beats a clever default that silently targets the wrong
    # test and reports a mutant as caught by a pin that never saw it.
    target = (f"plugins/dnse/tests/{test_id}" if test_id.endswith(".py") or "::" in test_id
              else f"{_TESTS}::{test_id}")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q",
         "-o", "addopts=--import-mode=importlib", "-p", "venue_mutants"],
        cwd=repo, env=env, capture_output=True, text=True)
    summary = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.strip()), "")
    # A mutant is CAUGHT only when the targeted test actually ran and FAILED. A collection
    # error, an import failure or "no tests ran" is a broken harness, not a catch.
    if "error" in summary.lower() or "no tests ran" in summary.lower():
        return False, f"HARNESS BROKEN: {summary}"
    return (proc.returncode == 1 and "1 failed" in summary), summary


def main() -> int:
    repo = Path(__file__).resolve().parents[3]
    print(f"repo={repo}\nmutants={len(MUTANTS)}  controls={len(CONTROLS)}\n")

    # The CONTROL runs first and on its own line. Read it before believing any mutant row:
    # if a no-op change is reported CAUGHT, the runner is failing tests for reasons unrelated
    # to the patch and the whole table is measuring the harness.
    broken = []
    for name, (test_id, _) in sorted(CONTROLS.items()):
        caught, summary = _run_one(name, test_id, repo)
        print(f"{'CAUGHT ' if caught else 'ESCAPED'}  {name:28s}  {summary}"
              f"   <- control, MUST escape")
        if caught:
            broken.append(name)
    print()
    if broken:
        print(f"HARNESS BROKEN: control(s) {', '.join(broken)} were reported CAUGHT.")
        print("A no-op change cannot fail a correct pin. Every mutant row below would be noise,")
        print("so none is reported. Fix the harness before reading any result.")
        return 2

    escaped = []
    for name, (test_id, _) in sorted(MUTANTS.items()):
        caught, summary = _run_one(name, test_id, repo)
        print(f"{'CAUGHT ' if caught else 'ESCAPED'}  {name:28s}  {summary}")
        if not caught:
            escaped.append(name)
    print()
    if escaped:
        print(f"FAIL: {len(escaped)} mutant(s) escaped: {', '.join(escaped)}")
        print("A pin that cannot fail is not a pin.")
        return 1
    print(f"OK: all {len(MUTANTS)} mutants caught, control escaped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
