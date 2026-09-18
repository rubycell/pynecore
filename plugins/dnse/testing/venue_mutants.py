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


def _mutant_http_binds_all_interfaces(venue_core):
    """WRONG: the adapter binds any address, so the fake becomes an unauthenticated order
    endpoint reachable from the network."""
    import venue_http
    original = venue_http.VenueHTTP.__init__

    def patched(self, venue, *, host="127.0.0.1", port=0):
        return original(self, venue, host="127.0.0.1", port=port)   # silently accepts anything

    venue_http.VenueHTTP.__init__ = patched
    _ = venue_core


def _mutant_http_accepts_production_url(venue_core):
    """WRONG: a production hostname passes the guard, so a runner can drive DNSE believing it is
    driving the fake."""
    import venue_http
    venue_http.VenueHTTP.assert_not_production = staticmethod(lambda base_url: None)
    _ = venue_core


def _mutant_http_refuses_every_url(venue_core):
    """WRONG in the other direction: the guard refuses everything, including its own loopback.

    The control's mutant. A guard that refuses every url would satisfy the production-refusal pin
    while making the fake unusable, so the loopback-accepted control has to be able to fail.
    """
    import venue_http

    def patched(base_url):
        raise venue_http.ProductionRefused(base_url)

    venue_http.VenueHTTP.assert_not_production = staticmethod(patched)
    _ = venue_core


def _mutant_http_invents_executions(venue_core):
    """WRONG: the executions endpoint answers 200 with a plausible payload.

    Production answers 404 on this account, which is why the plugin books at cumulative VWAP.
    Inventing a payload sends the plugin down a path production never gives it — a fake being
    helpfully wrong, which is the failure this whole suite exists to prevent.
    """
    import venue_http
    original = venue_http._Handler.do_GET

    def patched(self):
        from urllib.parse import urlparse
        if venue_http._EXEC_PATH.match(urlparse(self.path).path):
            return self._send(200, {"executions": [{"price": 1980.0, "quantity": 1}]})
        return original(self)

    venue_http._Handler.do_GET = patched
    _ = venue_core


def _mutant_http_collapses_reject_code(venue_core):
    """WRONG: a refusal returns a bare 400 with no code, so the engine cannot branch on it."""
    import venue_http

    def patched(self, exc):
        return self._send(400, {"message": "bad request"})

    venue_http._Handler._reject = patched
    _ = venue_core


def _mutant_fake_broker_skips_config_endpoint_check(venue_core):
    """WRONG: the broker does not check the CONFIG's endpoints, only the server's bind address.

    The over-trusting version. Because the broker overwrites base_url with its own loopback port
    moments later, a production host in the toml is usually harmless BY ACCIDENT, and this mutant
    shows that the pin is what turns that accident into a guarantee.
    """
    from pynecore_dnse import fake_broker as fb
    fb.VenueHTTP.assert_not_production = staticmethod(lambda url: None)
    _ = venue_core


def _mutant_terminal_cancel_refusal_is_transient(venue_core):
    """WRONG: the refusal on a venue-amended terminal id clears after a retry.

    This is precisely the assumption the engine's deferral guard was built on, and #162 is what
    happened when the venue did not honour it. If this mutant escapes, the replay is not pinning
    the property that cost 66 seconds of unprotected position.
    """
    original = venue_core.FakeVenue.cancel
    seen = {"n": 0}

    def patched(self, order_id):
        order = self._orders.get(order_id)
        if order is not None and order["orderStatus"] in venue_core._TERMINAL:
            seen["n"] += 1
            if seen["n"] > 1:
                return dict(order)
        return original(self, order_id)

    venue_core.FakeVenue.cancel = patched
    _ = venue_core


def _mutant_wire_side_is_internal_vocabulary(venue_core):
    """WRONG (R1): rows carry buy/sell instead of NB/NS.

    The plugin's read funnel defaults an unknown side to "buy" (broker.py:1134), so this mutant
    does not raise anywhere — it silently books every SELL as a BUY. That is what made the
    stage-D "position=flat" claim untrustworthy.
    """
    original = venue_core.FakeVenue._make

    def patched(self, order_id, book, wire_side, qty, price, stop_price,
                stop_order_price, status, category=None):
        order = original(self, order_id, book, wire_side, qty, price, stop_price,
                         stop_order_price, status, category)
        order["side"] = "buy" if wire_side == venue_core.BUY else "sell"
        return order

    venue_core.FakeVenue._make = patched


def _mutant_stop_amends_to_the_print(venue_core):
    """WRONG (R2): the triggered child is rewritten to the PRINT, so it always fills.

    The gap-through case then becomes unreachable and "triggered, unfilled, still exposed" can
    never be reproduced. This is the model the first version of this fake shipped.
    """
    original = venue_core.FakeVenue._trigger_conditionals

    def patched(self, price):
        for order in list(self._orders.values()):
            if (order["book"] == "STOP_BOOK" and order["_category"] == "OCO"
                    and order["stopPrice"] is not None and self._crossed(order, price)):
                order["stopOrderPrice"] = price
        original(self, price)

    venue_core.FakeVenue._trigger_conditionals = patched


def _mutant_listing_leaks_external_order_id(venue_core):
    """WRONG (R3): the LISTING volunteers externalOrderId, which is detail-only."""
    venue_core.FakeVenue._listing = venue_core.FakeVenue._detail


def _mutant_wrong_book_resolves(venue_core):
    """WRONG (R4): orderCategory is ignored, so an id resolves on either book."""
    venue_core.FakeVenue._on_book = staticmethod(lambda order, category: True)


def _mutant_filled_rows_have_no_average_price(venue_core):
    """WRONG (R5): filled rows omit averagePrice, so fill_price is None on every fill."""
    original = venue_core.FakeVenue._detail

    def patched(self, order):
        row = original(self, order)
        row["averagePrice"] = None
        return row

    venue_core.FakeVenue._detail = patched


def _mutant_oco_spawns_two_legs_at_birth(venue_core):
    """WRONG: an OCO places a TP order AND a stop order at birth — the two-leg model.

    The plausible wrong venue, and until now nothing pinned against it: oco_spawns proved the
    AMEND pin, but the one-child-at-birth pin had no mutant at all.
    """
    original = venue_core.FakeVenue.place

    def patched(self, *, category, side, qty, price=None, stop_price=None,
                stop_order_price=None):
        row = original(self, category=category, side=side, qty=qty, price=price,
                       stop_price=stop_price, stop_order_price=stop_order_price)
        if category == "OCO":
            self._make(self._new_normal_id(), "NORMAL", venue_core._TO_WIRE.get(side, side),
                       qty, stop_price, None, None, venue_core.NEW)
        return row

    venue_core.FakeVenue.place = patched


def _mutant_activated_conditional_can_be_cancelled(venue_core):
    """WRONG: an Activated conditional accepts a cancel instead of answering CO-ORD-013."""
    original = venue_core.FakeVenue.cancel

    def patched(self, order_id, *, category=None):
        order = self._orders.get(order_id)
        if order is not None and order["orderStatus"] == venue_core.ACTIVATED:
            order["orderStatus"] = venue_core.CANCELED
            return self._detail(order)
        return original(self, order_id, category=category)

    venue_core.FakeVenue.cancel = patched


def _mutant_stop_entry_does_not_spawn_a_child(venue_core):
    """WRONG: a STOP entry activates without creating its normal-book child (#39 blindness)."""
    original = venue_core.FakeVenue._trigger_conditionals

    def patched(self, price):
        for order in list(self._orders.values()):
            if (order["book"] == "STOP_BOOK" and order["_category"] != "OCO"
                    and order["stopPrice"] is not None and self._crossed(order, price)
                    and order["orderStatus"] == venue_core.NEW):
                order["orderStatus"] = venue_core.ACTIVATED
                self._record(order)
                return
        original(self, price)

    venue_core.FakeVenue._trigger_conditionals = patched


def _mutant_partial_fill_ignores_volume(venue_core):
    """WRONG: a resting order fills completely regardless of the print's traded volume."""
    original = venue_core.FakeVenue._match_resting

    def patched(self, price, volume):
        original(self, price, max(volume, 1e9))

    venue_core.FakeVenue._match_resting = patched


def _mutant_closed_session_accepts_orders(venue_core):
    """WRONG: placement succeeds after the close instead of being refused."""
    venue_core._CLOSED_PHASES.clear()


def _mutant_atc_allows_cancels(venue_core):
    """WRONG: ATC accepts a cancel; measured is a refusal."""
    original = venue_core.FakeVenue.cancel

    def patched(self, order_id, *, category=None):
        saved, self.phase = self.phase, "continuous"
        try:
            return original(self, order_id, category=category)
        finally:
            self.phase = saved

    venue_core.FakeVenue.cancel = patched



def _mutant_normal_amend_answers_500(venue_core):
    """WRONG: a NORMAL-book amend answers 500 — the model the fake shipped for one revision.

    It is the plausible wrong answer, because a 500 on amend IS a measured venue fact; it just
    belongs to the CONDITIONAL book. Serving it on the normal book makes the plugin park on a
    refusal the venue never sends, which is exactly what the staged probe did at T6.
    """
    original = venue_core.FakeVenue.amend

    def patched(self, order_id, *, price=None, qty=None, category=None):
        order = self._orders.get(order_id)
        if order is not None and order["book"] == "NORMAL":
            raise venue_core.VenueServerError(500, "amend is not supported")
        return original(self, order_id, price=price, qty=qty, category=category)

    venue_core.FakeVenue.amend = patched


def _mutant_conditional_amend_succeeds_in_place(venue_core):
    """WRONG in the other direction: a CONDITIONAL amend succeeds instead of answering 500.

    The tidy version. It would leave the plugin's conditional cancel+replace and its exit park —
    both of which exist BECAUSE the venue 500s there — untested offline.
    """
    original = venue_core.FakeVenue.amend

    def patched(self, order_id, *, price=None, qty=None, category=None):
        order = self._orders.get(order_id)
        if order is not None and order["book"] == "STOP_BOOK":
            if price is not None:
                order["price"] = price
            return self._detail(order)
        return original(self, order_id, price=price, qty=qty, category=category)

    venue_core.FakeVenue.amend = patched


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



# --------------------------------------------------------------------------- the 1m-derived day

def _mutant_derived_day_is_just_synthetic(venue_core):
    """WRONG: a day built from downloaded history is labelled SYNTHETIC.

    The tidy-looking option, and the one that loses the whole point: SYNTHETIC means derived
    from bars already tracked in this repo, and a reader could no longer tell that apart from a
    dated download of a real session at the venue.
    """
    import venue_day
    original = venue_day.derive_day_from_1m_bars

    def patched(bars, **kwargs):
        day = original(bars, **kwargs)
        day.label = venue_day.DayLabel.SYNTHETIC
        return day

    venue_day.derive_day_from_1m_bars = patched
    _ = venue_core


def _mutant_derived_day_needs_no_provenance(venue_core):
    """WRONG: the new label loads with no provenance behind it.

    The label then claims a specific session downloaded at a specific time while carrying
    nothing that could be checked — a longer string pretending to be evidence.
    """
    import venue_day
    original = venue_day.VenueDay.from_dict.__func__

    def patched(cls, raw):
        raw = dict(raw)
        if raw.get("label") == venue_day.DayLabel.DERIVED_FROM_1M.value:
            raw["provenance"] = dict(raw.get("provenance") or {},
                                     **{key: "?" for key in venue_day.REQUIRED_PROVENANCE})
        return original(cls, raw)

    venue_day.VenueDay.from_dict = classmethod(patched)
    _ = venue_core


def _mutant_derived_day_accepts_coarse_bars(venue_core):
    """WRONG: 5m bars are stamped DERIVED-FROM-1M.

    The derived day then replays a fifth of the price path it claims to carry, so every stop the
    real session triggered between the sampled minutes never triggers in the replay.
    """
    import venue_day
    original = venue_day.derive_day_from_1m_bars

    def patched(bars, **kwargs):
        bar_list = [dict(b) for b in bars]
        first = bar_list[0]["timestamp"] if bar_list else 0
        squeezed = [dict(b, timestamp=first + index * venue_day.MINUTE_MS)
                    for index, b in enumerate(bar_list)]
        return original(squeezed, **kwargs)

    venue_day.derive_day_from_1m_bars = patched
    _ = venue_core


def _mutant_history_keeps_venue_seconds(venue_core):
    """WRONG: the history parser leaves the venue's SECONDS unconverted.

    Measured consequence, not a theoretical one: a day stamped in seconds replays its whole
    session inside a second of wall clock, the engine's wall-clock anchoring sees a missed
    timeframe boundary every real minute, and it substitutes flat synthetic bars for the entire
    run. The run looks healthy and produces no trades.
    """
    import venue_day_from_history as helper
    original = helper.bars_from_ohlc_body

    def patched(body):
        return [dict(bar, timestamp=bar["timestamp"] // 1000) for bar in original(body)]

    helper.bars_from_ohlc_body = patched
    _ = venue_core


def _mutant_history_reads_empty_as_a_quiet_day(venue_core):
    """WRONG: a 200 answer with no bars is read as a session in which nothing traded.

    Measured 2026-09-18: that is how /price/ohlc reports a symbol it does not serve — the dated
    contract code answers 200 with every array empty. Reading it as a quiet day writes files for
    sessions that never happened.
    """
    import venue_day_from_history as helper
    original = helper.bars_from_ohlc_body

    def patched(body):
        if isinstance(body, dict) and body.get("t") == []:
            return []
        return original(body)

    helper.bars_from_ohlc_body = patched
    _ = venue_core


def _mutant_history_writes_anywhere(venue_core):
    """WRONG: the destination guard accepts a path inside workdir/data.

    That directory holds the tracked .ohlcv bar stores every offline backtest in this repo
    reads, and they are shared with the main checkout's working tree.
    """
    import venue_day_from_history as helper
    helper.refuse_bar_store_paths = lambda target: target
    _ = venue_core



def _mutant_venue_record_written_once(venue_core):
    """WRONG: the record file is written at the FIRST transition only.

    The plausible cheap implementation, and it loses exactly what a grade needs: the entry's
    activation, its child's fill and the flatten all land after the first row.
    """
    original = venue_core.FakeVenue._record

    def patched(self, order):
        already = bool(self._records)
        file_before = self._record_file
        if already:
            self._record_file = None
        try:
            original(self, order)
        finally:
            self._record_file = file_before

    venue_core.FakeVenue._record = patched


def _mutant_dataset_writes_any_name(venue_core):
    """WRONG: the dataset builder accepts any name, including a shared bar store.

    ``dnse_VN30F1M_1`` is the accumulated 1m history every offline backtest reads; overwriting
    it destroys data no session here owns.
    """
    import venue_day_dataset
    venue_day_dataset.refuse_foreign_dataset = lambda name: name
    _ = venue_core


def _mutant_parity_window_from_a_trade_stamp(venue_core):
    """WRONG: the parity window is derived from the fake's first TRADE time again.

    Measured 2026-09-18: that derivation put the window 39 s past a bar boundary and the
    acceptance test reported "backtest 3 vs fake 4" when both engines had produced 4. It can
    equally trim a window until a real difference disappears, which is the worse direction.
    """
    import sys as _sys
    from pathlib import Path as _Path
    offline = _Path(__file__).resolve().parent / "fixtures" / "offline"
    if str(offline) not in _sys.path:
        _sys.path.insert(0, str(offline))
    import trade_list_parity
    original = trade_list_parity.compare

    def patched(bt, fake, offset_s, window_start):
        from datetime import datetime, timedelta
        if fake:
            window_start = (datetime.fromisoformat(fake[0]["Date/Time"].strip())
                            - timedelta(seconds=offset_s) + timedelta(seconds=39))
        return original(bt, fake, offset_s, window_start)

    trade_list_parity.compare = patched
    _ = venue_core



def _mutant_ohlc_serves_the_old_dict_shape(venue_core):
    """WRONG: /price/ohlc answers {"data": [ {timestamp, open, ...} ]} again.

    THE REAL HISTORICAL BUG, restored. This is what the fake served until the second pass of the
    suite run, and it survived a round-1 review that rated it LOW because FakeVenueBroker
    overrides download_ohlcv, so the payload "could not be reached". It could: the direct-client
    scripts are another door, and through that door it broke the L0 gate loudly and
    _stop_already_crossed SILENTLY, the latter by failing open to False.

    The mutant targets the SILENT reader. A mutant caught by an exception is a weaker
    demonstration than one caught by a safety check quietly giving the wrong answer.
    """
    import venue_http
    original = venue_http._Handler._send

    def patched(self, status, payload):
        if isinstance(payload, dict) and "t" in payload and "nextTime" in payload:
            payload = {"data": [
                {"timestamp": t * 1000, "open": o, "high": h, "low": low, "close": c,
                 "volume": v}
                for t, o, h, low, c, v in zip(payload["t"], payload["o"], payload["h"],
                                              payload["l"], payload["c"], payload["v"])]}
        return original(self, status, payload)

    venue_http._Handler._send = patched
    _ = venue_core



def _mutant_phase_is_always_continuous(venue_core):
    """WRONG: the venue is open at every instant, which is what it did before the second pass.

    The consequence is not a wrong answer but a MISSING QUESTION: with the venue permanently
    open, no closed-hours or ATC behaviour can be posed at all, so T33 could only ever fail and
    the lunch-queue case could not be reproduced.
    """
    venue_core.phase_at = lambda timestamp_ms: "continuous"
    original = venue_core.FakeVenue.advance_to

    def patched(self, timestamp_ms):
        self.phase = "continuous"
        return self.phase

    venue_core.FakeVenue.advance_to = patched
    _ = original



def _mutant_oco_accepted_on_a_stock(venue_core):
    """WRONG: OCO is accepted on a STOCK, which the venue's category matrix forbids.

    This is what the fake did until the 2026-08-06 changelog matrix was checked. A fake MORE
    PERMISSIVE than the venue is the dangerous direction: the engine learns a bracket it can
    place, every offline pin agrees, and the refusal arrives live.
    """
    original = venue_core.FakeVenue.place

    def patched(self, *, category, **kwargs):
        if category == "OCO" and self.market_type == "STOCK":
            market_type = self.market_type
            self.market_type = "DERIVATIVE"
            try:
                return original(self, category=category, **kwargs)
            finally:
                self.market_type = market_type
        return original(self, category=category, **kwargs)

    venue_core.FakeVenue.place = patched


def _mutant_history_ignores_paging(venue_core):
    """WRONG: every page of order history is the whole result again.

    It does not crash the plugin's paging loop — the loop terminates on the first call because
    total equals what was served. It removes the ability to exercise the loop's MULTI-PAGE branch
    at all, which is the branch that proves completeness.
    """
    import venue_http
    original = venue_http.parse_qs

    def patched(query_string, *args, **kwargs):
        parsed = original(query_string, *args, **kwargs)
        # Drop exactly the paging keys, so the route reads every other parameter as before and
        # simply never sees these two. A first attempt rewrote the RESPONSE envelope instead and
        # ESCAPED, because the pin asserts on the rows: the mutant has to remove the behaviour,
        # not relabel its output.
        parsed.pop("pageSize", None)
        parsed.pop("pageIndex", None)
        return parsed

    venue_http.parse_qs = patched
    _ = venue_core


def _mutant_close_position_only_acknowledges(venue_core):
    """WRONG: closing a position answers OK without placing the opposing order.

    The guide states the mechanic: a close IS an order, opposite side, type LO, at the ceiling or
    floor. A bare acknowledgement leaves the position open while the caller believes it closed —
    the worst possible direction for a flatten.
    """
    import venue_http
    original = venue_http._Handler._send

    def patched(self, status, payload):
        if isinstance(payload, dict) and payload.get("orderType") == "LO":
            payload = {"message": "closed"}
        return original(self, status, payload)

    venue_http._Handler._send = patched
    _ = venue_core


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
    # stage C2: the socket adapter
    "http_binds_all_interfaces": (
        "test_venue_http.py::__test_the_adapter_refuses_to_bind_a_non_loopback_address__",
        _mutant_http_binds_all_interfaces),
    "http_accepts_production_url": (
        "test_venue_http.py::__test_the_adapter_refuses_a_production_looking_base_url__",
        _mutant_http_accepts_production_url),
    "http_refuses_every_url": (
        "test_venue_http.py::__test_the_adapter_accepts_its_own_loopback_url__",
        _mutant_http_refuses_every_url),
    "http_invents_executions": (
        "test_venue_http.py::__test_an_uncalled_endpoint_answers_404_as_production_does__",
        _mutant_http_invents_executions),
    "http_collapses_reject_code": (
        "test_venue_http.py::__test_a_cancel_refusal_carries_the_venue_code_over_the_wire__",
        _mutant_http_collapses_reject_code),
    "terminal_cancel_refusal_is_transient": (
        "test_162_park_replay.py::__test_the_forced_cancel_is_refused_and_stays_refused__",
        _mutant_terminal_cancel_refusal_is_transient),
    "wire_side_is_internal_vocabulary": (
        "test_venue_http.py::__test_a_wire_round_trip_reads_back_a_sell_as_a_sell__",
        _mutant_wire_side_is_internal_vocabulary),
    "stop_amends_to_the_print": (
        "__test_a_gap_through_the_stop_limit_leaves_the_child_resting_and_unprotecting__",
        _mutant_stop_amends_to_the_print),
    "listing_leaks_external_order_id": (
        "test_venue_http.py::__test_the_listing_hides_external_order_id_and_the_detail_shows_it__",
        _mutant_listing_leaks_external_order_id),
    "wrong_book_resolves": (
        "test_venue_http.py::__test_an_id_looked_up_on_the_wrong_book_is_not_found__",
        _mutant_wrong_book_resolves),
    "filled_rows_have_no_average_price": (
        "test_venue_http.py::__test_a_filled_row_carries_an_average_price__",
        _mutant_filled_rows_have_no_average_price),
    "oco_spawns_two_legs_at_birth": (
        "__test_an_oco_umbrella_is_activated_from_birth_and_spawns_exactly_one_child__",
        _mutant_oco_spawns_two_legs_at_birth),
    "activated_conditional_can_be_cancelled": (
        "__test_cancelling_an_activated_conditional_is_refused_as_done__",
        _mutant_activated_conditional_can_be_cancelled),
    "stop_entry_does_not_spawn_a_child": (
        "__test_a_print_through_the_trigger_activates_the_stop_and_spawns_a_normal_child__",
        _mutant_stop_entry_does_not_spawn_a_child),
    "partial_fill_ignores_volume": (
        "__test_a_resting_limit_partially_fills_by_print_volume_then_completes__",
        _mutant_partial_fill_ignores_volume),
    "closed_session_accepts_orders": (
        "__test_placing_after_the_close_is_refused_with_the_measured_code__",
        _mutant_closed_session_accepts_orders),
    "atc_allows_cancels": (
        "__test_cancelling_during_atc_is_refused_with_the_measured_code__",
        _mutant_atc_allows_cancels),
    "normal_amend_answers_500": (
        "test_venue_http.py::__test_a_normal_book_derivative_amend_succeeds_IN_PLACE__",
        _mutant_normal_amend_answers_500),
    "conditional_amend_succeeds_in_place": (
        "test_venue_http.py::__test_a_conditional_amend_answers_500_whatever_the_asset__",
        _mutant_conditional_amend_succeeds_in_place),
    "fake_broker_skips_config_endpoint_check": (
        "test_venue_http.py::__test_the_fake_broker_refuses_a_production_endpoint_from_the_config__",
        _mutant_fake_broker_skips_config_endpoint_check),
    # the 1m-derived day and its downloader
    "derived_day_is_just_synthetic": (
        "test_venue_day_derived.py::__test_a_day_built_from_downloaded_1m_history_is_labelled_derived_from_1m__",
        _mutant_derived_day_is_just_synthetic),
    "derived_day_needs_no_provenance": (
        "test_venue_day_derived.py::__test_a_derived_day_with_no_provenance_is_REFUSED_on_load__",
        _mutant_derived_day_needs_no_provenance),
    "derived_day_accepts_coarse_bars": (
        "test_venue_day_derived.py::__test_a_file_of_five_minute_bars_is_REFUSED_rather_than_stamped_1m__",
        _mutant_derived_day_accepts_coarse_bars),
    "history_keeps_venue_seconds": (
        "test_venue_day_from_history.py::__test_the_venue_seconds_become_milliseconds__",
        _mutant_history_keeps_venue_seconds),
    "history_reads_empty_as_a_quiet_day": (
        "test_venue_day_from_history.py::__test_a_200_answer_with_no_bars_at_all_is_REFUSED__",
        _mutant_history_reads_empty_as_a_quiet_day),
    "history_writes_anywhere": (
        "test_venue_day_from_history.py::__test_an_output_path_under_the_shared_bar_store_is_REFUSED__",
        _mutant_history_writes_anywhere),
    "venue_record_written_once": (
        "test_venue_record_file.py::__test_every_later_transition_reaches_the_file_too__",
        _mutant_venue_record_written_once),
    "dataset_writes_any_name": (
        "test_venue_day_dataset.py::__test_the_builder_itself_refuses_a_foreign_name_before_writing__",
        _mutant_dataset_writes_any_name),
    "parity_window_from_a_trade_stamp": (
        "test_trade_list_parity.py::__test_the_window_starts_at_the_first_live_bar_not_at_a_wall_clock_trade_time__",
        _mutant_parity_window_from_a_trade_stamp),
    "ohlc_serves_the_old_dict_shape": (
        "test_venue_http_ohlc_shape.py::__test_a_stop_below_the_market_is_seen_as_already_crossed__",
        _mutant_ohlc_serves_the_old_dict_shape),
    "phase_is_always_continuous": (
        "test_venue_phase_from_bar_time.py::__test_a_placement_during_the_replayed_lunch_still_works_and_one_after_the_close_does_not__",
        _mutant_phase_is_always_continuous),
    "oco_accepted_on_a_stock": (
        "test_venue_http_sdk_guide_endpoints.py::__test_an_oco_order_is_refused_on_a_stock__",
        _mutant_oco_accepted_on_a_stock),
    "history_ignores_paging": (
        "test_venue_http_honours_parameters.py::__test_order_history_honours_page_size_and_page_index__",
        _mutant_history_ignores_paging),
    "close_position_only_acknowledges": (
        "test_venue_http_sdk_guide_endpoints.py::__test_closing_a_position_places_the_opposing_order_and_flattens_it__",
        _mutant_close_position_only_acknowledges),
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
