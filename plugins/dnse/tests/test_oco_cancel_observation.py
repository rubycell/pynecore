"""#124 instrumentation — a CANCELLED on an OCO-origin order is OBSERVED.

A bracket is two venue records (the OCO umbrella and its NORMAL-book child),
and the cancel event names only the child, so nothing in the event says which
record the venue ended or whether the position moved with it. The broker now
reads the umbrella detail + the positions and logs ONE ``#124-OBS`` line
before emitting the event.

The read is also the EVIDENCE the classification needs (#124 fix): when the
umbrella itself reads back TERMINAL the child died with it, so the emitted
event carries the venue-driven ``cancel_reason`` the engine's guard
recognises. Everything else is FAIL-CLOSED — a still-live (``Activated``)
umbrella, an unknown status or a failed read emit the event UNTAGGED, and
every read failure degrades to ``#124-OBS read-failed`` — never to a blocked
or missing event.

Same fake-client seam as ``test_cancel_disposition.py``: no network, no files.
"""
import asyncio
import logging

import pynecore.lib as lib

lib.bar_index = 0  # let the [BROKER] log formatter render during broker._emit()

from pynecore_dnse import broker
from pynecore.core.broker.models import (
    CANCEL_REASON_VENUE_OCO_UMBRELLA_TERMINAL, VENUE_DRIVEN_CANCEL_REASONS,
    LegType,
)

_SECDEF_ROW = [{"ceilingPrice": "1550", "floorPrice": "1450", "securityGroupId": "FU"}]
_LOAN_OK = (200, {"loanPackages": [{"id": 42}]})
_FLAT_POSITIONS = (200, {"positions": [], "total": 0})


def _broker(fake_client, tmp_path, **client_responses):
    responses = {"get_security_definition": (200, _SECDEF_ROW),
                 "get_loan_packages": _LOAN_OK,
                 "get_positions": _FLAT_POSITIONS}
    responses.update(client_responses)
    config = broker.DNSEBrokerConfig(
        api_key="k", api_secret="s", account_no="ACC001", trading_token="tok-A",
        token_file=str(tmp_path / "missing_token.json"))
    instance = broker.DNSEBroker(symbol="VN30F1M", timeframe="15", config=config)
    instance._client = fake_client(**responses)
    return instance


def _track_oco_child(instance, child_id="C1", umbrella_id="UMB1"):
    """The post-place tracking shape of a bracket: the NORMAL-book child is
    tracked, and the umbrella it came from is retained in memory (#124)."""
    instance._identity[child_id] = ("L", "L", LegType.TAKE_PROFIT)
    instance._order_category[child_id] = "NORMAL"
    instance._placed_category[child_id] = "OCO"
    instance._oco_umbrella_ids[child_id] = umbrella_id


def _cancelled_row(order_id="C1"):
    return {"id": order_id, "symbol": "VN30F1M", "side": "NS", "quantity": 1,
            "orderStatus": "Canceled", "fillQuantity": 0}


def _observation_lines(caplog):
    return [record.message for record in caplog.records
            if "#124-OBS" in record.message]


def __test_oco_child_cancel_logs_one_observation_line__(
        fake_client, tmp_path, caplog):
    """The umbrella detail + the position are read and reported once, and the
    CANCELLED event still reaches the engine unchanged."""
    b = _broker(fake_client, tmp_path, get_order_detail=(
        200, {"id": "UMB1", "orderStatus": "Activated", "externalOrderId": "C1"}))
    _track_oco_child(b)

    with caplog.at_level(logging.DEBUG):
        events = asyncio.run(b._scan_row(_cancelled_row()))

    lines = _observation_lines(caplog)
    assert len(lines) == 1, f"exactly one #124-OBS line expected, got {lines!r}"
    assert "umbrella=UMB1" in lines[0] and "umbrella_status=Activated" in lines[0]
    assert "externalOrderId=C1" in lines[0] and "position=0" in lines[0]
    detail_calls = [c for c in b._client.calls if c[0] == "get_order_detail"]
    assert detail_calls and detail_calls[-1][2]["order_category"] == "OCO", \
        "the umbrella must be read on the OCO book"
    assert b._client.count("get_positions") == 1, "the position must be read too"
    assert len(events) == 1 and events[0].event_type == "cancelled", \
        "the observation must not change what is emitted"
    assert events[0].pine_id == "L"


def __test_observation_read_failure_still_delivers_the_event__(
        fake_client, tmp_path, caplog):
    """A failing umbrella read logs ``read-failed`` and never blocks or
    corrupts the event path — the whole point of instrumentation that runs
    inside the live event loop."""
    def _boom(*_args, **_kwargs):
        raise RuntimeError("umbrella read exploded")

    b = _broker(fake_client, tmp_path, get_order_detail=_boom)
    _track_oco_child(b)

    with caplog.at_level(logging.DEBUG):
        events = asyncio.run(b._scan_row(_cancelled_row()))

    lines = _observation_lines(caplog)
    assert len(lines) == 1 and "read-failed" in lines[0], \
        f"a failed read must say so exactly once, got {lines!r}"
    assert len(events) == 1 and events[0].event_type == "cancelled", \
        "the event is emitted unchanged even when the observation fails"


def __test_non_oco_cancel_is_not_observed__(fake_client, tmp_path, caplog):
    """The discriminating control: a plain NORMAL-book order's cancel carries
    no umbrella, so it must cost no extra venue reads and log nothing."""
    b = _broker(fake_client, tmp_path)
    b._identity["N1"] = ("L", None, LegType.ENTRY)
    b._order_category["N1"] = "NORMAL"

    with caplog.at_level(logging.DEBUG):
        events = asyncio.run(b._scan_row(_cancelled_row("N1")))

    assert not _observation_lines(caplog), "no umbrella -> no observation"
    assert b._client.count("get_positions") == 0, "and no extra venue reads"
    assert len(events) == 1 and events[0].event_type == "cancelled"


def __test_terminal_umbrella_tags_the_event_venue_driven__(
        fake_client, tmp_path, caplog):
    """POSITIVE evidence: the umbrella the child was born from reads back
    ``Canceled`` -> the child died WITH its bracket, so the event is stamped
    :data:`CANCEL_REASON_VENUE_OCO_UMBRELLA_TERMINAL` and the engine trims +
    re-arms instead of quarantining a live position.

    Discriminating against the fail-closed default: the ONLY difference from
    ``__test_oco_child_cancel_logs_one_observation_line__`` (umbrella
    ``Activated`` -> no tag) is the umbrella's status.
    """
    b = _broker(fake_client, tmp_path, get_order_detail=(
        200, {"id": "UMB1", "orderStatus": "Canceled", "externalOrderId": "C1"}))
    _track_oco_child(b)

    with caplog.at_level(logging.DEBUG):
        events = asyncio.run(b._scan_row(_cancelled_row()))

    assert len(events) == 1
    assert events[0].cancel_reason == CANCEL_REASON_VENUE_OCO_UMBRELLA_TERMINAL, (
        "a TERMINAL umbrella is positive venue evidence and must be tagged, "
        f"got {events[0].cancel_reason!r}")
    assert events[0].cancel_reason in VENUE_DRIVEN_CANCEL_REASONS, (
        "the tag must be a member of the set the engine's guard reads")


def __test_live_umbrella_emits_no_tag_and_flags_a_respawn_candidate__(
        fake_client, tmp_path, caplog):
    """FAIL-CLOSED: an ``Activated`` (from-birth phantom, #41) umbrella is NOT
    evidence the venue ended the bracket — no tag, so an operator's app cancel
    keeps being treated as external. A live umbrella whose ``externalOrderId``
    names a DIFFERENT child is logged as a respawn candidate (adoption is
    deliberately deferred), and the event still flows untagged.
    """
    b = _broker(fake_client, tmp_path, get_order_detail=(
        200, {"id": "UMB1", "orderStatus": "Activated", "externalOrderId": "C2"}))
    _track_oco_child(b)

    with caplog.at_level(logging.DEBUG):
        events = asyncio.run(b._scan_row(_cancelled_row()))

    assert events[0].cancel_reason is None, (
        "a live umbrella must NOT be reclassified — operator cancels stay "
        f"external, got {events[0].cancel_reason!r}")
    respawn = [r.message for r in caplog.records if "respawn-candidate" in r.message]
    assert len(respawn) == 1 and "externalOrderId=C2" in respawn[0], (
        f"a moved child must be logged loudly, got {respawn!r}")


def __test_unreadable_umbrella_emits_no_tag__(fake_client, tmp_path, caplog):
    """FAIL-CLOSED on the read itself: no evidence -> no tag -> the engine's
    bounded re-arm handles it. The ``read-failed`` line is the audit trail."""
    def _boom(*_args, **_kwargs):
        raise RuntimeError("umbrella read exploded")

    b = _broker(fake_client, tmp_path, get_order_detail=_boom)
    _track_oco_child(b)

    with caplog.at_level(logging.DEBUG):
        events = asyncio.run(b._scan_row(_cancelled_row()))

    assert events[0].cancel_reason is None, "an unreadable umbrella proves nothing"
    assert any("read-failed" in line for line in _observation_lines(caplog))


class _FakeWSSource:
    """Stands in for :class:`WSOrderSource`: hands the watch loop one batch of
    already-normalised raw rows (what ``_order_model_to_raw_row`` produces),
    then stays quiet."""

    def __init__(self, rows):
        self._batches = [rows]

    async def collect(self, _timeout):
        return self._batches.pop(0) if self._batches else []


def __test_ws_delivered_cancel_carries_the_tag_and_the_poll_copy_does_not_reread__(
        fake_client, tmp_path, caplog):
    """#124 must NOT be transport-dependent: the #121 WS order feed is default-ON
    and on prod a child-cancel can reach us over WS BEFORE the REST poll sees it.
    A WS-delivered CANCELLED for a terminal-umbrella bracket therefore has to
    carry the same venue-driven tag the poll path stamps — otherwise the
    classification would silently depend on which transport won the race.

    It does, structurally: ``_collect_ws_order_events`` funnels every WS frame
    through the SAME ``_scan_row`` (on the single ``watch_orders`` task, never
    the SDK callback), so the observation runs there and nowhere else — no
    second emit path, no blocking of the WS callback.

    The second half is the dedup discipline: the poll then re-reads the same
    row, and the shared ``_last_seen`` watermark (``(cumulative, raw_status)``,
    transport-agnostic) drops it — so ONE cancel costs exactly ONE umbrella
    read, never two.
    """
    b = _broker(fake_client, tmp_path, get_order_detail=(
        200, {"id": "UMB1", "orderStatus": "Canceled", "externalOrderId": "C1"}))
    _track_oco_child(b)
    b._ws_order_source = _FakeWSSource([_cancelled_row()])

    with caplog.at_level(logging.DEBUG):
        ws_events = asyncio.run(b._collect_ws_order_events(0.01))

    assert len(ws_events) == 1, f"the WS frame must yield its event, got {ws_events!r}"
    assert ws_events[0].event_type == "cancelled"
    assert ws_events[0].cancel_reason == CANCEL_REASON_VENUE_OCO_UMBRELLA_TERMINAL, (
        "#124: a WS-delivered cancel must carry the same venue-driven tag as "
        f"the poll-delivered one, got {ws_events[0].cancel_reason!r}")
    reads_after_ws = b._client.count("get_order_detail")
    assert reads_after_ws == 1, f"one umbrella read for one cancel, got {reads_after_ws}"

    # The REST poll now delivers the SAME row (both transports saw the cancel).
    poll_events = asyncio.run(b._scan_row(_cancelled_row()))

    assert poll_events == [], "the duplicate poll delivery must dedup to nothing"
    assert b._client.count("get_order_detail") == reads_after_ws, (
        "the duplicate delivery must NOT cost a second umbrella read — the "
        "observation is one-shot per cancel via the _last_seen watermark")


def __test_umbrella_retained_at_place_time__(fake_client, tmp_path):
    """``_place`` must retain the umbrella id in memory — previously it only
    reached the journal as a ref, so the observation had nothing to read."""
    from pynecore.core.broker.models import DispatchEnvelope, ExitIntent
    b = _broker(fake_client, tmp_path, post_order=(
        201, {"id": "UMB1", "symbol": "VN30F1M", "side": "NS", "quantity": 1,
              "orderStatus": "New"}), get_order_detail=(
        200, {"id": "C1", "symbol": "VN30F1M", "side": "NS", "quantity": 1,
              "orderStatus": "New", "externalOrderId": "C1"}))
    # A TP + SL bracket is the OCO shape (tp_price AND sl_price together).
    intent = ExitIntent(pine_id="X", from_entry="L", symbol="VN30F1M",
                        side="sell", qty=1, sl_price=1480.0, tp_price=1540.0)
    envelope = DispatchEnvelope(intent=intent, run_tag="abcd",
                                bar_ts_ms=1_700_000_000_000)

    orders = asyncio.run(b.execute_exit(envelope))

    tracked = str(orders[0].id)
    assert b._oco_umbrella_ids.get(tracked) == "UMB1", (
        f"the umbrella id must be retained for the tracked order {tracked!r}, "
        f"got {b._oco_umbrella_ids!r}")


def __test_every_outgoing_cancel_logs_before_the_wire_call__(
        fake_client, tmp_path, caplog):
    """Forensics control: the bool ``execute_cancel`` path sent a wire cancel
    with NO broker-side log at all, so "no cancel was sent" could not be told
    apart from "the log line is missing". Absence of this line is now proof.
    """
    from pynecore.core.broker.models import CancelIntent, DispatchEnvelope
    b = _broker(fake_client, tmp_path, cancel_order=(200, {}), get_order_detail=(
        200, {"id": "N1", "symbol": "VN30F1M", "side": "NB", "quantity": 1,
              "orderStatus": "Canceled", "fillQuantity": 0}))
    b._cancel_verify_attempts, b._cancel_verify_delay = 1, 0.0
    b._identity["N1"] = ("L", None, LegType.ENTRY)
    b._order_category["N1"] = "NORMAL"
    envelope = DispatchEnvelope(
        intent=CancelIntent(pine_id="L", symbol="VN30F1M"), run_tag="abcd",
        bar_ts_ms=1_700_000_000_000)
    b._order_ids[envelope.intent.intent_key] = ["N1"]

    with caplog.at_level(logging.DEBUG):
        cancelled = asyncio.run(b.execute_cancel(envelope))

    wire_lines = [r.message for r in caplog.records if "cancel -> wire" in r.message]
    assert len(wire_lines) == 1, \
        f"one line per outgoing cancel expected, got {wire_lines!r}"
    assert "order=N1" in wire_lines[0] and "book=NORMAL" in wire_lines[0]
    assert "pine=L" in wire_lines[0], "the intent must be named when known"
    assert cancelled is True
