"""#124 instrumentation — a CANCELLED on an OCO-origin order is OBSERVED.

A bracket is two venue records (the OCO umbrella and its NORMAL-book child),
and the cancel event names only the child, so nothing in the event says which
record the venue ended or whether the position moved with it. The broker now
reads the umbrella detail + the positions and logs ONE ``#124-OBS`` line
before emitting the event. Observation only: the event itself is unchanged,
and every read failure degrades to ``#124-OBS read-failed`` — never to a
blocked or missing event.

Same fake-client seam as ``test_cancel_disposition.py``: no network, no files.
"""
import asyncio
import logging

import pynecore.lib as lib

lib.bar_index = 0  # let the [BROKER] log formatter render during broker._emit()

from pynecore_dnse import broker
from pynecore.core.broker.models import LegType

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
