"""`record_frame` must persist every frame, and persist it MASKED (#157, #113).

WHY THESE EXIST. On 2026-09-18 an afternoon market-data capture received 351,731
frames, counted them, and kept none: the loop did ``counts[key] += 1`` and nothing
durable. The log was 6,812 bytes holding six sample payloads. It read as a
successful capture — correct symbol, clean exit, a per-channel table — and there was
no venue day in it. ``--dump`` fixes that, and these pins hold the two properties
that make a dump safe rather than merely present:

* **every counted frame is written** — a counter and a writer are two code paths,
  and this is the week where two paths silently disagreeing has cost repeatedly;
* **what is written is MASKED** — the masking exists because a real account number
  reached an evidence file on 2026-09-16, caught by the scrub gate rather than by
  the code. A dump written before the mask reintroduces that once per frame.

They drive ``record_frame`` directly. The capture loop around it only exists inside
a live websocket session, so anything left in the loop is untestable by
construction — which is why the per-frame body was factored out.

Run explicitly: ``pytest plugins/dnse/tests/test_probe_frame_record.py -q``.
"""
from __future__ import annotations

import io
import json
import pathlib
import sys
from collections import Counter

import pytest

PROBE_DIR = (pathlib.Path(__file__).resolve().parents[1]
             / "testing" / "live_test")
sys.path.insert(0, str(PROBE_DIR))

import probe_ws_market_data as probe  # noqa: E402

#: A real account-shaped identifier, nested where the venue actually puts it.
ACCOUNT = "0001234567"
INVESTOR = "9876543210"


def _order_frame(seq: int) -> dict:
    """An order frame with identifiers INSIDE the nested payload.

    Nested on purpose: the 2026-09-16 leak happened because a top-level-only
    lookup masked nothing while looking like it had worked.
    """
    return {"T": "do", "seq": seq,
            "order": {"id": f"x{seq}", "accountNo": ACCOUNT,
                      "custodyCode": "ABC123", "investorId": INVESTOR}}


def _market_frame(seq: int) -> dict:
    return {"T": "t", "symbol": "41I1GA000", "matchPrice": 1990.0 + seq}


def __test_every_counted_frame_is_written__():
    """N frames in -> N lines out, and the counter agrees with the writer."""
    counts: Counter = Counter()
    samples: dict = {}
    sink = io.StringIO()
    frames = [_market_frame(i) for i in range(25)] + [_order_frame(i) for i in range(7)]

    for frame in frames:
        probe.record_frame(frame, counts, samples, (), sink)

    lines = sink.getvalue().splitlines()
    assert len(lines) == len(frames), (
        f"{len(frames)} frames in, {len(lines)} lines out — the writer and the "
        f"counter disagree"
    )
    assert sum(counts.values()) == len(lines), (
        f"summary count {sum(counts.values())} != dump line count {len(lines)}; "
        f"one of the two paths is dropping frames silently"
    )
    for line in lines:
        json.loads(line)  # each line must stand alone as JSON


def __test_the_dump_is_masked__():
    """THE SAFETY PROPERTY. No account-shaped identifier reaches the file."""
    counts: Counter = Counter()
    sink = io.StringIO()
    probe.record_frame(_order_frame(1), counts, {}, (INVESTOR,), sink)

    written = sink.getvalue()
    assert ACCOUNT not in written, (
        f"account number {ACCOUNT!r} reached the dump — this is the 2026-09-16 "
        f"leak, once per frame"
    )
    assert INVESTOR not in written, "investor id reached the dump"
    assert "ABC123" not in written, "custody code reached the dump"
    assert "<masked>" in written, "nothing was masked at all"


def __test_a_frame_without_identifiers_is_written_intact__():
    """CONTROL — masking must not be achieved by mangling everything.

    A pin that only asserts an absence passes on an implementation that writes
    empty lines. This one requires the payload to survive.
    """
    counts: Counter = Counter()
    sink = io.StringIO()
    probe.record_frame(_market_frame(3), counts, {}, (), sink)

    row = json.loads(sink.getvalue().strip())
    assert row["symbol"] == "41I1GA000"
    assert row["matchPrice"] == 1993.0
    assert "<masked>" not in sink.getvalue()


def __test_no_dump_handle_still_counts__():
    """Counting must not depend on dumping — the pre-#157 behaviour still works."""
    counts: Counter = Counter()
    samples: dict = {}
    probe.record_frame(_market_frame(1), counts, samples, (), None)
    assert sum(counts.values()) == 1
    assert samples, "the sample was not recorded"


@pytest.mark.parametrize("field", ["accountNo", "custodyCode", "investorId"])
def __test_the_pin_catches_a_write_before_the_mask__(field):
    """RED-FIRST, against the real hazard rather than an invented one.

    Simulates the ordering mistake — serialise and write, THEN mask — and asserts
    the identifier would have reached the file. If this ever stops holding, the
    masking assertion above has stopped discriminating and is decorative.
    """
    frame = _order_frame(1)
    raw_first = json.dumps(frame)          # what a write-before-mask would emit
    assert str(frame["order"][field]) in raw_first, (
        f"{field} is absent from the unmasked text, so the masking pin proves "
        f"nothing about it"
    )
