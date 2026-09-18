"""#157 conformance suite for the offline fake DNSE venue — WRITTEN BEFORE THE VENUE EXISTS.

Test-first by operator instruction (2026-09-18). Every test below pins a venue fact that was
MEASURED against the real venue and is cited in its docstring; none is invented. The suite is
therefore the acceptance criterion for the fake, written before the design is chosen, so it is
deliberately design-INDEPENDENT: whichever of S1 (standalone server), S2 (in-process transport
fake) or S3 (cassettes) the panel picks must satisfy exactly these behaviours. The API used here
is the contract the tests impose, not a design decision smuggled in early.

Facts are pinned against BEHAVIOUR, never against the fake's configuration (card #157 stage E).
A test that asserted "the fake was configured to reject X" would pass on a fake that never
rejects anything.

Sources, per fact:
- two books, string ids for conditionals vs integer ids for NORMAL: CLAUDE.md "DNSE has TWO order books"
- Activated spawns a NORMAL child carrying externalOrderId; the fill lands on the child: same, plus #39/#41
- Activated is TERMINAL for the conditional row: operator ruling 2026-09-18 (f88c7e43)
- OCO umbrella Activated from birth, umbrella cancel answers CO-ORD-013: CLAUDE.md, measured 2026-09-15
- partial fills by matched print volume, qty-1 derivatives never partial-fill: card #157 section 4, leader review item 15
- session-phase refusal codes: plugins/dnse/testing/live_test/README.md venue facts
- Activated shells persist on the STOP book for the rest of the day: #41, leader review item 14
- determinism (two replays byte-identical): card #157 stage C acceptance
"""
import pytest

# A PLAIN import, deliberately. This started as `pytest.importorskip`, which was wrong: once
# the module existed the guard was inert, but any future import error would have turned all
# twelve pins into twelve SKIPS, and a skipped pin reads as a passing suite. An import failure
# here must be LOUD (panel review 3/3, 2026-09-18).
from pynecore_dnse.venue_core import FakeVenue, VenueReject  # noqa: E402


# --------------------------------------------------------------------------- helpers

def _venue(**kwargs):
    """A venue positioned in the continuous session on the VN30 front-month contract."""
    kwargs.setdefault("phase", "continuous")
    kwargs.setdefault("symbol", "41I1G9000")
    kwargs.setdefault("market_type", "DERIVATIVE")
    kwargs.setdefault("last_price", 1980.0)
    return FakeVenue(**kwargs)


# --------------------------------------------------------------------------- the two books

def __test_normal_order_gets_an_integer_id_and_a_conditional_gets_a_string_id__():
    """CLAUDE.md: the NORMAL book issues integer ids (437346), the conditional book issues
    long string ids (da203hg6p09g1n1vipog). The id SHAPE is how every reader tells the books
    apart, so it is a behaviour, not a cosmetic detail."""
    venue = _venue()
    normal = venue.place(category="NORMAL", side="buy", qty=1, price=1975.0)
    stop = venue.place(category="STOP", side="buy", qty=1, stop_price=1990.0, price=1990.2)

    assert str(normal["id"]).isdigit(), f"NORMAL id must be integer-shaped, got {normal['id']!r}"
    assert not str(stop["id"]).isdigit(), f"conditional id must be string-shaped, got {stop['id']!r}"


# --------------------------------------------------------------------------- activation

def __test_a_resting_stop_does_not_activate_before_a_print_crosses_its_trigger__():
    """The discriminating half of the activation pin: without it, a fake that activated every
    conditional immediately would pass the next test and still be wrong."""
    venue = _venue()
    stop = venue.place(category="STOP", side="buy", qty=1, stop_price=1990.0, price=1990.2)

    venue.feed_print(price=1985.0, volume=10)   # below the trigger

    assert venue.order(stop["id"])["orderStatus"] == "New"
    assert venue.orders(book="NORMAL") == [], "no child may exist before the trigger is crossed"


def __test_a_print_through_the_trigger_activates_the_conditional_and_spawns_a_normal_child__():
    """CLAUDE.md + #39: on trigger the conditional becomes Activated (CLOSED, not filled) and the
    venue creates a NEW order on the NORMAL book; the activated conditional names its child in
    externalOrderId. Anything mapping venue records to Pine ids by the placed id alone goes blind
    here, which is exactly what #39 measured live."""
    venue = _venue()
    stop = venue.place(category="STOP", side="buy", qty=1, stop_price=1990.0, price=1990.2)

    venue.feed_print(price=1990.5, volume=10)   # through the trigger

    parent = venue.order(stop["id"])
    assert parent["orderStatus"] == "Activated"
    child_id = parent["externalOrderId"]
    assert child_id, "an activated conditional must name its NORMAL-book child"
    assert str(child_id).isdigit(), "the child lives on the NORMAL book, so its id is integer-shaped"
    assert venue.order(child_id)["orderStatus"] in ("New", "PartiallyFilled", "Filled")


def __test_an_activated_conditional_never_fills_and_stays_terminal__():
    """Operator ruling 2026-09-18: Activated is TERMINAL for the conditional row — it made its
    normal order and can do nothing else. The fill belongs to the child; the parent must never
    acquire a fill quantity however many prints arrive."""
    venue = _venue()
    stop = venue.place(category="STOP", side="buy", qty=1, stop_price=1990.0, price=1990.2)
    venue.feed_print(price=1990.5, volume=10)
    venue.feed_print(price=1991.0, volume=50)
    venue.feed_print(price=1992.0, volume=50)

    parent = venue.order(stop["id"])
    assert parent["orderStatus"] == "Activated", "Activated is terminal; it must not advance"
    assert float(parent.get("fillQuantity", 0)) == 0.0, "the parent conditional must never fill"


def __test_an_activated_shell_persists_on_the_stop_book_for_the_rest_of_the_day__():
    """#41: a triggered conditional stays Activated on the STOP book all day while its child does
    the work. venue.py status relies on still seeing it, so the list endpoint must keep returning
    it rather than dropping it once terminal."""
    venue = _venue()
    stop = venue.place(category="STOP", side="buy", qty=1, stop_price=1990.0, price=1990.2)
    venue.feed_print(price=1990.5, volume=10)
    for _ in range(50):
        venue.feed_print(price=1991.0, volume=5)

    ids = [o["id"] for o in venue.orders(book="STOP")]
    assert stop["id"] in ids, "the activated shell must remain listed on the conditional book"


# --------------------------------------------------------------------------- OCO

def __test_an_oco_umbrella_is_activated_from_birth_and_spawns_its_tp_child_at_placement__():
    """CLAUDE.md, measured 2026-09-15: an OCO umbrella is Activated FROM BIRTH — placement
    immediately spawns the normal-book TP child, with no trigger involved. So Activated by itself
    never means 'triggered'."""
    venue = _venue()
    oco = venue.place(category="OCO", side="sell", qty=1, price=1990.0, stop_price=1970.0)

    record = venue.order(oco["id"])
    assert record["orderStatus"] == "Activated", "the umbrella is Activated at the first read"
    assert record["externalOrderId"], "placement spawns the TP child immediately"


def __test_cancelling_an_oco_umbrella_is_refused_and_the_child_cancel_succeeds__():
    """CLAUDE.md: a cancel of the umbrella answers CO-ORD-013 'order status is not new' from
    second one; the CHILD id is what you cancel instead."""
    venue = _venue()
    oco = venue.place(category="OCO", side="sell", qty=1, price=1990.0, stop_price=1970.0)
    child_id = venue.order(oco["id"])["externalOrderId"]

    with pytest.raises(VenueReject) as excinfo:
        venue.cancel(oco["id"])
    assert excinfo.value.code == "CO-ORD-013"

    venue.cancel(child_id)
    assert venue.order(child_id)["orderStatus"] == "Canceled"


# --------------------------------------------------------------------------- fills

def __test_a_resting_limit_partially_fills_by_print_volume_then_completes__():
    """Card #157 section 4: partial fills are by MATCHED PRINT VOLUME against the resting
    quantity, not by the sandbox's fixed timer. A stock lot of 100 against a print of 30 leaves
    70 working."""
    venue = _venue(symbol="HPG", market_type="STOCK", last_price=26.5)
    order = venue.place(category="NORMAL", side="buy", qty=100, price=26.5)

    venue.feed_print(price=26.5, volume=30)
    mid = venue.order(order["id"])
    assert mid["orderStatus"] == "PartiallyFilled"
    assert float(mid["fillQuantity"]) == 30.0

    venue.feed_print(price=26.5, volume=70)
    done = venue.order(order["id"])
    assert done["orderStatus"] == "Filled"
    assert float(done["fillQuantity"]) == 100.0


def __test_a_quantity_one_derivative_never_partially_fills__():
    """Leader review item 15: qty-1 derivatives cannot partial-fill, so the partial path is only
    reachable through a stock series. A fake that emitted a partial tick for qty 1 would invent a
    state the venue cannot produce."""
    venue = _venue()
    order = venue.place(category="NORMAL", side="buy", qty=1, price=1980.0)

    venue.feed_print(price=1980.0, volume=1)

    assert venue.order(order["id"])["orderStatus"] == "Filled", "qty 1 goes straight to Filled"


# --------------------------------------------------------------------------- session phases

def __test_placing_after_the_close_is_refused_with_the_measured_code__():
    """live_test/README.md venue facts: post-close writes are refused. The code is part of the
    contract because the engine branches on it."""
    venue = _venue(phase="post_close")

    with pytest.raises(VenueReject) as excinfo:
        venue.place(category="NORMAL", side="buy", qty=1, price=1980.0)
    assert excinfo.value.code in ("CO-ORD-006", "CANNOT_PLACE_ORDER_IN_THE_CLOSED_SESSION")


def __test_cancelling_during_atc_is_refused_with_the_measured_code__():
    """live_test/README.md: ATC refuses cancels and fills whatever rests. This is the phase that
    has burned live runs, so the fake must reproduce the refusal rather than silently allowing it."""
    venue = _venue()
    order = venue.place(category="NORMAL", side="buy", qty=1, price=1975.0)
    venue.phase = "atc"

    with pytest.raises(VenueReject) as excinfo:
        venue.cancel(order["id"])
    assert excinfo.value.code == "CANNOT_CANCEL_THE_ORDER_IN_THE_ATC_SESSION"


# --------------------------------------------------------------------------- determinism

def __test_two_identical_replays_produce_identical_records__():
    """Card #157 stage C acceptance. A nondeterministic fake cannot pin anything: a pin that
    passes only sometimes is worse than no pin, because it is read as evidence. Ids must come
    from a seeded generator, not from a clock or a random source."""
    def _run():
        venue = _venue(seed=1234)
        stop = venue.place(category="STOP", side="buy", qty=1, stop_price=1990.0, price=1990.2)
        venue.feed_print(price=1990.5, volume=10)
        venue.feed_print(price=1991.0, volume=10)
        return venue.records()

    assert _run() == _run(), "two identical replays must produce identical venue records"
