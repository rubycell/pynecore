"""#91 — the fill-tier flatten tool must never leave protection armed.

Measured live 2026-09-08 13:51-13:52 (F6 re-grade, operator-caught): after
the old close-only flatten, the position's protective buy-stop stayed
ARMED on a FLAT account until the still-running engine's next sync
cancelled it (~25 s). Engine down = armed until the 14:45 DAY expiry; a
spike to trigger OPENS a position on a flat account (the #82 class).

Pins the panel-adjudicated contract of ``plugins/dnse/tools/flatten.py``:
close-FIRST (sequence-pinned), sweep only after a confirmed flat, owned
orders only (shared account — foreign untouched), attribution from a
read-only store query that NEVER creates the store file, unresolved
dispositions degrade loudly to exit 1, unavailable attribution to exit 2.
"""
import asyncio
import importlib.util
import pathlib
import sqlite3

import pynecore.lib as lib

lib.bar_index = 0

from pynecore_dnse import broker

_SECDEF_ROW = [{"ceilingPrice": "2100", "floorPrice": "1800", "securityGroupId": "FU"}]
_LOAN_OK = (200, {"loanPackages": [{"id": 42}]})

_TOOL_PATH = (pathlib.Path(__file__).resolve().parents[1]
              / "tools" / "flatten.py")

# WARNING — MUTATION TESTING THIS TOOL. It is loaded BY PATH, so it gets a
# ``plugins/dnse/tools/__pycache__/flatten.cpython-*.pyc``. A mutant run writes
# that cache, and restoring the source afterwards does NOT invalidate it
# reliably: the tests then execute the MUTANT while the file on disk is correct
# (hit 2026-09-17 during the sign fix — a correct fix looked broken).
# ``inspect.getsource()`` CANNOT detect it: getsource reads the .py while the
# code object comes from the .pyc, so the check that feels authoritative is the
# one that is blind to this. Verify BEHAVIOURALLY — call the function and
# assert on its RETURN VALUE — and run mutants with PYTHONDONTWRITEBYTECODE=1
# or move that __pycache__ to backup/deleteable/ before the restore run.
spec = importlib.util.spec_from_file_location("dnse_flatten_tool", _TOOL_PATH)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def _broker(fake_client, tmp_path, **client_responses):
    responses = {"get_security_definition": (200, _SECDEF_ROW),
                 "get_loan_packages": _LOAN_OK}
    responses.update(client_responses)
    config = broker.DNSEBrokerConfig(
        api_key="k", api_secret="s", account_no="ACC001", trading_token="tok-A",
        token_file=str(tmp_path / "missing_token.json"))
    instance = broker.DNSEBroker(symbol="VN30F1M", timeframe="1", config=config)
    instance._client = fake_client(**responses)
    instance._cancel_verify_attempts = 1
    instance._cancel_verify_delay = 0.0
    return instance


def _short_position_book(state):
    """Canned venue: short 1 while ``state['pos']``; a marketable close
    flips it flat; one protective conditional rests until cancelled."""
    def _positions(*_a, **_k):
        if state["pos"]:
            return (200, {"positions": [{"symbol": "VN30F1M", "side": "NS",
                                         "openQuantity": 1,
                                         "costPrice": 1964.6}]})
        return (200, {"positions": []})

    def _post(*_a, **_k):
        state["pos"] = 0
        return (201, {"id": "900001", "symbol": "VN30F1M", "side": "NB",
                      "quantity": 1, "orderStatus": "Filled",
                      "fillQuantity": 1})

    def _orders(_acct, _mkt, order_category=None, **_k):
        if state.get("swept") or order_category == "NORMAL":
            return (200, {"orders": [], "totalPages": 1})
        return (200, {"orders": [{"id": "prot-cond-1", "symbol": "VN30F1M",
                                  "side": "NB", "quantity": 1,
                                  "fillQuantity": 0, "orderStatus": "New",
                                  "stopPrice": 1966.5}], "totalPages": 1})

    def _cancel(_acct, oid, _mkt, _tok, order_category=None):
        if str(oid) == "prot-cond-1":
            state["swept"] = True
            return (200, {"orderStatus": "Canceled"})
        return (400, {"code": "ORDER_NOT_FOUND"})

    def _detail(_acct, oid, _mkt, order_category=None):
        status = "Canceled" if state.get("swept") else "New"
        return (200, {"id": oid, "symbol": "VN30F1M", "side": "NB",
                      "quantity": 1, "fillQuantity": 0,
                      "orderStatus": status, "stopPrice": 1966.5})

    return dict(get_positions=_positions, post_order=_post,
                get_orders=_orders, cancel_order=_cancel,
                get_order_detail=_detail)


def __test_flatten_closes_first_then_sweeps_own_protection__(
        fake_client, tmp_path, monkeypatch):
    """The measured live gap + the panel's ordering ruling in one pin:
    (a) the protection IS swept (old tool: zero cancel_order calls — the
    armed-stop-on-flat-account gap); (b) the close is placed BEFORE any
    cancel reaches the venue (catches a protection-first impl: a rejected
    close there = unbounded naked open)."""
    monkeypatch.setattr(tool.time, "sleep", lambda _s: None)
    state = {"pos": 1}
    b = _broker(fake_client, tmp_path, **_short_position_book(state))

    rc = tool.flatten(b, "VN30F1M", {"prot-cond-1"})

    assert rc == 0, f"flatten reported failure (rc={rc})"
    writes = [(c[0], c[1][1] if c[0] == "cancel_order" else None)
              for c in b._client.calls
              if c[0] in ("post_order", "cancel_order")]
    assert writes and writes[0][0] == "post_order", (
        f"first venue WRITE must be the close (close-first, #91 panel "
        f"unanimous); sequence was {writes}")
    assert ("cancel_order", "prot-cond-1") in writes, (
        "the position's protective conditional was left ARMED after the "
        "flatten — naked entry-stop on a flat account (#91, measured live)")


def __test_unconfirmed_close_withholds_the_sweep__(fake_client, tmp_path,
                                                   monkeypatch):
    """Panel rider: the sweep runs ONLY after a confirmed flat. A close
    that never confirms -> exit 1 with ZERO cancels (catches sweep-anyway,
    which cancels protection over a possibly-open position)."""
    monkeypatch.setattr(tool.time, "sleep", lambda _s: None)
    clock = {"t": 0.0}
    monkeypatch.setattr(tool.time, "time", lambda: clock.__setitem__(
        "t", clock["t"] + 5.0) or clock["t"])
    state = {"pos": 1}
    responses = _short_position_book(state)

    def _post_no_fill(*_a, **_k):        # close accepted but never fills
        return (201, {"id": "900001", "symbol": "VN30F1M", "side": "NB",
                      "quantity": 1, "orderStatus": "New",
                      "fillQuantity": 0})
    responses["post_order"] = _post_no_fill

    b = _broker(fake_client, tmp_path, **responses)
    rc = tool.flatten(b, "VN30F1M", {"prot-cond-1"}, close_wait_s=1)

    assert rc == 1
    assert b._client.count("cancel_order") == 0, (
        "the sweep ran on an UNCONFIRMED close — protection cancelled "
        "over a possibly-open position (#91 panel rider)")


def __test_flatten_never_cancels_foreign_orders__(fake_client, tmp_path,
                                                  monkeypatch):
    """SHARED account hard rule: a working order outside the owned set is
    the operator's — reported, never cancelled (catches any blanket
    sweep)."""
    monkeypatch.setattr(tool.time, "sleep", lambda _s: None)
    b = _broker(fake_client, tmp_path,
                get_positions=(200, {"positions": []}),
                get_orders=(200, {"orders": [
                    {"id": "operator-42", "symbol": "VN30F1M", "side": "NB",
                     "quantity": 3, "fillQuantity": 0,
                     "orderStatus": "New"}], "totalPages": 1}))

    rc = tool.flatten(b, "VN30F1M", set())

    assert rc == 0, "already-flat with no owned orders must succeed"
    assert b._client.count("cancel_order") == 0, (
        "the flatten sweep cancelled an order it does not own — the "
        "netting account is SHARED (#91 hard rule)")


def __test_attribution_unavailable_is_exit_2_never_a_clean_pass__(
        fake_client, tmp_path, monkeypatch):
    """Seat 1's vacuous-success guard: owned_ids=None means 'could not
    prove ownership', which must be exit 2 — never 0 (catches treating
    unavailable attribution as an empty owned set)."""
    monkeypatch.setattr(tool.time, "sleep", lambda _s: None)
    b = _broker(fake_client, tmp_path,
                get_positions=(200, {"positions": []}))

    rc = tool.flatten(b, "VN30F1M", None)

    assert rc == 2, (
        f"attribution UNAVAILABLE returned rc={rc} — 'ownership never "
        f"loaded' must never read as 'no owned orders' (#91 seat 1)")
    assert b._client.count("cancel_order") == 0


def __test_unresolved_disposition_degrades_loudly_to_exit_1__(
        fake_client, tmp_path, monkeypatch):
    """#51 shape: the sweep cancel is refused and the readback stays
    working -> UNKNOWN -> exit 1, exactly ONE cancel write (catches
    retry-hammering, #58, and catches swallowing the unresolved id)."""
    monkeypatch.setattr(tool.time, "sleep", lambda _s: None)
    state = {"pos": 0}
    responses = _short_position_book(state)
    responses["get_positions"] = (200, {"positions": []})
    responses["cancel_order"] = (400, {"code": "INVALID_TRADING_TOKEN"})
    responses["get_order_detail"] = (200, {
        "id": "prot-cond-1", "symbol": "VN30F1M", "side": "NB",
        "quantity": 1, "fillQuantity": 0, "orderStatus": "New"})

    b = _broker(fake_client, tmp_path, **responses)
    rc = tool.flatten(b, "VN30F1M", {"prot-cond-1"})

    assert rc == 1, "an UNKNOWN disposition must surface as failure"
    # An unhinted id probes each book ONCE (#45: NORMAL/STOP/OCO) — three
    # writes total, but never a RETRY of a refused book (#58).
    books = [c[2].get("order_category") for c in b._client.calls
             if c[0] == "cancel_order"]
    assert len(books) == len(set(books)) <= 3, (
        f"a refused sweep write was retried — #58: one refusal per book, "
        f"one write; got {books}")


def __test_owned_live_ids_scoped_contract__(tmp_path):
    """#96 attribution contract: (a) missing store -> None, file NEVER
    created; (b) unknown account -> None (never an empty 'clean' set —
    the #91 vacuous-pass guard); (c) account-scoped; (d) digit (NORMAL-
    class) ids need TODAY activity on MAX(created,updated) — a reopened
    row (old created, fresh updated) STAYS owned (#77/T16 pattern), a
    stale prior-day digit id drops; (e) string (conditional) ids are
    day-unscoped; (f) still NO symbol filter (#77 wire-symbol lesson)."""
    import time
    missing = tmp_path / "nope" / "broker.sqlite"
    assert tool.owned_live_ids(missing, "ACC001") is None
    assert not missing.exists(), (
        "owned_live_ids CREATED the store file — absence now reads as a "
        "clean account (#91 seat 2 guard)")

    db = tmp_path / "broker.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE orders (exchange_order_id TEXT, symbol TEXT,"
                 " closed_ts_ms INTEGER, created_ts_ms INTEGER,"
                 " updated_ts_ms INTEGER, run_instance_id INTEGER)")
    conn.execute("CREATE TABLE runs (run_instance_id INTEGER,"
                 " account_id TEXT, plugin_name TEXT)")
    old_ms = int((time.time() - 26 * 3600) * 1000)
    now_ms = int(time.time() * 1000)
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)", [
        ("115586", "41I1G9000", None, old_ms, old_ms, 1),    # stale digit -> OUT
        ("222222", "41I1G9000", None, old_ms, now_ms, 1),    # reopened -> OWNED
        ("333333", "41I1G9000", None, now_ms, now_ms, 1),    # today -> OWNED
        ("daf-old-cond", "41I1G9000", None, old_ms, old_ms, 1),  # string -> OWNED
        ("444444", "HPG", None, now_ms, now_ms, 1),          # other WIRE sym -> OWNED
        ("999999", "41I1G9000", None, now_ms, now_ms, 9),    # OTHER account -> OUT
        ("closed1", "41I1G9000", 123, now_ms, now_ms, 1),    # closed -> OUT
    ])
    conn.executemany("INSERT INTO runs VALUES (?,?,?)", [
        (1, "ACC001", "DNSE Broker"), (9, "OTHER", "Binance Broker")])
    conn.commit(); conn.close()

    assert tool.owned_live_ids(db, "ACC001") == {
        "222222", "333333", "daf-old-cond", "444444"}
    assert tool.owned_live_ids(db, "NO-SUCH-ACCOUNT") is None, (
        "an unmatched account key must be UNAVAILABLE (exit 2), never a "
        "clean empty set (#96 seat 1 guard)")


def __test_stale_prior_day_row_must_not_claim_a_foreign_order__(
        fake_client, tmp_path, monkeypatch):
    """RED (#96, 2026-09-09 review finding 2): DNSE REUSES NORMAL ids
    across days (measured: 09-08 issued LOWER ids than 09-07) and the
    store holds multiple account identities. A stale un-closed journal
    row from a PRIOR day whose id the venue reissued TODAY to a FOREIGN
    order makes the unscoped attribution claim it — and the sweep cancels
    the operator's order on a shared netting account. Contract: a row
    from another day (or another account's run) must not contribute to
    the owned set the sweep acts on."""
    import sqlite3, time
    monkeypatch.setattr(tool.time, "sleep", lambda _s: None)

    db = tmp_path / "broker.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE orders (exchange_order_id TEXT, symbol TEXT,"
                 " closed_ts_ms INTEGER, created_ts_ms INTEGER,"
                 " updated_ts_ms INTEGER, run_instance_id INTEGER)")
    conn.execute("CREATE TABLE runs (run_instance_id INTEGER,"
                 " account_id TEXT, plugin_name TEXT)")
    yesterday_ms = int((time.time() - 26 * 3600) * 1000)
    today_ms = int(time.time() * 1000)
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)", [
        ("115586", "41I1G9000", None, yesterday_ms, yesterday_ms, 1),  # STALE
        ("fresh-1", "41I1G9000", None, today_ms, today_ms, 2),         # today
    ])
    conn.executemany("INSERT INTO runs VALUES (?,?,?)", [
        (1, "ACC001", "dnse_broker"), (2, "ACC001", "dnse_broker")])
    conn.commit(); conn.close()

    owned = tool.owned_live_ids(db, "ACC001")
    assert owned is not None

    # TODAY the venue reissued id 115586 to the OPERATOR's order:
    b = _broker(fake_client, tmp_path,
                get_positions=(200, {"positions": []}),
                get_orders=(200, {"orders": [
                    {"id": "115586", "symbol": "VN30F1M", "side": "NB",
                     "quantity": 3, "fillQuantity": 0,
                     "orderStatus": "New"}], "totalPages": 1}))
    rc = tool.flatten(b, "VN30F1M", owned)

    cancels = [c for c in b._client.calls if c[0] == "cancel_order"]
    assert not cancels, (
        "the sweep cancelled the OPERATOR's order via a stale prior-day "
        "id collision (#96) — attribution must be scoped by day/account")
    assert rc == 0


# === two agreeing reads before acting on the SIGN (live incident 09-16) =====

def __test_disagreeing_position_reads_refuse_to_act__(fake_client, tmp_path,
                                                      monkeypatch):
    """THE red-first pin for the 2026-09-16 incident.

    A single read decides the SIGN, and the sign decides BUY vs SELL. Get it
    wrong and the "flatten" DOUBLES the position: with the account really SHORT
    1, one read answered `long 1.0`, the tool sold 1, and the account went to
    SHORT 2. The read was never cached — `get_position` hits the venue each
    time — the VENUE served a stale view (the same non-monotonic read CLAUDE.md
    records for order details, 08-17).

    So disagreement must be fail-closed: exit 2 (could-not-determine) and NO
    order. Never split the difference, never prefer the newer read — we have no
    evidence which of the two is stale.
    """
    monkeypatch.setattr(tool.time, "sleep", lambda _s: None)
    reads = iter([1.0, -1.0])       # long 1, then short 1: irreconcilable
    monkeypatch.setattr(tool, "_read_position_size",
                        lambda *_a, **_k: next(reads, -1.0))
    b = _broker(fake_client, tmp_path)

    rc = tool.flatten(b, "VN30F1M", set())

    assert rc == 2, f"disagreeing reads must be could-not-determine, got {rc}"
    writes = [c[0] for c in b._client.calls
              if c[0] in ("post_order", "cancel_order")]
    assert writes == [], (
        f"NOTHING may be sent when the sign is unproven — got {writes}. "
        "Acting on the wrong sign doubles the position (live 2026-09-16)")


def __test_agreeing_reads_still_flatten__(fake_client, tmp_path, monkeypatch):
    """The over-block guard: the confirmation must not freeze the normal path.

    Two agreeing reads -> the close is still placed. Without this, a fix that
    simply refused everything would pass the test above and look correct.
    """
    monkeypatch.setattr(tool.time, "sleep", lambda _s: None)
    state = {"pos": 1}
    b = _broker(fake_client, tmp_path, **_short_position_book(state))

    rc = tool.flatten(b, "VN30F1M", {"prot-cond-1"})

    assert rc == 0, f"agreeing reads must proceed normally, got rc={rc}"
    assert any(c[0] == "post_order" for c in b._client.calls), \
        "the close must still be placed when both reads agree"


def __test_short_position_is_closed_by_BUYING_not_selling__(
        fake_client, tmp_path, monkeypatch):
    """#135: flattening a SHORT must send a BUY. It sent a SELL.

    `ExchangePosition.size` is a MAGNITUDE — the sign lives in `.side`
    (broker.py builds it as `size=abs(net)`, `side="long" if net > 0 else
    "short"`), and the engine re-derives a signed size from `.side` at three
    separate places. The tool did not: it read `float(pos.size)` and then
    branched on `size > 0`, which is TRUE for every non-flat position. So the
    `else "buy"` arm was unreachable on this venue and a real SHORT 1 was
    SOLD into — short 1 -> short 2, printed as "long 1.0", reported rc=0.

    This is the mechanism of the 2026-09-16 incident. It survived because
    every live test entry so far has been LONG, and because the existing
    short-fixture tests assert the SEQUENCE (close, then sweep) and the exit
    code, never the SIDE of the order actually sent.

    The two-agreeing-reads guard cannot catch it: both reads return +1.0 and
    agree with each other. That guard is for replica LAG and stays.
    """
    state = {"pos": 1}
    responses = _short_position_book(state)
    sent: "list[dict]" = []
    original_post = responses["post_order"]

    def _capturing_post(*a, **k):
        # payload is POSITIONAL arg 2:
        # post_order(account, market_type, payload, token, order_category=...)
        sent.append(a[2])
        return original_post(*a, **k)

    responses["post_order"] = _capturing_post
    b = _broker(fake_client, tmp_path, **responses)
    monkeypatch.setattr(tool.time, "sleep", lambda *_a: None)

    rc = tool.flatten(b, "VN30F1M", set())

    assert sent, "no order was sent at all"
    assert sent[0]["side"] == "NB", (
        f"flattening a SHORT sent side={sent[0]['side']!r} (NS=sell) — that "
        f"DOUBLES the position instead of closing it; a short is closed by "
        f"BUYING (NB)"
    )
    assert rc == 0


def __test_position_reader_returns_a_SIGNED_size__(fake_client, tmp_path):
    """The reader's contract is a SIGNED net; a short must read negative.

    Its docstring already promised "Signed net size" while it returned the
    unsigned magnitude — the docstring was right and the code was wrong.
    Pinned separately from the order-side pin so a future refactor that moves
    the sign derivation elsewhere still has to keep this contract.
    """
    state = {"pos": 1}
    b = _broker(fake_client, tmp_path, **_short_position_book(state))

    size = tool._read_position_size(b, "VN30F1M")

    assert size is not None, "the read failed; this pin needs a live read"
    assert size < 0, (
        f"a SHORT position read as {size} — the sign was dropped, and the "
        f"sign is what decides whether the flatten buys or sells"
    )


def __test_long_position_is_still_closed_by_SELLING__(
        fake_client, tmp_path, monkeypatch):
    """Control for the sign fix: a LONG must still be closed by SELLING.

    Without this, "flattening a short must buy" is satisfied by an
    implementation that simply inverted the branch — which would then double
    every LONG instead. The two pins together fix the mapping in both
    directions, and this is the case every live test so far has exercised,
    which is precisely why the short bug went unseen.
    """
    state = {"pos": 1}
    responses = _short_position_book(state)

    def _long_positions(*_a, **_k):
        if state["pos"]:
            return (200, {"positions": [{"symbol": "VN30F1M", "side": "NB",
                                         "openQuantity": 1,
                                         "costPrice": 1964.6}]})
        return (200, {"positions": []})

    sent: "list[dict]" = []
    original_post = responses["post_order"]

    def _capturing_post(*a, **k):
        sent.append(a[2])
        return original_post(*a, **k)

    responses["get_positions"] = _long_positions
    responses["post_order"] = _capturing_post
    b = _broker(fake_client, tmp_path, **responses)
    monkeypatch.setattr(tool.time, "sleep", lambda *_a: None)

    assert tool._read_position_size(b, "VN30F1M") > 0, (
        "a LONG must read positive"
    )
    rc = tool.flatten(b, "VN30F1M", set())

    assert sent and sent[0]["side"] == "NS", (
        f"flattening a LONG sent side={sent[0]['side'] if sent else None!r} "
        f"— a long is closed by SELLING (NS)"
    )
    assert rc == 0


def __test_close_quantity_matches_the_position_size__(
        fake_client, tmp_path, monkeypatch):
    """The wire QUANTITY must be the real size, not just the right side.

    Every fixture in this file held exactly 1 contract — the one magnitude
    where a correct size and a bare sign are the same number. A reader that
    returned only the SIGN (+1/-1) therefore passed every pin in this file
    while, against a real SHORT 3, buying 1: the position is left short 2,
    the tool never reads flat, and the protection sweep is withheld.

    So this fixture holds THREE, and the assertion is on what went out on
    the wire.
    """
    state = {"pos": 1}
    responses = _short_position_book(state)

    def _short_three(*_a, **_k):
        if state["pos"]:
            return (200, {"positions": [{"symbol": "VN30F1M", "side": "NS",
                                         "openQuantity": 3,
                                         "costPrice": 1964.6}]})
        return (200, {"positions": []})

    sent: "list[dict]" = []
    original_post = responses["post_order"]

    def _capturing_post(*a, **k):
        sent.append(a[2])
        return original_post(*a, **k)

    responses["get_positions"] = _short_three
    responses["post_order"] = _capturing_post
    b = _broker(fake_client, tmp_path, **responses)
    monkeypatch.setattr(tool.time, "sleep", lambda *_a: None)

    assert tool._read_position_size(b, "VN30F1M") == -3.0, (
        "the reader must carry the MAGNITUDE, not just the sign"
    )
    tool.flatten(b, "VN30F1M", set())

    assert sent, "no order was sent"
    assert sent[0]["side"] == "NB", "a short is closed by buying"
    assert float(sent[0]["quantity"]) == 3.0, (
        f"closed a SHORT 3 with quantity {sent[0]['quantity']!r} — a partial "
        f"close leaves the rest of the position open while the tool reports "
        f"it handled the flatten"
    )


def __test_unrecognised_side_label_refuses_to_guess_the_sign__(
        fake_client, tmp_path):
    """An unknown `.side` is could-not-determine, never a guessed sign.

    Mirrors the engine, which falls through to a halt rather than assume.
    Without this pin the whole refuse-to-guess branch has ZERO coverage:
    an implementation that silently treats an unrecognised label as SHORT
    passes every other test in this file, and would then BUY against an
    unknown-side position.
    """
    b = _broker(fake_client, tmp_path, **_short_position_book({"pos": 1}))

    class _OddPosition:
        size = 1.0
        side = "sideways"

    async def _odd(_symbol):
        return _OddPosition()

    b.get_position = _odd

    assert tool._read_position_size(b, "VN30F1M") is None, (
        "an unrecognised side label was given a sign instead of being "
        "reported as could-not-determine"
    )


def __test_one_stale_flat_read_must_not_authorise_the_sweep__(
        fake_client, tmp_path, monkeypatch):
    """The post-close FLAT verdict needs two agreeing reads, like the sign.

    The verification loop polls ~12 times over the close window and breaks on
    the FIRST flat observation — so it actively samples FOR a stale empty
    page. That verdict authorises cancelling the protective conditionals, so
    a single lagging response cancelled protection over a still-open position
    and returned 0. `size` and `emptiness` are the same rule.
    """
    state = {"pos": 1, "reads": 0}
    responses = _short_position_book(state)

    def _one_stale_blip(*_a, **_k):
        state["reads"] += 1
        # One lagging EMPTY page mid-poll; the position is still really short.
        if state["pos"] and state["reads"] == 3:
            return (200, {"positions": []})
        if state["pos"]:
            return (200, {"positions": [{"symbol": "VN30F1M", "side": "NS",
                                         "openQuantity": 1,
                                         "costPrice": 1964.6}]})
        return (200, {"positions": []})

    def _close_never_fills(*_a, **_k):
        return (201, {"id": "900001", "symbol": "VN30F1M", "side": "NB",
                      "quantity": 1, "orderStatus": "New", "fillQuantity": 0})

    responses["get_positions"] = _one_stale_blip
    responses["post_order"] = _close_never_fills
    b = _broker(fake_client, tmp_path, **responses)
    monkeypatch.setattr(tool.time, "sleep", lambda *_a: None)

    cancels_before = len(getattr(b._client, "calls", []))
    rc = tool.flatten(b, "VN30F1M", {"prot-cond-1"}, close_wait_s=6)

    assert rc != 0, (
        "one stale EMPTY read declared the account FLAT and returned success "
        "while the close had not filled — the protection sweep that follows "
        "cancels the conditionals over a still-open position"
    )
    assert cancels_before is not None
