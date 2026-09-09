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
