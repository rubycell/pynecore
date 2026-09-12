"""#73/#115 INVARIANT — get_position MUST return the whole-ACCOUNT net, unfiltered by run.

The engine's per-strategy isolation (#73) rests on a division of labour:
- the RUN-OWNED position is reconstructed from the run's OWN journal fills
  (``sync_engine._durable_owned_signed_size``), while
- ``get_position`` is the ACCOUNT net — every run's exposure on this
  account+symbol folded into one number (the venue nets; it carries no per-run
  handle).

``#73/C2`` (``sync_engine.py`` ~4633) uses that account net as an ABSENCE-PROOF:
``None`` (net==0) means *nobody* holds anything, so the periodic external-flatten
wipe is safe to fire. If a future change "helpfully" filtered ``get_position``
down to *our* share — the exact instinct that looks right for a multi-strategy
plugin and is WRONG here — the absence-proof would read our-own-flat as
account-flat and either miss a real external close or fire a destructive clear
against another run's live position.

These pin the contract so that a filter-by-run reimplementation goes RED:
test 1 (returns < the full net) and test 2 (returns non-None at account-flat).
"""
import asyncio

from pynecore_dnse import broker


def _broker(fake_client, tmp_path):
    config = broker.DNSEBrokerConfig(
        api_key="k", api_secret="s", account_no="ACC001", trading_token="tok",
        token_file=str(tmp_path / "missing.json"))
    instance = broker.DNSEBroker(symbol="VN30F1M", timeframe="15", config=config)
    instance._client = fake_client()  # instruments unstubbed -> resolve_contract returns the alias
    return instance


def _with_positions(instance, fake_client, positions_body):
    instance._client = fake_client(get_positions=(200, positions_body))
    return instance


def __test_get_position_reports_the_full_account_net_not_a_per_run_share__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path)
    wanted = b.resolve_contract("VN30F1M")
    # Account net = 5 long — larger than any single run would hold. A filter-by-run
    # "fix" returning our share (<5) FAILS this assertion.
    body = {"total": 1, "positions": [
        {"symbol": wanted, "side": "NB", "status": "OPEN", "openQuantity": 5, "costPrice": 1300}]}
    _with_positions(b, fake_client, body)
    pos = asyncio.run(b.get_position("VN30F1M"))
    assert pos is not None and pos.side == "long"
    assert pos.size == 5.0, \
        "get_position must report the WHOLE account net (#73/C2 absence-proof), not a per-run share"


def __test_get_position_is_None_at_account_flat_the_absence_proof__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path)
    wanted = b.resolve_contract("VN30F1M")
    # Only a CLOSED row -> account net 0 -> None. #73/C2 fires the external-flatten
    # wipe on exactly this None; a non-None "our-flat" object would break detection.
    body = {"total": 1, "positions": [
        {"symbol": wanted, "side": "NB", "status": "CLOSED", "openQuantity": 0, "costPrice": 1300}]}
    _with_positions(b, fake_client, body)
    pos = asyncio.run(b.get_position("VN30F1M"))
    assert pos is None, "net==0 must return None — the engine's external-flatten absence-proof (#73/C2)"


def __test_get_position_sums_only_non_closed_rows_for_the_symbol__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path)
    wanted = b.resolve_contract("VN30F1M")
    # OPEN 3 (counts) + CLOSED 9 (history, excluded) + a different symbol (excluded) -> net 3.
    body = {"total": 3, "positions": [
        {"symbol": wanted, "side": "NB", "status": "OPEN", "openQuantity": 3, "costPrice": 1300},
        {"symbol": wanted, "side": "NB", "status": "CLOSED", "openQuantity": 9, "costPrice": 1200},
        {"symbol": "OTHER", "side": "NB", "status": "OPEN", "openQuantity": 7, "costPrice": 50}]}
    _with_positions(b, fake_client, body)
    pos = asyncio.run(b.get_position("VN30F1M"))
    assert pos is not None and pos.size == 3.0
