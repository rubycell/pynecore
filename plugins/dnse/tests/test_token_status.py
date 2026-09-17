"""Tests for the DNSE token-status tool — pure logic, no live network."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import token_status as ts  # noqa: E402


def __test_read_state_missing_valid_and_malformed__(tmp_path):
    assert ts.read_state(tmp_path / "nope.json") is None
    good = tmp_path / "s.json"
    good.write_text(json.dumps({"trading_token": "T", "minted_at": 1}))
    assert ts.read_state(good)["trading_token"] == "T"
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert ts.read_state(bad) is None


def __test_resolve_account__(fake_client):
    assert ts.resolve_account(
        fake_client(get_accounts=(200, {"accounts": [{"id": "0001234567"}]}))) == "0001234567"
    assert ts.resolve_account(fake_client(get_accounts=(200, {"accounts": []}))) is None
    assert ts.resolve_account(fake_client(get_accounts=(500, {}))) is None


@pytest.mark.parametrize("reply, accepted", [
    ((400, {"code": "INVALID_TRADING_TOKEN"}), False),                       # token rejected
    ((401, {}), False),                                                      # auth failure
    ((404, {"code": "RESOURCE_NOT_FOUND"}), True),                           # accepted, id not found
    ((400, {"code": "CANNOT_CANCEL_THE_ORDER_IN_THE_ATO_SESSION"}), True),   # accepted, session
    ((0, {}), False),                                                        # could not reach DNSE
])
def __test_token_is_live__(fake_client, reply, accepted):
    fake = fake_client(cancel_order=reply)
    live, _why = ts.token_is_live(fake, "ACC", "tok")
    assert live is accepted, f"{reply} -> live should be {accepted}"


# --- #133: the cron log must reach the VERDICT, not sit beside it ---

def _log_with(tmp_path, *lines):
    state = tmp_path / "dnse_trading_token.json"
    (tmp_path / "refresh_token.log").write_text("\n".join(lines) + "\n")
    return state


def __test_133_cron_log_states_are_distinguished__(tmp_path):
    """absent / stale / ran must stay THREE answers, not two.

    "No log at all" means the schedule was probably never installed — this
    card's origin, and invisible to code review because a crontab lives
    outside the repo. "A log with nothing from today" means it is installed
    and did not run, or ran and failed. Collapsing them is what made the old
    "fresh cron: NO" wording read like a failure when the truth was an
    absence.
    """
    from datetime import date
    state = tmp_path / "dnse_trading_token.json"
    today = date(2026, 9, 17)

    assert ts.show_cron_log(state, today) == "absent"

    _log_with(tmp_path, "2026-09-16 08:00:03 GOOD token minted")
    assert ts.show_cron_log(state, today) == "stale"

    _log_with(tmp_path, "2026-09-16 08:00:03 GOOD", "2026-09-17 08:00:02 GOOD")
    assert ts.show_cron_log(state, today) == "ran"


def __test_133_a_good_token_with_a_dead_schedule_never_renders_unqualified__():
    """The whole point of the card, pinned.

    A GOOD verdict printed next to "the cron never ran" is how two mornings
    started with a dead token: the line was true, and nothing made it
    impossible to read past.
    """
    ran = ts.verdict_text(True, "ran")
    assert ran == "GOOD — the plugin can place orders", (
        "a healthy automated morning must stay clean — noise here is what "
        "trains an operator to stop reading the line"
    )

    for state, expected_hint in (("absent", "not installed"),
                                 ("stale", "did not run today")):
        text = ts.verdict_text(True, state)
        assert text.startswith("GOOD"), "the token IS usable — say so first"
        assert "BUT" in text and expected_hint in text, (
            f"a GOOD token with cron_state={state!r} must be QUALIFIED: {text}"
        )
        assert "not produced by the automation" in text

    assert ts.verdict_text(False, "ran") == "NOT GOOD — refresh needed", (
        "a bad token stays bad regardless of the schedule"
    )
