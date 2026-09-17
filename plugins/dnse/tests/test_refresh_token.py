"""Tests for the DNSE trading-token minter — no live network, no real Gmail, tmp FS only."""
import json
import os
import sys
from pathlib import Path

import pytest

# tools/ is not a package — put it on the path to import the minter.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import refresh_token as rt  # noqa: E402


def _config(tmp_path):
    cfg = tmp_path / "dnse.toml"
    cfg.write_text('api_key = "k"\napi_secret = "s"\n')
    return cfg


class _FakeClient:
    """Stand-in for DNSEClient: the OTP endpoints, no network."""
    def __init__(self, *a, **k):
        self.sent = 0

    def send_email_otp(self):
        self.sent += 1
        return (200, {})

    def create_trading_token(self, otp_type, passcode):
        assert otp_type == "email_otp", "the minter must request an email OTP"
        return (200, {"tradingToken": f"tok-{passcode}"})


def __test_write_token_is_atomic_and_private__(tmp_path):
    state = tmp_path / "state" / "dnse_trading_token.json"
    rt.write_token(state, "TKN-123")

    data = json.loads(state.read_text())
    assert data["trading_token"] == "TKN-123", "plugin reads the 'trading_token' key"
    assert isinstance(data["minted_at"], int)
    assert oct(os.stat(state).st_mode & 0o777) == "0o600", "token file must be 0600"
    assert not state.with_name(state.name + ".tmp").exists(), "no temp file left behind"


@pytest.mark.parametrize("text, want", [
    ("Mã OTP của bạn là 123456", "123456"),
    ("Your OTP code: 654321. Do not share.", "654321"),
    ("passcode 246810 expires soon", "246810"),
    ("order 111222 total 999999", "111222"),   # loose fallback: first 6-digit
    ("no six digit code here 12345", None),     # 5 digits -> no match
])
def __test_extract_otp__(text, want):
    assert rt._extract_otp(text) == want


def __test_message_text_strips_html__():
    import email
    msg = email.message_from_string(
        "Content-Type: text/html\n\n<p>Your code is <b>135790</b></p>")
    assert rt._extract_otp(rt._message_text(msg)) == "135790"


def __test_manual_mode_writes_token_without_sending__(tmp_path, monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(rt, "DNSEClient", lambda *a, **k: fake)
    state = tmp_path / "dnse_trading_token.json"

    rc = rt.main(["--otp", "424242", "--config", str(_config(tmp_path)), "--state", str(state)])

    assert rc == 0
    assert json.loads(state.read_text())["trading_token"] == "tok-424242"
    assert fake.sent == 0, "manual --otp must NOT send a fresh OTP (that would invalidate it)"


def __test_auto_mode_sends_then_scrapes_gmail__(tmp_path, monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(rt, "DNSEClient", lambda *a, **k: fake)
    monkeypatch.setattr(rt, "read_otp_from_gmail", lambda after_ts, **k: "777888")
    state = tmp_path / "s.json"

    rc = rt.main(["--config", str(_config(tmp_path)), "--state", str(state)])

    assert rc == 0
    assert fake.sent == 1, "auto mode sends the OTP before scraping"
    assert json.loads(state.read_text())["trading_token"] == "tok-777888"


def __test_send_and_otp_are_mutually_exclusive__(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "DNSEClient", lambda *a, **k: _FakeClient())
    with pytest.raises(SystemExit):
        rt.main(["--send", "--otp", "1", "--config", str(_config(tmp_path))])


def __test_load_dotenv_sets_unset_keys_only__(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# a comment\nDNSE_GMAIL_USER=me@x.com\n'
                   'DNSE_GMAIL_APP_PASSWORD="already-set-in-file"\nMALFORMED LINE\n')
    monkeypatch.delenv("DNSE_GMAIL_USER", raising=False)
    monkeypatch.setenv("DNSE_GMAIL_APP_PASSWORD", "from-shell")

    rt._load_dotenv(env)

    assert os.environ["DNSE_GMAIL_USER"] == "me@x.com", "unset key filled from .env"
    assert os.environ["DNSE_GMAIL_APP_PASSWORD"] == "from-shell", \
        "an already-exported var must win over .env (setdefault, not override)"


# --- #133: the OTP selection rule, pinned WITHOUT a fake IMAP server ---
#
# `select_otp` is pure — it takes already-fetched messages — so these pins
# exercise the real decision logic directly. That is deliberate: a
# hand-written imaplib fake pins its own imitation (`search` returns one
# space-separated blob, not a list; a fake that gets it wrong makes the code
# fetch a single OLDEST message while the test passes green), which is the
# #84 lesson — a control that shares an assumption with the code under test
# pins nothing.
#
# Fixtures are shaped like the headers measured on a real mint (2026-09-17):
# From a `mail.dnse.com.vn` domain, Date timezone-AWARE (+0000).

import email.message  # noqa: E402
import email.utils  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402


def _dnse_mail(code: str | None, sent_at: float | None,
               sender: str = "no-reply@mail.dnse.com.vn"):
    """A message shaped like a real DNSE OTP mail."""
    message = email.message.EmailMessage()
    message["From"] = f"DNSE <{sender}>"
    message["Subject"] = "Ma xac thuc giao dich"
    if sent_at is not None:
        message["Date"] = email.utils.format_datetime(
            datetime.fromtimestamp(sent_at, timezone.utc))
    message.set_content(
        f"Ma OTP cua ban la {code}" if code else "Thong bao tai khoan")
    return message


def __test_133_a_mail_dated_before_the_request_by_clock_skew_is_accepted__():
    """30 s "before" our request is CLOCK SKEW, not an old code.

    The Date header is DNSE's mail-server clock and `after_ts` is ours, so a
    small negative gap means the clocks disagree — not that the mail predates
    the request. Measured on a real mint, DNSE ran +3 s AHEAD, i.e. 8 s from
    the old hard-coded 5 s boundary. Discarding here is the silent failure
    this pin exists to prevent: the poll runs to its budget and reports "no
    OTP arrived" while the code sits unread in the inbox.
    """
    import refresh_token as rt

    request_at = time.time()
    mail = _dnse_mail("123456", request_at - 30)
    assert rt.select_otp([mail], request_at, "dnse") == "123456"


def __test_133_a_genuinely_previous_otp_is_still_discarded__():
    """The control for the pin above: hours-old codes must NOT be accepted.

    Widening the tolerance is only safe because the gap that actually
    discriminates is HOURS — measured, the real predecessors in the mailbox
    were 3.5 h and 19.5 h old. This pins that the widening did not reach
    back far enough to pick one up.
    """
    import refresh_token as rt

    request_at = time.time()
    stale = _dnse_mail("111111", request_at - 3.5 * 3600)
    assert rt.select_otp([stale], request_at, "dnse") is None


def __test_133_newest_first_ordering_decides__():
    """The FIRST acceptable message wins, so the caller's order is the rule."""
    import refresh_token as rt

    request_at = time.time()
    newest = _dnse_mail("222222", request_at + 1)
    older = _dnse_mail("333333", request_at + 0.5)
    assert rt.select_otp([newest, older], request_at, "dnse") == "222222"


def __test_133_a_lookalike_sender_domain_is_rejected__():
    """`dnsefake.com` must NOT satisfy a search for `dnse`.

    The IMAP `(FROM "dnse")` criterion is a SERVER-side substring match, and
    so was the original client-side re-check — both would accept this. The
    re-check now requires the token to be a whole dot-delimited LABEL of the
    sender domain, which is strictly stronger than what the server does.

    KNOWN LIMIT, pinned as such below: a SUFFIX attack is still accepted,
    because the configured value is a 4-character token rather than a domain,
    so there is nothing to anchor the end of the name against.
    """
    import refresh_token as rt

    request_at = time.time()
    lookalike = _dnse_mail("444444", request_at + 1,
                           sender="no-reply@dnsefake.com")
    assert rt.select_otp([lookalike], request_at, "dnse") is None

    genuine = _dnse_mail("555555", request_at + 1,
                         sender="no-reply@mail.dnse.com.vn")
    assert rt.select_otp([genuine], request_at, "dnse") == "555555"

    # The limit, stated honestly rather than left to be discovered: closing
    # this needs DNSE_OTP_FROM to become a full domain (a config change).
    suffix_attack = _dnse_mail("666666", request_at + 1,
                               sender="no-reply@dnse.com.vn.evil.test")
    assert rt.select_otp([suffix_attack], request_at, "dnse") == "666666"


def __test_133_an_undated_mail_is_not_skipped__():
    """A missing Date must not cost us the OTP — nor crash the mint.

    `parsedate_to_datetime` RAISES on a missing header rather than returning
    None, so the original `if date and ...` guard was dead code and an
    undated mail aborted the whole mint. IMAP ids are assigned in arrival
    order, so the caller's newest-first ordering is a better recency signal
    than a header the sender may have mangled.
    """
    import refresh_token as rt

    request_at = time.time()
    undated = _dnse_mail("777777", None)
    assert undated.get("Date") is None
    assert rt.select_otp([undated], request_at, "dnse") == "777777"
