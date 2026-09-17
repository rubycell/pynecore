#!/usr/bin/env python3
"""Mint a DNSE trading token into the state file the plugin reads.

The plugin is a pure CONSUMER of ``workdir/state/dnse_trading_token.json``; this is
the ONLY producer. A trading token is valid ~8h and self-invalidating (requesting a
new OTP kills the previous code), so run this once each trading morning — one daily
cron at 06:45 ICT covers the whole session: an 8h TTL from 06:45 lasts to 14:45,
the close. (The time has moved repeatedly — nothing in this tool depends on it.)

Modes
-----
* ``refresh_token.py``            auto (cron): send an email OTP, read the newest DNSE
                                  OTP from Gmail (IMAP), create the token, write it.
* ``refresh_token.py --otp CODE`` manual: you read the code yourself and pass it in
                                  (no Gmail creds needed — the reliable fallback).
* ``refresh_token.py --send``     just send the OTP email, then exit (read it, then
                                  re-run with ``--otp CODE``).

Auto mode needs a Gmail **app password** (not your login password). Put it in the
repo-root ``.env`` — ``DNSE_GMAIL_USER`` + ``DNSE_GMAIL_APP_PASSWORD`` (optionally
``DNSE_OTP_FROM``); this script auto-loads that file, so no manual sourcing is needed.
See ``.env.example`` for the template.

Security: the state file is order-placement authority. It is written ``0600`` under
``workdir/state/`` (gitignored) via an atomic temp+rename; the token is never printed.
"""
from __future__ import annotations

import argparse
import email
import email.utils
import imaplib
import json
import os
import re
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

# The minter reuses the plugin's version-pinned, TLS-verifying client for signing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # plugins/dnse
from pynecore_dnse.client import DNSEClient  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO / "workdir/config/plugins/dnse_broker.toml"
DEFAULT_STATE = REPO / "workdir/state/dnse_trading_token.json"
_SIX_DIGITS = re.compile(r"\b(\d{6})\b")


def load_credentials(config_path: Path) -> tuple[str, str]:
    if not config_path.exists():
        sys.exit(f"config not found: {config_path} (pass --config)")
    cfg = tomllib.loads(config_path.read_text())
    api_key, api_secret = cfg.get("api_key"), cfg.get("api_secret")
    if not api_key or not api_secret:
        sys.exit(f"{config_path} is missing api_key / api_secret")
    return api_key, api_secret


def write_token(state_path: Path, token: str) -> None:
    """Atomically write the state file the plugin reads (temp + os.replace, 0600)."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"trading_token": token,
                          "minted_at": int(time.time()),
                          "otp_type": "email_otp"})
    tmp = state_path.with_name(state_path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, payload.encode())
    finally:
        os.close(fd)
    os.replace(tmp, state_path)          # atomic on the same filesystem
    os.chmod(state_path, 0o600)


def send_otp(client: DNSEClient) -> float:
    """Send the email OTP; return the send timestamp (to reject older codes)."""
    sent_at = time.time()
    status, body = client.send_email_otp()
    if status not in (200, 201):
        sys.exit(f"send_email_otp failed: {status} {body}")
    return sent_at


def _message_text(message: email.message.Message) -> str:
    """Best-effort plaintext of an email (text/plain, else de-tagged text/html)."""
    parts = message.walk() if message.is_multipart() else [message]
    chunks = []
    for part in parts:
        if part.get_content_type() in ("text/plain", "text/html"):
            payload = part.get_payload(decode=True) or b""
            chunks.append(payload.decode(part.get_content_charset() or "utf-8", "replace"))
    text = "\n".join(chunks)
    return re.sub(r"<[^>]+>", " ", text)  # strip any HTML tags


def _extract_otp(text: str) -> str | None:
    """The 6-digit code, preferring one that follows an OTP/code/mã keyword."""
    keyed = re.search(r"(?:otp|code|m[aã]|passcode)[^0-9]{0,20}(\d{6})", text, re.I)
    if keyed:
        return keyed.group(1)
    loose = _SIX_DIGITS.search(text)
    return loose.group(1) if loose else None


def _load_dotenv(path: "Path | None" = None) -> None:
    """Load ``KEY=value`` lines from the repo-root ``.env`` into ``os.environ`` (without
    overriding already-set vars), so the Gmail creds work without a manual
    ``set -a && . .env``. Comment and malformed lines are skipped."""
    env_path = path or (REPO / ".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


#: DNSE's emailed OTP lives ~2 minutes. The poll must expire INSIDE that
#: window: polling longer can only return a code that is already dead, which
#: then fails at the venue with a confusing error instead of a clean
#: "expired, resend" here.
OTP_POLL_BUDGET_S = 110.0

#: How far BEFORE our OTP request a mail may be dated and still be accepted.
#:
#: This is CLOCK SKEW, not age: the ``Date`` header is DNSE's mail-server
#: clock while ``after_ts`` is ours, so the gap between them is whatever the
#: two clocks disagree by. Measured 2026-09-17 on a real mint: DNSE ran **+3 s
#: AHEAD** of this host — the safe side, but only 8 s from the old hard-coded
#: 5 s boundary, i.e. one NTP correction away from silently discarding a fresh
#: code and reporting "no OTP arrived" while it sat unread in the inbox.
#:
#: 60 s is bounded well under the ~120 s OTP TTL, so it can never reach back
#: to a genuinely previous code. The measurement says that margin is free:
#: the real predecessors in the same mailbox were 3.5 h and 19.5 h old, so
#: the gap that actually discriminates is HOURS, and a 5 s boundary bought
#: no protection while manufacturing a cliff.
CLOCK_SKEW_TOLERANCE_S = 60.0

#: A Gmail app password is 16 characters. Google DISPLAYS it as four groups of
#: four, so a pasted one arrives 19 characters long with three spaces — and
#: ``imaplib`` sends whatever it is given, verbatim.
_APP_PASSWORD_LEN = 16


def gmail_credentials() -> tuple[str, str]:
    """``(user, app_password)`` from the environment, whitespace stripped.

    Exits with a typed message when unusable. Never prints a value: the shape
    line reports lengths only, which is enough to tell an account password
    (not 16) from an app password and is the difference between a five-second
    diagnosis and a morning lost to a dead token.
    """
    _load_dotenv()  # pick up DNSE_GMAIL_* from the repo-root .env
    user = (os.environ.get("DNSE_GMAIL_USER") or "").strip()
    raw = os.environ.get("DNSE_GMAIL_APP_PASSWORD") or ""
    app_pw = "".join(raw.split())
    if not user or not app_pw:
        sys.exit("auto mode needs DNSE_GMAIL_USER + DNSE_GMAIL_APP_PASSWORD "
                 "(a Gmail APP password, not the account password). "
                 "Or use manual mode: --otp <code>.")
    if len(app_pw) != _APP_PASSWORD_LEN:
        # A warning, not a hard failure: the login itself is the real test,
        # and refusing here on a shape rule would block a working credential
        # Google decided to format differently.
        print(f"warning: DNSE_GMAIL_APP_PASSWORD is {len(app_pw)} characters "
              f"after stripping whitespace, not {_APP_PASSWORD_LEN}. Google "
              f"app passwords are {_APP_PASSWORD_LEN} characters; an account "
              f"password will NOT work — Gmail refuses plain-password IMAP on "
              f"accounts with 2-step verification.", file=sys.stderr)
    return user, app_pw


def _imap_login(user: str, app_pw: str) -> imaplib.IMAP4_SSL:
    """Connect and authenticate, mapping every failure to a typed reason.

    ``imaplib.IMAP4.error`` covers protocol errors ONLY. A DNS failure, a
    refused connection or a TLS timeout raises ``OSError`` / ``SSLError`` /
    ``TimeoutError`` and would otherwise escape as a traceback in the cron log —
    a stack trace where the log should have said why.
    """
    try:
        imap = imaplib.IMAP4_SSL("imap.gmail.com")
    except (OSError, imaplib.IMAP4.error) as error:
        sys.exit(f"Gmail unreachable ({type(error).__name__}): {error}")
    try:
        imap.login(user, app_pw)
    except imaplib.IMAP4.error as error:
        sys.exit(f"Gmail rejected the app password ({type(error).__name__}): "
                 f"{error}. Check DNSE_GMAIL_APP_PASSWORD is a 16-character "
                 f"APP password and that 2-step verification is ON.")
    except (OSError, imaplib.IMAP4.abort) as error:
        sys.exit(f"Gmail login failed ({type(error).__name__}): {error}")
    return imap


def preflight_gmail() -> tuple[str, str]:
    """Prove we can READ the mailbox BEFORE asking DNSE to send an OTP.

    Ordering matters and used to be wrong: ``main`` called ``send_otp`` first
    and the credential check lived inside the reader, so a misconfiguration
    burned a real code before anything was validated — and DNSE invalidates
    the previous OTP on each request, so that also poisoned the manual
    fallback the operator relies on at 08:20.
    """
    user, app_pw = gmail_credentials()
    imap = _imap_login(user, app_pw)
    try:
        imap.logout()
    except Exception:  # noqa: BLE001 — teardown of a probe connection
        pass
    return user, app_pw


def _message_sent_at(message: email.message.Message) -> float | None:
    """Epoch seconds from a ``Date`` header, or ``None`` when unusable.

    Two traps, both live:

    * ``-0000`` means "no timezone information" (RFC 5322), and
      ``parsedate_to_datetime`` returns a NAIVE datetime for it.
      ``datetime.timestamp()`` then reads that as LOCAL time, so on this
      ICT host a mail sent seconds ago computes as seven hours old and the
      freshness filter silently discards it — the mint then reports "no OTP
      arrived" while the OTP sits in the inbox. Measured: DNSE currently
      sends ``+0000``, so this is reachable rather than firing today; the
      default output of Python's own ``email.utils.formatdate`` is
      ``-0000``, so a sender change would reintroduce it silently.
    * a MISSING or malformed header makes ``parsedate_to_datetime`` RAISE
      ``ValueError`` — it does not return ``None`` — so the old
      ``if date and ...`` guard was dead code and an undated mail aborted
      the whole mint.
    """
    raw = message.get("Date", "")
    try:
        parsed = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _sender_matches(message: email.message.Message, sender: str) -> bool:
    """Re-check the sender HERE, client-side.

    The IMAP ``(FROM "x")`` criterion is evaluated by the SERVER and is a
    substring match, so it is both untestable locally and looser than it
    looks. Re-checking in our own code makes the rule ours: it can be pinned,
    and it cannot be widened by a server-side quirk.
    """
    from_header = (message.get("From", "") or "").lower()
    token = sender.lower()
    domain = from_header.rsplit("@", 1)[-1].strip().strip(">").strip()
    if not domain:
        return False
    # A LABEL match, not a substring: the server's ``(FROM "dnse")`` criterion
    # would happily accept ``dnsefake.com``, and so would ``token in header``.
    # Requiring the token to be a whole dot-delimited label rejects that.
    #
    # LIMIT, stated rather than implied: this does NOT reject a suffix attack
    # such as ``dnse.com.vn.evil.com``, because the configured value is a
    # 4-character token rather than a domain, so there is nothing to anchor
    # the end of the name against. Closing that needs DNSE_OTP_FROM to become
    # a full domain, which is a config change, not a code one.
    return token in domain.split(".")


def select_otp(messages, after_ts: float, sender: str) -> str | None:
    """PURE: pick the OTP from already-fetched messages, newest FIRST.

    Takes parsed messages rather than an IMAP connection so the DECISION can
    be pinned without a fake IMAP server. That is deliberate: a hand-written
    imaplib fake pins its own imitation — ``search`` returns one
    space-separated blob, not a list, and a fake that gets that wrong makes
    the code fetch a single OLDEST message while the test passes green.

    A message whose ``Date`` is unusable is NOT skipped: IMAP ids are
    assigned in arrival order, so the caller's newest-first ordering is a
    better recency signal than a header the sender may have mangled, and
    skipping would turn a cosmetic header fault into a total failure.
    """
    for message in messages:
        if not _sender_matches(message, sender):
            continue
        sent_at = _message_sent_at(message)
        if sent_at is not None and sent_at < after_ts - CLOCK_SKEW_TOLERANCE_S:
            continue  # older than our request — a previous OTP, already dead
        code = _extract_otp(_message_text(message))
        if code:
            return code
    return None


def read_otp_from_gmail(after_ts: float, *,
                        timeout: float = OTP_POLL_BUDGET_S,
                        poll: int = 10,
                        credentials: "tuple[str, str] | None" = None) -> str:
    """Poll Gmail for the newest DNSE OTP that arrived AFTER ``after_ts``."""
    user, app_pw = credentials if credentials else gmail_credentials()
    sender = os.environ.get("DNSE_OTP_FROM", "dnse")
    deadline = time.time() + timeout
    while time.time() < deadline:
        imap = _imap_login(user, app_pw)
        try:
            imap.select("INBOX")
            _typ, data = imap.search(None, f'(FROM "{sender}")')
            messages = []
            for msg_id in reversed((data[0] or b"").split()[-20:]):  # newest first
                _typ, raw = imap.fetch(msg_id, "(RFC822)")
                if raw and raw[0]:
                    messages.append(email.message_from_bytes(raw[0][1]))
            code = select_otp(messages, after_ts, sender)
            if code:
                return code
        except (imaplib.IMAP4.error, OSError) as error:
            sys.exit(f"Gmail IMAP error ({type(error).__name__}): {error}")
        finally:
            try:
                imap.logout()
            except Exception:  # noqa: BLE001
                pass
        time.sleep(poll)
    sys.exit(f"no DNSE OTP arrived within {timeout:.0f}s (the code expires in "
             f"about 2 minutes, so a longer wait could only return a dead "
             f"one). Retry, or use manual mode: --otp <code>.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mint a DNSE trading token.")
    parser.add_argument("--otp", metavar="CODE", help="OTP you read yourself (manual mode)")
    parser.add_argument("--send", action="store_true", help="send the OTP email, then exit")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    args = parser.parse_args(argv)

    if args.otp and args.send:
        parser.error("--send and --otp are mutually exclusive")

    client = DNSEClient(*load_credentials(args.config))

    if args.send:  # send-only leg
        send_otp(client)
        print("OTP email sent — read it, then re-run: refresh_token.py --otp <code>")
        return 0

    if args.otp:  # manual leg (OTP already delivered)
        code = args.otp
    else:          # auto leg
        # Prove the mailbox is readable BEFORE asking DNSE for a code: each
        # request invalidates the previous OTP, so failing after send_otp
        # burns a real code and poisons the manual fallback too.
        gmail = preflight_gmail()
        print("Gmail reachable; requesting the OTP…")
        sent_at = send_otp(client)
        print("OTP email sent; reading the newest DNSE OTP from Gmail…")
        code = read_otp_from_gmail(sent_at, credentials=gmail)

    status, body = client.create_trading_token("email_otp", code)
    if status not in (200, 201) or not isinstance(body, dict) or not body.get("tradingToken"):
        sys.exit(f"create_trading_token failed: {status} {body}")
    write_token(args.state, body["tradingToken"])
    now = datetime.now(timezone.utc).astimezone()
    print(f"✓ token minted {now:%Y-%m-%d %H:%M %Z} -> {args.state} "
          f"(valid ~8h; the plugin picks it up on its next read)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
