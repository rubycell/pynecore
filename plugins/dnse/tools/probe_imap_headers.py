#!/usr/bin/env python3
"""#133 read-only IMAP headers probe — consumes nothing, reveals nothing.

Answers three questions that no amount of code reading can settle, WITHOUT
minting a token or burning an OTP:

1. Does the stored app password authenticate at all (after whitespace is
   stripped — Google displays app passwords as four groups of four, so a
   pasted one is 19 characters and ``imap.login`` would reject it verbatim)?
2. What timezone spelling does the OTP mail's ``Date`` header carry?
   ``parsedate_to_datetime`` returns a NAIVE datetime for ``-0000``, and
   ``.timestamp()`` then reads it as LOCAL time — on an ICT host that makes a
   fresh mail look 7 hours old, so ``refresh_token.read_otp_from_gmail``
   silently discards it and reports "no OTP arrived" while it sits in the
   inbox. This probe prints BOTH interpretations side by side so the skew is
   visible rather than inferred.
3. Does the forwarded mail land in INBOX at all? A Gmail filter with
   "skip the inbox" would make ``select("INBOX")`` see nothing — invisible to
   code reading, and indistinguishable at 08:00 from "the OTP never arrived".

SAFETY — this reads the operator's mailbox, so it is deliberately blind:
  * it NEVER prints a credential, an address, a subject line or a body;
  * it prints the From DOMAIN only, never the local part;
  * the subject and body are reduced to a BOOLEAN "contains a 6-digit run",
    because a DNSE OTP mail carries the code in exactly those places;
  * it opens mailboxes READ-ONLY (``select(..., readonly=True)``) so nothing
    is marked as read and no state changes.

Usage:  .venv/bin/python plugins/dnse/tools/probe_imap_headers.py
"""
from __future__ import annotations

import email
import email.utils
import imaplib
import os
import re
import sys
import time
from datetime import timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SIX_DIGITS = re.compile(r"\b\d{6}\b")
FOLDERS = ("INBOX", "[Gmail]/All Mail", "[Gmail]/Spam")
MAX_SHOWN = 5


def load_env_value(name: str) -> str:
    """Read one value from the repo-root .env. Never returns it to a log."""
    env_path = REPO_ROOT / ".env"
    if name in os.environ:
        return os.environ[name]
    if not env_path.exists():
        return ""
    for raw in env_path.read_text().splitlines():
        if raw.startswith(f"{name}="):
            return raw.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def describe_date_header(raw_date: str) -> str:
    """Report the header's tz SPELLING and the skew it causes — no content."""
    if not raw_date:
        return "Date: MISSING (parsedate_to_datetime raises ValueError here)"
    offset = raw_date.strip()[-5:]
    try:
        parsed = email.utils.parsedate_to_datetime(raw_date)
    except Exception as exc:  # noqa: BLE001 — the probe reports, never crashes
        return f"Date: offset={offset!r} UNPARSEABLE ({type(exc).__name__})"
    if parsed.tzinfo is None:
        as_local = parsed.timestamp()
        as_utc = parsed.replace(tzinfo=timezone.utc).timestamp()
        skew_h = (as_utc - as_local) / 3600.0
        return (f"Date: offset={offset!r} -> tzinfo=None (NAIVE)  "
                f"skew if read as local: {skew_h:+.1f}h  <-- THE BUG")
    return f"Date: offset={offset!r} -> tzinfo={parsed.tzinfo} (aware, correct)"


def main() -> int:
    user = load_env_value("DNSE_GMAIL_USER")
    raw_pw = load_env_value("DNSE_GMAIL_APP_PASSWORD")
    sender = load_env_value("DNSE_OTP_FROM") or "dnse"
    app_pw = "".join(raw_pw.split())

    print("=== credential SHAPE (no values) ===")
    print(f"  DNSE_GMAIL_USER          : {'set' if user else 'MISSING'}"
          f"{f', {len(user)} chars' if user else ''}")
    print(f"  DNSE_GMAIL_APP_PASSWORD  : {'set' if raw_pw else 'MISSING'}, "
          f"{len(raw_pw)} chars raw -> {len(app_pw)} stripped "
          f"({'16 = app-password shape' if len(app_pw) == 16 else 'NOT 16 — Google app passwords are 16'})")
    print(f"  DNSE_OTP_FROM            : {'set' if sender else 'MISSING'}, "
          f"{len(sender)} chars")
    if not user or not app_pw:
        print("\nFAIL: credentials incomplete — nothing probed.")
        return 2

    print("\n=== IMAP login (read-only session) ===")
    try:
        imap = imaplib.IMAP4_SSL("imap.gmail.com")
    except (OSError, imaplib.IMAP4.error) as exc:
        print(f"  FAIL connecting: {type(exc).__name__}: {exc}")
        return 2
    try:
        try:
            imap.login(user, app_pw)
        except imaplib.IMAP4.error as exc:
            print(f"  FAIL login: {type(exc).__name__}: {exc}")
            print("  (if this says AUTHENTICATIONFAILED, the app password is "
                  "wrong or 2-step verification is off)")
            return 1
        print("  OK — the stripped app password authenticates")

        for folder in FOLDERS:
            print(f"\n=== {folder} ===")
            try:
                typ, _ = imap.select(f'"{folder}"', readonly=True)
            except imaplib.IMAP4.error as exc:
                print(f"  cannot select: {exc}")
                continue
            if typ != "OK":
                print("  cannot select (not OK)")
                continue
            typ, data = imap.search(None, f'(FROM "{sender}")')
            ids = (data[0] or b"").split() if data else []
            print(f"  mails matching FROM {sender!r}: {len(ids)}")
            if not ids:
                continue
            for msg_id in reversed(ids[-MAX_SHOWN:]):
                typ, raw = imap.fetch(
                    msg_id, "(BODY.PEEK[HEADER.FIELDS (DATE FROM SUBJECT)])")
                if not raw or not raw[0]:
                    continue
                msg = email.message_from_bytes(raw[0][1])
                from_hdr = msg.get("From", "")
                domain = from_hdr.rsplit("@", 1)[-1].strip(">").strip() or "?"
                subject_has_code = bool(SIX_DIGITS.search(msg.get("Subject", "")))
                print(f"   - from domain: {domain:28} "
                      f"subject has 6-digit run: {subject_has_code}")
                print(f"     {describe_date_header(msg.get('Date', ''))}")
    finally:
        try:
            imap.logout()
        except Exception:  # noqa: BLE001
            pass

    print("\nDone — nothing was consumed, marked read, or modified.")
    print(f"(local clock offset now: {time.strftime('%z')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
