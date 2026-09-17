#!/usr/bin/env python3
"""Check the DNSE trading-token status, and refresh it by hand if it isn't good.

Meant to be run manually at ~08:05 (just after the 08:00 cron) to answer one question:
**did the job leave us with a token that actually works?** It shows:

  * mint time + age vs the 8h TTL, and whether it was minted after 08:00 today,
  * the tail of the cron log (what the morning job actually did),
  * a LIVE liveness probe — a harmless cancel of a bogus order id. DNSE checks the
    trading-token header before it looks the order up, so an ``INVALID_TRADING_TOKEN``
    reply means the token is bad, while ANY other reply (not-found, session-closed, …)
    means the token was accepted, i.e. good.

If the token is missing / stale / rejected (or you pass ``--refresh``), it walks you
through a manual mint: it sends the email OTP, you read it and type the code, it writes
the new token to the file the plugin reads.

    .venv/bin/python plugins/dnse/tools/token_status.py [--refresh]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tools/ (for refresh_token)
import refresh_token as rt  # noqa: E402
from pynecore_dnse import errors  # noqa: E402
from pynecore_dnse.client import DNSEClient  # noqa: E402

ICT = timezone(timedelta(hours=7))
TTL_HOURS = 8
_PROBE_ORDER_ID = "TOKENCHECK0000"  # a bogus id — a cancel of it never touches a real order


def read_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None


def resolve_account(client: DNSEClient) -> str | None:
    status, body = client.get_accounts()
    if status == 200 and isinstance(body, dict):
        accounts = body.get("accounts") or []
        if accounts:
            return accounts[0].get("id")
    return None


def token_is_live(client: DNSEClient, account: str, token: str) -> tuple[bool, str]:
    """Probe the token with a harmless cancel of a bogus order id.

    Returns ``(accepted, reason)``. Only ``INVALID_TRADING_TOKEN`` / auth means the
    token is bad; every other reply means DNSE accepted it.
    """
    status, body = client.cancel_order(account, _PROBE_ORDER_ID, "DERIVATIVE",
                                       token, order_category="STOP")
    if status == 0:
        return False, f"could not reach DNSE ({errors.code_of(body) or 'network error'})"
    classified = errors.classify(status, body, is_write=True)
    if classified and classified.disposition is errors.Disposition.AUTH:
        # #68: AUTH here is the credential layer (key/secret/clock), NOT the
        # trading token — say so, or a dead secret sends the operator minting.
        return False, f"DNSE rejected the CREDENTIALS, not the token ({classified.code})"
    if classified and classified.disposition is errors.Disposition.AUTH_TOKEN:
        return False, f"DNSE rejected the token ({classified.code})"
    return True, f"accepted (probe -> {errors.code_of(body) or ('http ' + str(status))})"


def verdict_text(good: bool, cron_state: str) -> str:
    """The VERDICT line, as a value a test can read.

    Extracted so the qualification is pinned rather than asserted in a
    comment: a GOOD token whose schedule never ran must NEVER render
    unqualified, because that parenthetical-beside-GOOD is exactly how a
    dead cron stayed invisible for two mornings.
    """
    if not good:
        return "NOT GOOD — refresh needed"
    if cron_state == "ran":
        return "GOOD — the plugin can place orders"
    detail = ("no cron log exists — the schedule is probably not installed"
              if cron_state == "absent"
              else "the schedule did not run today")
    return (f"GOOD — the plugin can place orders, BUT {detail}. "
            f"This token was not produced by the automation.")


def show_cron_log(state_path: Path, today=None) -> str:
    """Print the cron log tail and RETURN what it says about today.

    Returns ``"ran"`` / ``"stale"`` / ``"absent"``.

    It returns a value rather than only printing because it used to return
    ``None`` and never touch the verdict — so "the scheduled refresh never
    ran" rendered as a parenthetical NEXT TO a ``GOOD`` verdict and exit 0.
    That is this card's own defect one layer in: a check reporting truthfully
    while the caller reads a different object as the answer. #133 found it at
    the callers (they pipe the exit status into ``tail``); this one was
    inside the tool.

    ``absent`` and ``stale`` stay DISTINCT on purpose. No log file at all
    means the schedule was probably never installed — this card's origin
    story, and invisible to code review because a crontab lives outside the
    repo. A log with nothing from today means it is installed and did not
    run, or ran and failed. Different actions; collapsing them is what made
    the old "fresh cron: NO" wording read like a failure when the truth was
    an absence.
    """
    today = today or datetime.now(ICT).date()
    log = state_path.parent / "refresh_token.log"
    print(f"\ncron log ({log}):")
    if not log.exists():
        print("  (NO LOG FILE — the scheduled refresh has never written here; "
              "it is probably not installed at all)")
        return "absent"
    lines = log.read_text().splitlines()
    for line in lines[-6:]:
        print(f"  {line}")
    stamp = today.isoformat()
    if any(stamp in line for line in lines):
        return "ran"
    print(f"  (NOTHING DATED {stamp} — the schedule did not run today, or "
          f"ran without logging)")
    return "stale"


def interactive_refresh(client: DNSEClient, state_path: Path) -> bool:
    """Send an OTP, prompt for the code, mint + write. Returns True on success."""
    if not sys.stdin.isatty():
        print("  (not a terminal — run this yourself to refresh interactively)")
        return False
    print("\nSending an email OTP to your DNSE account…")
    rt.send_otp(client)
    code = input("Enter the OTP code from your email (blank to abort): ").strip()
    if not code:
        print("aborted.")
        return False
    status, body = client.create_trading_token("email_otp", code)
    if status not in (200, 201) or not isinstance(body, dict) or not body.get("tradingToken"):
        print(f"✗ mint failed: {status} {body}")
        return False
    rt.write_token(state_path, body["tradingToken"])
    print(f"✓ new token written to {state_path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Check + manually refresh the DNSE trading token.")
    parser.add_argument("--config", type=Path, default=rt.DEFAULT_CONFIG)
    parser.add_argument("--state", type=Path, default=rt.DEFAULT_STATE)
    parser.add_argument("--refresh", action="store_true", help="refresh even if the token looks good")
    parser.add_argument("--require-cron", action="store_true",
                        help="also fail (exit 1) when the scheduled refresh "
                             "did not run today, even if the token is GOOD")
    args = parser.parse_args()

    client = DNSEClient(*rt.load_credentials(args.config))
    now = datetime.now(ICT)
    print(f"=== DNSE token status @ {now:%Y-%m-%d %H:%M %Z} ===")

    state = read_state(args.state)
    good = False
    if not state or not state.get("trading_token"):
        print(f"token file:  MISSING or empty  ({args.state})")
    else:
        minted_at = state.get("minted_at")
        # A corrupt/hand-edited state file must read as NOT GOOD, never crash:
        # a traceback here gives the caller no verdict line at all (measured
        # 2026-08-19 with a string minted_at).
        if isinstance(minted_at, str):
            try:
                minted_at = float(minted_at)
            except ValueError:
                print(f"minted:      (UNREADABLE — minted_at={minted_at!r})")
                minted_at = None
        if not isinstance(minted_at, (int, float)):
            minted_at = None
        if minted_at:
            minted = datetime.fromtimestamp(minted_at, ICT)
            age_h = (time.time() - minted_at) / 3600
            within_ttl = age_h < TTL_HOURS
            # NOT "fresh cron". This measures the TOKEN's mint time and knows
            # nothing about whether a schedule ran — the old label claimed the
            # latter while computing the former, so it printed "fresh cron:
            # yes" three lines above "NO LOG FILE — the scheduled refresh has
            # never written here". Two lines of the same output contradicting
            # each other is how an operator learns to stop reading both.
            # Whether the schedule ran is show_cron_log's answer, and it now
            # reaches the verdict.
            minted_today = minted.date() == now.date() and minted.hour >= 8
            print(f"minted:      {minted:%Y-%m-%d %H:%M %Z}  (age {age_h:.1f}h; "
                  f"TTL {TTL_HOURS}h -> {'within' if within_ttl else 'EXPIRED'})")
            print(f"minted today: {'yes — today, after 08:00' if minted_today else 'NO — not minted after 08:00 today'}"
                  f"  (token age only; whether the SCHEDULE ran is the cron log below)")
        else:
            within_ttl = False
            print("minted:      (unknown — file has no minted_at)")

        account = resolve_account(client)
        if not account:
            good = within_ttl
            print("liveness:    (could not resolve account to probe — using the TTL heuristic)")
        else:
            live, why = token_is_live(client, account, state["trading_token"])
            good = live
            print(f"liveness:    {'GOOD — ' if live else 'BAD — '}{why}")

    cron_state = show_cron_log(args.state)
    # The verdict answers "can the plugin place orders?", so a token that IS
    # good keeps exit 0 even when the schedule is broken: a manual mint is a
    # legitimate way to arrive here, and failing a working morning would train
    # the operator to ignore the exit code — the same cry-wolf failure the
    # 07:55-vs-`hour >= 8` check already had. But it must never render
    # UNQUALIFIED, because a dead schedule is what this card exists to surface.
    print(f"\nVERDICT: {verdict_text(good, cron_state)}")
    if args.require_cron and cron_state != "ran":
        # Opt-in strict mode for an automated caller that wants the SCHEDULE
        # verified, not merely the token. Off by default so a human running
        # this ad hoc is never blocked by it.
        print("  (--require-cron: the schedule did not run today -> exit 1)")
        return 1

    if args.refresh or not good:
        if not sys.stdin.isatty():
            return 0 if good else 1
        default_yes = args.refresh or not good
        answer = input(f"\nRefresh the token now? [{'Y/n' if default_yes else 'y/N'}] ").strip().lower()
        if answer == "y" or (default_yes and answer == ""):
            interactive_refresh(client, args.state)
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
