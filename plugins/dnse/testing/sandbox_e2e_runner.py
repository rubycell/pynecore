#!/usr/bin/env python3
"""Sandbox Replay E2E runner (#114) — drive a .pine strategy through the real engine + sandbox.

Reads a ``.pine`` (the source of truth for WHAT is exercised), verifies it is
SANDBOX-COMPATIBLE, transpiles it with the local pine2pyne, runs the Sandbox Replay E2E
(replay bars -> real strategy -> engine -> sandbox orders), and grades the run log against
the milestones the ``.pine`` declares in a ``// @sandbox-expect:`` header line.

WHAT THE SANDBOX PROVES (and this runner grades): the ORDER PATH — the order reaches the
venue, ``CREATED -> FILLED`` on the WS, the engine processes the fill (position update), and
a protective exit / close is placed and fills. It does NOT prove SL/TP triggering, matching,
netting, or P&L (no price sim; positions accumulate, never net). So ``// @sandbox-expect:``
milestones are ORDER-PATH events, never a final position or P&L figure.

Known SANDBOX-BOUNDARY artifacts (NOT engine bugs — the runner names them so a run is not
misread):
  * position oscillates 1->0->-1->0 (the sandbox does not net; repeated closes flip the net)
  * cancel-retry loop on a filled order ("already in terminal state Filled") — message-blind
    classification (#116); the sandbox lacks the read-back endpoints #55 needs to resolve it
  * ``executions read http=404 -> VWAP`` — the sandbox has no executions endpoint

The ``// @sandbox-expect:`` vocabulary (space-separated tokens on one comment line):
    entry.placed  entry.filled  exit.placed  close.placed  close.filled  position>=N

Usage:
    # full run + grade (needs network to the sandbox):
    python sandbox_e2e_runner.py --pine plugins/dnse/testing/live_test/l2_fill_flatten.pine
    # grade an existing (already-captured) log without re-running:
    python sandbox_e2e_runner.py --pine <p.pine> --grade-only path/to/run.log

The L0 gate does NOT apply (this is the mock sandbox, not prod). OTP is the public constant
666666, so a stale token is re-minted automatically. Secrets move file-to-file — never echoed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PINE2PYNE = Path("/home/mike/workspace/github/pine2pyne")
SANDBOX_CONFIG = REPO / "workdir" / "config" / "plugins" / "dnse_replay_sandbox.toml"
DEFAULT_FIXTURE = REPO / "plugins" / "dnse" / "testing" / "replay_fixtures" / "vn30f1m_15_smoke.json"
TOKEN_TTL_S = 8 * 3600          # DNSE trading-token TTL (re-mint past this)

#: milestone token -> regex over the STRIPPED [BROKER] log. Order-path only.
_MILESTONE_PATTERNS = {
    "entry.placed": r"dispatch(?:ed|ing) ENTRY",
    "entry.filled": r"(?:FILLED\b.*leg=entry|after filled pine='?E)",
    "exit.placed":  r"dispatch(?:ed|ing) EXIT",
    "close.placed": r"dispatch(?:ed|ing) CLOSE",
    "close.filled": r"FILLED\b.*leg=close",
}
#: things that look like feed bugs but are the sandbox boundary — reported, never failed.
_SANDBOX_ARTIFACTS = {
    "oscillation": (r"position size=-", "position oscillation (sandbox does not net)"),
    "cancel-loop": (r"already in terminal state Filled", "cancel-retry loop (#116, message-blind)"),
    "vwap-404":    (r"executions read http=404", "no executions endpoint -> VWAP booking"),
}
#: conditional-order idioms the sandbox REJECTS (orderCategory STOP/OCO) — a compat gate.
_INCOMPAT = re.compile(r"strategy\.(entry|exit|order)\b[^\n]*\b(stop|oca_)", re.IGNORECASE)


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def read_pine(pine: Path) -> tuple[list[str], list[str]]:
    """Return (expected-milestone tokens, sandbox-compat warnings) from the .pine."""
    text = pine.read_text()
    expect: list[str] = []
    for line in text.splitlines():
        m = re.search(r"//\s*@sandbox-expect:\s*(.+)$", line)
        if m:
            expect = m.group(1).split()
            break
    warnings: list[str] = []
    if _INCOMPAT.search(text):
        warnings.append("uses a STOP/OCA idiom — the sandbox REJECTS conditional orders; "
                        "those legs cannot be exercised here (use market/limit).")
    if not expect:
        warnings.append("no `// @sandbox-expect:` line — will grade only the universal "
                        "order-path milestones and report observed behavior.")
    return expect, warnings


def ensure_fresh_token(cfg: dict) -> None:
    """Re-mint the sandbox trading token if missing/older than the TTL (OTP 666666)."""
    tf = REPO / cfg["token_file"] if not os.path.isabs(cfg["token_file"]) else Path(cfg["token_file"])
    if tf.exists():
        age = time.time() - tf.stat().st_mtime
        if age < TOKEN_TTL_S and json.loads(tf.read_text()).get("trading_token"):
            print(f"[token] fresh ({age/3600:.1f}h old) — reusing")
            return
        print(f"[token] stale ({age/3600:.1f}h) — re-minting")
    else:
        print("[token] missing — minting")
    sys.path.insert(0, str(REPO / "plugins" / "dnse"))
    from pynecore_dnse.client import DNSEClient
    client = DNSEClient(cfg["api_key"], cfg["api_secret"], base_url=cfg["base_url"])
    status, body = client.create_trading_token("smart_otp", "666666")   # sandbox public OTP
    token = (body.get("tradingToken") or body.get("trading-token")) if isinstance(body, dict) else None
    if status != 200 or not token:
        sys.exit(f"[token] sandbox mint failed: HTTP {status}")
    tf.parent.mkdir(parents=True, exist_ok=True)
    tf.write_text(json.dumps({"trading_token": token, "minted_at": int(time.time())}))
    print("[token] fresh token minted")


def transpile(pine: Path) -> Path:
    out = pine.with_suffix(".py")
    print(f"[transpile] {pine.name} -> {out.name}")
    result = subprocess.run(
        [str(PINE2PYNE / ".venv" / "bin" / "python"), "-m", "pine2pyne",
         str(pine), "-o", str(out)],
        capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"[transpile] FAILED:\n{result.stdout}\n{result.stderr}")
    return out


def run_e2e(py: Path, data: str, fixture: Path, contract: str, market_type: str,
            timeout_s: int, log_path: Path) -> None:
    env = dict(os.environ,
               REPLAY_SANDBOX_FIXTURE=str(fixture),
               REPLAY_SANDBOX_MARKET_TYPE=market_type,
               REPLAY_SANDBOX_CONTRACT=contract)
    print(f"[run] pyne run {py.name} {data} --broker (timeout {timeout_s}s; "
          f"a timeout-kill after the bars drain is EXPECTED)")
    with open(log_path, "wb") as fh:
        proc = subprocess.run(
            [str(REPO / ".venv" / "bin" / "pyne"), "run", str(py), data, "--broker"],
            env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout_s + 30)  # noqa
    print(f"[run] exit={proc.returncode} (124/143 = drain timeout — grade from the log)")


def grade(log_text: str, expect: list[str]) -> int:
    text = _strip_ansi(log_text)
    print("\n===== ORDER-PATH milestones =====")
    failed = 0
    if not expect:
        for name, pat in _MILESTONE_PATTERNS.items():
            hit = bool(re.search(pat, text))
            print(f"  {'seen ' if hit else '  -- '} {name}")
        print("  (no @sandbox-expect line -> observation only, no pass/fail verdict)")
    else:
        for token in expect:
            pos = re.fullmatch(r"position>=(\d+(?:\.\d+)?)", token)
            if pos:
                want = float(pos.group(1))
                sizes = [float(x) for x in re.findall(r"position size=([0-9.]+)", text)]
                ok = any(s >= want for s in sizes)
                print(f"  {'PASS' if ok else 'FAIL'}  {token}  (max seen={max(sizes, default=0.0)})")
            else:
                pat = _MILESTONE_PATTERNS.get(token)
                if pat is None:
                    print(f"  ????  {token}  (unknown milestone token — check the @sandbox-expect line)")
                    failed += 1
                    continue
                ok = bool(re.search(pat, text))
                print(f"  {'PASS' if ok else 'FAIL'}  {token}")
            failed += 0 if ok else 1
    print("\n===== sandbox-boundary artifacts (expected; NOT engine bugs) =====")
    for _key, (pat, desc) in _SANDBOX_ARTIFACTS.items():
        n = len(re.findall(pat, text))
        if n:
            print(f"  seen x{n}: {desc}")
    return failed


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Sandbox Replay E2E runner (#114)")
    ap.add_argument("--pine", required=True, type=Path, help="strategy .pine (source of truth)")
    ap.add_argument("--data", default="dnse_replay_sandbox:VN30F1M@15")
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--contract", default="41I1G9000", help="sandbox dated code for the symbol")
    ap.add_argument("--market-type", default="DERIVATIVE", choices=("DERIVATIVE", "STOCK"))
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--grade-only", type=Path, help="grade an existing log; skip token/transpile/run")
    args = ap.parse_args(argv[1:])

    if not args.pine.exists():
        sys.exit(f"pine not found: {args.pine}")
    expect, warnings = read_pine(args.pine)
    print(f"[pine] {args.pine.name}  expect={expect or '(none)'}")
    for w in warnings:
        print(f"[pine] WARN: {w}")

    if args.grade_only:
        log_text = args.grade_only.read_text(errors="replace")
    else:
        if not SANDBOX_CONFIG.exists():
            sys.exit(f"sandbox config not found: {SANDBOX_CONFIG}")
        cfg = tomllib.loads(SANDBOX_CONFIG.read_text())
        ensure_fresh_token(cfg)
        py = transpile(args.pine)
        log_path = args.pine.with_suffix(".sandbox_e2e.log")
        try:
            run_e2e(py, args.data, args.fixture, args.contract, args.market_type,
                    args.timeout, log_path)
        except subprocess.TimeoutExpired:
            print("[run] hard timeout — the run parked after drain (expected); grading the log")
        log_text = log_path.read_text(errors="replace")

    failed = grade(log_text, expect)
    verdict = "PASS" if (expect and failed == 0) else ("FAIL" if failed else "OBSERVED")
    print(f"\n===== VERDICT: {verdict}  ({failed} milestone(s) missing) =====")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
