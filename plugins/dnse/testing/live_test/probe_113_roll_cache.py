#!/usr/bin/env python3
"""#113 — does a LONG-LIVED process serve a STALE dated contract across the roll?

THE QUESTION
------------
``DNSEProvider.resolve_contract`` maps the rolling alias (``VN30F1M``) to the
dated KRX contract (``41I1G9000``) by matching ``symbolType`` on
``/market/instruments``, and caches the answer in ``self._contract_cache``
(provider.py). That cache has **no TTL and no day boundary** — unlike the
sibling ``_secdef_cache``, which already retries an empty read after
``_SECDEF_RETRY_S``. Operator-confirmed venue mechanic: the aliases repoint to
the next dated contract on the MORNING AFTER expiry. Expiry is Thu 2026-09-17,
so the repoint lands Fri 2026-09-18.

Therefore a process that resolved the alias BEFORE the repoint keeps returning
the EXPIRED contract for its whole life — and that value is what orders and the
streaming channels use. This probe measures whether that actually happens, and
WHEN the venue flips.

THE A/B, IN ONE PROCESS
-----------------------
Each cycle asks the same question three ways and logs all three side by side:

  AGED    — one provider instance created at START-UP, asked again every cycle.
            After its first answer this is pure cache; it is the "long-lived
            engine" arm.
  FRESH   — a BRAND-NEW provider instance per cycle, so its cache is empty and
            the call really hits ``/market/instruments``. This is the
            "just-restarted engine" arm.
  VENUE   — the raw ``symbolType -> symbol`` rows, read directly. Ground truth,
            independent of our cache, so a divergence can be attributed to US
            rather than to the venue changing under both arms.

AGED != FRESH is #113 CONFIRMED, and the first cycle where it happens timestamps
the repoint. AGED == FRESH == VENUE all watch long enough is evidence the cache
is harmless FOR THE WATCHED WINDOW and nothing longer.

WHY IT MUST START THE DAY BEFORE
--------------------------------
The AGED arm is only meaningful if it cached BEFORE the repoint. Start it
Thursday and let it run into Friday's open. Starting it Friday morning risks
caching the ALREADY-REPOINTED value, which makes the two arms agree for an
uninteresting reason — the probe says so in its verdict rather than claiming a
pass.

SAFETY
------
Read-only. It places no orders, needs NO trading token (``/market/instruments``
is an api-key read), and holds no position. The only cost is polling.

SELF-TEST (run this before trusting a real watch)
-------------------------------------------------
``--self-test`` injects a simulated repoint into the comparison and asserts the
detector REPORTS it. A detector that has only ever seen "no divergence" has
proven nothing, so this is the red-first control: it must print DIVERGENCE for
the forced case and agreement for the matching case, with no venue calls.

    .venv/bin/python plugins/dnse/testing/live_test/probe_113_roll_cache.py --self-test
    .venv/bin/python plugins/dnse/testing/live_test/probe_113_roll_cache.py --hours 20

Exit codes: 0 the watch ran and reported a verdict (divergence or not);
2 setup failed / the probe refuses to conclude.
"""
from __future__ import annotations

import argparse
import sys
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "plugins" / "dnse"))
from pynecore.core.config import ensure_config                        # noqa: E402
from pynecore_dnse.provider import DNSEProvider, DNSEConfig           # noqa: E402

ICT = timezone(timedelta(hours=7))
ALIASES = ("VN30F1M", "VN30F2M")

# The repoint is expected around Friday's open, so poll tightly across the
# morning and loosely overnight — a 30 s grid all night would be ~2,800 wasted
# reads and would not time the flip any better.
HOT_START, HOT_END = 8, 11          # ICT hours, inclusive-exclusive
HOT_POLL_S, COLD_POLL_S = 30, 300


def _now() -> datetime:
    return datetime.now(ICT)


def _poll_interval(moment: datetime) -> int:
    return HOT_POLL_S if HOT_START <= moment.hour < HOT_END else COLD_POLL_S


def venue_mapping(provider) -> "dict[str, str]":
    """Raw ``symbolType -> symbol`` straight off /market/instruments.

    Ground truth. Returns {} when the read fails — the caller reports that as a
    FAILED READ, never as "no mapping", so a dead endpoint can never be
    mistaken for a venue that stopped serving the alias.
    """
    status, body = provider.client.get_instruments(limit=200)
    if status != 200 or not isinstance(body, dict):
        return {}
    out = {}
    for row in body.get("data") or []:
        symbol_type = row.get("symbolType")
        if symbol_type in ALIASES and row.get("symbol"):
            out[symbol_type] = row["symbol"]
    return out


def compare(aged: "dict[str, str]", fresh: "dict[str, str]",
            venue: "dict[str, str]") -> "list[str]":
    """The detector. Returns one finding line per alias that DISAGREES.

    Kept pure and free of venue calls so ``--self-test`` can force a repoint
    through it and prove it fires.
    """
    findings = []
    for alias in ALIASES:
        aged_value, fresh_value = aged.get(alias), fresh.get(alias)
        if aged_value and fresh_value and aged_value != fresh_value:
            findings.append(
                f"DIVERGENCE {alias}: AGED(cached)={aged_value} != "
                f"FRESH(uncached)={fresh_value} "
                f"| venue now serves {venue.get(alias) or 'UNREADABLE'} "
                f"-> #113 CONFIRMED: a long-lived process is holding the "
                f"pre-roll contract")
    return findings


def self_test() -> int:
    """Red-first control: force a repoint and assert the detector reports it."""
    print("=== SELF-TEST (no venue calls) ===")
    before = {"VN30F1M": "41I1G9000", "VN30F2M": "41I1GA000"}
    after = {"VN30F1M": "41I1GA000", "VN30F2M": "41I1GB000"}

    agree = compare(before, before, before)
    print(f"  matching arms          -> {len(agree)} finding(s): {agree or 'none (correct)'}")
    diverge = compare(before, after, after)
    print(f"  simulated repoint      -> {len(diverge)} finding(s)")
    for line in diverge:
        print(f"     {line}")
    partial = compare({"VN30F1M": "41I1G9000"}, {}, {})
    print(f"  fresh read FAILED      -> {len(partial)} finding(s): "
          f"{partial or 'none (correct: a failed read is never a divergence)'}")

    ok = (not agree) and len(diverge) == 2 and not partial
    print("SELF-TEST", "PASS — the detector fires on a real repoint, stays "
          "quiet when arms agree, and never turns a failed read into a finding"
          if ok else "FAIL — do NOT trust a watch run from this build")
    return 0 if ok else 2


def main(argv: "list[str]") -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=20.0,
                    help="how long to watch (default 20 — Thursday afternoon "
                         "through Friday's open)")
    ap.add_argument("--poll-seconds", type=int, default=None,
                    help="fixed poll interval; default is adaptive "
                         f"({HOT_POLL_S}s in {HOT_START:02d}-{HOT_END:02d} ICT, "
                         f"{COLD_POLL_S}s otherwise)")
    ap.add_argument("--self-test", action="store_true",
                    help="prove the divergence detector fires, then exit "
                         "(no venue calls, no watch)")
    args = ap.parse_args(argv[1:])

    if args.self_test:
        return self_test()

    try:
        cfg = ensure_config(DNSEConfig, REPO / "workdir/config/plugins/dnse.toml")
        aged_provider = DNSEProvider(symbol="VN30F1M", timeframe="1", config=cfg)
    except Exception as exc:                                          # noqa: BLE001
        print(f"SETUP FAILED ({type(exc).__name__}: {exc}) — refusing to watch")
        return 2

    started = _now()
    aged: "dict[str, str]" = {}
    for alias in ALIASES:
        aged[alias] = aged_provider.resolve_contract(alias)
    if not any(aged.values()):
        print("SETUP FAILED — the AGED arm could not resolve ANY alias; "
              "a watch from here would compare nothing")
        return 2

    print("=" * 78)
    print(f"#113 roll-cache watch — started {started:%Y-%m-%d %H:%M:%S} ICT")
    print(f"  AGED arm cached at start: "
          + ", ".join(f"{a}={aged[a]}" for a in ALIASES))
    print(f"  watching {args.hours:.1f}h; expiry Thu 2026-09-17, "
          f"repoint expected Fri 2026-09-18 morning")
    print("  AGED != FRESH  =>  #113 CONFIRMED (long-lived process holds the "
          "pre-roll contract)")
    print("=" * 78, flush=True)

    deadline = _time.monotonic() + args.hours * 3600
    first_divergence: "tuple[datetime, list[str]] | None" = None
    cycles = failed_reads = 0

    try:
        while _time.monotonic() < deadline:
            moment = _now()
            cycles += 1
            # FRESH arm: a new instance every cycle, so its cache is cold and
            # the call genuinely re-reads the venue.
            fresh_provider = DNSEProvider(symbol="VN30F1M", timeframe="1",
                                          config=cfg)
            fresh = {a: fresh_provider.resolve_contract(a) for a in ALIASES}
            venue = venue_mapping(fresh_provider)
            aged_now = {a: aged_provider.resolve_contract(a) for a in ALIASES}

            if not venue:
                failed_reads += 1
                print(f"[{moment:%m-%d %H:%M:%S}] venue read FAILED "
                      f"(read #{failed_reads}) — reporting, NOT concluding",
                      flush=True)

            findings = compare(aged_now, fresh, venue)
            if findings and first_divergence is None:
                first_divergence = (moment, findings)
                print("\n" + "!" * 78, flush=True)
                for line in findings:
                    print(f"[{moment:%m-%d %H:%M:%S}] {line}", flush=True)
                print("!" * 78 + "\n", flush=True)
            else:
                print(f"[{moment:%m-%d %H:%M:%S}] "
                      + " ".join(
                          f"{a}: aged={aged_now.get(a)} fresh={fresh.get(a)} "
                          f"venue={venue.get(a) or '?'}" for a in ALIASES)
                      + ("  <-- STILL DIVERGED" if findings else ""), flush=True)
            _time.sleep(args.poll_seconds or _poll_interval(moment))
    except KeyboardInterrupt:
        print("\ninterrupted — reporting what was measured", flush=True)

    ended = _now()
    print("=" * 78)
    print(f"#113 VERDICT after {cycles} cycles "
          f"({started:%m-%d %H:%M} -> {ended:%m-%d %H:%M} ICT)")
    if first_divergence:
        moment, findings = first_divergence
        print(f"  CONFIRMED — first divergence at {moment:%Y-%m-%d %H:%M:%S} ICT")
        for line in findings:
            print(f"    {line}")
        print("  => resolve_contract's per-instance cache MUST be invalidated "
              "per trading day (the _secdef_cache TTL is the in-repo pattern).")
    elif started.date() == ended.date():
        print("  INCONCLUSIVE — the watch never crossed a trading-day boundary, "
              "so the roll could not have happened inside it.")
    else:
        print(f"  NO DIVERGENCE across {cycles} cycles spanning the boundary. "
              f"Evidence the cache was harmless FOR THIS WINDOW only.")
        print("  Check the AGED start value above against the venue's CURRENT "
              "mapping: if they were ALREADY equal at start, the probe began "
              "AFTER the repoint and the arms agreed for an uninteresting "
              "reason — that is not a pass.")
    if failed_reads:
        print(f"  NOTE: {failed_reads} venue read(s) failed during the watch; "
              f"those cycles proved nothing either way.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
