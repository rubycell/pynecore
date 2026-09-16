#!/usr/bin/env python3
"""#50 capture probe — Market-Data WS with the DOCUMENTED auth handshake.

The earlier "WS channels are silent" measurement predates this probe and is
SUSPECT METHODOLOGY (operator, 2026-08-26): the server requires an explicit
HMAC auth message within 30 s and refuses subscribes before auth succeeds
(sdk-build_websocket.md) — a probe that skipped or failed that handshake sees
exactly "silent channels". This probe follows the documented flow precisely:

    connect -> ready -> auth (HMAC-SHA256 over "{api_key}:{ts}:{nonce}")
            -> subscribe -> capture frames

#92 (measured 2026-09-08): the server CLOSED a trading-channel session
mid-capture (1000 OK) and the probe crashed with a raw traceback, losing
the close timing — so we could not tell whether the close predated the
account activity. A server close is now a REPORTED measurement (wall time,
elapsed, code/reason, frames-so-far) with exit 3. ``--dual`` measures the
single-session hypothesis (the engine's own WS may contend with a probe):
two sequential authenticated sessions on the same channels; the report
shows which one the server drops.

#121 / plan step 5 (2026-09-16): ``--trading`` now captures the SHORT order
channel (``order.{MT}.json`` — the only one that has ever delivered on prod,
4 ``do`` frames on 09-15) AND the BROKER channel
(``order.broker.{MT}.{investorId}.json`` — subscribed by ``ws_order_source.py``
since #121 and NEVER yet captured), so one place+cancel decides which channel
the plugin should trust.

HOW A FRAME IS ATTRIBUTED TO A CHANNEL (the honest part)
--------------------------------------------------------
There is no channel tag to read. Evidence, from the VENDORED SDK
(``_vendor/dnse/websocket/client.py``): ``_message_handler`` decodes a frame and
dispatches it purely on ``data["T"]`` through ``_MSG_TYPE_MAP``, where BOTH
``"do"`` and ``"eo"`` map to the single event name ``"order_event"``; nothing in
the client ever reads a channel field from an incoming frame, and the documented
payloads (guide-market-data-trading_connect.md / -broker_connect.md) carry none
either. So neither an SDK callback nor a raw frame can say which subscription
delivered it.

This probe therefore attributes by **SOCKET**: each channel group is captured on
its OWN authenticated session, so "which channel delivered" is answered by
construction, with no assumption about the envelope. It ALSO records each
frame's ``T`` code, so the capture measures for free whether the envelope
happens to be self-describing after all (short vs broker delivering different
``T`` codes would be exactly that).

``--shared-session`` is the fallback: everything on ONE socket (use it if the
server refuses two concurrent sessions, #92). Its counts are then labelled
AMBIGUOUS, because on one socket a frame cannot be attributed at all.

Read-only; credentials come from the broker toml and are NEVER printed. The
investor id is masked everywhere it is printed.
Usage:
    .venv/bin/python plugins/dnse/testing/live_test/probe_ws_market_data.py \
        [--seconds N] [--trading] [--dual] [--shared-session] \
        [--no-broker-channel]

Exit codes: 0 capture ran to its deadline; 2 auth/setup failed; 3 the
server closed the connection early (the close itself is the finding).
"""
import argparse
import asyncio
import datetime
import hashlib
import hmac
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "plugins" / "dnse" / "tools"))

from pynecore.core.config import ensure_config                      # noqa: E402
from pynecore_dnse.broker import DNSEBrokerConfig                   # noqa: E402

import websockets                                                    # noqa: E402

WS_URL = "wss://ws-openapi.dnse.com.vn/v1/stream?encoding=json"  # path from the vendored SDK (websocket/client.py)
CHANNELS = [
    {"name": "tick.G1.json", "symbols": ["41I1G9000", "HPG"]},
    {"name": "tick_extra.G1.json", "symbols": ["41I1G9000", "HPG"]},
    {"name": "ohlc.1.json", "symbols": ["41I1G9000", "HPG"]},
    {"name": "top_price.G1.json", "symbols": ["41I1G9000"]},
]
TRADING_CHANNELS = [
    # market_type is UPPERCASE per the official docs (Trading Data WebSocket) AND the
    # vendored SDK default (subscribe_order_event market_type="STOCK"). Lowercase
    # (order.derivative.json) is silently accepted by the server as a subscription but
    # streams NOTHING — that produced a false "trading WS is silent" verdict on
    # 2026-09-11 despite in-window account activity (#107). Channel = order.{market_type}.{encoding}.
    {"name": "order.DERIVATIVE.json", "symbols": []},
    {"name": "order.STOCK.json", "symbols": []},
    {"name": "position.DERIVATIVE.json", "symbols": []},
]


def broker_channels(investor_id: str) -> list:
    """The BROKER trading channels for one investor.

    Shape from the vendored SDK (``subscribe_broker_order_event`` /
    ``subscribe_broker_position_event``) — the same names ``ws_order_source.py``
    subscribes in the live plugin, so what this probe measures is what the
    plugin would receive.
    """
    return [
        {"name": f"order.broker.DERIVATIVE.{investor_id}.json", "symbols": []},
        {"name": f"order.broker.STOCK.{investor_id}.json", "symbols": []},
        {"name": f"position.broker.DERIVATIVE.{investor_id}.json", "symbols": []},
    ]


def mask(value) -> str:
    """Last 4 characters only — ids are never printed in full (same rule as
    ``ws_order_source._mask``)."""
    text = str(value or "")
    return ("*" * max(len(text) - 4, 0)) + text[-4:] if text else ""


def channel_names(channels, investor_id: "str | None") -> str:
    """Printable channel list with the investor id masked out."""
    names = [channel["name"] for channel in channels]
    if investor_id:
        names = [name.replace(str(investor_id), mask(investor_id))
                 for name in names]
    return ",".join(names)


def resolve_investor_id(cfg) -> "str | None":
    """The account's ``investorId`` — the BROKER channel key.

    Uses the PLUGIN's own resolver (``DNSEBroker._resolve_investor_id``, which
    reads it off ``/accounts``) rather than a hand-rolled REST call, so the
    probe subscribes exactly the channel the live plugin subscribes. Returns
    None on any failure — the caller then reports the broker channel as
    UNTESTED rather than silently capturing only the short one.
    """
    try:
        from pynecore_dnse.broker import DNSEBroker
        broker = DNSEBroker(symbol="VN30F1M", timeframe="1", config=cfg)
        return broker._resolve_investor_id()
    except Exception as exc:                                        # noqa: BLE001
        print(f"investor-id read FAILED ({type(exc).__name__}: {exc}) — the "
              f"BROKER channel cannot be subscribed")
        return None


def frame_key(frame: dict) -> str:
    """A per-frame accounting key.

    ``T`` is the venue's message type and the ONLY thing the vendored client
    dispatches on: ``do``/``eo`` are both order events (they collapse to one
    ``order_event`` callback there), ``dp``/``ep`` both position events. Keeping
    them apart HERE is free and measures whether the two order channels happen
    to use different type codes — the only way a frame could be self-describing.
    """
    msg_type = str(frame.get("T") or "")
    if msg_type in ("do", "eo"):
        order = frame.get("order") if isinstance(frame.get("order"), dict) else {}
        return f"T={msg_type} order marketType={order.get('marketType') or '?'}"
    if msg_type in ("dp", "ep"):
        position = (frame.get("position")
                    if isinstance(frame.get("position"), dict) else {})
        return f"T={msg_type} position marketType={position.get('marketType') or '?'}"
    if msg_type:
        return (f"T={msg_type}:"
                f"{frame.get('symbol') or frame.get('s') or '?'}")
    # If the venue ever DOES tag frames with their channel, this is where it
    # would show up — and the report would then say so instead of guessing.
    return str(frame.get("channel") or frame.get("action") or "<data>")


def _now() -> str:
    return datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=7))).strftime("%H:%M:%S")


async def _capture(cfg, channels, seconds: int, label: str,
                   secret_ids: "tuple[str, ...]" = ()) -> tuple:
    """One authenticated capture session on ONE socket.

    Returns ``(exit_code, counts, samples)`` — the counts are what makes a
    session's channel group accountable separately from every other session's.
    ``secret_ids`` are additionally masked out of the printed samples.
    """
    counts: Counter = Counter()
    samples: dict[str, str] = {}
    started = time.monotonic()
    closed_early: "str | None" = None

    ts = int(time.time())
    nonce = str(int(time.time() * 1_000_000))
    signature = hmac.new(cfg.api_secret.encode(),
                         f"{cfg.api_key}:{ts}:{nonce}".encode(),
                         hashlib.sha256).hexdigest()

    async with websockets.connect(WS_URL, open_timeout=15) as ws:
        try:
            ready = json.loads(await asyncio.wait_for(ws.recv(), 15))
            print(f"[{label}] ready: action={ready.get('action')} "
                  f"session={str(ready.get('session_id'))[:12]}…")
            await ws.send(json.dumps({"action": "auth", "api_key": cfg.api_key,
                                      "signature": signature, "timestamp": ts,
                                      "nonce": nonce}))
            auth = json.loads(await asyncio.wait_for(ws.recv(), 15))
            print(f"[{label}] auth: action={auth.get('action')} "
                  f"code={auth.get('code')} rate_limit={auth.get('rate_limit')}")
            if auth.get("action") != "auth_success":
                print(f"[{label}] AUTH FAILED: {auth.get('message')}")
                return 2, counts, samples
            await ws.send(json.dumps({"action": "subscribe",
                                      "channels": channels}))
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                try:
                    raw = await asyncio.wait_for(
                        ws.recv(), max(0.5, deadline - time.monotonic()))
                except asyncio.TimeoutError:
                    break
                try:
                    frame = json.loads(raw)
                except (ValueError, TypeError):
                    counts["<binary/unparsed>"] += 1
                    continue
                action = frame.get("action")
                if action in ("subscribed", "error", "ping", "pong"):
                    print(f"[{label}] control: {json.dumps(frame)[:160]}")
                    if action == "ping":
                        await ws.send(json.dumps({"action": "pong"}))
                    continue
                key = frame_key(frame)
                counts[key] += 1
                text = json.dumps(frame)
                for field in ("accountNo", "custodyCode", "investorId"):
                    value = frame.get(field)
                    if value:
                        text = text.replace(str(value), "<masked>")
                for secret in secret_ids:
                    if secret:
                        text = text.replace(str(secret), "<masked>")
                samples.setdefault(key, text[:600])
        except websockets.exceptions.ConnectionClosed as exc:
            # #92: the close IS the measurement — report, never traceback.
            elapsed = time.monotonic() - started
            closed_early = (f"SERVER CLOSED the connection at {_now()} "
                            f"(+{elapsed:.1f}s into the session): "
                            f"code={exc.rcvd.code if exc.rcvd else '?'} "
                            f"reason={(exc.rcvd.reason if exc.rcvd else '') or '<none>'}")
            print(f"[{label}] {closed_early}")

    print(f"\n[{label}] === capture ({seconds}s requested, "
          f"{time.monotonic() - started:.1f}s actual) ===")
    for key, n in counts.most_common():
        print(f"[{label}] {n:5d}  {key}")
    for key, sample in samples.items():
        print(f"\n[{label}] sample [{key}]: {sample}")
    if closed_early:
        print(f"[{label}] frames before the close: {sum(counts.values())} — "
              f"a close BEFORE account activity means the channel was never "
              f"tested; re-run and note WHAT ELSE holds a session "
              f"(engine WS? another probe?)")
        return 3, counts, samples
    if not counts:
        print(f"[{label}] ZERO data frames — silent WITH correct auth "
              f"(methodology sound; for order./position. channels this is "
              f"EXPECTED unless account events occurred during capture)")
    return 0, counts, samples


def _report_per_channel(results, investor_id) -> None:
    """The step-5 answer: how many frames each SESSION's channel group got.

    One session = one socket = one channel group, so the attribution needs no
    assumption about the frame envelope (see the module docstring). The ``T``
    breakdown underneath each session is what would prove the envelope IS
    self-describing, if the two order channels ever used different codes.
    """
    print("\n" + "=" * 78)
    print("PER-CHANNEL FRAME ACCOUNTING")
    for label, channels, rc, counts, samples in results:
        total = sum(counts.values())
        print(f"\n[{label}] channels: {channel_names(channels, investor_id)}")
        print(f"[{label}] rc={rc}  frames={total}")
        if not total:
            print(f"[{label}]   (no frames — for order/position channels this is "
                  f"only meaningful if an account event happened during the "
                  f"capture; with none, this is UNTESTED, not silent)")
        for key, n in counts.most_common():
            print(f"[{label}]   {n:5d}  {key}")
        for key, sample in samples.items():
            print(f"[{label}]   sample [{key}]: {sample}")
    delivering = [label for label, _c, _rc, counts, _s in results if sum(counts.values())]
    print("\nverdict: " + (
        f"frames arrived on: {', '.join(delivering)}" if delivering
        else "NO session received a frame — nothing is decided about either "
             "channel (was there an account event inside the window?)"))
    print("=" * 78, flush=True)


async def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=int, default=45)
    ap.add_argument("--trading", action="store_true",
                    help="subscribe the TRADING channels (order/position "
                         "events) instead of market data — passive, read-only. "
                         "Captures the SHORT channels and the BROKER channels "
                         "on SEPARATE sockets, so each frame is attributable")
    ap.add_argument("--no-broker-channel", action="store_true",
                    help="with --trading, skip the broker channel session "
                         "(the pre-2026-09-16 behaviour: short channels only)")
    ap.add_argument("--shared-session", action="store_true",
                    help="put every channel on ONE socket instead. Use it if "
                         "the server refuses two concurrent sessions (#92); "
                         "the counts are then AMBIGUOUS — a frame cannot be "
                         "attributed to a channel on a shared socket")
    ap.add_argument("--dual", action="store_true",
                    help="#92 single-session measurement: run TWO "
                         "authenticated sessions on the same channels, "
                         "second joining 5s late — the report shows which "
                         "one the server drops")
    args = ap.parse_args()

    cfg = ensure_config(DNSEBrokerConfig,
                        REPO / "workdir" / "config" / "plugins" / "dnse_broker.toml")
    channels = TRADING_CHANNELS if args.trading else CHANNELS

    investor_id: "str | None" = None
    broker_group: list = []
    if args.trading and not args.no_broker_channel:
        investor_id = resolve_investor_id(cfg)
        if investor_id:
            broker_group = broker_channels(investor_id)
            print(f"broker channels for investor {mask(investor_id)}: "
                  f"{channel_names(broker_group, investor_id)}")
        else:
            print("WARNING: no investor id -> the BROKER channel is UNTESTED "
                  "this run (absence of frames there proves nothing)")

    secret_ids = (investor_id,) if investor_id else ()

    if args.dual:
        # Unchanged #92 measurement: same channels, two sessions.
        async def _late_second():
            await asyncio.sleep(5)
            return await _capture(cfg, channels + broker_group, args.seconds,
                                  "ws-B", secret_ids)

        (rc_a, *_), (rc_b, *_) = await asyncio.gather(
            _capture(cfg, channels + broker_group, args.seconds + 5, "ws-A",
                     secret_ids),
            _late_second())
        print(f"\n=== dual verdict: A rc={rc_a}, B rc={rc_b} — "
              f"3 marks the session the server dropped; two 0s = no "
              f"single-session limit observed ===")
        return max(rc_a, rc_b)

    if args.shared_session or not broker_group:
        all_channels = channels + broker_group
        rc, counts, samples = await _capture(cfg, all_channels, args.seconds,
                                             "ws", secret_ids)
        if broker_group:
            print("\nNOTE: --shared-session — every channel shared ONE socket, "
                  "so these counts are AMBIGUOUS: a frame cannot be attributed "
                  "to the short or the broker channel (the envelope carries no "
                  "channel tag; see the module docstring).")
        _report_per_channel([("ws", all_channels, rc, counts, samples)],
                            investor_id)
        return rc

    # Default for --trading: ONE SOCKET PER CHANNEL GROUP = exact attribution.
    (rc_short, counts_short, samples_short), (rc_broker, counts_broker,
                                              samples_broker) = await asyncio.gather(
        _capture(cfg, channels, args.seconds, "short", secret_ids),
        _capture(cfg, broker_group, args.seconds, "broker", secret_ids))
    _report_per_channel(
        [("short", channels, rc_short, counts_short, samples_short),
         ("broker", broker_group, rc_broker, counts_broker, samples_broker)],
        investor_id)
    return max(rc_short, rc_broker)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
