"""DNSE broker plugin — native conditional order execution for PyneCore (v2).

Builds on :class:`DNSEProvider` (history + metadata) and implements the
``BrokerPlugin`` abstracts using DNSE's **native conditional orders**
(``orderCategory=STOP|OCO`` on the account-scoped ``/accounts/{accountNo}/orders``
endpoints). Server-side stops fire even if the plugin is offline.

Transport: the ORDER path (place / cancel / fill detection) is REST poll-based —
there is no WS order-event transport yet (#107). Market data is mixed: 1m+ bars
are REST closed-bar polls, while sub-minute bars are synthesized from the venue
WebSocket per-print stream (#100). So the plugin is NOT WebSocket-free.

Design:

* ``position_port = None`` — DNSE derivatives are netted per symbol.
* Pine intent -> native order:
  ``entry(stop)`` / ``entry(limit,stop)`` -> STOP; ``entry(limit)`` -> NORMAL LO;
  ``exit(limit,stop)`` -> OCO; ``exit(stop)`` -> STOP; ``exit(limit)`` -> NORMAL LO;
  ``close`` / market -> marketable LO (band edge).
* ``conditionOperator`` = ``>=`` for buy stops (trigger on the way up), ``<=`` for
  sell stops (trigger on the way down).
* Cancel/replace only while ``New`` (once ``Activated`` a conditional is a NORMAL
  order and is managed there). ``version`` is pinned to 2026-07-23 in the client.
* Trading token is read from the state file written by the OTP minter.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, replace as _dc_replace
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path

from pynecore.core.plugin import override
from pynecore.core.plugin.broker import BrokerPlugin
from pynecore.core.broker.models import (
    CancelDispositionOutcome, CapabilityLevel, ExchangeCapabilities,
    CANCEL_REASON_VENUE_EXPIRED,
    ExchangeOrder, ExchangePosition, LegType, OrderEvent, OrderStatus,
    OrderType,
)

from .cancel_disposition import (
    aggregate as _aggregate_dispositions,
    classify_readback as _classify_readback,
)
from .feed_health import FeedHealth
from .fill_slices import parse_reports, select_events
from .journal_wiring import (
    iter_journal_identities, journal_child_ref, journal_disposition_unknown,
    journal_fill_progress, journal_rejected, journal_server_ref,
    journal_submitted, journal_terminal,
)
from .price_units import (
    DERIVATIVE_WIRE_SCALE, STOCK_WIRE_SCALE, StockOrdersDisabledError,
    UnverifiedClassificationError, from_wire, quantize_wire, to_wire,
)
from .recovery_ladder import classify_recovery
from .residue_detector import ResidueTracker
from .transport_errors import guard as _guard_transport
from .page_completeness import (
    BOOK_READ_DEADLINE_S, POSITIONS_PAGE_SIZE,
    book_page_count, is_exposure_row, positions_complete,
)
from pynecore.types.ohlcv import OHLCV
from pynecore.lib import log
from pynecore.core.broker.exceptions import (
    AuthenticationError, BrokerManualInterventionError, ExchangeConnectionError,
    ExchangeOrderRejectedError, ExchangeRateLimitError, InsufficientMarginError,
    OrderDispositionUnknownError,
)
from pynecore.core.broker.idempotency import (
    KIND_ENTRY, KIND_EXIT_TP, KIND_EXIT_SL, KIND_CLOSE)

from .provider import DNSEConfig, DNSEProvider
from . import errors, expiry


def _midnight_utc(day: _date) -> datetime:
    """Midnight UTC of ``day`` — the GTD form DNSE accepts (see ``_clamp_gtd_to_expiry``)."""
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)


_SIDE_TO_DNSE = {"buy": "NB", "sell": "NS"}
_DNSE_TO_SIDE = {"NB": "buy", "NS": "sell"}

#: DNSE order status -> PyneCore OrderStatus (keys uppercased, ``_``/``-`` stripped).
#: Covers NORMAL statuses + the STOP/OCO lifecycle (New -> Activated -> terminal).
_STATUS_MAP = {
    "PENDING": OrderStatus.PENDING, "PENDINGNEW": OrderStatus.PENDING,
    "NEW": OrderStatus.OPEN, "OPEN": OrderStatus.OPEN,
    "ACTIVATED": OrderStatus.OPEN,       # conditional triggered -> now a working order
    "PENDINGCANCEL": OrderStatus.OPEN,   # cancel in flight; still live
    "PARTIALLYFILLED": OrderStatus.PARTIALLY_FILLED,
    "FILLED": OrderStatus.FILLED,
    "CANCELLED": OrderStatus.CANCELLED, "CANCELED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED, "EXPIRED": OrderStatus.EXPIRED,
    "FAILED": OrderStatus.REJECTED,
}

_TERMINAL_STATUSES = frozenset({
    OrderStatus.FILLED, OrderStatus.CANCELLED,
    OrderStatus.REJECTED, OrderStatus.EXPIRED,
})

#: Restore ``LegType`` from the journal's ``leg_kind`` extras (#36).
_LEG_TYPE_BY_NAME = {member.name: member for member in LegType}

#: Read-side dispositions that mean the CREDENTIAL is refused (#54): the only
#: failure kinds that can satisfy the feed-health all-books halt condition.
_AUTH_DISPOSITIONS = (errors.Disposition.AUTH, errors.Disposition.AUTH_TOKEN)

#: TradingView timeframe -> seconds (bar-period math for the closed-bar poll).
_TF_SECONDS = {"1": 60, "3": 180, "5": 300, "15": 900, "30": 1800,
               "60": 3600, "1H": 3600, "1D": 86400}

#: Order-book categories to poll/scan for WORKING orders + fills. An OCO's real
#: working order is the spawned NORMAL LO (tracked via its externalOrderId), and a
#: STOP is its own working order — so we scan NORMAL + STOP and skip the OCO
#: umbrella book (whose records would double-count the LO / linger as zombies).
#: Do NOT "fix" #43 by adding "OCO" here (S3 on that card — rejected: it would
#: double-count get_open_orders and cost +50% Get-Orders budget forever): an
#: umbrella whose LO is unknown at place time goes into ``_pending_oco`` and is
#: drained by ``watch_orders`` instead.
_CATEGORIES = ("NORMAL", "STOP")

#: Books a CANCEL may need to probe for an id with no category record (#45) —
#: unlike the scan set above this must include OCO: an unrecorded umbrella id
#: answers 404 on both scanned books, and "not found everywhere probed" is
#: treated as already-gone.
_CANCEL_PROBE_BOOKS = ("NORMAL", "STOP", "OCO")

#: LegType.name -> idempotency KIND, for the disposition-unknown client_order_id.
_LEG_KIND = {
    "ENTRY": KIND_ENTRY, "TAKE_PROFIT": KIND_EXIT_TP,
    "STOP_LOSS": KIND_EXIT_SL, "CLOSE": KIND_CLOSE,
}


# Phase 0a (#66): DNSEBrokerConfig lives in config.py; re-exported here
# so existing imports keep working.
from .config import DNSEBrokerConfig  # noqa: F401  (re-export)

class DNSEBroker(DNSEProvider[DNSEBrokerConfig], BrokerPlugin[DNSEBrokerConfig]):
    """DNSE broker: Vietnam derivatives (native STOP/OCO) and stocks."""

    plugin_name = "DNSE Broker"
    #: Phase 0a (#66): the one config type both bases agree on — pyright's
    #: structural incompatibility came from the provider pinning DNSEConfig.
    config: DNSEBrokerConfig
    Config = DNSEBrokerConfig

    #: Netting-native venue — no hedged-leg emulation.
    position_port = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._connected = False
        self._account_no: str | None = None
        #: intent_key -> [venue order ids] (the handle a later cancel/amend uses).
        self._order_ids: dict[str, list[str]] = {}
        #: venue order id -> (pine_id, from_entry, leg_type) for event tagging.
        self._identity: dict[str, tuple] = {}
        #: venue order id -> the book it lives in ("NORMAL"/"STOP") for cancel/amend.
        self._order_category: dict[str, str] = {}
        #: venue order id -> (cumulative_fill, status) from the last poll.
        #: #121: SHARED across the REST poll AND the WS order feed — both diff
        #: their observed cumulative against THIS one watermark inside
        #: ``_scan_row``, so the same fill on both transports advances it once
        #: and emits one delta (cross-transport dedup; a cumulative <= watermark
        #: is a duplicate -> dropped).
        self._last_seen: dict[str, tuple] = {}
        #: #121 dual-transport failsafe (see config.enable_ws_order_events).
        #: Lazily-started PROD WS order-event source; None until first
        #: ``watch_orders`` cycle (or when disabled). Both transports feed the
        #: engine's ONE event queue via ``watch_orders``.
        self._ws_order_source = None
        self._ws_start_task = None  # in-flight background connect+subscribe
        self._ws_disabled_logged = False
        #: Investor id for the WS broker channel, resolved once from /accounts.
        self._investor_id: str | None = None
        #: Monotonic backoff deadline after a WS start failure — poll-only until
        #: then (bounded by ws_order_reconnect_interval); a mid-stream drop is
        #: handled by the vendored client's auto_reconnect, not this.
        self._ws_next_retry_monotonic: float = 0.0
        _cfg = getattr(self, "config", None)
        self._poll_interval: float = float(
            getattr(_cfg, "order_poll_interval", None) or 0.5)
        self._bar_poll_interval: float = float(
            getattr(_cfg, "bar_poll_interval", None) or 3.0)
        #: #121 dual-transport WS order feed (config; see enable_ws_order_events).
        self._enable_ws_order_events: bool = bool(
            getattr(_cfg, "enable_ws_order_events", True))
        self._ws_reconnect_interval: float = float(
            getattr(_cfg, "ws_order_reconnect_interval", None) or 3.0)
        # --- #37 dual-mode feed (S1' — dispatch inside watch_ohlcv) ---
        self._feed_mode: str = str(getattr(_cfg, "feed_mode", None) or "ohlc")
        #: #100 LTF (sub-minute) feed state: lazily-started WS tick source +
        #: aggregator + pending closed bars; separate LTF store writer.
        self._ltf_state: "dict | None" = None
        self._ltf_writer = None
        if self._feed_mode not in ("ohlc", "tick"):
            raise ValueError(
                f"feed_mode must be 'ohlc' or 'tick', got {self._feed_mode!r} "
                f"(fail-fast: a typo silently falling back would hide tick mode)")
        self._tick_poll_interval: float = float(
            getattr(_cfg, "tick_poll_interval", None) or 2.0)
        self._tick_close_timeout: float = float(
            getattr(_cfg, "tick_close_timeout", None) or 20.0)
        #: Session-cumulative totalVolumeTraded of the newest accepted print,
        #: PER BOARD — measured live 2026-08-25 (probe_trades_latest.py): the
        #: endpoint returns G1 (continuous) AND T1 (put-through/block) rows
        #: interleaved, each with its OWN independent, non-comparable counter (a
        #: single global cursor silently dropped valid G1 prints as replays
        #: whenever a lower-cursor T1 row interleaved). Only G1 feeds synthesis.
        self._tick_board = "G1"
        self._tick_cursor: float = 0.0
        self._tick_slot: int = 0          # bar-start ts (s) of the forming bar
        self._tick_bar: dict | None = None
        self._tick_close_deadline: float | None = None
        self._tick_throttled: bool = False
        #: Cancel-disposition read-back poll (see ``_readback_disposition``);
        #: 4 reads / 3 sleeps = a <=3 s budget (#55 panel) — a leftover UNKNOWN
        #: resolves on the watch/poll cadence instead of blocking the loop.
        self._cancel_verify_attempts: int = 4
        self._cancel_verify_delay: float = 0.7
        #: Wall-clock deadline for one off-loop positions/book read (#62);
        #: module default sits under the engine's ~30 s execute budget.
        self._book_read_deadline_s: float = BOOK_READ_DEADLINE_S
        #: #54 feed-health thresholds, in watch CYCLES (0.5 s cadence): first
        #: warning ~10 s into a persistent failure, re-warn ~60 s, and the
        #: all-books-AUTH halt only after ~60 s (a latched halt is
        #: irreversible — an auth blip must warn, never halt). Tunable for
        #: tests like ``_cancel_verify_attempts``.
        self._feed_warn_after: int = 20
        self._feed_rewarn_every: int = 120
        self._feed_halt_after: int = 120
        #: Single-flight wait per cycle on the in-flight poll read; a hung
        #: socket (60 s timeout) counts as stuck cycles, never a stack of
        #: abandoned worker threads on the SHARED default executor.
        self._watch_read_deadline_s: float = 10.0
        #: #56: executions read (slice prices) — own 10k/h bucket; ~2.5 s
        #: wait then the VWAP-delta fallback (a fill is NEVER lost to a slow
        #: slice read); a 429 backs the reads off until the cooldown passes.
        self._executions_read_deadline_s: float = 2.5
        self._executions_cooldown_until: float = 0.0
        #: #85: intent keys already warned about the conditional-modify
        #: limitation (exit park) — once per key per episode; an entry
        #: replace clears its key so the next episode warns again.
        self._modify_warned_keys: set = set()
        #: #93 G1: venue id -> INTENT-time book ("OCO"/"STOP"/"NORMAL").
        #: The tracked category of a resolved OCO child is "NORMAL", which
        #: erases the bracket origin exit-modify routing needs; this map
        #: (journal-rooted, restart-hydrated) preserves it.
        self._placed_category: dict = {}
        #: #81 bar-feed poll-failure ladder — counters on the INSTANCE (the
        #: runner re-enters watch_ohlcv every ≤2 s, so coroutine locals
        #: reset per entry). Warn after N consecutive failures, re-warn
        #: every M, reset on any success.
        self._bar_poll_failures: int = 0
        self._bar_poll_warn_after: int = 20
        self._bar_poll_rewarn_every: int = 120
        #: #74 residue detector: grace before a vanished journalled id is
        #: even ASKED about — 30 s flat, 3x the measured ~10 s stale-replica
        #: lag, deliberately NOT a cadence formula (a 5x-cadence rule gives
        #: 2.5 s at our 0.5 s poll, below the lag: measured false cancel).
        self._residue_grace_s: float = 30.0
        #: Child-adoption retry shape for an Activated conditional whose
        #: ``externalOrderId`` is not published yet (#42-A). Counted in POLLS, not
        #: seconds, so the schedule is deterministic in tests and cannot hot-spin
        #: on a wall clock. At the 0.5 s default cadence: retry every poll for
        #: 10 s (the measured stale-replica lag is ~10 s), then once per 10 s so a
        #: permanently unresolvable shell cannot drain the Get-Order-Detail budget
        #: (10,000/h — shared with ``_readback_disposition``, so starving it would
        #: turn a fill-visibility bug into a cancel-verification bug), and give up
        #: at ~2 min by escalating for manual intervention.
        self._adopt_fast_polls: int = 20
        self._adopt_slow_every: int = 20
        self._adopt_give_up_polls: int = 240
        #: parent order id -> adoption attempts so far. ADVISORY only: losing it
        #: (restart) costs a faster retry, never correctness.
        self._adopt_attempts: dict[str, int] = {}
        #: OCO umbrella ids whose working LO was unresolved at PLACE time (#43);
        #: they live on a book ``_CATEGORIES`` never scans, so ``watch_orders``
        #: drains this set every cycle. In-memory like the rest (#36): lost on a
        #: restart, so recovery must re-derive from the venue books.
        self._pending_oco: set[str] = set()
        self._last_bar_ts: int = 0
        self._loan_id: int | None = None
        #: #119/G1: a READ that hit a guessed STOCK classification warns once
        #: per instance (the read keeps the identity scale — see _wire_scale).
        self._unverified_scale_warned: bool = False

    # --- account / token ---

    @property
    def account_id(self) -> str:
        if self._account_no:
            return self._account_no
        assert self.config is not None
        if self.config.account_no:
            self._account_no = self.config.account_no
            return self._account_no
        status, body = self.client.get_accounts()
        accounts = (body.get("accounts") or []) if isinstance(body, dict) else []
        if status != 200 or not accounts:
            raise RuntimeError(f"cannot resolve account: {status} {body}")
        self._account_no = accounts[0]["id"]
        return self._account_no

    def _token(self) -> str:
        """Trading token: the OTP-minter state file first, config as fallback."""
        assert self.config is not None
        path = Path(self.config.token_file)
        if path.exists():
            try:
                token = json.loads(path.read_text()).get("trading_token")
                if token:
                    return token
            except (ValueError, OSError):
                pass
        if self.config.trading_token:
            return self.config.trading_token
        raise RuntimeError(
            f"no trading_token — run the OTP minter (writes {self.config.token_file}) "
            f"or set trading_token in the plugin config"
        )

    # --- capabilities ---

    @override
    def get_capabilities(self) -> ExchangeCapabilities:
        return ExchangeCapabilities(
            # Native server-side STOP (stays New until triggered -> cleanly
            # cancellable). Used for entry-stops and the SL-only exit.
            stop_order=CapabilityLevel.NATIVE,
            # Native OCO bracket: one server-side order (a TP LO that auto-amends
            # to the SL price if the SL condition hits). The venue runs the
            # one-cancels-other; the plugin tracks the OCO's working LO via
            # externalOrderId. See execute_exit.
            tp_sl_bracket=CapabilityLevel.NATIVE,
            # #82b (measured live 2026-09-07): DNSE protective exits are
            # STANDALONE conditional-book orders that execute with no
            # position behind them — the engine must withhold them until
            # the parent entry fills and clamp qty to the live position.
            exit_orders_execute_standalone=True,
            # #121 (operator directive: "we cannot place an order without
            # protection"): arm the dependent SL/TP the instant the entry
            # fills instead of at the next bar-close sync. ON by default for
            # DNSE — an entry must never sit unprotected for a whole bar
            # (~60s at 1m). The engine's #121 wake worker fires the arm ~1-2s
            # after the fill event (0.5s watch_orders poll + wake dispatch +
            # REST place). A zero window is impossible on DNSE (the SL is a
            # separate conditional placed only AFTER the fill), but this closes
            # the ~1-bar window measured in #107.
            arm_protection_on_fill=True,
            # #87: a both-set entry is executed HERE as one stop-limit
            # (native conditional STOP; crossed-at-placement -> immediate
            # capped LO, #34). The engine must not also arm its software
            # entry-stop watch — dual ownership measured live (F6): the
            # watch cancelled the plugin's own placement and a poisoned
            # id scope misread the disposition. Pine-semantics question
            # (OCO vs stop-limit) stays open on card #14.
            entry_stop_limit_native=True,
            # SOFTWARE, not NATIVE (#33): the OCO above is the single-exit
            # bracket ONLY — no DNSE payload can link separate orders into a
            # group (Live-L1-T11: oca members are venue-strangers). Declaring
            # NATIVE suppresses the sync engine's fill-time sibling cancel
            # (_cascade_oca_cancel), stranding the far leg of an oca.cancel
            # ENTRY group as a live working order after the near leg fills.
            oca_cancel=CapabilityLevel.SOFTWARE,
            # Not natively supported by DNSE conditional orders.
            trailing_stop=CapabilityLevel.SOFTWARE,
            partial_qty_bracket_exit=CapabilityLevel.SOFTWARE,
            partial_qty_bracket_exit_pyramiding=CapabilityLevel.SOFTWARE,
            # PUT /orders/{id} amends price+quantity atomically (while New).
            amend_order=CapabilityLevel.NATIVE,
            cancel_all=CapabilityLevel.SOFTWARE,
            reduce_only=CapabilityLevel.SOFTWARE,
            # REST-poll of the order books, not a live push channel.
            watch_orders=CapabilityLevel.SOFTWARE,
            fetch_position=CapabilityLevel.NATIVE,
            # No client-supplied order id in the place payload.
            idempotency=CapabilityLevel.SOFTWARE,
            short_selling=CapabilityLevel.NATIVE,
        )

    # --- live plumbing (REST order path; sub-minute market data is WS, #100) ---

    @override
    async def connect(self) -> None:
        # No persistent ORDER socket to open — the order path is REST, and the
        # #100 sub-minute market-data WS is started lazily inside watch_ohlcv, not
        # here. So connect() only touches the client to log the endpoint banner.
        # NOTE: this validates NOTHING — a dead credential surfaces on the first
        # classified read/write (#68), not here.
        _ = self.client
        self._connected = True
        self._restore_identity_from_journal()

    def _restore_identity_from_journal(self) -> None:
        '''#36: journal-ROOTED restart adoption (Live-L1-T16).

        Restores the three in-memory maps from this run identity's live
        journal rows so a same-label relaunch re-owns its resting venue
        orders (identity, book category, cancellability). Panel rules:
        adoption starts from JOURNALLED rows only — a book row with no
        journal root is FOREIGN (the operator's) and is never touched; a
        conditional parent whose child ref is missing (crash window) gets a
        best-effort child CHASE via the parent detail's externalOrderId;
        idempotent — existing in-memory entries are never overwritten (the
        engine's own store_ctx.replay() re-points envelopes separately).
        '''
        if self.store_ctx is None:
            return
        adopted = 0
        for journal_row in iter_journal_identities(self.store_ctx):
            leg_type = _LEG_TYPE_BY_NAME.get(journal_row.leg_kind)
            primary = journal_row.venue_ids[0] if journal_row.venue_ids else None
            # #87: a terminal-marked row is the EXPOSURE LEDGER (#73/#74 keep
            # it live after a FILL terminal), not a working order. It keeps
            # identity + the _last_seen watermark below (late-event dedup),
            # but must never re-enter _order_ids: prior-session fills adopted
            # under a recurring pine key made every cancel of that key
            # aggregate ALREADY_FILLED (any-fill-wins), so the entry-stop
            # watch read a venue-REJECTED order as "limit won" (measured F6).
            working = journal_row.terminal_status is None
            for venue_id in journal_row.venue_ids:
                if venue_id in self._identity:
                    continue
                self._identity[venue_id] = (journal_row.pine_id or None,
                                            journal_row.from_entry, leg_type)
                self._order_category[venue_id] = (
                    "NORMAL" if venue_id == journal_row.child_id
                    else journal_row.category)
                # #93 G1: the placed shape survives restart — without it a
                # relaunch silently re-opens the bracket-modify CRITICAL.
                self._placed_category.setdefault(
                    venue_id, journal_row.placed_category)
                if working and journal_row.intent_key:
                    self._order_ids.setdefault(
                        journal_row.intent_key, []).append(venue_id)
                adopted += 1
            if journal_row.filled_qty > 0 and journal_row.last_raw_status:
                # #56: seed the fill watermark so the first post-restart poll
                # re-emits nothing. ONLY the filling id (the child, or a
                # NORMAL primary) — never a conditional shell, whose
                # Activated transition must stay un-deduped (#42-A).
                seed_id = journal_row.last_fill_venue_id or (
                    journal_row.child_id if journal_row.child_id is not None
                    else (primary if journal_row.category == "NORMAL" else None))
                if seed_id is not None and seed_id not in self._last_seen:
                    self._last_seen[seed_id] = (journal_row.filled_qty,
                                                journal_row.last_raw_status)
            if (working and primary is not None and journal_row.child_id is None
                    and journal_row.category in ("STOP", "OCO")):
                # Crash-window chase: the parent may have triggered while we
                # were down — its economics live on the un-journalled child.
                try:
                    detail = self._resolve_child_detail(primary,
                                                        journal_row.category)
                except Exception:                                   # noqa: BLE001
                    detail = None        # venue unreachable: next poll retries
                child = (detail or {}).get("externalOrderId")
                if child and str(child) not in self._identity:
                    child = str(child)
                    self._identity[child] = (journal_row.pine_id or None,
                                             journal_row.from_entry, leg_type)
                    self._order_category[child] = "NORMAL"
                    if journal_row.intent_key:
                        self._order_ids.setdefault(
                            journal_row.intent_key, []).append(child)
                    journal_child_ref(self.store_ctx, parent_venue_id=primary,
                                      child_id=child)
                    adopted += 1
        if adopted:
            log.broker_info("%s", (
                f"journal restore: re-owned {adopted} venue id(s) from the "
                f"run's journal rows (#36) — foreign book rows untouched"))
        self._recovery_report()

    def _recovery_report(self) -> None:
        '''#71 (Phase B1): the report-only recovery verdict ladder.

        Loud per-row verdicts for this run's unresolved lost-reply rows
        (quarantine-marked; the still-unknown rule forbids every write) and
        for sibling-label strands (#60 — reported, never adopted, never
        cancelled). WARN, not halt: a halt would also block the operator's
        own recovery commands while preventing nothing the still-unknown
        rule does not already forbid (panel G4).
        '''
        if self.store_ctx is None:
            return
        from pynecore.core.broker.store_helpers import STATE_DISPOSITION_UNKNOWN
        du_rows = [row for row in self.store_ctx.iter_live_orders()
                   if row.state == STATE_DISPOSITION_UNKNOWN]
        evidence_complete = True
        strand_ids: "set[str]" = set()
        try:
            strand_ids = self.store_ctx.foreign_live_exchange_order_ids(
                symbol=self.resolve_contract())
        except Exception:                                           # noqa: BLE001
            evidence_complete = False   # widens doubt, never narrows it
        for verdict in classify_recovery(du_rows=du_rows,
                                         strand_ids=sorted(strand_ids),
                                         evidence_complete=evidence_complete):
            log.broker_warning("%s", verdict.message)
            if verdict.kind == "still_unknown":
                # Quarantine mark: every later reader of the row knows the
                # run trades beside unknown exposure.
                from .journal_wiring import _merged_extras
                self.store_ctx.upsert_order(
                    verdict.subject,
                    extras=_merged_extras(self.store_ctx, verdict.subject,
                                          recovery_verdict="still_unknown"))

    @override
    async def disconnect(self) -> None:
        self._connected = False
        await self._stop_ws_order_source()  # #121: close the WS order feed, if any

    # --- #121 dual-transport WS order feed (ADDITIVE failsafe over the poll) ---

    def _mask_id(self, value) -> str:
        s = str(value or "")
        return ("*" * max(len(s) - 4, 0)) + s[-4:] if s else ""

    def _resolve_investor_id(self) -> "str | None":
        """The account's investorId (the WS broker channel key), from /accounts.

        Cached. Returns None on any failure — the caller degrades to poll-only.
        The account body already carries ``investorId`` (docs
        dnse-get-accounts.md); the same /accounts read that resolves account_no.
        """
        if self._investor_id:
            return self._investor_id
        try:
            status, body = self.client.get_accounts()
        except Exception as exc:                                  # noqa: BLE001
            log.broker_warning("WS order feed: /accounts read failed (%s) — poll-only",
                               type(exc).__name__)
            return None
        accounts = (body.get("accounts") or []) if isinstance(body, dict) else []
        if status != 200 or not accounts:
            return None
        investor_id = accounts[0].get("investorId")
        if investor_id:
            self._investor_id = str(investor_id)
            log.broker_info("[BROKER] resolved investorId=%s for the WS order feed",
                            self._mask_id(investor_id))
        return self._investor_id

    def _ensure_ws_order_source(self) -> None:
        """Kick off the PROD WS order source in the BACKGROUND (never blocking
        the poll loop). Connect + investor-id read run in a detached task, so a
        slow/failed WS handshake can NEVER delay the REST poll floor — the whole
        contract of the #121 failsafe. Idempotent: at most one start in flight;
        after a start failure it backs off ``ws_order_reconnect_interval``. A
        mid-stream drop is handled by the vendored client's ``auto_reconnect``."""
        if self._ws_order_source is not None or self._ws_start_task is not None:
            return
        if not self._enable_ws_order_events:
            if not self._ws_disabled_logged:
                self._ws_disabled_logged = True
                log.broker_info("[BROKER] WS order feed disabled by config — "
                                "poll-only (#121)")
            return
        if time.monotonic() < self._ws_next_retry_monotonic:
            return  # in start-failure backoff; poll is the floor meanwhile
        self._ws_start_task = asyncio.ensure_future(self._start_ws_order_source_bg())

    async def _start_ws_order_source_bg(self) -> None:
        """Background: resolve investor id (off-loop — it is a blocking REST
        read), construct + start the WS source. On success publishes
        ``_ws_order_source``; on any failure logs and arms the backoff. NEVER
        raises out (it is a detached task) except a designed halt."""
        try:
            investor_id = await asyncio.to_thread(self._resolve_investor_id)
            if not investor_id:
                self._ws_next_retry_monotonic = (
                    time.monotonic() + self._ws_reconnect_interval)
                return
            from .ws_order_source import WSOrderSource
            src = WSOrderSource(self.config.api_key, self.config.api_secret,
                                investor_id, self.market_type)
            await src.start()
            self._ws_order_source = src
        except asyncio.CancelledError:
            raise
        except Exception as exc:                                  # noqa: BLE001
            log.broker_warning(
                "WS order feed start failed (%s: %s) — poll remains the floor; "
                "retrying in %ss", type(exc).__name__, exc,
                self._ws_reconnect_interval)
            self._ws_next_retry_monotonic = (
                time.monotonic() + self._ws_reconnect_interval)
        finally:
            self._ws_start_task = None

    async def _collect_ws_order_events(self, timeout: float) -> "list":
        """Wait up to ``timeout`` s for WS order frames and return their
        OrderEvents (deduped via the shared ``_last_seen`` watermark in
        ``_scan_row``). Replaces the poll's fixed sleep. WS off/quiet/down/
        still-connecting -> waits ``timeout`` and returns [] (poll-only
        behaviour, unchanged).

        A WS exception is NEVER allowed to kill the watch loop — the whole point
        of the failsafe is that WS degrades latency, never protection. A
        BrokerManualInterventionError from ``_scan_row`` (the designed halt) is
        the one thing that still propagates, exactly as on the poll path."""
        self._ensure_ws_order_source()      # background start; returns at once
        src = self._ws_order_source
        if src is None:
            await asyncio.sleep(timeout)     # WS not ready -> exactly the old sleep
            return []
        try:
            raw_rows = await src.collect(timeout)
        except asyncio.CancelledError:
            raise
        except Exception as exc:                                  # noqa: BLE001
            log.broker_warning("WS order collect failed (%s: %s) — poll remains "
                               "the floor", type(exc).__name__, exc)
            await asyncio.sleep(timeout)  # keep the poll cadence; no tight loop
            return []
        events = []
        for raw in raw_rows:
            try:
                scanned = await self._scan_row(raw)
            except BrokerManualInterventionError:
                raise                    # designed escalation: the engine halts
            except Exception as exc:                              # noqa: BLE001
                # A poisoned WS frame must not kill detection — the poll re-reads
                # the same order and its watermark advances there instead. NOT
                # marked seen here (the raise skips _scan_row's advance).
                log.broker_warning("WS frame scan raised (%s: %s) — poll will "
                                   "re-detect", type(exc).__name__, exc)
                continue
            events.extend(scanned)
        return events

    async def _stop_ws_order_source(self) -> None:
        task = self._ws_start_task
        self._ws_start_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):           # noqa: BLE001
                pass
        src = self._ws_order_source
        self._ws_order_source = None
        if src is not None:
            try:
                await src.stop()
            except Exception:                                     # noqa: BLE001
                pass

    @property
    @override
    def is_connected(self) -> bool:
        return self._connected

    @override
    async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
        """The engine's one bar-feed entry point (live_runner hard-calls it):
        dispatch by ``feed_mode`` — the parity-proven closed-bar path stays
        byte-identical and isolated from the tick body (#37 S1'). Sub-minute
        timeframes have exactly ONE source (#100): WS per-print synthesis —
        the venue's REST floor is 1m, and ``/trades/latest`` is a sampled
        feed no bar may ever be built from (panel-measured ~10%% capture)."""
        from .provider import DNSEProvider
        if DNSEProvider.is_sub_minute(timeframe):
            return await self._watch_ohlcv_ltf(symbol, timeframe)
        if self._feed_mode == "tick":
            return await self._watch_ohlcv_tick(symbol, timeframe)
        return await self._watch_ohlcv_closed(symbol, timeframe)

    #: #100: seconds of tick silence tolerated OUTSIDE quiet phases before
    #: the LTF feed raises (the runner's retry path). Generous vs the
    #: measured active rate (~18 prints/s) yet far under the watchdog.
    _LTF_OUTAGE_GRACE_S = 45.0

    def _in_feed_quiet_phase(self) -> bool:
        """Provider-declared no-print venue phases (DNSE ATC 14:30-14:45,
        measured L4-T03) — tick silence inside them is NORMAL, never an
        outage. Same table the core staleness clock pauses on (#100/#98)."""
        from .provider import DNSEProvider
        now = datetime.now(timezone(timedelta(hours=7))).strftime("%H:%M")
        return any(start <= now < end
                   for start, end in DNSEProvider.feed_quiet_phases)

    async def _watch_ohlcv_ltf(self, symbol: str, timeframe: str) -> OHLCV:
        """#100: sub-minute closed bars synthesized from the WS per-print
        stream (dv=0 parity vs venue bars, measured offline AND live).

        Contract (panel-adjudicated):
        - CLOSED bars only; empty windows emit nothing (venue shape).
        - Outage (silence past grace outside a quiet phase) RAISES
          ``ExchangeConnectionError`` — never hangs: a hanging feed makes
          the runner's idle-synth fabricate frozen zero-volume bars.
        - Every closed bar is appended to the SEPARATE LTF store
          (``workdir/data/ltf/``) — never the provider's shared ``.ohlcv``,
          which provider-mode warmup atomically truncates.
        - A ``suspect`` bar (cumulative-volume mismatch = missed prints)
          is delivered but LOUDLY logged; the strategy layer decides.
        """
        from pynecore.lib import timeframe as tf_lib
        from pynecore.core.tick_aggregator import TickAggregator
        from .tick_source import WSTickSource
        if self._ltf_state is None:
            wire = str(self._secdef(self.symbol or symbol).get("symbol")
                       or self.symbol or symbol)
            source = WSTickSource(self.config.api_key,
                                  self.config.api_secret, wire)
            await source.start()
            self._ltf_state = {
                "source": source,
                "agg": TickAggregator(int(tf_lib.in_seconds(timeframe))),
                "pending": [],
            }
        state = self._ltf_state
        while True:
            if state["pending"]:
                bar = state["pending"].pop(0)
                if bar.suspect:
                    log.broker_warning(
                        "LTF bar %d is SUSPECT (missed prints — cumulative "
                        "volume mismatch); high/low may be wrong (#100)",
                        bar.time)
                self._persist_ltf_bar(bar, timeframe)
                return OHLCV(timestamp=bar.time * 1000, open=bar.open,
                             high=bar.high, low=bar.low, close=bar.close,
                             volume=bar.volume, is_closed=True)
            try:
                ts, price, qty, cum = await state["source"].next_tick(
                    timeout=self._LTF_OUTAGE_GRACE_S)
            except (asyncio.TimeoutError, TimeoutError):
                if self._in_feed_quiet_phase():
                    continue            # ATC-class silence: normal, wait on
                raise ExchangeConnectionError(
                    f"LTF tick feed silent > "
                    f"{self._LTF_OUTAGE_GRACE_S:.0f}s during an active "
                    f"phase — refusing to fabricate sub-minute bars (#100)")
            state["pending"].extend(
                state["agg"].add(ts, price, qty, cumulative=cum))

    def _persist_ltf_bar(self, bar, timeframe: str) -> None:
        """Append to the SEPARATE LTF store (self-accumulating history —
        the #80 warmup answer). Best-effort: a store failure must never
        stall the live feed; it logs and moves on."""
        try:
            from pynecore.core.ohlcv import OHLCVWriter
            if self._ltf_writer is None:
                root = Path("workdir/output") if not Path("workdir/data").exists() \
                    else Path("workdir/data")
                ltf_dir = root / "ltf"
                ltf_dir.mkdir(parents=True, exist_ok=True)
                # Same naming as the CLI warmup leg (run.py #100 skip):
                # ONE store per (symbol, tf), warmup reads what live wrote.
                path = type(self).get_ohlcv_path(
                    str(self.symbol), timeframe, ltf_dir)
                self._ltf_writer = OHLCVWriter(path, timeframe).open()
            self._ltf_writer.write(OHLCV(
                timestamp=bar.time * 1000, open=bar.open, high=bar.high,
                low=bar.low, close=bar.close, volume=bar.volume,
                is_closed=True))
        except Exception as exc:                              # noqa: BLE001
            log.broker_warning("LTF store append failed (%s: %s) — live "
                               "feed continues, history loses this bar",
                               type(exc).__name__, exc)

    async def _watch_ohlcv_closed(self, symbol: str, timeframe: str) -> OHLCV:
        """Yield the next CLOSED bar by polling REST ``/price/ohlc``.

        #81: poll-failure accounting lives on ``self``, NEVER in coroutine
        locals — the live runner re-enters ``watch_ohlcv`` under a ≤2 s
        ``wait_for``, so each coroutine instance sees ~one poll and a local
        counter can never accumulate (panel-measured). A failed-poll streak
        warns (throttled by re-warn interval); success resets. The
        staleness/wedge half is the CORE watchdog (``feed_timeout_bars``,
        armed at 16 in provider.py) — this ladder covers the
        failing-but-answering venue shape the watchdog cannot attribute.
        """
        resolution = self.to_exchange_timeframe(timeframe)
        period = _TF_SECONDS.get(timeframe, 300)
        while True:
            now = int(time.time())
            status, body = await asyncio.to_thread(lambda: _guard_transport(
                lambda: self.client.get_ohlc(
                    self.market_type,
                    {"symbol": self.symbol, "resolution": resolution,
                     "from": now - period * 5, "to": now})))
            if status != 200 or not isinstance(body, dict):
                self._bar_poll_failures += 1
                if (self._bar_poll_failures >= self._bar_poll_warn_after
                        and (self._bar_poll_failures
                             % self._bar_poll_rewarn_every
                             == self._bar_poll_warn_after
                             % self._bar_poll_rewarn_every)):
                    log.broker_warning(
                        "bar feed: %d consecutive failed OHLC polls "
                        "(last http=%s) — the strategy is not receiving "
                        "prices (#81)",
                        self._bar_poll_failures, status)
            else:
                if self._bar_poll_failures >= self._bar_poll_warn_after:
                    log.broker_info(
                        "bar feed recovered after %d failed polls",
                        self._bar_poll_failures)
                self._bar_poll_failures = 0
            if status == 200 and isinstance(body, dict) and body.get("t"):
                times = body["t"]
                idx = len(times) - 1
                while idx >= 0 and int(times[idx]) + period > now:
                    idx -= 1            # skip the still-forming bar
                if idx >= 0:
                    ts = int(times[idx])
                    if ts > self._last_bar_ts:
                        self._last_bar_ts = ts
                        return OHLCV(
                            timestamp=ts * 1000,
                            open=float(body["o"][idx]), high=float(body["h"][idx]),
                            low=float(body["l"][idx]), close=float(body["c"][idx]),
                            volume=float(body["v"][idx]), is_closed=True)
            await asyncio.sleep(self._bar_poll_interval)

    async def _watch_ohlcv_tick(self, symbol: str, timeframe: str) -> OHLCV:
        """Tick mode (#37): poll ``/trades/latest``, synthesize the developing
        bar, emit ``is_closed=False`` on change; at rollover the venue's
        official closed bar is authoritative — fetched for up to
        ``tick_close_timeout`` seconds, after which the SYNTHESIZED bar is
        closed loudly (who-closed=SYNTH; the L4 red line grades this).

        Emit-ordering guard (#37 panel): no forming update for slot N+1 is
        emitted before slot N's close — a forming bar overtaking its close
        double-increments bar_index engine-side and moves time backwards.
        """
        period = _TF_SECONDS.get(timeframe, 300)
        while True:
            now = time.time()
            slot = int(now - (now % period))
            if self._tick_bar is not None and self._tick_slot < slot:
                # rollover: close slot N before any forming N+1
                if self._tick_close_deadline is None:
                    self._tick_close_deadline = now + self._tick_close_timeout
                official = await asyncio.to_thread(
                    self._tick_fetch_official_close, period)
                if official is not None:
                    return official
                if time.time() >= self._tick_close_deadline:
                    bar, ts = self._tick_bar, self._tick_slot
                    self._tick_bar, self._tick_close_deadline = None, None
                    self._last_bar_ts = ts
                    log.broker_warning(
                        "tick mode: official close for bar %d withheld past "
                        "%.0fs — closing the SYNTHESIZED bar (who-closed=SYNTH; "
                        "expected at session close, Live-L4-T03)",
                        ts, self._tick_close_timeout)
                    return OHLCV(timestamp=ts * 1000, open=bar["o"], high=bar["h"],
                                 low=bar["l"], close=bar["c"], volume=bar["v"],
                                 is_closed=True)
                await asyncio.sleep(self._tick_poll_interval)
                continue
            update = await asyncio.to_thread(self._tick_poll_once, slot)
            if update is not None:
                return update
            await asyncio.sleep(self._tick_poll_interval)

    def _tick_fetch_official_close(self, period: int) -> "OHLCV | None":
        """One attempt to read slot ``self._tick_slot``'s OFFICIAL closed bar."""
        ts = self._tick_slot
        status, body = _guard_transport(lambda: self.client.get_ohlc(
            self.market_type,
            {"symbol": self.symbol,
             "resolution": self.to_exchange_timeframe(str(period // 60 or 1)),
             "from": ts - period, "to": ts + 2 * period}))
        if status != 200 or not isinstance(body, dict) or not body.get("t"):
            return None
        for idx, row_ts in enumerate(body["t"]):
            if int(row_ts) == ts:
                self._tick_bar, self._tick_close_deadline = None, None
                self._last_bar_ts = ts
                return OHLCV(
                    timestamp=ts * 1000,
                    open=float(body["o"][idx]), high=float(body["h"][idx]),
                    low=float(body["l"][idx]), close=float(body["c"][idx]),
                    volume=float(body["v"][idx]), is_closed=True)
        return None

    def _tick_poll_once(self, slot: int) -> "OHLCV | None":
        """One ``/trades/latest`` poll: merge new prints into the forming bar.

        Dedup cursor: ``totalVolumeTraded`` is session-cumulative and monotone
        PER BOARD (measured live 2026-08-25: G1/T1 counters are independent and
        NOT comparable to each other — a global cursor silently dropped valid
        prints as replays). Only ``self._tick_board`` rows are compared/kept;
        strictly-greater-than-cursor (not >=) so a same-volume edge case cannot
        be mistaken for a replay, and cannot double-count either.
        Board filtering: T1 (put-through/block trades) is EXCLUDED — its prices
        are off-market negotiated trades that would contaminate H/L.
        """
        status, body = self.client.get_latest_trade(self.resolve_contract())
        if status == 429:
            if not self._tick_throttled:
                self._tick_throttled = True
                log.broker_warning(
                    "tick mode: /trades/latest throttled (429) — degraded to "
                    "poll-and-hope cadence; forming updates may stall (#37)")
            return None
        if status != 200:
            return None
        if self._tick_throttled:
            self._tick_throttled = False
            log.broker_info("tick mode: /trades/latest throttle cleared")
        rows = (body if isinstance(body, list)
                else (body.get("trades") or body.get("data") or [])
                if isinstance(body, dict) else [])
        changed = False
        board_rows = [r for r in rows if str(r.get("boardId")) == self._tick_board]
        for raw in sorted(board_rows, key=lambda r: float(r.get("totalVolumeTraded") or 0)):
            total = float(raw.get("totalVolumeTraded") or 0)
            if total <= self._tick_cursor:
                continue                      # replayed print from an earlier poll
            price = float(raw.get("matchPrice") or 0)
            qty = float(raw.get("matchQtty") or 0)
            self._tick_cursor = total
            if price <= 0:
                continue
            if self._tick_bar is None or self._tick_slot != slot:
                self._tick_slot = slot
                self._tick_bar = {"o": price, "h": price, "l": price,
                                  "c": price, "v": 0.0}
                self._tick_close_deadline = None
            bar = self._tick_bar
            bar["h"] = max(bar["h"], price)
            bar["l"] = min(bar["l"], price)
            bar["c"] = price
            bar["v"] += qty
            changed = True
        if not changed or self._tick_bar is None:
            return None
        bar = self._tick_bar
        return OHLCV(timestamp=self._tick_slot * 1000, open=bar["o"],
                     high=bar["h"], low=bar["l"], close=bar["c"],
                     volume=bar["v"], is_closed=False)

    # --- price unit codec (#119) ---

    def _wire_scale(self, *, writing: bool) -> float:
        """Feed->wire scale for THIS symbol, with both #119 hard guards.

        Derivatives and indices quote the same number on both sides of the
        boundary (scale 1). A STOCK's order book counts đồng while its feed
        counts thousands (scale 1000), but that factor is applied ONLY when
        the classification is authoritative and stock trading is enabled:

        * **G1** — ``classify_market_type`` answers STOCK for ANY symbol whose
          secdef read came back empty, including a dated derivative contract
          code. Scaling a derivative by 1000 would book a 1000x fill and mint
          SL/TP levels 1000x away — an unprotected position, strictly worse
          than the bug being fixed. A write therefore refuses; a READ cannot
          refuse (it would kill reconcile on one bad poll), so it keeps the
          identity scale — the pre-#119 behaviour — and warns once.
        * **G2** — even an authoritative stock write needs
          ``enable_stock_orders``; the readback units are still unconfirmed.
        """
        try:
            market_type, authoritative = self.classify_market_type()
        except Exception:                                          # noqa: BLE001
            if writing:
                raise            # a write must never guess (G1)
            # A read must never die on one unreadable secdef: treat it as the
            # unprovable case below (warn once, keep the identity scale).
            market_type, authoritative = "STOCK", False
        if market_type != "STOCK":
            return DERIVATIVE_WIRE_SCALE
        if not authoritative:
            if writing:
                raise UnverifiedClassificationError(
                    f"DNSE refuses a STOCK-scaled write on {self.symbol!r}: the "
                    f"classification is a GUESS (secdef carried no "
                    f"securityGroupId), and a guessed STOCK is exactly how a "
                    f"dated derivative contract would get its price multiplied "
                    f"by {STOCK_WIRE_SCALE:.0f} (#119/G1)")
            if not self._unverified_scale_warned:
                self._unverified_scale_warned = True
                log.broker_warning(
                    "price readbacks on %s are UNSCALED: the symbol classifies "
                    "STOCK only by guess (secdef empty or unreadable), so the "
                    "đồng->thousands conversion is not provable — reporting "
                    "venue prices verbatim (#119/G1)", self.symbol)
            return DERIVATIVE_WIRE_SCALE
        if writing and not bool(getattr(self.config, "enable_stock_orders", False)):
            raise StockOrdersDisabledError(
                f"DNSE live STOCK orders are disabled for {self.symbol!r}: set "
                f"enable_stock_orders=true in the plugin config to allow them. "
                f"The wire unit is measured, but the fill/position readback "
                f"units are not yet confirmed by a real stock fill (#119/G2)")
        return STOCK_WIRE_SCALE

    def _wire_price(self, price: float, scale: float) -> float:
        """FEED price -> the quantized WIRE price the order book accepts."""
        return quantize_wire(to_wire(float(price), scale), scale)

    def _from_wire(self, price: "float | None") -> "float | None":
        """WIRE price -> the FEED unit the engine and the strategy speak."""
        if price is None:
            return None
        return from_wire(float(price), self._wire_scale(writing=False))

    # --- order construction ---

    def _to_exchange_order(self, raw: dict) -> ExchangeOrder:
        filled = float(raw.get("fillQuantity") or 0)
        qty = float(raw.get("quantity") or 0)
        stop_price = raw.get("stopPrice")
        # A conditional row carries a stopPrice; a plain LO does not (#78).
        # OrderType has no STOP_LIMIT, so a stop-limit maps to STOP too — the
        # engine's restart reconstruction reads stop_price for a STOP and
        # price for a LIMIT, so a hardcoded LIMIT here rebuilt a re-owned
        # stop entry as a limit AT the stop price (marketable, wrong book).
        # Reachable since #77 made the restart snapshot non-empty.
        has_stop = stop_price is not None and float(stop_price or 0) != 0.0
        # #119: this is the READ funnel — every venue price row the engine ever
        # sees comes through here, so the wire->feed conversion happens once.
        return ExchangeOrder(
            id=str(raw.get("id")),
            symbol=raw.get("symbol") or self.symbol or "",
            side=_DNSE_TO_SIDE.get(raw.get("side", ""), "buy"),
            order_type=OrderType.STOP if has_stop else OrderType.LIMIT,
            qty=qty, filled_qty=filled,
            remaining_qty=float(raw.get("leaveQuantity") or max(qty - filled, 0)),
            price=self._from_wire(float(raw.get("price") or 0) or None),
            stop_price=self._from_wire(float(stop_price) if stop_price else None),
            average_fill_price=self._from_wire(
                float(raw.get("averagePrice") or 0) or None),
            status=_STATUS_MAP.get(
                str(raw.get("orderStatus", "")).upper().replace("_", "").replace("-", ""),
                OrderStatus.PENDING),
            timestamp=int(time.time() * 1000),
            fee=0.0, fee_currency="VND",
        )

    def _marketable_price(self, side: str) -> float:
        """Band-edge price for a market intent — ceiling to buy, floor to sell."""
        ceiling, floor = self._band()
        return ceiling if side == "buy" else floor

    def _band(self) -> tuple[float, float]:
        """(ceilingPrice, floorPrice) for the symbol — the venue's hard price limits."""
        row = self._secdef(self.symbol or "")
        ceiling = float(row.get("ceilingPrice") or 0)
        floor = float(row.get("floorPrice") or 0)
        if not ceiling or not floor:
            raise RuntimeError(
                f"cannot read the price band: secdef has no ceiling/floor for "
                f"{self.symbol!r}")
        return ceiling, floor

    def _stop_already_crossed(self, side: str, stop_price: float) -> bool:
        """Is a stop's trigger condition already TRUE at placement time?

        Pine treats a crossed stop as an IMMEDIATE entry (the backtest oracle
        fills it at the next open — measured 2026-08-18, #34). Detection reads
        the venue's last 1-minute close (one REST call, stop entries only).
        Fails OPEN: any read problem returns False, i.e. the conditional path —
        today's behaviour — so a market-data hiccup can never block an order.
        """
        try:
            now = int(time.time())
            status, body = _guard_transport(lambda: self.client.get_ohlc(
                self.market_type, {
                    "symbol": self.symbol, "resolution": "1",
                    "from": now - 600, "to": now}))
            if status != 200 or not isinstance(body, dict) or not body.get("c"):
                return False
            last = float(body["c"][-1])
        except Exception:                                          # noqa: BLE001
            return False
        return last >= stop_price if side == "buy" else last <= stop_price

    def _stop_fill_price(self, side: str, stop_price: float) -> float:
        """Limit price for the LO a triggered stop emits — trigger + 2x slippage.

        Pine's ``entry(stop=)`` / ``exit(stop=)`` mean a stop-**market**: once the
        trigger prints, you want out (or in) and accept slippage. DNSE has no
        stop-market — every order is an ``LO``, and a conditional order emits that LO
        at ``price`` when ``stopPrice`` is crossed. Posting it *at* the trigger (the
        old behaviour) makes it a stop-**limit**: if price gaps through, the LO never
        fills, so the stop silently does nothing — triggered, unfilled, still exposed.

        The LO is therefore offset **through** the trigger by ``2 x slippage`` ticks
        (the strategy's own ``strategy(slippage=)``, in ticks) so it can cross the
        spread. Doubling gives room for the book to move between trigger and arrival
        while still bounding the worst fill — unlike a band-edge order, which always
        fills but can print up to +/-7% away.

        Falls back to :attr:`stop_slippage_ticks` when the script declares no slippage
        (Pine's default is 0, which would reproduce the never-fills bug), and is
        always clamped into the venue band so the order cannot be rejected.
        """
        from pynecore import lib
        script = getattr(lib, "_script", None)
        ticks = int(getattr(script, "slippage", 0) or 0) * 2
        if ticks <= 0:
            ticks = int(getattr(self.config, "stop_slippage_ticks", 0) or 0)
        offset = ticks * self._mintick()
        price = stop_price + offset if side == "buy" else stop_price - offset
        ceiling, floor = self._band()
        # Band + offset are FEED units; the tick snap happens once, in wire
        # units, inside ``_place`` (#119) — rounding to 0.1 here would quietly
        # coarsen a stock to a 100 đ grid whose real tick is 10 or 50 đ.
        return min(max(price, floor), ceiling)

    def _mintick(self) -> float:
        """Tick size for the traded contract (VN30F1M derivatives: 0.1)."""
        try:
            return float(self.get_symbol_info().mintick) or 0.1
        except Exception:                                          # noqa: BLE001
            return 0.1

    def _gtd(self, days: int = 7) -> str:
        """RFC3339 expiry for a GTD STOP, CLAMPED to the contract's final trade date.

        A derivatives contract stops trading on its ``finalTradeDate``; DNSE rejects any
        order whose GTD reaches past it with ``CO-ORD-006 Validate Order Failed``.
        Blindly adding 7 days therefore makes every native STOP/OCO — including protective
        stop-losses — unplaceable during the last week of each contract month.

        Measured 2026-08-14: VN30F1M's finalTradeDate was 2026-08-20, +7 days gave
        2026-08-21, and every conditional place was refused. The same call succeeded on
        2026-08-13 (+7 = 2026-08-20, exactly the final trade date), which is why this
        surfaced as a sudden, whole-day failure rather than a gradual one.

        **STOCKS keep the plain +days window** (they have no final trade date), so the
        whole clamp is DERIVATIVE-gated. ``durationType=DAY`` is NOT an option for a bare
        STOP: measured on prod 2026-09-14 the venue answers ``400 CO-ORD-004`` for it
        (while GTD 2026-09-17T00:00:00Z, the real final trade date, placed 201).
        """
        now = datetime.now(timezone.utc)
        target = now + timedelta(days=days)
        if self.market_type == "DERIVATIVE":
            target = self._clamp_gtd_to_expiry(target, now=now)
        return target.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _clamp_gtd_to_expiry(self, target: datetime, *, now: datetime) -> datetime:
        """Clamp a candidate GTD into ``[next open day, final trade date]``.

        Two bounds, both of which the old code lacked (#118):

        * **Ceiling** — midnight UTC of the final trade date, which is how DNSE itself
          reports it. Not 23:59Z: the venue reads the date in ICT (UTC+7), so 23:59Z on
          the final date is already 07:00 the NEXT day there and is refused. Measured
          2026-08-14 — GTD 2026-08-20T04:00Z was accepted, 2026-08-20T23:59Z was not.
        * **Floor** — the next open day. Without it a stale/past expiry (an alias-keyed
          secdef cache serving a rolled-away contract, #113) produced a GTD **in the
          past**, which the venue refuses just as hard. The floor winning is itself an
          operator-facing event, so it logs.
        """
        floor = _midnight_utc(expiry.next_open_day_after(now.date()))
        final = self._final_trade_date(now.date())
        if final is not None:
            target = min(target, _midnight_utc(final))
        if target < floor:
            log.broker_warning(
                "GTD floored: final trade date %s is not in the future (now %s) — "
                "using %s. The contract expires today or the secdef is stale (#113); "
                "this conditional may still be refused (#118)",
                final, now.strftime("%Y-%m-%d"), floor.strftime("%Y-%m-%d"))
            target = floor
        return target

    def _final_trade_date(self, today: _date) -> _date | None:
        """The contract's final trade date — venue value first, computed second.

        The venue's ``finalTradeDate`` is INTERMITTENT (present 2026-08-14, absent in the
        2026-08-04 fixture and in the 2026-09-14 live read), so a missing field is the
        normal case, not an error. Resolution order:

        1. ``finalTradeDate`` from the secdef, parsed strictly
           (:func:`expiry.parse_venue_date` — an unknown format now WARNS instead of
           being swallowed by a bare ``except: pass``).
        2. the last venue value seen this run, cached **by the DATED contract code** —
           never by the alias, which is what makes the existing permanent ``_secdef``
           cache dangerous across a roll (#113).
        3. computed from the dated code: 3rd Thursday of its month, walked back off
           weekends/holidays (:func:`expiry.computed_final_trade_date`). Logged once per
           contract, because a computed date carries the holiday-coverage caveat.

        ``None`` when the symbol is not a dated contract and the venue serves nothing —
        the caller then keeps the plain window (fail-open, as before).
        """
        contract = self.resolve_contract()
        cache = getattr(self, "_final_trade_date_cache", None)
        if cache is None:
            cache = self._final_trade_date_cache = {}
        cached = cache.get(contract)
        if cached is not None:
            return cached

        raw = self._secdef(self.symbol or "").get("finalTradeDate")
        try:
            served = expiry.parse_venue_date(raw)
        except ValueError as exc:
            served = None
            log.broker_warning(
                "unreadable finalTradeDate for %s (%s) — falling back to the computed "
                "expiry; a NEW venue date format needs a parser update (#118)",
                contract, exc)
        if served is not None:
            # Cache ONLY under a genuinely dated code: caching under an unresolved
            # alias would rebuild the #113 hazard (a permanent entry surviving a roll).
            if expiry.contract_month(contract) is not None:
                cache[contract] = served
            return served

        computed = expiry.computed_final_trade_date(contract, today=today)
        self._warn_computed_expiry_once(contract, computed)
        return computed

    def _warn_computed_expiry_once(self, contract: str, computed: _date | None) -> None:
        """One loud line per contract when the venue serves no expiry.

        A silent fallback is how #118 survived from 2026-08-04 to 2026-09-14: the plugin
        placed unplaceable conditionals and nothing in the output said why.
        """
        warned = getattr(self, "_computed_expiry_warned", None)
        if warned is None:
            warned = self._computed_expiry_warned = set()
        if contract in warned:
            return
        warned.add(contract)
        if computed is None:
            log.broker_warning(
                "no venue finalTradeDate for %s and it is not a dated VN30 contract "
                "code — GTD stays the plain window, so a conditional placed in expiry "
                "week can be refused with CO-ORD-006 (#118)", contract)
        elif not expiry.holiday_coverage_is_verified(computed):
            log.broker_warning(
                "no venue finalTradeDate for %s — using COMPUTED %s (3rd Thursday). "
                "This is PAST the verified holiday table (%s), so an unlisted (lunar) "
                "closure could make it a day late (#118)",
                contract, computed, expiry.HOLIDAY_TABLE_VERIFIED_THROUGH)
        else:
            log.broker_warning(
                "no venue finalTradeDate for %s — using COMPUTED %s (3rd Thursday, "
                "walked back off weekends/holidays) (#118)", contract, computed)

    def _loan_package_id(self) -> int:
        if self._loan_id is None:
            status, body = self.client.get_loan_packages(self.account_id, self.market_type)
            if status != 200 or not isinstance(body, dict) or not body.get("loanPackages"):
                raise RuntimeError(f"cannot resolve loanPackageId: {status} {body}")
            self._loan_id = body["loanPackages"][0]["id"]
        return self._loan_id

    # --- error handling (see errors.py + docs/plan/dnse-error-handling.md) ---

    def _emit(self, classified, *, action: str, ident: str) -> None:
        """Emit the one structured ``[BROKER]`` line for a classified error."""
        fn = {"error": log.broker_error, "warning": log.broker_warning,
              "info": log.broker_info}.get(classified.level, log.broker_warning)
        fn("%s", classified.log_message(action, ident))

    def _write(self, call):
        """``call(token) -> (status, body)`` — ONE venue write, never a retry.

        The retired "token-reread (retry once)" on INVALID_TRADING_TOKEN was
        a second IDENTICAL write: ``_token()`` reads the state file fresh on
        EVERY call, so the first attempt already carried the freshest token,
        and the measured #51/#46 windows showed retrying (and re-minting)
        never reclaims a refusal — it only doubled writes into the lockout
        (#58). The refusal surfaces through the classify path, whose message
        carries the operator action. #67: a raw transport exception here
        becomes the ``(0, NO_RESPONSE)`` sentinel so the classify path parks
        the write (OrderDispositionUnknownError) instead of killing the run.
        """
        return _guard_transport(lambda: call(self._token()))

    @staticmethod
    def _ident_str(envelope, leg_type) -> str:
        intent = getattr(envelope, "intent", None)
        pine = getattr(intent, "pine_id", None) or "?"
        leg = getattr(leg_type, "value", None) or getattr(leg_type, "name", None) or "?"
        key = getattr(intent, "intent_key", None) or "?"
        return f"{pine}/{leg} intent={key}"

    @staticmethod
    def _coid(envelope, leg_type) -> str:
        kind = _LEG_KIND.get(getattr(leg_type, "name", ""), KIND_ENTRY)
        try:
            return envelope.client_order_id(kind)
        except Exception:
            return getattr(getattr(envelope, "intent", None), "intent_key", "") or ""

    def _raise_write_error(self, status, body, *, action: str, ident: str,
                           coid: str) -> None:
        """Classify a WRITE reply; on failure emit its log line and raise the
        matching ``BrokerError``. Returns quietly on a 2xx success."""
        classified = errors.classify(status, body, is_write=True)
        if classified is None:
            return
        self._emit(classified, action=action, ident=ident)
        disposition = classified.disposition
        detail = f"{classified.code} {classified.message}".strip()
        if disposition is errors.Disposition.MARGIN:
            raise InsufficientMarginError(f"DNSE margin reject on {action}: {detail}")
        if disposition is errors.Disposition.RATE_LIMIT:
            raise ExchangeRateLimitError(f"DNSE rate limit on {action}: {detail}",
                                         classified.retry_after)
        if disposition is errors.Disposition.DISPOSITION_UNKNOWN:
            raise OrderDispositionUnknownError(
                f"DNSE disposition unknown on {action}: {detail}", client_order_id=coid)
        if disposition in (errors.Disposition.AUTH, errors.Disposition.AUTH_TOKEN):
            raise AuthenticationError(f"DNSE auth on {action}: {detail}",
                                      reason=classified.code)
        if disposition is errors.Disposition.CONNECTION:
            raise ExchangeConnectionError(f"DNSE transient on {action}: {detail}")
        # REJECT / SESSION_REJECT (+ any TERMINAL/NOT_FOUND reaching a place/amend)
        raise ExchangeOrderRejectedError(f"DNSE rejected {action}: {detail}")

    def _place(self, envelope, side: str, qty: float, *, price: float,
               category: str = "NORMAL", stop_price: float | None = None,
               stop_order_price: float | None = None, leg_type=None,
               coid_suffix: str = "") -> list[ExchangeOrder]:
        """Place one native order (NORMAL / STOP / OCO) and record its identity.

        #119: this is the WRITE funnel — every price the plugin ever puts on
        the venue's order book is converted and tick-snapped here (and in
        ``_amend_normal``), so intents arrive in the FEED unit and leave in the
        WIRE unit exactly once. ``_wire_scale`` also enforces both hard guards,
        BEFORE the journal row is written and the POST leaves the process.
        """
        scale = self._wire_scale(writing=True)
        payload = {
            "symbol": self.resolve_contract(),   # tradable KRX contract, not the alias
            "side": _SIDE_TO_DNSE[side],
            "orderType": "LO",
            "price": self._wire_price(price, scale),
            "quantity": int(qty),
            "loanPackageId": self._loan_package_id(),
        }
        if category == "STOP":
            payload.update({
                "stopPrice": self._wire_price(stop_price, scale),
                "conditionOperator": ">=" if side == "buy" else "<=",
                "durationType": "GTD",
                "durationDateTime": self._gtd(),
            })
        elif category == "OCO":
            payload.update({
                "stopPrice": self._wire_price(stop_price, scale),
                "stopOrderPrice": self._wire_price(
                    stop_order_price or stop_price, scale),
                "durationType": "DAY",
            })

        ident = self._ident_str(envelope, leg_type)
        coid = self._coid(envelope, leg_type)
        # #123 add-a-leg: a protective EXIT that grows with a partial entry fill
        # places an ADDITIONAL conditional leg for the newly filled slice (the
        # conditional book cannot amend qty — it PARKs, #18/#85/#93 — so a grow
        # is a fresh leg placed alongside the armed one, never a cancel+replace
        # that would bare the whole position). Each extra leg needs its OWN
        # journal row / restart identity, so disambiguate its coid. The coid is
        # DNSE-LOCAL (never sent to the venue — the venue assigns its own id),
        # so a suffix is a safe, collision-free journal key.
        if coid_suffix:
            coid = f"{coid}{coid_suffix}"
        intent = envelope.intent
        # #36 persist-FIRST: the row exists BEFORE the POST leaves the process
        # — a crash in ANY later window leaves an auditable restart bridge.
        journal_submitted(
            self.store_ctx, coid=coid, symbol=payload["symbol"], side=side,
            qty=qty, intent_key=getattr(intent, "intent_key", None),
            pine_id=getattr(intent, "pine_id", None),
            from_entry=getattr(intent, "from_entry", None),
            leg_kind=getattr(leg_type, "name", None),
            category=category, order_type=payload["orderType"],
            # WIRE unit (đồng for stocks), like the venue rows a future
            # recovery matcher would compare it against (#119).
            price=payload.get("price"))
        status, body = self._write(lambda tok: self.client.post_order(
            self.account_id, self.market_type, payload, tok, order_category=category))
        try:
            self._raise_write_error(status, body, action="place", ident=ident,
                                    coid=coid)
        except OrderDispositionUnknownError:
            # #67 join: the reply was lost — the order may exist. phase comes
            # from the sentinel BODY (it never escapes classify, #36 panel).
            sentinel = body if isinstance(body, dict) else {}
            journal_disposition_unknown(
                self.store_ctx, coid=coid, phase=sentinel.get("phase"),
                transport=sentinel.get("transport"))
            raise
        except Exception:
            journal_rejected(self.store_ctx, coid=coid)  # write provably not live
            raise
        if not isinstance(body, dict):
            journal_rejected(self.store_ctx, coid=coid)
            raise ExchangeOrderRejectedError(
                f"DNSE place: non-dict success body: {body!r}")

        order = self._to_exchange_order(body)
        tracked_category = category
        if category == "OCO":
            # The OCO record is an umbrella; its real working order is a spawned
            # NORMAL LO. Track THAT — fills + cancels route by it — via the OCO
            # detail's externalOrderId (which appears ~instantly on activation).
            working = self._resolve_oco_lo(order.id)
            if working is not None:
                order = working
                tracked_category = "NORMAL"  # the live order lives on the NORMAL book
            else:
                # Stale replica: the LO is not visible yet. Track the umbrella
                # (cancel/amend must target the OCO book) and queue it for the
                # watch_orders drain — merely tracking it as "OCO" would park it
                # on a book no scan ever reads and its child's fill would stay
                # invisible (#43).
                self._pending_oco.add(str(order.id))
                log.broker_warning(
                    "OCO %s placed but its working LO is unresolved — queued for "
                    "poll-loop adoption (#43)", order.id)
        journal_server_ref(
            self.store_ctx, coid=coid, venue_id=order.id,
            category=tracked_category,
            umbrella_id=body.get("id") if tracked_category == "NORMAL"
            and category == "OCO" else None)
        key = getattr(intent, "intent_key", None)
        if key:
            self._order_ids.setdefault(key, []).append(order.id)
        self._placed_category[str(order.id)] = category   # #93 G1
        self._identity[order.id] = (
            getattr(intent, "pine_id", None),
            getattr(intent, "from_entry", None),
            leg_type,
        )
        # Cancel/amend must target the book the tracked order actually lives in
        # (a STOP entry -> STOP; an OCO -> its NORMAL working LO). Guessing
        # NORMAL-first let a wrong-book RESOURCE_NOT_FOUND look like a clean cancel.
        self._order_category[order.id] = tracked_category
        return [order]

    def _resolve_child_detail(self, parent_id: str, category: str) -> "dict | None":
        """One 429-aware DETAIL read of a conditional parent (or None).

        Only the DETAIL carries ``externalOrderId`` (the list view omits it).
        The single resolver core behind BOTH watch-loop feeders — the
        Activated-STOP adoption (#42-A) and the pending-OCO drain (#43) — so
        the two paths cannot drift (panel guard on #43).

        ONE read per call — unlike :meth:`_resolve_oco_lo`, which polls 6x
        because it runs at PLACE time inside a synchronous call with no outer
        retry. This runs inside the ``watch_orders`` poll loop, which already
        retries every cycle (see ``_adopt_child``); a nested retry would only
        burn a worker thread and the Detail budget.
        """
        status, body = self.client.get_order_detail(
            self.account_id, parent_id, self.market_type, order_category=category)
        if status == 429:
            # A 429 body is a dict WITHOUT externalOrderId, i.e. it looks
            # exactly like "child not published yet". Retrying a rate-limited
            # endpoint is how a stall becomes an outage — stop, let the poll
            # loop come back later.
            return None
        return body if isinstance(body, dict) else None

    async def _adopt_child(self, parent_id: str, pine_id, from_entry,
                           leg_type, category: str = "STOP"
                           ) -> "tuple[str, dict | None]":
        """Adopt the NORMAL-book child of a conditional parent (STOP or OCO).

        :return: ``("adopted", detail)`` once the child id is known (adopted now,
            or already adopted); ``("dead", detail)`` when the parent is terminal
            WITHOUT ever naming a child — nothing will ever come, the caller
            retires it; ``("pending", detail)`` otherwise — and on ``pending``
            the caller MUST leave the parent row out of ``_last_seen``, because a
            shell's status and fill never change again: a row marked seen here
            would be deduped forever and the child's fill would stay invisible
            (#42-A, the latent form of #39).
        """
        attempts = self._adopt_attempts.get(parent_id, 0) + 1
        self._adopt_attempts[parent_id] = attempts
        # Degrade the cadence past the fast window instead of hammering forever.
        if attempts > self._adopt_fast_polls and attempts % self._adopt_slow_every:
            return ("pending", None)
        detail = await asyncio.to_thread(
            self._resolve_child_detail, parent_id, category)
        external = detail.get("externalOrderId") if detail else None
        if external:
            child = str(external)
            if child not in self._identity:
                self._identity[child] = (pine_id, from_entry, leg_type)
                self._order_category[child] = "NORMAL"
                # #36: the child is a SECOND ref on the parent's journal row —
                # the crash-window chase (journal-ROOTED adoption) needs it.
                journal_child_ref(self.store_ctx, parent_venue_id=parent_id,
                                  child_id=child)
                for ids in self._order_ids.values():
                    if parent_id in ids and child not in ids:
                        ids.append(child)
                log.broker_info(
                    "conditional ACTIVATED -> tracking child | parent=%s child=%s "
                    "pine=%s polls=%d", parent_id, child, pine_id, attempts)
            self._adopt_attempts.pop(parent_id, None)
            return ("adopted", detail)   # adopted now or already adopted
        if detail is not None:
            try:
                terminal = self._to_exchange_order(detail).status in _TERMINAL_STATUSES
            except Exception:                                        # noqa: BLE001
                terminal = False         # unparseable row: keep retrying
            if terminal:
                # Terminal WITHOUT a child: the parent died before spawning its
                # working order (DAY expiry, operator cancel, reject). Retire it
                # instead of grinding to the 2-minute escalation (#43 guard).
                self._adopt_attempts.pop(parent_id, None)
                return ("dead", detail)
        if attempts == 1 or attempts == self._adopt_fast_polls:
            log.broker_warning(
                "conditional [%s] has no externalOrderId after %d poll(s) | "
                "parent=%s pine=%s — the child fill is NOT trackable yet; retrying",
                category, attempts, parent_id, pine_id)
        if attempts >= self._adopt_give_up_polls:
            self._adopt_attempts.pop(parent_id, None)
            raise BrokerManualInterventionError(
                f"DNSE conditional {parent_id} (pine={pine_id}) never published "
                f"its child order id after {attempts} polls: its fill CANNOT be "
                f"tracked and the account may hold an unmanaged position — check "
                f"the venue and flatten manually")
        return ("pending", detail)

    def _resolve_oco_lo(self, oco_id: str, attempts: int = 6, delay: float = 0.15
                        ) -> "ExchangeOrder | None":
        """Return the OCO's working NORMAL LO as an ``ExchangeOrder`` (or None).

        The OCO spawns a NORMAL LO on activation (~instant); the OCO *detail*'s
        ``externalOrderId`` names it (the list view omits it), and the LO's own
        ``metadata.conditionOrderId`` points back. Poll briefly for activation,
        then fetch the LO. (Synchronous; the brief poll blocks the caller.)
        """
        for i in range(attempts):
            status, body = self.client.get_order_detail(
                self.account_id, oco_id, self.market_type, order_category="OCO")
            if status == 429:
                # A 429 body is a dict WITHOUT externalOrderId — keep polling and
                # a stall becomes an outage. Give up: the place-time caller queues
                # the umbrella for the watch-loop drain instead (#43).
                return None
            external = body.get("externalOrderId") if isinstance(body, dict) else None
            if external:
                lo_id = str(external)
                _, detail = self.client.get_order_detail(
                    self.account_id, lo_id, self.market_type, order_category="NORMAL")
                if isinstance(detail, dict):
                    return self._to_exchange_order(detail)
                return self._to_exchange_order({"id": lo_id})
            if i + 1 < attempts:
                time.sleep(delay)   # never after the LAST attempt
        return None

    # --- BrokerPlugin abstracts: execution ---

    @override
    async def execute_entry(self, envelope) -> list[ExchangeOrder]:
        from pynecore.core.broker.models import LegType
        intent = envelope.intent
        if intent.stop is not None:
            if self._stop_already_crossed(intent.side, intent.stop):
                # Crossed at placement (#34): Pine semantics = enter NOW (oracle
                # fills at the next open). A conditional here would either be
                # refused or emit its LO at trigger±slippage — arbitrarily far
                # behind the market. Plain stop -> marketable LO (band edge,
                # the same shape as a market intent); stop-limit -> LO at the
                # user's cap, which may rest (exactly TV's crossed stop-limit).
                log.broker_warning(
                    "crossed stop at placement -> immediate %s LO (Pine: instant "
                    "entry) | %s stop=%s",
                    "capped" if intent.limit is not None else "marketable",
                    self._ident_str(envelope, LegType.ENTRY), intent.stop)
                return self._place(envelope, intent.side, intent.qty,
                                   price=(intent.limit if intent.limit is not None
                                          else self._marketable_price(intent.side)),
                                   leg_type=LegType.ENTRY)
            # stop or stop-limit entry -> native STOP. An explicit ``limit`` is the
            # user asking for a stop-LIMIT, so it is honoured verbatim; a bare
            # ``stop`` means stop-MARKET, which DNSE cannot express, so the emitted
            # LO is priced through the trigger (see ``_stop_fill_price``).
            return self._place(envelope, intent.side, intent.qty,
                               price=(intent.limit if intent.limit is not None
                                      else self._stop_fill_price(intent.side, intent.stop)),
                               category="STOP",
                               stop_price=intent.stop, leg_type=LegType.ENTRY)
        if intent.limit is not None:
            return self._place(envelope, intent.side, intent.qty,
                               price=intent.limit, leg_type=LegType.ENTRY)
        return self._place(envelope, intent.side, intent.qty,
                           price=self._marketable_price(intent.side),
                           leg_type=LegType.ENTRY)

    @override
    async def execute_exit(self, envelope) -> list[ExchangeOrder]:
        """Bracket exit via native OCO (one server-side order), else STOP / LO.

        - tp + sl -> native OCO (``orderCategory=OCO``): the venue places a TP LO
          that auto-amends to the SL price if the SL condition hits, running the
          one-cancels-other server-side. The OCO spawns a working NORMAL LO on
          activation; ``_place`` tracks THAT (via ``externalOrderId``) so fills and
          cancels route by the id that actually acts.
        - sl only -> native STOP; tp only -> NORMAL LO.
        """
        return self._place_exit(envelope)

    def _place_exit(self, envelope, *, coid_suffix: str = "") -> list[ExchangeOrder]:
        """Place ONE protective exit leg (OCO / STOP / LO) for ``envelope``.

        Shared by :meth:`execute_exit` (the first arm) and the #123 add-a-leg
        grow in :meth:`_amend` (each later partial-entry slice). ``coid_suffix``
        disambiguates the extra legs' journal identity — see :meth:`_place`.
        """
        from pynecore.core.broker.models import LegType
        from pynecore.core.broker.exceptions import OrderSkippedByPlugin
        intent = envelope.intent
        tp, sl = intent.tp_price, intent.sl_price
        # A stop-loss must FILL when it fires — price the LO it emits through the
        # trigger, never at it (see ``_stop_fill_price``). The TP leg keeps its exact
        # limit: a take-profit is a limit order by nature and must not slip.
        if tp is not None and sl is not None:
            return self._place(envelope, intent.side, intent.qty, price=tp,
                               category="OCO", stop_price=sl,
                               stop_order_price=self._stop_fill_price(intent.side, sl),
                               leg_type=LegType.TAKE_PROFIT, coid_suffix=coid_suffix)
        if sl is not None:
            return self._place(envelope, intent.side, intent.qty,
                               price=self._stop_fill_price(intent.side, sl),
                               category="STOP", stop_price=sl,
                               leg_type=LegType.STOP_LOSS, coid_suffix=coid_suffix)
        if tp is not None:
            return self._place(envelope, intent.side, intent.qty, price=tp,
                               leg_type=LegType.TAKE_PROFIT, coid_suffix=coid_suffix)
        raise OrderSkippedByPlugin(
            "DNSE plugin cannot express this exit: no tp_price/sl_price "
            "(trailing stops are not implemented)",
            intent_key=getattr(intent, "intent_key", ""))

    @override
    async def execute_close(self, envelope) -> ExchangeOrder:
        from pynecore.core.broker.models import LegType
        intent = envelope.intent
        return self._place(envelope, intent.side, intent.qty,
                           price=self._marketable_price(intent.side),
                           leg_type=LegType.CLOSE)[0]

    def _identity_for(self, order_id: str) -> tuple:
        return self._identity.get(order_id, (None, None, None))

    def _ids_for(self, envelope) -> list[str]:
        key = getattr(envelope.intent, "intent_key", None)
        return list(self._order_ids.get(key, [])) if key else []

    def _order_category_for(self, order_id: str):
        """The book a placed order lives in, recorded at place time (authoritative).

        No record -> None, so a cancel probes EVERY book. There is no safe
        leg-based guess (#45): the old "NORMAL" catch-all made `_cancel_one`
        probe one wrong book whose 404 then read as gone-from-every-book —
        three false cancel_one=True on a live conditional, measured 2026-08-24.
        Even STOP_LOSS cannot narrow to "STOP": an OCO umbrella carries an SL
        leg too. (Identity and category records share a lifecycle — both
        written at place, both in-memory #36 — so a leg-known/category-unknown
        id does not occur in practice anyway.)
        """
        return self._order_category.get(order_id) or None

    @override
    async def execute_cancel(self, envelope) -> bool:
        """Bool contract over the same disposition core as
        :meth:`execute_cancel_with_outcome` (#55).

        DECLARED CHANGE (#55 panel): True now means every id resolved to a
        POSITIVE cancelled-with-no-fill class (``CANCEL_CONFIRMED`` /
        ``TOO_LATE_TO_CANCEL``). Absence-from-every-book and fill-raced
        cancels previously returned True — the same lie the outcome variant
        told the engine. Core collapses both bools to UNKNOWN on the outcome
        path anyway (core/plugin/broker.py), and False means "retry" — the
        safe direction.
        """
        ids = self._ids_for(envelope)
        if not ids:
            return False
        outcomes = [await self._cancel_one_disposition(str(order_id))
                    for order_id in ids]
        return all(outcome in (CancelDispositionOutcome.CANCEL_CONFIRMED,
                               CancelDispositionOutcome.TOO_LATE_TO_CANCEL)
                   for outcome in outcomes)

    def _cancel_dependent_exits(self, entry_order_id: str) -> None:
        """RETIRED — do not call. Kept only as documentation of a measured dead end.

        This plugin-side cascade was reverted on 2026-08-14 after test T5 measured it
        breaking the ENGINE's ownership model live: the engine never asked for the exit
        leg's cancel, so its next sync saw a bot-owned order cancelled on the venue,
        RE-PLACED the exit (a brand-new orphan, id 575506 — the very thing the cascade
        was meant to prevent) and QUARANTINED the account, refusing every further
        dispatch ("Entry dispatch blocked by quarantine ... signal dropped").

        A cancel of a dependent exit must be ENGINE-initiated so its order ids land in
        ``sync_engine._strategy_cancel_expected_ids`` and its intent is retired — i.e.
        the fix for rubycell/pynecore#19 belongs in the engine's diff/cancel path, not
        here. Until then: strategies must cancel exit ids explicitly, as staged test T3
        does.

        Original rationale (still true, just mis-layered): DNSE does not cascade an
        entry cancel to its exit legs.

        DNSE does NOT cascade. Measured 2026-08-13 (rubycell/pynecore#19): an entry and
        its ``strategy.exit`` stop were placed in one OCA group; cancelling only the entry
        left the exit ``New`` at the venue — a naked buy-stop above the market, with no
        position to protect. Left there it does not merely linger: if it triggers it OPENS
        a position on a flat account, turning protection into exposure.

        The plugin already knows the binding — ``_identity`` maps every venue order id to
        ``(pine_id, from_entry, leg_type)`` — so an entry's dependants are exactly the
        tracked orders whose ``from_entry`` is that entry's ``pine_id``. Best-effort and
        idempotent: an already-terminal leg simply reports gone.
        """
        raise NotImplementedError(
            "RETIRED 2026-08-14 — see docstring. The executable cascade was "
            "removed with #55: its _cancel_one dependency was replaced by the "
            "disposition core, and this path must never run anyway.")

    async def _readback_disposition(self, order_id: str, category: str
                                    ) -> CancelDispositionOutcome:
        """Poll the order detail for a POSITIVE terminal classification.

        DNSE answers a conditional cancel with **200 and the order object** —
        an acknowledgement only; a resting STOP stayed ``New`` for >12 s after
        three ACKed cancels (measured 2026-08-13). So the venue itself must
        answer, and the answer must say WHY the order is done: ``Canceled``/
        ``Expired`` with zero fill confirms, any fill is ``ALREADY_FILLED``
        (G6), and a still-working read-back stays ``UNKNOWN`` (G5) so the
        engine retries on its own cadence — never ``STILL_OPEN``, which the
        engine treats as a confirmed cancel (models.py naming trap).

        Client calls run in a worker thread (G9: the thread touches ONLY the
        client — every broker-map mutation stays on the loop side) and pacing
        is ``await asyncio.sleep``, so the loop stays live for the very
        ``watch_orders`` fill feed that resolves the race (#55 panel). A
        detail 404 falls through to the history book: absence is not a
        disposition.
        """
        for attempt in range(self._cancel_verify_attempts):
            status, body = await asyncio.to_thread(
                lambda: _guard_transport(lambda: self.client.get_order_detail(
                    self.account_id, order_id, self.market_type,
                    order_category=category)))
            if status == 200 and isinstance(body, dict):
                try:
                    order = self._to_exchange_order(body)
                except Exception:                                   # noqa: BLE001
                    order = None            # unparseable row: fall through, retry
                if order is not None:
                    outcome = _classify_readback(order.status, order.filled_qty)
                    if outcome is not CancelDispositionOutcome.UNKNOWN:
                        return outcome
            elif errors.code_of(body) in errors.NOT_FOUND_CODES:
                return await self._history_disposition(order_id)
            if attempt + 1 < self._cancel_verify_attempts:
                await asyncio.sleep(self._cancel_verify_delay)
        return CancelDispositionOutcome.UNKNOWN

    def _read_history_rows_sync(self) -> "tuple[list[dict], bool]":
        """Yesterday+today's ``/orders/history`` rows, paginated (#69).

        Returns ``(rows, complete)``: ``complete`` only when the drain
        covered a provable ``total`` with no failed page. A POSITIVE row
        found in an incomplete read is still positive evidence (truncation
        hides rows, never fabricates them) — only ABSENCE claims would need
        ``complete``, and no caller concludes from absence. The endpoint has
        no category parameter (doc/SDK verified, #74 P2): one call covers
        whatever the venue puts there; conditional string ids appearing in
        it is an UNMEASURED premise, so their absence stays inconclusive."""
        today = datetime.now(timezone(timedelta(hours=7))).date()
        rows: list[dict] = []
        page_index = 0
        while True:
            status, body = _guard_transport(
                lambda: self.client.get_order_history(
                    self.account_id, self.market_type,
                    from_date=str(today - timedelta(days=1)),
                    to_date=str(today),
                    page_size=200, page_index=page_index))
            if status != 200 or not isinstance(body, dict):
                return rows, False
            page = body.get("data") or []
            rows.extend(page)
            total = body.get("total")
            if not isinstance(total, int) or total < 0:
                return rows, False       # completeness unprovable
            if len(rows) >= total:
                return rows, True
            if not page or page_index >= 49:
                return rows, False       # dead page / runaway cap
            page_index += 1

    async def _history_disposition(self, order_id: str
                                   ) -> CancelDispositionOutcome:
        """Absence is not a disposition — only a POSITIVE ``/orders/history``
        row may classify an id no book answers for (rows are date-prefixed
        ``20260818_538916`` under ``data``, measured 2026-08-19). No row ->
        ``UNKNOWN``: same-day row timeliness is an UNPROVEN venue premise
        (#55), and this failure shape degrades to a retry, never to a wrong
        terminal verdict. Paginated to exhaustion via the shared reader
        (#69: the old single-page read silently truncated at 200 rows)."""
        rows, _complete = await asyncio.to_thread(self._read_history_rows_sync)
        for row in rows:
            if str(row.get("id", "")).split("_")[-1] != str(order_id):
                continue
            try:
                order = self._to_exchange_order(row)
            except Exception:                                   # noqa: BLE001
                return CancelDispositionOutcome.UNKNOWN
            return _classify_readback(order.status, order.filled_qty)
        return CancelDispositionOutcome.UNKNOWN

    async def _residue_confirm(self, order_id: str) -> "OrderEvent | None":
        """#74: CANCELLED only from a positive, ``_classify_readback``-graded
        history row — a fill on the row outranks its ``Canceled`` status
        string (panel P1: never let a residue verdict contradict executed
        quantity). Everything else -> ``None`` (INCONCLUSIVE)."""
        from pynecore.core.broker.models import OrderEvent
        rows, _complete = await asyncio.to_thread(self._read_history_rows_sync)
        for row in rows:
            if str(row.get("id", "")).split("_")[-1] != str(order_id):
                continue
            try:
                order = self._to_exchange_order({**row, "id": str(order_id)})
                outcome = _classify_readback(order.status, order.filled_qty)
            except Exception:                                   # noqa: BLE001
                return None
            if outcome is not CancelDispositionOutcome.CANCELLED:
                return None
            pine_id, from_entry, leg_type = self._identity_for(order_id)
            journal_terminal(self.store_ctx, venue_id=order_id,
                             terminal_status="Canceled")
            self._last_seen[order_id] = (float(order.filled_qty or 0.0),
                                         "Canceled")
            if pine_id is None:
                log.broker_warning(
                    "residue: order %s is Canceled per history but carries "
                    "no identity — journal closed, no event emitted",
                    order_id)
                return None
            return OrderEvent(
                order=order, event_type="cancelled", fill_price=None,
                fill_qty=None, timestamp=int(time.time()),
                pine_id=pine_id, from_entry=from_entry, leg_type=leg_type)
        return None

    async def _residue_step(self, residue_tracker, present_ids,
                            all_books_readable) -> "list":
        """#74: the residue backstop, one watch cycle. The POPULATION encodes
        the exclusions (panel-adjudicated): exposure-ledger rows
        (``filled_qty>0`` / terminal extras — #73 keeps filled rows LIVE, and
        their ids legitimately leave the day book) are never residue
        subjects; a #41 shell is tracked by its CHILD ref only. At most one
        confirm per cycle, deadline-bounded (a hung history read must not
        cost fill blindness — P2 G5)."""
        if self.store_ctx is None:
            return []
        tracked: dict = {}
        for journal_row in iter_journal_identities(self.store_ctx):
            if journal_row.filled_qty > 0 or journal_row.terminal_status:
                continue
            if journal_row.child_id is not None:
                tracked[str(journal_row.child_id)] = journal_row
            elif journal_row.venue_ids:
                tracked[str(journal_row.venue_ids[0])] = journal_row
        due = residue_tracker.observe(
            tracked_ids=set(tracked), present_ids=present_ids,
            all_books_readable=all_books_readable, now=time.monotonic())
        if due is None:
            return []
        try:
            event = await asyncio.wait_for(
                self._residue_confirm(due), self._watch_read_deadline_s)
        except (asyncio.TimeoutError, TimeoutError):
            event = None                 # hung read = INCONCLUSIVE (G5)
        except Exception as exc:                                # noqa: BLE001
            log.broker_warning("residue confirm raised for %s: %s: %s",
                               due, type(exc).__name__, exc)
            event = None
        warn = residue_tracker.record_confirm(due, concluded=event is not None)
        if warn:
            log.broker_warning("%s", warn)
        return [event] if event is not None else []

    async def _terminal_reject_disposition(self, order_id: str, category: str
                                           ) -> CancelDispositionOutcome:
        """The venue refused the cancel because the order is DONE — read WHY
        before classifying (the old ``TERMINAL_CODES -> treated-gone`` short
        circuit is the #55 double-open). On a conditional book the order may
        be an Activated shell whose economics moved to the NORMAL-book child
        (#41): classification follows the CHILD via ``externalOrderId`` (G2);
        a shell that names no child yet stays ``UNKNOWN``."""
        if category in ("STOP", "OCO"):
            detail = await asyncio.to_thread(
                self._resolve_child_detail, order_id, category)
            child_id = (detail or {}).get("externalOrderId")
            if child_id:
                return await self._readback_disposition(str(child_id), "NORMAL")
            if detail:
                try:
                    order = self._to_exchange_order(detail)
                except Exception:                                   # noqa: BLE001
                    return CancelDispositionOutcome.UNKNOWN
                return _classify_readback(order.status, order.filled_qty)
            return CancelDispositionOutcome.UNKNOWN
        return await self._readback_disposition(order_id, category)

    async def _cancel_one_disposition(self, order_id: str
                                      ) -> CancelDispositionOutcome:
        """Cancel by the recorded book (probe each if unknown, #45) and return
        a POSITIVE-observation disposition.

        Replaces the bool ``_cancel_one``, whose three True paths could not
        tell "cancelled, no fill" from "filled before the cancel landed"
        (#55): a FILLED read-back "took effect", ``TERMINAL_CODES`` was
        "treated-gone" unread, and absence from every book counted as
        success. A write refusal (#51 session binding, transient, reject) is
        a FAILED WRITE, not a disposition (G3) -> ``UNKNOWN`` so the engine
        retries; absence from every book asks the history — never concluded
        from silence.
        """
        hinted = self._order_category_for(order_id)
        categories = [hinted] if hinted else list(_CANCEL_PROBE_BOOKS)
        write_refused = False
        for category in categories:
            status, body = await asyncio.to_thread(
                self._write,
                lambda tok, category=category: self.client.cancel_order(
                    self.account_id, order_id, self.market_type, tok,
                    order_category=category))
            if status in (200, 204):
                # A 2xx from DNSE's cancel is an ACKNOWLEDGEMENT, not a
                # completion — the venue read-back decides the disposition.
                outcome = await self._readback_disposition(order_id, category)
                if outcome is CancelDispositionOutcome.UNKNOWN:
                    log.broker_warning("%s", (
                        f"cancel[{category}] http={status} ACKED but no terminal "
                        f"read-back within the budget -> UNKNOWN so the engine "
                        f"retries | order={order_id}"))
                else:
                    self._pending_oco.discard(order_id)
                    journal_terminal(self.store_ctx, venue_id=order_id,
                                     terminal_status=outcome.value)
                return outcome
            code = errors.code_of(body)
            if code in errors.TERMINAL_CODES:
                log.broker_info("%s", (
                    f"cancel[{category}] code={code} http={status} -> order is "
                    f"done; reading WHY before classifying | order={order_id}"))
                outcome = await self._terminal_reject_disposition(order_id, category)
                if outcome is not CancelDispositionOutcome.UNKNOWN:
                    self._pending_oco.discard(order_id)
                    journal_terminal(self.store_ctx, venue_id=order_id,
                                     terminal_status=outcome.value)
                return outcome
            if code in errors.NOT_FOUND_CODES:  # not in THIS book -> probe the next
                continue
            classified = errors.classify(status, body, is_write=True)
            if classified is not None:
                self._emit(classified, action=f"cancel[{category}]", ident=order_id)
            write_refused = True                # G3: refusal is not a disposition
        if write_refused:
            return CancelDispositionOutcome.UNKNOWN
        outcome = await self._history_disposition(order_id)
        if outcome is not CancelDispositionOutcome.UNKNOWN:
            self._pending_oco.discard(order_id)
        return outcome

    def _prune_order_id(self, order_id: str) -> None:
        """#87 S3: a terminal order leaves ``_order_ids`` by TARGETED removal
        (#85-M2 style — never reassign, a concurrent #41 child must survive)
        so per-key cancel/modify scopes stop aggregating historical
        terminals. ``_identity`` / ``_last_seen`` stay — late-event dedup
        and phantom-shell classification still need them."""
        for tracked_ids in self._order_ids.values():
            try:
                tracked_ids.remove(order_id)
            except ValueError:
                continue

    @override
    async def execute_cancel_with_outcome(self, envelope):
        ids = self._ids_for(envelope)
        if not ids:
            return CancelDispositionOutcome.UNKNOWN
        # Every id the envelope maps to (#47): after an adoption ids is
        # [consumed parent shell, working child]. Aggregation is conservative
        # (cancel_disposition.aggregate): any ALREADY_FILLED wins, then any
        # UNKNOWN keeps the engine retrying, then confirmed-class.
        outcomes = [await self._cancel_one_disposition(str(order_id))
                    for order_id in ids]
        # #87 G-R6 (panel 2/3): the multi-id sweep was invisible — 4 of the 6
        # wasted ids in the measured F6 episode took a silent NOT_FOUND ->
        # history branch. One line per cancel envelope, always.
        log.broker_info(
            "cancel scope %s: ids=%s outcomes=%s",
            getattr(envelope.intent, "pine_id", "?"), list(map(str, ids)),
            [o.value for o in outcomes])
        # #87 S3 (restricted): an id whose cancel ANSWERED terminally is
        # historical — prune it so the next per-key cancel/modify never
        # re-aggregates it (measured: bar 509 re-swept both bar-503/504
        # rejects). UNKNOWN stays mapped for the engine's retry, and
        # ALREADY_FILLED stays mapped too (#95, mirroring _scan_row's own
        # FILLED exemption): pruning a filled parent severed _adopt_child's
        # 'parent_id in ids' join (the #41 child never entered the key
        # scope) and degraded the next per-key ask to UNKNOWN — engine
        # legs stuck in cancel_tentative over an open position (#55).
        for order_id, outcome in zip(ids, outcomes):
            if outcome not in (CancelDispositionOutcome.UNKNOWN,
                               CancelDispositionOutcome.ALREADY_FILLED):
                self._prune_order_id(str(order_id))
        return _aggregate_dispositions(outcomes)

    async def _cancel_replace_entry(self, old, new, order_id: str
                                    ) -> list[ExchangeOrder]:
        """#85: move a conditional ENTRY the only way the venue allows.

        Outcome-gated: ONLY a positive ``CANCEL_CONFIRMED`` proceeds to the
        replacement. ``ALREADY_FILLED`` means the entry executed mid-race —
        replacing would DOUBLE-OPEN; ``TOO_LATE``/``UNKNOWN`` cannot prove
        the old order is gone (measured: the venue serves a cancelled STOP
        as ``New`` for >12 s) — replacing could rest TWO live stops on the
        netting account. All non-positive outcomes raise the
        disposition-unknown PARK (the engine keeps the OLD intent active —
        sync_engine.py:13565 — and the watch events resolve reality;
        self-limiting: a filled entry leaves Pine's book, so the modify is
        not re-attempted). On confirmed cancel the old id is pruned by
        TARGETED removal (reassigning the list would drop a concurrently
        adopted #41 child, panel-probed) and the replacement goes through
        the FULL ``execute_entry`` — persist-first journaling, and the #34
        crossed-at-placement semantics (a replacement whose stop the market
        crossed mid-gap fires immediately, exactly as TV treats a
        modified-to-crossed stop).
        """
        outcome = await self._cancel_one_disposition(order_id)
        if outcome is not CancelDispositionOutcome.CANCEL_CONFIRMED:
            raise OrderDispositionUnknownError(
                f"DNSE conditional entry modify: predecessor cancel not "
                f"positively confirmed (outcome={outcome.value}) — parked, "
                f"old order {order_id} treated as possibly live",
                client_order_id=order_id)
        key = getattr(old.intent, "intent_key", None)
        if key and key in self._order_ids:
            try:
                self._order_ids[key].remove(order_id)
            except ValueError:
                pass
        self._modify_warned_keys.discard(key)
        return await self.execute_entry(new)

    @override
    async def modify_entry(self, old, new) -> list[ExchangeOrder]:
        return await self._amend(old, new, is_exit=False)

    @override
    async def modify_exit(self, old, new) -> list[ExchangeOrder]:
        return await self._amend(old, new, is_exit=True)

    async def _amend(self, old, new, *, is_exit: bool) -> list[ExchangeOrder]:
        ids = self._ids_for(old)
        if not ids:
            return await (super().modify_exit(old, new) if is_exit
                          else super().modify_entry(old, new))
        order_id = str(ids[0])
        # #123 add-a-leg: a protective EXIT whose qty GROWS (same levels) as a
        # partial entry fills cannot be amended on the conditional/OCO book — a
        # qty amend PARKs (#18/#85/#93). Parking would leave the newly filled
        # lot NAKED; a cancel+replace would bare the WHOLE position between the
        # cancel and the replace. Instead place an ADDITIONAL protective leg
        # sized to the delta and leave the armed leg(s) untouched — zero naked
        # window (the delta legs together with the original exactly cover the
        # grown position, so no reversal-through-flat either). Scoped to the
        # conditional (STOP) / OCO books; a NORMAL-book exit (pure TP LO) amends
        # qty in place through ``_amend_normal`` below.
        is_conditional_exit = (
            is_exit and (
                (self._order_category_for(order_id) or "NORMAL") != "NORMAL"
                or self._placed_category.get(order_id) == "OCO"))
        if is_conditional_exit and self._is_pure_qty_grow(old.intent, new.intent):
            return self._add_protective_leg(old, new)
        # #85 (operator-identified, panel-adjudicated): the conditional book
        # CANNOT be amended (#18: HTTP 500 always — re-measured live
        # 2026-09-08), so a per-bar Pine modify silently never reached the
        # venue (the old trigger stayed armed). Conditionals now route:
        # ENTRY -> plugin-local outcome-gated cancel+replace; EXIT -> the
        # proven disposition-unknown PARK (old stop stays armed at the
        # stale level — never naked) plus a loud once-per-key warning.
        # NEVER super().modify_*: it discards the cancel outcome (a cancel
        # the venue answers 'order is done' may mean FILLED — replacing
        # then would double-open), and the venue was measured serving a
        # cancelled STOP as `New` for >12 s.
        if (self._order_category_for(order_id) or "NORMAL") != "NORMAL":
            if is_exit:
                self._park_exit_modify(old, order_id)
            return await self._cancel_replace_entry(old, new, order_id)
        # #93 (CRITICAL, review 2026-09-09): a TP+SL bracket's working child
        # is tracked NORMAL, but its SL lives in the OCO UMBRELLA — no
        # child PUT can move it. Routing by the recorded book alone sent
        # trailing-SL modifies into `_amend_normal`, which diffed the TP
        # (unchanged), wrote NOTHING, and fabricated success. Route by the
        # journal-rooted PLACED shape instead: anything but a pure TP move
        # on an OCO-origin exit takes the #85 loud park (stale bracket
        # stays ARMED — never naked). A TP-only change legitimately amends
        # the child LO's price (its price IS the TP).
        if is_exit and self._placed_category.get(order_id) == "OCO":
            old_i, new_i = old.intent, new.intent
            tp_only = (getattr(old_i, "sl_price", None)
                       == getattr(new_i, "sl_price", None)
                       and int(old_i.qty) == int(new_i.qty))
            if not tp_only:
                self._park_exit_modify(old, order_id)
        return self._amend_normal(old, new, order_id)

    def _park_exit_modify(self, old, order_id: str) -> "None":
        """The #85/#93 loud exit park: warn once per key per EPISODE, then
        raise the disposition-unknown park. ``predecessor_cancel_ids=()``
        is load-bearing (#93 G4): an UNDECLARED modify shape makes the
        engine register EVERY mapped id as engine-initiated-cancel
        expected, so the operator's own app-cancel of the frozen bracket
        would be consumed silently and never fire ``on_unexpected_cancel``.
        Always raises."""
        if old.intent.intent_key not in self._modify_warned_keys:
            self._modify_warned_keys.add(old.intent.intent_key)
            log.broker_warning(
                "conditional/bracket EXIT %s cannot be amended on DNSE "
                "(venue limitation, #18/#85/#93): the protection stays "
                "ARMED at its ORIGINAL level(s) (order %s) — trailing "
                "exits do not move on this venue. Parking the modify; "
                "manage trailing via strategy logic if the stale level "
                "is unacceptable.",
                old.intent.intent_key, order_id)
        raise OrderDispositionUnknownError(
            f"DNSE conditional/bracket exit amend unsupported "
            f"(#18/#93) — parked, old order {order_id} stays armed",
            client_order_id=order_id,
            predecessor_cancel_ids=())

    @staticmethod
    def _is_pure_qty_grow(old_intent, new_intent) -> bool:
        """A protective-exit modify that only RAISES qty (levels unchanged).

        The #123 partial-entry extend: as later slices fill, the engine grows
        the whole-row exit's qty while every price level stays put. Any level
        change (a trailing SL move) is NOT this and still PARKs (#85/#93)."""
        if int(new_intent.qty) <= int(old_intent.qty):
            return False
        return all(
            getattr(old_intent, attr, None) == getattr(new_intent, attr, None)
            for attr in ("sl_price", "tp_price", "limit", "stop"))

    def _add_protective_leg(self, old, new) -> list[ExchangeOrder]:
        """#123: place an EXTRA protective leg for the newly filled slice.

        The armed leg(s) stay LIVE (never cancelled), so the position is never
        bared; the delta leg protects the freshly filled lot(s). Together they
        exactly cover the grown position (no over-protection -> no reversal on a
        netting account). Returns EVERY tracked leg id for the key so the
        engine's order map covers them all — a later engine-initiated cancel of
        this exit then cancels the whole set (``execute_cancel`` iterates
        ``_order_ids[key]``)."""
        key = old.intent.intent_key
        pre_count = len(self._order_ids.get(key, []))
        delta = int(new.intent.qty) - int(old.intent.qty)
        # A fresh envelope carrying ONLY the delta qty at the SAME levels; the
        # coid suffix (session-unique via the leg count) gives it its own
        # journal identity without a venue-visible change.
        leg_env = _dc_replace(new, intent=_dc_replace(new.intent, qty=float(delta)))
        self._place_exit(leg_env, coid_suffix=f"~g{pre_count}")
        ids = list(self._order_ids.get(key, []))
        log.broker_info(
            "#123 extended protection for %s: placed an additional %d-lot leg "
            "for the newly filled slice; %d leg(s) now armed (no cancel — the "
            "existing protection stayed live)", key, delta, len(ids))
        return [self._exchange_order_stub(oid, new.intent) for oid in ids]

    def _exchange_order_stub(self, order_id: str, intent) -> ExchangeOrder:
        """A minimal :class:`ExchangeOrder` naming an already-tracked leg id.

        The engine only reads ``.id`` off a ``modify_exit`` result (to refresh
        its order map); the remaining fields are cosmetic placeholders."""
        return ExchangeOrder(
            id=str(order_id),
            symbol=self.resolve_contract(),
            side=getattr(intent, "side", "sell"),
            order_type=OrderType.LIMIT,
            qty=float(getattr(intent, "qty", 0.0) or 0.0),
            filled_qty=0.0,
            remaining_qty=float(getattr(intent, "qty", 0.0) or 0.0),
            price=None, stop_price=None, average_fill_price=None,
            status=OrderStatus.OPEN, timestamp=0.0, fee=0.0, fee_currency="")

    @staticmethod
    def _intent_price(intent) -> float:
        """The one price a NORMAL-book amend can carry, in FEED units.

        Unrounded since #119: the tick snap belongs in WIRE units (a stock's
        real tick is 10-50 đ = 0.01-0.05 thousands, which ``round(_, 1)``
        flattened to a 100 đ grid). Callers quantize via ``_wire_price``.
        """
        price = (getattr(intent, "limit", None) or getattr(intent, "stop", None)
                 or getattr(intent, "tp_price", None) or getattr(intent, "sl_price", None))
        return float(price) if price else 0.0

    @staticmethod
    def _intent_signature(intent) -> tuple:
        """Every field a modify can carry — the #93 contradiction guard
        compares old vs new on ALL of them, not just the one
        ``_intent_price`` happens to select."""
        return (getattr(intent, "limit", None), getattr(intent, "stop", None),
                getattr(intent, "tp_price", None),
                getattr(intent, "sl_price", None), int(intent.qty))

    def _order_detail_dict(self, order_id: str) -> "dict | None":
        status, body = self.client.get_order_detail(
            self.account_id, order_id, self.market_type, order_category="NORMAL")
        return body if status == 200 and isinstance(body, dict) else None

    def _amend_normal(self, old, new, order_id: str) -> list[ExchangeOrder]:
        """One-changed-field-per-PUT amend (#86, measured live 2026-09-08).

        The venue rejects a PUT changing BOTH price and quantity
        (``400 INVALID_INPUT "Only allow edit order quantity or price"``)
        yet requires the payload to CARRY both keys (omitting quantity ->
        ``400 EDIT_ORDER_QUANTITY_NOT_ENOUGH``). So: diff, then one PUT per
        changed field, price first. The diff runs against the VENUE's
        resting values (one detail read), not the engine's old envelope —
        after an earlier partial application the envelope lies, and a stale
        diff would emit the venue's UNMEASURED no-op PUT shape (#86 panel).
        A modify that changes neither field writes nothing for the same
        reason.

        #119: the diff is computed entirely in WIRE units — the venue detail
        is already wire, the intent is converted the same way ``_place``
        converts it. Mixing the two (detail in đồng vs intent in thousands)
        made every stock diff true, so a NO-CHANGE modify emitted a
        wrong-unit PUT and the half-applied check could never confirm.
        """
        intent = new.intent
        scale = self._wire_scale(writing=True)
        new_price = self._wire_price(self._intent_price(intent), scale)
        new_qty = int(intent.qty)
        detail = self._order_detail_dict(order_id)
        try:
            cur_price = quantize_wire(float(detail.get("price")), scale)  # type: ignore[union-attr, arg-type]
            cur_qty = int(detail.get("quantity"))             # type: ignore[union-attr, arg-type]
        except (AttributeError, TypeError, ValueError):
            # Venue truth unreadable this instant — best effort from the
            # old envelope (correct in every case except a prior
            # half-applied amend, which the next successful read heals).
            cur_price = self._wire_price(self._intent_price(old.intent), scale)
            cur_qty = int(old.intent.qty)
        payloads = []
        if new_price != cur_price:
            payloads.append({"price": new_price, "quantity": cur_qty})
        if new_qty != cur_qty:
            payloads.append({"price": new_price, "quantity": new_qty})
        if not payloads:
            if self._intent_signature(old.intent) != self._intent_signature(intent):
                # #93 contradiction guard: the INTENT changed but the diff
                # produced nothing this path can write — some changed field
                # (an SL living in an umbrella, a leg this book cannot
                # express) is unreachable from here. Fabricating success
                # froze a trailing stop silently for a whole live trade
                # (probe-measured). Park loudly instead.
                log.broker_error(
                    "amend CONTRADICTION on %s: the intent changed (%s -> "
                    "%s) but no NORMAL-book payload can express it — "
                    "parking, never fabricating success (#93)",
                    order_id, self._intent_signature(old.intent),
                    self._intent_signature(intent))
                raise OrderDispositionUnknownError(
                    f"amend cannot express the intent change on "
                    f"{order_id} (#93) — parked, old order stays armed",
                    client_order_id=order_id,
                    predecessor_cancel_ids=())
            return [self._to_exchange_order(detail or {"id": order_id})]
        for leg_index, payload in enumerate(payloads):
            status, body = self._write(lambda tok, _p=payload: self.client.put_order(
                self.account_id, order_id, self.market_type, _p, tok,
                order_category="NORMAL"))
            if status in (200, 201) and isinstance(body, dict):
                last_body = body
                continue
            if leg_index == 0:
                # Nothing applied yet — the pre-#86 failure semantics hold.
                self._raise_write_error(
                    status, body, action="amend",
                    ident=f"{order_id} intent={getattr(intent, 'intent_key', '?')}",
                    coid=order_id)
                raise ExchangeOrderRejectedError(
                    f"DNSE amend: non-dict success body: {body!r}")
            return self._recover_half_applied_amend(order_id, payload, status)
        return [self._to_exchange_order(last_body)]

    def _recover_half_applied_amend(self, order_id: str, qty_payload: dict,
                                    first_status) -> list[ExchangeOrder]:
        """The qty leg failed AFTER the price leg landed: the venue rests at
        NEW price + OLD quantity — a state neither intent describes. Never
        raise here (#86 adjudication): a reject propagates out of ``sync()``
        and kills the run, and a disposition-unknown park is UNRESOLVABLE on
        DNSE (no ``client_order_id`` for the engine's promotion path) while
        the engine promotes the NEW intent regardless. Resolve locally:
        a terminal order means the fill/cancel outran the amend and the
        event stream owns reconciliation; a working order gets ONE retry,
        then a loud half-applied warning — any later modify re-diffs
        against venue truth and self-heals the qty leg.
        """
        detail = self._order_detail_dict(order_id)
        if detail is not None:
            order = self._to_exchange_order(detail)
            if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED,
                                OrderStatus.REJECTED, OrderStatus.EXPIRED):
                log.broker_warning(
                    "amend qty leg overtaken on %s (order already %s, first "
                    "refusal http=%s) — venue truth wins, events reconcile",
                    order_id, order.status, first_status)
                return [order]
            try:
                # Both sides WIRE units: the detail as served, the payload as
                # ``_amend_normal`` built it (#119).
                landed = (int(detail.get("quantity")) == int(qty_payload["quantity"])
                          and quantize_wire(float(detail.get("price")),
                                            self._wire_scale(writing=True))
                          == float(qty_payload["price"]))
            except (TypeError, ValueError):
                landed = False
            if landed:
                # The write reached the venue and only the RESPONSE was lost
                # (transport blip) — a retry here would be the unmeasured
                # no-op PUT shape. Venue truth already matches the intent.
                return [order]
        status, body = self._write(lambda tok: self.client.put_order(
            self.account_id, order_id, self.market_type, qty_payload, tok,
            order_category="NORMAL"))
        if status in (200, 201) and isinstance(body, dict):
            return [self._to_exchange_order(body)]
        log.broker_error(
            "HALF-APPLIED amend on %s: price leg landed, qty leg refused "
            "twice (http=%s then %s) — venue rests at NEW price/OLD qty; "
            "the next modify re-diffs against venue truth and self-heals",
            order_id, first_status, status)
        detail = self._order_detail_dict(order_id)
        return [self._to_exchange_order(detail or {"id": order_id})]

    # --- BrokerPlugin abstracts: state ---

    def _read_book_rows_sync(self, category: str
                             ) -> "tuple[list[dict] | None, object | None]":
        """``(rows, None)`` for a complete book; ``(None, classified)`` on a
        failed read; ``(None, None)`` when pagination is unprovable.

        Drains ``totalPages`` (documented in the orders envelope; ignored
        until #61 — a book past 100 rows silently truncated). None-rows on
        any failed page, and on an over-cap/unparseable ``totalPages`` (G5'):
        a partial drain must never be returned as the book. The failure
        classification is RETURNED, never self-recorded (#54 panel): this
        helper serves both the watch loop and ``get_open_orders``, and only
        the watch loop feeds the feed-health ladder.
        """
        rows: list[dict] = []
        pages = 1
        page_index = 0
        while page_index < pages:
            status, body = _guard_transport(lambda: self.client.get_orders(
                self.account_id, self.market_type, order_category=category,
                page_index=page_index, page_size=100))
            if status != 200 or not isinstance(body, dict):
                # High-frequency poll -> DEBUG so a transient blip stays
                # inspectable without flooding the operator's log.
                log.broker_debug("read:orders[%s] p%s -> transient | http=%s code=%s",
                                 category, page_index, status, errors.code_of(body))
                return None, errors.classify(status, body, is_write=False)
            rows.extend(body.get("orders") or [])
            if page_index == 0:
                page_plan = book_page_count(body.get("totalPages"))
                if page_plan is None:
                    log.broker_debug("read:orders[%s] totalPages=%s -> unprovable",
                                     category, body.get("totalPages"))
                    return None, None
                pages = page_plan
            page_index += 1
        return rows, None

    def _iter_orders(self):
        """Yield raw order rows across NORMAL + conditional books (best-effort).

        A book whose completeness is unprovable yields NOTHING this cycle —
        the 0.5 s watch loop self-heals next poll (change-detector), so the
        tolerant semantics stay here; the strict ANY-book-unreadable raise
        lives in :meth:`get_open_orders`, the decision read (#62 G3')."""
        for category in _CATEGORIES:
            rows, _classified = self._read_book_rows_sync(category)
            if rows is None:
                continue
            yield from rows

    async def _drain_pending_oco(self):
        """Retry child-resolution for OCO umbrellas unresolved at PLACE time.

        Such an umbrella lives on a book ``_CATEGORIES`` never scans, so nothing
        row-driven ever retries it (#43) — this runs every ``watch_orders``
        cycle instead, on the same poll-counted cadence as the Activated-STOP
        adoption. Yields a terminal :class:`OrderEvent` for an umbrella that
        died childless, so the engine's exit intent is released rather than
        staying blind forever.
        """
        from pynecore.core.broker.models import OrderEvent
        for parent_id in list(self._pending_oco):
            pine_id, from_entry, leg_type = self._identity_for(parent_id)
            try:
                adoption, detail = await self._adopt_child(
                    parent_id, pine_id, from_entry, leg_type, category="OCO")
            except BrokerManualInterventionError:
                self._pending_oco.discard(parent_id)
                raise                # designed escalation: the engine halts
            except Exception as exc:                              # noqa: BLE001
                # One odd reply must never kill fill detection for every other
                # order — mirror the per-row guard in ``watch_orders``.
                log.broker_warning(
                    "pending-OCO drain raised for parent=%s: %s: %s — retrying",
                    parent_id, type(exc).__name__, exc)
                continue
            if adoption == "adopted":
                self._pending_oco.discard(parent_id)
            elif adoption == "dead":
                self._pending_oco.discard(parent_id)
                order = self._to_exchange_order(detail)
                event_type = ("cancelled" if order.status is OrderStatus.CANCELLED
                              else "rejected" if order.status is OrderStatus.REJECTED
                              else "filled" if order.status is OrderStatus.FILLED
                              else "cancelled")   # EXPIRED releases as cancelled
                yield OrderEvent(
                    order=order, event_type=event_type, fill_price=None,
                    fill_qty=None, timestamp=int(time.time()),
                    pine_id=pine_id, from_entry=from_entry, leg_type=leg_type)

    async def watch_orders(self):
        """Detect fills/cancels by polling the order books (REST, off-loop).

        #54 feed-health: a persistently failing poll must not leave the feed
        PERMANENTLY SILENT (measured: 18,936 DEBUG-only polls under a dead
        credential). Per-book consecutive-failure counters drive
        feed-attributed warnings (throttled by the ladder), and an ALL-books
        AUTH streak raises the DESIGNED halt — the engine latches
        ``BrokerManualInterventionError`` via ``_record_halt``; any other
        raise would kill the stream task with one log line and no restart.
        The poll is SINGLE-FLIGHT with a wait deadline: a hung socket read
        counts as stuck cycles and is re-awaited, never abandoned per cycle
        (the shared default executor must not fill with dead workers).
        """
        health = FeedHealth(
            warn_after=self._feed_warn_after,
            rewarn_every=self._feed_rewarn_every,
            halt_after=self._feed_halt_after,
            books=tuple(_CATEGORIES))
        residue = ResidueTracker(grace_s=self._residue_grace_s)
        poll_inflight = None
        while True:
            # #121 dual transport: this REPLACES the old top-of-loop
            # ``await asyncio.sleep(self._poll_interval)``. It waits up to one
            # poll period for WS-detected fills and yields them PROMPTLY (deduped
            # against the REST poll via the shared _last_seen watermark inside
            # _scan_row). When the WS is off / quiet / down / still-connecting it
            # simply sleeps ~_poll_interval and returns [], i.e. it degrades to
            # exactly the old sleep — the poll floor's cadence is unchanged. It
            # then ALWAYS falls through to the poll below (a WS frame just makes
            # this cycle return early, adding at most a prompt extra poll during
            # a fill burst; the poll stays single-flight, so no parallel reads).
            for ws_event in await self._collect_ws_order_events(self._poll_interval):
                yield ws_event
            if self._pending_oco:
                try:
                    # Umbrellas queued at place time live on the unscanned OCO
                    # book — drain BEFORE the row scan so an adopted child's
                    # fill can surface in this same cycle (#43).
                    async for pending_event in self._drain_pending_oco():
                        yield pending_event
                    health.record_success("drain")
                except BrokerManualInterventionError:
                    raise            # designed escalation: the engine halts
                except Exception as exc:                          # noqa: BLE001
                    # G7: one drain failure must not kill the stream via the
                    # engine's terminate-on-raise supervisor; persistent
                    # failure escalates through the ladder instead.
                    health.record_failure("drain", type(exc).__name__)
            if poll_inflight is None:
                poll_inflight = asyncio.ensure_future(
                    asyncio.to_thread(self._poll_books_sync))
            try:
                rows, book_outcomes = await asyncio.wait_for(
                    asyncio.shield(poll_inflight), self._watch_read_deadline_s)
                poll_inflight = None
            except (asyncio.TimeoutError, TimeoutError):
                # Single-flight: the SAME read is re-awaited next cycle — a
                # hung socket costs stuck observations, never a growing stack
                # of abandoned worker threads.
                for book in _CATEGORIES:
                    health.record_failure(book, "stuck-read")
                self._emit_feed_warnings(health)
                continue
            except Exception as exc:                              # noqa: BLE001
                poll_inflight = None
                health.record_failure("poll", type(exc).__name__)
                self._emit_feed_warnings(health)
                continue
            health.record_success("poll")
            for book, outcome in book_outcomes.items():
                if outcome is None:
                    health.record_success(book)
                else:
                    failure_kind, failure_is_auth = outcome
                    health.record_failure(book, failure_kind,
                                          is_auth=failure_is_auth)
            self._emit_feed_warnings(health)
            halt_message = health.halt_due()
            if halt_message is not None:
                raise BrokerManualInterventionError(halt_message)
            scan_crash_kind = None
            for raw in rows:
                try:
                    events = await self._scan_row(raw)
                except BrokerManualInterventionError:
                    raise            # designed escalation: the engine halts
                except Exception as exc:                          # noqa: BLE001
                    # G7: a poisoned row must not kill fill detection for
                    # every other order. NOT marked seen -> retried next poll.
                    scan_crash_kind = type(exc).__name__
                    continue
                for event in events:
                    yield event
            if scan_crash_kind is None:
                health.record_success("scan")
            else:
                health.record_failure("scan", scan_crash_kind)
            self._emit_feed_warnings(health)
            try:
                residue_events = await self._residue_step(
                    residue,
                    {str(raw.get("id")) for raw in rows},
                    all(o is None for o in book_outcomes.values()))
            except BrokerManualInterventionError:
                raise                # designed escalation: the engine halts
            except Exception as exc:                              # noqa: BLE001
                # G7: the backstop must never kill fill detection.
                log.broker_warning("residue step raised: %s: %s",
                                   type(exc).__name__, exc)
                residue_events = []
            for residue_event in residue_events:
                yield residue_event

    @staticmethod
    def _emit_feed_warnings(health: FeedHealth) -> None:
        for message in health.warnings_due():
            log.broker_warning("%s", message)

    def _poll_books_sync(self) -> "tuple[list[dict], dict[str, tuple | None]]":
        """One watch cycle's raw rows + per-book outcome (None = healthy;
        otherwise ``(kind, is_auth)`` for the feed-health ladder)."""
        rows: list[dict] = []
        outcomes: dict = {}
        for category in _CATEGORIES:
            book_rows, classified = self._read_book_rows_sync(category)
            if book_rows is None:
                if classified is not None:
                    outcomes[category] = (
                        f"{classified.code} http={classified.http_status}",
                        classified.disposition in _AUTH_DISPOSITIONS)
                else:
                    outcomes[category] = ("unprovable-pagination", False)
            else:
                rows.extend(book_rows)
                outcomes[category] = None
        return rows, outcomes

    async def _scan_row(self, raw: dict) -> "list[OrderEvent]":
        """Process ONE polled row; return its OrderEvents (usually 0 or 1 —
        a multi-slice fill emits one event PER slice, #56).

        Extracted from the watch loop so each row runs under the G7 guard
        (#54) — the control flow is the loop body's, with ``continue``
        translated to ``return []``.
        """
        order_id = str(raw.get("id"))
        order = self._to_exchange_order(raw)
        cumulative = float(raw.get("fillQuantity") or 0)
        # Dedup on the RAW venue status, not the mapped OrderStatus: the
        # map collapses New and Activated to OPEN, which made a stop's
        # trigger transition invisible — the exact moment the child
        # normal-book order must be adopted (#39, measured live 08-18).
        raw_status = str(raw.get("orderStatus") or "")
        previous, prev_status = self._last_seen.get(order_id, (0.0, None))
        if cumulative == previous and raw_status == prev_status:
            return []
        pine_id, from_entry, leg_type = self._identity_for(order_id)
        if pine_id is None:
            # NOT marked seen: identity can arrive later (a stop's child
            # is adopted only at the parent's Activated transition — #39),
            # and a row marked seen pre-adoption would dedup its fill away.
            return []  # not ours (yet)
        if (raw_status.upper() == "ACTIVATED"
                and self._order_category.get(order_id) == "STOP"):
            # Two-book mechanic (CLAUDE.md): Activated = the conditional
            # CLOSED and a NEW order now works the NORMAL book. The fill
            # will arrive under the CHILD's id — adopt it into the
            # parent's identity so the scan's normal path reports it.
            try:
                adoption, _ = await self._adopt_child(
                    order_id, pine_id, from_entry, leg_type)
            except BrokerManualInterventionError:
                raise            # designed escalation: the engine halts
            except Exception as exc:                      # noqa: BLE001
                # One odd reply must never kill fill detection for every
                # other order (this loop has no other per-row guard).
                log.broker_warning(
                    "child adoption raised for parent=%s: %s: %s — retrying",
                    order_id, type(exc).__name__, exc)
                adoption = "pending"
            # "dead" is treated as pending here: an Activated shell stays
            # Activated forever (#41), and if a stale row read Activated
            # while the detail is already terminal, the row itself will
            # report the real status on a later poll via the normal path.
            if adoption != "adopted":
                # Deliberately NOT marked seen: the shell never changes
                # again, so this is the only thing that makes the next
                # poll retry (#42-A).
                return []
            self._last_seen[order_id] = (cumulative, raw_status)
            return []  # events come from the child row, not the shell
        # Any other status retires a pending adoption for this parent.
        self._adopt_attempts.pop(order_id, None)
        delta = max(cumulative - previous, 0.0)
        # WIRE unit (#119) — converted once, at the OrderEvent boundary below
        # and inside ``_fill_slice_events`` (whose slice prices are wire too).
        average_price_wire = float(raw.get("averagePrice") or 0)
        # #56: fill-bearing transitions book PER-SLICE events at the slice's
        # own price (the executions feed) — never the cumulative VWAP. The
        # read is gated on delta > 0 (status-only transitions skip it, P2).
        slice_events: "list[tuple[float, float]]" = []
        if delta > 0:
            slice_events = await self._fill_slice_events(
                order_id, previous, cumulative, average_price_wire)
        # read-before-mark-seen: _last_seen advances only after the slice
        # outcome is decided (the fallback keeps quantity conserved).
        self._last_seen[order_id] = (cumulative, raw_status)
        if order.status in _TERMINAL_STATUSES:
            journal_terminal(self.store_ctx, venue_id=order_id,
                             terminal_status=raw_status, filled_qty=cumulative)
            # #87 S3 (restricted): a NO-FILL terminal (cancelled / rejected /
            # expired) leaves the per-key scope immediately. FILLED stays
            # mapped — a fill-raced cancel must still answer ALREADY_FILLED
            # (#55); the next restore's terminal filter (#87 S2) retires it.
            if order.status is not OrderStatus.FILLED:
                self._prune_order_id(order_id)
            # #93 G3: the exit episode ended — re-arm the once-per-key
            # modify warning so the NEXT position's frozen bracket is loud
            # again (the docstring's per-EPISODE contract, previously
            # process-lifetime for exits).
            pine_id_w, from_entry_w, _leg_w = self._identity.get(
                order_id, (None, None, None))
            if pine_id_w is not None and from_entry_w is not None:
                self._modify_warned_keys.discard(
                    f"{pine_id_w}\x00{from_entry_w}")
        elif delta > 0:
            # Persist the watermark on the live partial (#56/item 5 collapse):
            # a restart seeds _last_seen from the row and re-emits nothing.
            journal_fill_progress(self.store_ctx, venue_id=order_id,
                                  filled_qty=cumulative, raw_status=raw_status)
        event_type = ("filled" if order.status is OrderStatus.FILLED
                      else "partial" if order.status is OrderStatus.PARTIALLY_FILLED
                      else "cancelled" if order.status is OrderStatus.CANCELLED
                      else "rejected" if order.status is OrderStatus.REJECTED
                      # #87 panel precondition: EXPIRED fell through to
                      # "created", so a GTD-expired order never retired its
                      # engine-side tracking. No-fill expiry == cancelled —
                      # but MARKED venue-driven (#94): a bare cancel is
                      # indistinguishable from an operator's, and the
                      # unexpected-cancel policy quarantined the run at the
                      # ordinary 14:45 expiry (probe-measured).
                      else "cancelled" if order.status is OrderStatus.EXPIRED
                      else "created")
        cancel_reason = (CANCEL_REASON_VENUE_EXPIRED
                         if order.status is OrderStatus.EXPIRED else None)
        if delta > 0 and slice_events:
            events = []
            for index, (slice_qty, slice_price) in enumerate(slice_events):
                final_slice = index == len(slice_events) - 1
                events.append(OrderEvent(
                    order=order,
                    event_type=event_type if final_slice else "partial",
                    fill_price=slice_price or None,
                    fill_qty=slice_qty or None, timestamp=int(time.time()),
                    pine_id=pine_id, from_entry=from_entry, leg_type=leg_type,
                    cancel_reason=cancel_reason if final_slice else None))
            return events
        return [OrderEvent(
            order=order, event_type=event_type,
            fill_price=self._from_wire(average_price_wire or None),
            fill_qty=delta or None, timestamp=int(time.time()),
            pine_id=pine_id, from_entry=from_entry, leg_type=leg_type,
            cancel_reason=cancel_reason)]

    async def _fill_slice_events(self, order_id: str, previous: float,
                                 cumulative: float, average_price_wire: float
                                 ) -> "list[tuple[float, float]]":
        '''#56: the per-slice (qty, price) emissions for one fill delta.

        Availability beats precision on EVERY failure path (panel G2): a
        failed/slow/429 executions read falls back to ONE average-priced
        delta event — a fill is never lost to the slice feed. Selection is
        the budget clamp in fill_slices.select_events.

        #119 units: ``average_price_wire`` and every ``lastPrice`` the
        executions feed serves are WIRE prices (đồng for stocks), so the whole
        selection runs in wire units and the RESULT is converted to the feed
        unit here — ``fill_slices`` stays a pure, unit-agnostic selector.
        '''
        fallback = [(max(cumulative - previous, 0.0),
                     from_wire(average_price_wire,
                               self._wire_scale(writing=False)))]
        if time.monotonic() < self._executions_cooldown_until:
            return fallback
        category = self._order_category.get(order_id, "NORMAL")
        try:
            status, body = await asyncio.wait_for(
                asyncio.to_thread(lambda: _guard_transport(
                    lambda: self.client.get_execution_detail(
                        self.account_id, order_id, self.market_type,
                        order_category=category))),
                timeout=self._executions_read_deadline_s)
        except (asyncio.TimeoutError, TimeoutError):
            log.broker_warning("%s", (
                f"slice pricing degraded: executions read timed out -> "
                f"booking the delta at the cumulative VWAP | order={order_id}"))
            return fallback
        if status == 429:
            self._executions_cooldown_until = time.monotonic() + 60.0
            log.broker_warning("%s", (
                f"slice pricing degraded: executions rate-limited -> VWAP "
                f"fallback + 60 s cooldown | order={order_id}"))
            return fallback
        if status != 200:
            log.broker_warning("%s", (
                f"slice pricing degraded: executions read http={status} -> "
                f"booking the delta at the cumulative VWAP | order={order_id}"))
            return fallback
        events = select_events(parse_reports(body), booked_cum=previous,
                               venue_cum=cumulative,
                               average_price=average_price_wire)
        if not events:
            return fallback
        scale = self._wire_scale(writing=False)
        return [(qty, from_wire(price, scale)) for qty, price in events]

    @override
    async def get_open_orders(self, symbol: str | None = None) -> list[ExchangeOrder]:
        """Union of the NORMAL + conditional books; a failed fetch never looks empty."""
        wanted = self.resolve_contract(symbol) if symbol else None
        orders: list[ExchangeOrder] = []
        for category in _CATEGORIES:
            try:
                rows, _classified = await asyncio.wait_for(
                    asyncio.to_thread(self._read_book_rows_sync, category),
                    timeout=self._book_read_deadline_s)
            except (asyncio.TimeoutError, TimeoutError):
                rows = None
            if rows is None:
                # #62 G3': ONE unreadable/unprovable book poisons the whole
                # answer — the old any_ok union returned a PARTIAL book set
                # as complete (false-clean `flat`, restart COID collision).
                log.broker_warning("%s", (
                    f"read:orders[{category}] unreadable/unprovable -> "
                    f"refusing a partial book union"))
                raise ExchangeConnectionError(f"DNSE {category} book unreadable")
            for raw in rows:
                if wanted and raw.get("symbol") != wanted:
                    continue
                order = self._to_exchange_order(raw)
                if order.status not in _TERMINAL_STATUSES:
                    orders.append(order)
        return orders

    @override
    async def get_position(self, symbol: str) -> ExchangePosition | None:
        """Net position for ``symbol`` from ``/positions`` (netting venue)."""
        try:
            status, body = await asyncio.wait_for(
                asyncio.to_thread(lambda: _guard_transport(
                    lambda: self.client.get_positions(
                        self.account_id, self.market_type, POSITIONS_PAGE_SIZE))),
                timeout=self._book_read_deadline_s)
        except (asyncio.TimeoutError, TimeoutError):
            raise ExchangeConnectionError("DNSE positions read timed out")
        if status != 200 or not isinstance(body, dict):
            classified = errors.classify(status, body, is_write=False)
            if classified is not None:
                self._emit(classified, action="read:positions", ident=symbol)
            raise ExchangeConnectionError(f"DNSE positions unavailable: {status}")
        raw_rows = body.get("positions") or body.get("data") or []
        # #57/#62 G1/G2b: judged on the RAW row count, before the CLOSED
        # filter — a truncated page must NEVER look FLAT (None arms the
        # engine's external-flatten wipe). Absent ``total`` (STOCK) never
        # infers truncation.
        if not positions_complete(len(raw_rows), body.get("total")):
            raise ExchangeConnectionError(
                f"DNSE positions page truncated: {len(raw_rows)} rows delivered "
                f"but total={body.get('total')} — refusing to conclude")
        wanted = self.resolve_contract(symbol)
        net, cost = 0.0, 0.0
        for row in raw_rows:
            if row.get("symbol") != wanted or not is_exposure_row(row):
                continue
            size = float(row.get("openQuantity") or row.get("quantity") or 0)
            signed = size if str(row.get("side", "")).upper() in ("NB", "LONG") else -size
            net += signed
            # WIRE unit (#119): the positions book prices in đồng for stocks,
            # like the order book. Converted once, on entry_price below.
            cost += abs(signed) * float(row.get("costPrice")
                                        or row.get("averagePrice") or row.get("price") or 0)
        if net == 0:
            return None
        volume = abs(net)
        # "long"/"short" is the ExchangePosition contract (models.py:322) and the
        # ONLY vocabulary the engine decodes — "buy"/"sell" silently disabled
        # startup size adoption and could halt a defensive-close settle (#49).
        return ExchangePosition(
            symbol=symbol, side="long" if net > 0 else "short", size=volume,
            entry_price=float(self._from_wire(cost / volume) or 0.0) if volume else 0.0,
            unrealized_pnl=0.0, liquidation_price=None,
            leverage=1.0, margin_mode="cross")

    @override
    async def get_balance(self) -> dict[str, float]:
        status, body = self.client.get_balances(self.account_id)
        if status != 200 or not isinstance(body, dict):
            return {}
        derivative = body.get("derivative") or {}
        stock = body.get("stock") or {}
        return {"VND": float(derivative.get("remainSecure")
                             or stock.get("availableCash") or 0)}
