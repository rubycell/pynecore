"""#132 W0 — the naked-position invariant, as PURE logic.

An OPEN, BOT-OWNED position must have at least one RESTING protective order
at the venue that can reduce it. This module answers that question and
NOTHING else: no venue calls, no sqlite, no asyncio, no clock of its own.
Every input arrives in an :class:`Observation`; the clock is passed in.
``naked_watch.py`` is the I/O shell that fills those in.

WHY THE SPLIT (panel, card #132, lens 3): the two cautionary examples in
this repo are both seam failures. ``test_flatten_tool.py``'s disagreeing-
reads pin has to monkeypatch the very reader whose bug it would need to
exercise, because reading and deciding live in one module with no seam; and
an off-session rule is only testable during the 15 minutes a day the clock
says so, unless the phase is injected. Everything below is therefore a pure
function of its arguments, and the pins drive it with no venue at all.

THE VERDICTS, and why there are six rather than two:

  OK           the invariant holds — nothing owned is open, or cover rests
  UNSTOPPED    cover rests, but none of it is stop-class (see below)
  NAKED        owned exposure is open and NOTHING covers it
  UNDETERMINED a read did not answer, or the evidence contradicts itself
  BLIND        the watcher cannot prove it can see (#145) — never OK
  HOLD         outside continuous trading; the population is FROZEN

``UNDETERMINED`` and ``BLIND`` exist because the repo's rule is exit-2-never-
no: a read that FAILED is not evidence of safety. Applied here it also points
inward — a watchdog that cannot prove its own sight must say so rather than
report calm. ``HOLD`` exists because off-session the order books answer 200
with ZERO rows, so every order legitimately "vanishes" and an evaluating
watchdog would alarm on every held position (card #132, HC3).

``UNSTOPPED`` is lens 1's split. The invariant as frozen asks "is anything
resting?", but the operator's real question is "is my STOPLOSS there?", and on
this venue an OCO bracket's only VISIBLE leg is its take-profit child (the
umbrella lives on a book ``_CATEGORIES`` never scans, broker.py:160/326).
So take-profit-only cover satisfies the frozen invariant and still deserves
its own signal — at its own severity, without weakening NAKED.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

#: Exit codes, MEANINGS identical to ``plugins/dnse/tools/venue.py``
#: (0 affirmative / 1 negative / 2 could-not-determine). Duplicated rather
#: than imported because importing ``venue`` would drag config loading and a
#: ``sys.path`` rewrite into this pure module; ``test_naked_watch.py`` pins
#: the two definitions equal so they cannot drift apart silently.
EXIT_OK, EXIT_NEGATIVE, EXIT_UNKNOWN = 0, 1, 2

#: The canonical marker is IMPORTED by the shell from the engine, never
#: retyped. This tag rides ALONGSIDE it: the engine already emits that marker
#: from four sites (sync_engine.py ~7211/~7219/~7227 and the re-place backoff
#: gate at ~16590), so without a distinguishing tag a grader grepping evidence
#: could not tell the sidecar's finding from the engine's own.
SIDECAR_TAG = "#132 W0 SIDECAR"

#: ``extras.leg_kind`` values that are STOP-class protection. Used ONLY to
#: split NAKED from UNSTOPPED — never to decide whether an order is cover.
#: That decision is ``from_entry``, a typed column that is required on
#: ExitIntent and absent by construction on EntryIntent/CloseIntent, so it
#: covers TP, SL, trailing and the partials WITHOUT an allowlist. An
#: allowlist here is safe precisely because being wrong only mislabels a
#: severity; the same allowlist used as the cover predicate was measured
#: wrong (it omitted TRAILING_STOP, tp_partial and trail_partial — three of
#: eight real values — which would have paged a trailing strategy NAKED
#: continuously; card #132 adjudication).
STOP_CLASS_LEG_KINDS = frozenset({
    "STOP_LOSS", "sl_partial", "TRAILING_STOP", "trail_partial",
})

#: Floor for the NAKED-confirm window, in seconds. A FLAT CONSTANT, never
#: derived from the poll cadence: ``broker.py:301-305`` records a 5x-cadence
#: rule producing a MEASURED FALSE CANCEL, and ``residue_detector`` answered
#: it with a flat 30 s. The real window is ``max(this, one bar period)``
#: because a reactively placed exit arms A BAR LATE by measured design
#: (CLAUDE.md, live 2026-09-15) — at 15m that is 900 s of LEGITIMATE
#: nakedness after every entry, and a watchdog that pages on every trade is
#: a watchdog that gets muted.
NAKED_CONFIRM_FLOOR_S = 30.0

#: Session phases in which the invariant IS evaluated. `lunch` and `atc` are
#: here deliberately: a position held across them is exactly as naked as one
#: held at 10:00, and the ATC in particular is when an unprotected position
#: meets the auction print.
EVALUATED_PHASES = frozenset({"continuous", "lunch", "atc"})

#: Phases in which the population is FROZEN (HOLD) — off session an empty book
#: is not evidence of a missing order. Matched on the FIRST WORD, because
#: ``venue.session_phase`` appends a display annotation for holidays
#: ("closed (exchange holiday)") while keeping the token stable.
#:
#: Why an allowlist and not "anything that is not continuous" (review finding
#: F2): ``venue.session_phase`` returns ``"UNKNOWN (ImportError)"`` when the L0
#: import fails (venue.py:81-82). A ``not in CONTINUOUS_PHASES`` test routed
#: that FAILED READ to HOLD — exit 0, silent, for the whole session — which is
#: this module's own law broken in its first gate. A phase we do not recognise
#: is now UNDETERMINED.
#: ONLY `closed`. `lunch` and `atc` were here on the assumption that the books
#: answer empty off-session — but that measurement was taken on a WEEKEND
#: (live_test/README.md:387), and T9 (README:179) shows a PendingCancel row
#: served THROUGH lunch, so the lunch books are populated. Holding there meant
#: exit 0 and silence for 90 minutes at lunch and 15 at ATC, the latter being
#: exactly when a naked position is live to the auction print. The confirm
#: window already absorbs order-queueing noise, so evaluating those phases costs
#: nothing and buys back 105 minutes of coverage per day.
KNOWN_OFF_SESSION_PHASES = frozenset({"closed"})


class Verdict(Enum):
    OK = "OK"
    UNSTOPPED = "UNSTOPPED"
    NAKED = "NAKED"
    UNATTRIBUTED = "UNATTRIBUTED"
    UNDETERMINED = "UNDETERMINED"
    BLIND = "BLIND"
    HOLD = "HOLD"

    @property
    def exit_code(self) -> int:
        if self is Verdict.NAKED:
            return EXIT_NEGATIVE
        if self in (Verdict.UNDETERMINED, Verdict.BLIND, Verdict.UNATTRIBUTED):
            return EXIT_UNKNOWN
        return EXIT_OK


@dataclass(frozen=True)
class RestingOrder:
    """One order the VENUE reports as working, plus what we know of it.

    ``from_entry`` and ``leg_kind`` come from the run journal (attribution);
    everything else from the venue record. ``classifiable`` is False when the
    venue detail read did not answer for this order — which must never be
    confused with "it is not protection".
    """
    venue_id: str
    side: str                      # "buy" | "sell"
    qty: float
    owned: bool
    from_entry: str | None = None
    leg_kind: str | None = None
    classifiable: bool = True
    book: str | None = None        # log content only, never a gate
    stop_price: float | None = None  # log content only, never a gate


@dataclass(frozen=True)
class CoverUmbrella:
    """An OCO umbrella, reduced to what the invariant needs.

    The core stays pure: the SHELL reads the OCO book and maps
    ``venue.Umbrella`` onto this, exactly as it maps working orders onto
    :class:`RestingOrder`. ``state`` is the venue-resolved ARMED / SPENT /
    UNKNOWN, decided by the umbrella's CHILD — never by its own row, which
    reads ``Activated`` with a populated ``stopPrice`` in all three cases
    (measured 2026-09-18: armed, spent-by-cancel and spent-by-fill are
    byte-identical at the umbrella).
    """

    id: str
    state: str
    stop_price: float | None
    side: str | None
    qty: float | None


def umbrella_covers(u: CoverUmbrella, exposure: float) -> bool:
    """Does this umbrella actually protect THIS exposure?

    ARMED is necessary and not sufficient: an umbrella on the same side as the
    exposure would ADD to the position, not reduce it. Direction is the whole
    content of "cover" — the same reason a reduce-side entry order is not
    cover in :func:`is_cover`.
    """
    if u.state != "ARMED" or not exposure:
        return False
    side = (u.side or "").upper()
    if side in ("NS", "SELL"):
        return exposure > 0          # a sell protects a long
    if side in ("NB", "BUY"):
        return exposure < 0          # a buy protects a short
    return False                     # unrecognised side -> never assume cover


@dataclass(frozen=True)
class Observation:
    """Everything one evaluation cycle needs, already read.

    ``None`` means "the read did not answer" for every field that has it —
    never "nothing there".
    """
    phase: str
    contract_proven: bool
    position_signed: float | None
    owned_exposure: float | None
    resting: tuple[RestingOrder, ...] | None
    bar_period_s: float = 0.0
    contract: str | None = None
    #: OCO umbrellas read from the venue this cycle, or ``None`` for NOT READ.
    #: The default is deliberately the doubting value: a caller that forgets to
    #: supply it gets "we have not consulted the OCO book" rather than "there
    #: are none", so a missing read can never be mistaken for proven absence.
    umbrellas: tuple[CoverUmbrella, ...] | None = None


@dataclass(frozen=True)
class Assessment:
    verdict: Verdict
    reason: str
    exposure: float = 0.0
    covers: tuple[str, ...] = ()
    stop_class_covers: tuple[str, ...] = ()
    covered_qty: float = 0.0


def reduces(side: str, exposure_signed: float) -> bool:
    """Can an order of this side REDUCE that exposure?

    The necessary half of the cover test (candidate S2). Necessary but never
    sufficient: a flip strategy's reverse-entry stop is reduce-side by
    construction, and ENTRY rows were 85 of 155 in the measured journal, so
    a side test ALONE accepts an entry order as protection — false SILENCE,
    the one direction this tool must not have.
    """
    if exposure_signed > 0.0:
        return side == "sell"
    if exposure_signed < 0.0:
        return side == "buy"
    return False


def is_cover(order: RestingOrder, exposure_signed: float) -> bool:
    """Does this resting order protect that exposure?

    All five conditions, in the order they were adjudicated (card #132):
    owned by us, classifiable, attributed as an EXIT by the journal
    (``from_entry``), and able to reduce the exposure. The venue already
    answered "resting" by returning it, and the shell has already removed
    #41 phantom shells.
    """
    if not order.owned or not order.classifiable:
        return False
    if not order.from_entry:
        return False
    return reduces(order.side, exposure_signed)


def owned_exposure_at_venue(
        venue_signed: float, owned_signed: float,
) -> tuple[float, bool]:
    """-> (exposure, determinate). The venue-clamped, bot-owned slice.

    Mirrors ``_clamp_adoption_to_owned`` (sync_engine.py:4542) with ONE
    deliberate difference, and it is the difference that matters here.

    That clamp returns ``0.0`` when the journal and the venue disagree about
    the SIGN, because for ADOPTION "do not guess" correctly means "claim
    nothing". Read by a watchdog, the same ``0.0`` means "we own nothing, all
    clear" — so the process would go SILENT over a position it cannot
    attribute, which is exactly the #135-class corruption it exists to catch.
    Identical arithmetic, inverted safety meaning. Here a sign disagreement
    is ``determinate=False`` -> UNDETERMINED, and stays loud.

    The two agreeing cases are unchanged: a flat VENUE is proof of absence
    (nothing to protect) whatever the journal believes, and owning nothing
    means any open position is foreign — neither is our invariant.
    """
    if venue_signed == 0.0 or owned_signed == 0.0:
        return 0.0, True
    if (venue_signed > 0.0) != (owned_signed > 0.0):
        return 0.0, False
    magnitude = min(abs(venue_signed), abs(owned_signed))
    return (magnitude if venue_signed > 0.0 else -magnitude), True


def evaluate(obs: Observation) -> Assessment:
    """The whole invariant, as a pure function of one observation."""
    if not obs.contract_proven:
        return Assessment(
            Verdict.BLIND,
            "cannot prove the watcher can SEE: the traded contract was not "
            "resolved from a successful instruments read (#145 — a cached "
            "alias makes every position read answer FLAT, silently)")

    phase_token = (obs.phase or "").split()[0] if (obs.phase or "").strip() else ""
    if phase_token not in EVALUATED_PHASES:
        if phase_token in KNOWN_OFF_SESSION_PHASES:
            return Assessment(
                Verdict.HOLD,
                f"phase={obs.phase!r} — population FROZEN (off session the "
                f"books answer 200 with zero rows, so an absent order is not "
                f"evidence)")
        # F2: an UNRECOGNISED phase is a FAILED READ, not an off-session one.
        # `venue.session_phase` answers "UNKNOWN (ImportError)" when the L0
        # import fails, and routing that to HOLD would mean exit 0 and silence
        # for an entire session.
        return Assessment(
            Verdict.UNDETERMINED,
            f"session phase is UNRECOGNISED ({obs.phase!r}) — that is a FAILED "
            f"READ, not an off-session window, and it must not be read as a "
            f"reason to stop looking")

    if obs.position_signed is None:
        return Assessment(
            Verdict.UNDETERMINED,
            "the venue position read did not answer (or two reads disagreed) "
            "— NOT evidence of flat, and NOT evidence of naked")

    if obs.owned_exposure is None:
        return Assessment(
            Verdict.UNDETERMINED,
            "attribution UNAVAILABLE: the run journal could not be read, so "
            "ownership of any open position is unproven (never a clean pass)")

    # F1 (review finding, and the docstring above used to overstate this): the
    # POSITION read is independent of engine belief; OWNERSHIP is not — it is
    # 100% the engine's journalled `filled_qty`. So an UNJOURNALLED fill reads
    # as foreign. The measured shape: a STOP entry journals the umbrella id and
    # its normal-book child is adopted only at the Activated poll, so if the
    # engine DIES between trigger and adoption (#39/#120 — the very case this
    # sidecar exists for) `filled_qty` stays 0 while the account holds a real
    # position. Collapsing that into exposure==0 printed [OK] every cycle over
    # a naked position. It gets its own token, and the string OK never appears.
    exposure, determinate = owned_exposure_at_venue(
        obs.position_signed, obs.owned_exposure)
    if not determinate:
        return Assessment(
            Verdict.UNDETERMINED,
            f"journal and venue disagree about the SIGN "
            f"(venue={obs.position_signed:+g}, journal={obs.owned_exposure:+g}) "
            f"— exposure unattributable; refusing to read that as 'own nothing'")

    # The venue holds MORE than our journal can account for. At owned == 0 this
    # is the #39/#120 shape — a stop entry whose normal-book child the engine
    # never adopted because it died between trigger and adoption — but the same
    # mechanism produces a NONZERO remainder whenever only part of the position
    # is journalled: a pyramiding stop entry, or the #105 frozen-2 flip, with
    # the engine down for one of the legs. The first cut tested `owned == 0.0`
    # exactly and so graded "venue +2, journal +1 covered" as a clean [OK],
    # silently carrying an unaccounted contract.
    unaccounted = abs(obs.position_signed) - abs(obs.owned_exposure)
    if unaccounted > 1e-9:
        return Assessment(
            Verdict.UNATTRIBUTED,
            f"the venue holds {obs.position_signed:+g} but our journal accounts "
            f"for only {obs.owned_exposure:+g} — {unaccounted:g} contract(s) "
            f"unattributed. Cannot tell an operator position from our own "
            f"UNJOURNALLED fill (engine down between a stop trigger and the "
            f"child's adoption, #39/#120). NOT a clean account.",
            exposure=obs.position_signed)

    if exposure == 0.0:
        return Assessment(
            Verdict.OK,
            f"nothing bot-owned is open (venue={obs.position_signed:+g}, "
            f"journal={obs.owned_exposure:+g})", exposure=0.0)

    if obs.resting is None:
        return Assessment(
            Verdict.UNDETERMINED,
            f"owned exposure {exposure:+g} is open but the working-order "
            f"books did not answer — cover unknown, which is not cover "
            f"absent", exposure=exposure)

    covers = tuple(o for o in obs.resting if is_cover(o, exposure))
    covered_qty = sum(o.qty for o in covers)
    if covers:
        stop_class = tuple(
            o.venue_id for o in covers if o.leg_kind in STOP_CLASS_LEG_KINDS)
        ids = tuple(o.venue_id for o in covers)
        if stop_class:
            return Assessment(
                Verdict.OK,
                f"exposure {exposure:+g} covered by {len(covers)} resting "
                f"order(s) ({covered_qty:g}), {len(stop_class)} stop-class",
                exposure, ids, stop_class, covered_qty)
        # #152: the stop MAY be on the OCO book, which `get_open_orders` never
        # scans. Resolve it rather than reporting could-not-see as a verdict
        # the ladder resets on — that combination produced 33 UNSTOPPED lines
        # and ZERO alarms on 2026-09-18 while a bracket was genuinely absent.
        if obs.umbrellas is None:
            return Assessment(
                Verdict.UNSTOPPED,
                f"exposure {exposure:+g} has cover ({covered_qty:g}) but NO "
                f"stop-class leg among it, and the OCO BOOK WAS NOT READ — a "
                f"take-profit is not a stoploss, and an umbrella's stop leg is "
                f"invisible to the NORMAL/STOP books. This is could-not-see, "
                f"NOT proven absence: do not read it as naked.",
                exposure, ids, (), covered_qty)
        protecting = tuple(u for u in obs.umbrellas
                           if umbrella_covers(u, exposure))
        if protecting:
            return Assessment(
                Verdict.OK,
                f"exposure {exposure:+g} covered by {len(covers)} resting "
                f"order(s) ({covered_qty:g}); the stop is on the OCO book — "
                f"umbrella {protecting[0].id} at {protecting[0].stop_price}",
                exposure, ids, tuple(u.id for u in protecting), covered_qty)
        return Assessment(
            Verdict.NAKED,
            f"exposure {exposure:+g} has cover ({covered_qty:g}) but it is "
            f"TAKE-PROFIT ONLY: the OCO book WAS read and holds no armed "
            f"umbrella protecting this side. Nothing is below this position.",
            exposure, ids, (), covered_qty)

    # No proven cover. Before calling it naked, an order we could not
    # classify must poison the verdict rather than its own candidacy: a
    # reduce-side owned order whose detail read failed MIGHT be the
    # protection. ``venue.classify_working`` files such an order under LIVE,
    # which is the safe direction for ``venue.py flat`` (it cries wolf) and
    # the WRONG one here (it would be admitted as cover), so the shell marks
    # it unclassifiable and this branch degrades to could-not-determine.
    # Ordering matters: a PROVEN cover above already returned, so this can
    # never make a genuinely covered position look uncertain.
    unclassifiable = tuple(
        o.venue_id for o in obs.resting
        if o.owned and not o.classifiable and reduces(o.side, exposure))
    if unclassifiable:
        return Assessment(
            Verdict.UNDETERMINED,
            f"exposure {exposure:+g} has no PROVEN cover, but "
            f"{len(unclassifiable)} owned reduce-side order(s) could not be "
            f"classified ({', '.join(unclassifiable)}) — one of them may be "
            f"the protection", exposure)

    foreign = sum(1 for o in obs.resting if not o.owned)
    return Assessment(
        Verdict.NAKED,
        f"exposure {exposure:+g} is OPEN and UNPROTECTED: zero owned "
        f"reduce-side exit orders rest at the venue "
        f"({len(obs.resting)} working order(s) seen, {foreign} foreign)",
        exposure)


@dataclass
class AlarmLadder:
    """Confirm-then-throttle, so W0 neither cries wolf nor spams.

    TWO separate jobs, deliberately not one:

    * **confirm** — a NAKED verdict must persist for ``window_s`` of wall
      clock before it is an alarm. A reactively placed protective exit arms
      A BAR LATE by measured design, so every entry produces a legitimately
      naked interval; without this the tool pages on every trade and is
      muted within a day. The window is ``max(NAKED_CONFIRM_FLOOR_S, one bar
      period)`` and is never derived from the poll cadence.
    * **throttle** — once alarming, re-warn every ``rewarn_every`` cycles
      rather than every cycle, the ladder ``feed_health.py`` already uses
      (warn_after / rewarn_every), so a multi-hour outage costs bounded log
      volume. It is also the only thing standing between an unlisted
      exchange holiday (the session table is clock-only) and an all-day
      alarm storm.

    The FIRST suppressed cycle logs its reason, so a genuinely naked position
    that lands inside the confirm window is still visible in the transcript
    rather than silently swallowed.
    """
    window_s: float
    rewarn_every: int = 12
    _since: float = field(default=0.0)
    _armed: bool = field(default=False)
    _suppressed_logged: bool = field(default=False)
    _cycles_since_warn: int = field(default=0)

    def observe(self, verdict: Verdict, now: float) -> tuple[str | None, str | None]:
        """-> (alarm_line, note). Both may be None; the note is informational."""
        if verdict is not Verdict.NAKED:
            self._since = 0.0
            self._armed = False
            self._suppressed_logged = False
            self._cycles_since_warn = 0
            return None, None

        if self._since == 0.0:
            self._since = now
        held_for = now - self._since

        if not self._armed:
            if held_for < self.window_s:
                if not self._suppressed_logged:
                    self._suppressed_logged = True
                    return None, (
                        f"NAKED observed but WITHHELD for up to "
                        f"{self.window_s:.0f}s: a reactively placed exit arms "
                        f"a bar late by design, so this is the expected "
                        f"post-entry window. Alarms if it persists.")
                return None, None
            self._armed = True
            self._cycles_since_warn = 0
            return (f"CONFIRMED naked for {held_for:.0f}s "
                    f"(window {self.window_s:.0f}s)"), None

        self._cycles_since_warn += 1
        if self._cycles_since_warn >= self.rewarn_every:
            self._cycles_since_warn = 0
            return f"STILL naked after {held_for:.0f}s", None
        return None, None


def stale_numeric_id_verdict(
        journal_side: str | None, journal_qty: float | None,
        venue_side: str | None, venue_qty: float | None,
        venue_created_ms: float | None = None,
        day_start_ms: float | None = None,
) -> str:
    """-> "owned" | "unclassifiable". Corroborates a PRIOR-DAY numeric id.

    The problem (review finding N1) is a genuine fork, and both branches are
    silent in one direction:

    * Day-scope numeric ids OUT of the owned set (the #96 guard against DNSE
      REUSING NORMAL ids across days) and an OVERNIGHT bracket — real cover,
      resting, with yesterday's journal row — stops counting, so a protected
      position grades NAKED and re-warns all morning.
    * Day-scope them IN and a prior-day id the venue REISSUED today to the
      operator's order is read as OUR cover: false SILENCE over a naked
      position, the direction this tool must never have.

    The id alone cannot separate them, so corroborate with a record we have
    already fetched: OUR order, still resting, still has the side and quantity
    the journal recorded for it. A reissued id belongs to a different order and
    will generally differ in at least one. A mismatch is NOT reclassified as
    foreign (that would quietly restore the false alarm) — it is
    ``unclassifiable``, which poisons the cycle to UNDETERMINED.

    Deliberately NOT compared: price/stop level. A trailing stop legitimately
    moves, so a level mismatch would condemn exactly the leg type that protects
    a held position best.

    ``venue_created_ms`` is the decisive one and closes the coincidence side+qty
    alone cannot: derivative quantity is almost always 1, so on an id hit the
    corroboration degrades to side-only, and "our overnight ``501 sell 1`` was
    cancelled with the engine down, then the venue reissued ``501`` to the
    operator's ``sell 1``" would read as our cover over a naked position. A
    venue record CREATED TODAY under an id our journal recorded on a PRIOR day
    is a reissue by definition — the order we journalled cannot have been
    created after we wrote it down.
    """
    if day_start_ms is not None:
        if venue_created_ms is None:
            # ABSENT or UNPARSABLE createdDate is could-not-determine, not a
            # pass. The first cut ran this clause only `if venue_created_ms is
            # not None`, so a record without the field fell through to side+qty
            # — which on derivatives is effectively side-only, since quantity is
            # almost always 1. The doc marks createdDate OPTIONAL
            # (dnse-get-order-detail.md:208, required=false), so the harm state
            # is reachable without anything going wrong: our overnight
            # `501 sell 1` cancelled with the engine down, DNSE reissues 501 to
            # the operator's `sell 1` (#96), the detail record omits the date,
            # and the reissued order reads as OUR cover over a naked position.
            return "unclassifiable"
        if float(venue_created_ms) >= float(day_start_ms):
            return "unclassifiable"
    if journal_side is None or venue_side is None:
        return "unclassifiable"
    if str(journal_side).lower() != str(venue_side).lower():
        return "unclassifiable"
    if journal_qty is None or venue_qty is None:
        return "unclassifiable"
    if abs(float(journal_qty) - float(venue_qty)) > 1e-9:
        return "unclassifiable"
    return "owned"


def confirm_window_s(bar_period_s: float,
                     arm_grace_s: float | None = None) -> float:
    """How long a NAKED verdict must persist before it is an alarm.

    ``max(floor, one bar period)`` unless the caller DECLARES its vehicle's arm
    grace, in which case ``max(floor, that)``.

    WHY THE DECLARATION EXISTS (measured 2026-09-18). Deriving the window from
    the bar period assumes the vehicle arms its protection a bar late, which is
    true only of a REACTIVE exit — one gated on ``position_size > 0``, which
    cannot be placed until the fill is visible. A PRE-PLACED bracket arms on the
    fill: l2c's umbrella existed **0.874 s** after its entry child. Run A was
    @5m, so the derived window was **300 s** while the exposure was unprotected
    for **66.25 s** — a grace 4.5x longer than the entire episode. Four NAKED
    verdicts were observed, every one withheld, no alarm file was ever created,
    and the operator learned about the naked position from a human reading a
    monitor. A grace calibrated for a vehicle shape that is not being run is
    indistinguishable from having no alarm at all.

    ``None`` means NOT DECLARED and keeps the old derivation exactly, so this
    cannot silently re-tune a vehicle that genuinely needs a bar. The floor
    still applies to a declared value: a zero grace would page on the arm gap
    of every entry, and an alarm that fires on every trade is muted within a
    day — after which the real one is invisible too.
    """
    if arm_grace_s is not None:
        return max(NAKED_CONFIRM_FLOOR_S, float(arm_grace_s or 0.0))
    return max(NAKED_CONFIRM_FLOOR_S, float(bar_period_s or 0.0))


def worst_exit_code(seen: "list[Verdict] | tuple[Verdict, ...]") -> int:
    """Loop-mode exit code. ALARM outranks could-not-determine.

    A deliberate deviation from ``venue.py``'s ``max()`` precedence, with the
    MEANINGS unchanged: a run that alarmed must never report as merely
    inconclusive because some later read failed. Unanimous on the panel.
    """
    if any(v is Verdict.NAKED for v in seen):
        return EXIT_NEGATIVE
    # UNATTRIBUTED belongs here for the same reason it is exit 2 per-cycle, and
    # leaving it out made LOOP MODE — the mode Friday actually runs — exit 0
    # after an unattributed cycle while `--once` correctly answered 2. The
    # policy was encoded in two places and only one of them was updated; they
    # are now derived from one source.
    if any(v.exit_code == EXIT_UNKNOWN for v in seen):
        return EXIT_UNKNOWN
    return EXIT_OK
