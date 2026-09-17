"""#132 W0 — pins for the naked-position sidecar.

Every pin below names the WRONG IMPLEMENTATION it catches. A pin no wrong
implementation can fail pins nothing, and four of the five minimums this card
originally proposed were exactly that (lens 3): "naked -> alarm" is a smoke
test no ``is_cover`` mutation can redden, "protected -> no alarm" is passed by
a side-test-only implementation, and "heartbeat cadence" pins nothing at all.

!! MUTATION-TESTING WARNING — READ BEFORE YOU REDDEN ANYTHING !!
``naked_position.py`` and ``naked_watch.py`` are PATH-LOADED here
(``importlib.util.spec_from_file_location``), so a mutant run writes
``plugins/dnse/tools/__pycache__/*.pyc`` that can SURVIVE the source restore
and keep executing. ``inspect.getsource()`` CANNOT detect it — it reads the
``.py`` while the code object came from the ``.pyc``, so it prints the correct
source and gives a false all-clear. Measured 2026-09-17 on ``flatten.py``,
where it nearly got a correct fix declared broken. So: run mutants with
``PYTHONDONTWRITEBYTECODE=1`` (or move that ``__pycache__`` to
``backup/deleteable/`` before the restore run), and read every colour
BEHAVIOURALLY — call the function and assert its return value.
"""
import importlib.util
import pathlib
import sys
import time

import pynecore.lib as lib

lib.bar_index = 0

from pynecore_dnse import broker as bmod

_TOOLS = pathlib.Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    """Path-load a tool module, REGISTERED in ``sys.modules``.

    The registration is not optional here, unlike in ``test_flatten_tool.py``:
    ``naked_position.py`` uses ``@dataclass`` under
    ``from __future__ import annotations``, and ``dataclasses`` resolves those
    string annotations via ``sys.modules.get(cls.__module__).__dict__``. For an
    unregistered module that lookup returns ``None`` and the class body dies
    with ``AttributeError: 'NoneType' object has no attribute '__dict__'`` at
    import time.
    """
    spec = importlib.util.spec_from_file_location(
        f"dnse_{name}", _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ORDER MATTERS, and getting it wrong cost a non-discriminating pin.
# `naked_watch.py` puts tools/ on sys.path and does `import naked_position as
# core`, registering it in sys.modules under the PLAIN name. If the test also
# path-loaded it as "dnse_naked_position" there would be TWO distinct module
# objects with the same source: patching one (a mutant, a monkeypatch) would
# leave the shell running the other, and `RestingOrder` would be two different
# classes that only happen to duck-type alike. So load the SHELL first and take
# the core object it actually uses.
watch = _load("naked_watch")
core = sys.modules["naked_position"]
Verdict = core.Verdict


def _order(venue_id="prot-1", side="buy", qty=1.0, owned=True,
           from_entry="E", leg_kind="STOP_LOSS", classifiable=True):
    return core.RestingOrder(venue_id=venue_id, side=side, qty=qty, owned=owned,
                             from_entry=from_entry, leg_kind=leg_kind,
                             classifiable=classifiable)


def _obs(**over):
    base = dict(phase="continuous", contract_proven=True, position_signed=-1.0,
                owned_exposure=-1.0, resting=(), bar_period_s=0.0)
    base.update(over)
    return core.Observation(**base)


# === the cover predicate — the panel's actual disagreement ==================

def __test_reduce_side_entry_order_is_NOT_cover__():
    """THE pin that discriminates the whole panel (lens 3; it was missing
    from the card's stated minimum set).

    Catches: the venue-only reduce-side implementation (candidate S2 alone).
    A flip strategy's reverse-entry stop is reduce-side BY CONSTRUCTION — buy
    while short — and ENTRY rows were 85 of 155 in the measured journal. An
    implementation that tests only the side accepts it as protection and
    reports OK over a naked position: false SILENCE, the one direction this
    tool must never have.

    Mutation: drop the ``from_entry`` clause from ``is_cover`` -> this
    reddens (verdict becomes OK).
    """
    entry_stop = _order(venue_id="entry-stop-1", side="buy", from_entry=None,
                        leg_kind="ENTRY")
    assert not core.is_cover(entry_stop, -1.0), (
        "a reduce-side ENTRY order was accepted as protection")
    assert core.evaluate(_obs(resting=(entry_stop,))).verdict is Verdict.NAKED


def __test_protective_exit_IS_cover__():
    """The control, so an implementation that simply never finds cover (the
    trivially 'safe' watchdog that alarms always) cannot pass the pin above.

    Catches: ``is_cover`` hardwired to False / ``evaluate`` hardwired to NAKED.
    """
    assessment = core.evaluate(_obs(resting=(_order(),)))
    assert assessment.verdict is Verdict.OK, assessment.reason
    assert assessment.covers == ("prot-1",)


def __test_trailing_and_partial_legs_are_cover__():
    """Catches: the ``leg_kind`` ALLOWLIST implementation (candidate S1).

    The allowlist written on this very card — {STOP_LOSS, TAKE_PROFIT,
    sl_partial} — omitted THREE of the eight real values: TRAILING_STOP
    (models.py:129), tp_partial and trail_partial (store_helpers.py:287-297).
    A trailing-stop strategy would have been paged NAKED continuously by the
    shipped watchdog. The adjudicated predicate is ``from_entry``, which is
    required on ExitIntent and therefore covers every protective leg WITHOUT
    an allowlist.

    Mutation: replace the ``from_entry`` test with
    ``leg_kind in {"STOP_LOSS","TAKE_PROFIT","sl_partial"}`` -> this reddens.
    """
    for leg_kind in ("TRAILING_STOP", "trail_partial", "tp_partial", None):
        order = _order(leg_kind=leg_kind)
        assert core.is_cover(order, -1.0), (
            f"an exit leg journalled with leg_kind={leg_kind!r} was not "
            f"counted as cover — an enumerative predicate has drifted")


def __test_take_profit_only_cover_is_UNSTOPPED_not_OK__():
    """Lens 1's verdict split. An OCO bracket's only VISIBLE leg is its
    take-profit child (the umbrella lives on a book ``_CATEGORIES`` never
    scans, broker.py:160), so TP-only cover satisfies the frozen existence
    invariant while the stoploss may be absent.

    Catches: collapsing UNSTOPPED into OK, which would answer "protected" to
    an operator whose actual question is "is my STOPLOSS there?".
    """
    assessment = core.evaluate(_obs(resting=(_order(leg_kind="TAKE_PROFIT"),)))
    assert assessment.verdict is Verdict.UNSTOPPED, assessment.reason
    assert assessment.verdict.exit_code == core.EXIT_OK, (
        "UNSTOPPED must not escalate to the NAKED exit code — the frozen "
        "invariant is existence; this is a second, softer signal")


def __test_foreign_order_is_NOT_cover__():
    """The netting account is SHARED. Catches: dropping the ``owned`` test,
    which would let the operator's own resting order silence the watchdog."""
    foreign = _order(venue_id="operator-42", owned=False)
    assert not core.is_cover(foreign, -1.0)
    assert core.evaluate(_obs(resting=(foreign,))).verdict is Verdict.NAKED


# === evidence that did not answer is never evidence of safety ==============

def __test_sign_disagreement_is_UNDETERMINED_never_silent__():
    """Catches: reusing ``_clamp_adoption_to_owned`` (sync_engine.py:4542)
    WHOLESALE.

    That clamp returns 0.0 when journal and venue disagree about the sign,
    which is right for ADOPTION ("do not guess" = claim nothing) and inverts
    here: 0.0 reads as "we own nothing, all clear", so the watchdog goes
    SILENT over a position it cannot attribute — the #135-class corruption it
    exists to catch. Identical arithmetic, opposite safety meaning.

    Mutation: ``return 0.0, True`` for the opposing-sign branch -> verdict
    becomes OK and this reddens.
    """
    assessment = core.evaluate(_obs(position_signed=1.0, owned_exposure=-2.0))
    assert assessment.verdict is Verdict.UNDETERMINED, assessment.reason
    assert assessment.verdict.exit_code == core.EXIT_UNKNOWN


def __test_unreadable_position_is_UNDETERMINED_not_flat__():
    """exit-2-never-no. Catches: treating a failed/disagreeing position read
    as 0.0, which is the 2026-08-19 'account is FLAT' report that the whole
    venue toolkit exists to prevent."""
    assert core.evaluate(_obs(position_signed=None)).verdict is Verdict.UNDETERMINED


def __test_attribution_unavailable_is_UNDETERMINED_not_a_clean_pass__():
    """The #91 vacuous-pass guard. Catches: reading an unreadable journal as
    an empty owned set, i.e. 'we own nothing, so nothing can be naked'."""
    assert core.evaluate(_obs(owned_exposure=None)).verdict is Verdict.UNDETERMINED


def __test_unreadable_books_are_UNDETERMINED_not_NAKED__():
    """A book that did not answer is not an empty book. Catches: defaulting
    ``resting`` to () on a failed read, which manufactures a false alarm on
    every transient venue error (and trains the operator to ignore it)."""
    assessment = core.evaluate(_obs(resting=None))
    assert assessment.verdict is Verdict.UNDETERMINED, assessment.reason


def __test_unclassifiable_reduce_side_order_poisons_the_verdict__():
    """Two-way pin (lens 1 + lens 3's 18b). ``venue.classify_working`` files
    an order whose detail read FAILED under LIVE — safe for ``venue.py flat``
    (it cries wolf), wrong here (it would be admitted as cover).

    (a) alone, an unclassifiable reduce-side owned order must yield
        UNDETERMINED: it MIGHT be the protection, so neither NAKED nor OK.
        Catches: treating unclassifiable as not-cover -> NAKED (false alarm),
        and as cover -> OK (false silence).
    (b) alongside a PROVEN cover it must NOT downgrade the verdict, or the
        safe rule produces a permanently-exit-2, always-silent watchdog.
    """
    murky = _order(venue_id="murky-1", classifiable=False, from_entry=None)
    assert core.evaluate(_obs(resting=(murky,))).verdict is Verdict.UNDETERMINED
    both = core.evaluate(_obs(resting=(murky, _order())))
    assert both.verdict is Verdict.OK, (
        "an unclassifiable order downgraded a verdict that had PROVEN cover — "
        "that rule makes the watchdog permanently inconclusive")


def __test_blind_watcher_is_never_OK__():
    """#145: a cached unresolved alias makes every position read answer FLAT,
    silently and forever. Catches: any path where unproven sight still
    evaluates the invariant — it would report OK over a real position."""
    assessment = core.evaluate(_obs(contract_proven=False, position_signed=None))
    assert assessment.verdict is Verdict.BLIND
    assert assessment.verdict.exit_code == core.EXIT_UNKNOWN


def __test_off_session_FREEZES_rather_than_evaluating__():
    """With the books CLOSED an absent order is not evidence of a missing one,
    so the population is frozen rather than evaluated. Catches: alarming on
    every position held overnight.

    NOTE — this pin originally used `lunch`, on the premise that every
    non-continuous phase has empty books. Review round 2 traced that premise to
    a measurement taken on a WEEKEND (live_test/README.md:387), while T9
    (README:179) shows a PendingCancel row served THROUGH lunch. The contract
    changed deliberately: only `closed` freezes, and lunch/atc are evaluated by
    `__test_lunch_and_atc_are_EVALUATED_not_held__`. The assertion was not
    weakened to go green — the behaviour it described was wrong.
    """
    assessment = core.evaluate(_obs(phase="closed", resting=()))
    assert assessment.verdict is Verdict.HOLD, assessment.reason


def __test_flat_venue_is_OK_even_when_the_journal_claims_exposure__():
    """A flat VENUE is proof of absence whatever stale journal rows say —
    measured: a finished run sums to -2 with every contributing row
    Filled-but-unclosed (#89). Catches: alarming off journal belief alone,
    which would page continuously against an idle account."""
    assessment = core.evaluate(_obs(position_signed=0.0, owned_exposure=-2.0))
    assert assessment.verdict is Verdict.OK, assessment.reason


# === the alarm ladder ======================================================

def __test_naked_is_WITHHELD_for_the_confirm_window_then_alarms__():
    """Catches: alarming on the first NAKED cycle.

    A reactively placed protective exit arms A BAR LATE by measured design
    (CLAUDE.md, live 2026-09-15), so every entry produces a legitimately
    naked interval — 900s at 15m. A watchdog that pages on every trade is
    muted within a day, which is the same outcome as never building it.
    """
    ladder = core.AlarmLadder(window_s=30.0)
    alarm, note = ladder.observe(Verdict.NAKED, 1000.0)
    assert alarm is None, "alarmed on the FIRST naked cycle"
    assert note and "WITHHELD" in note, (
        "the first suppressed cycle must say so, or a real naked position "
        "inside the window leaves no trace in the transcript")
    assert ladder.observe(Verdict.NAKED, 1020.0)[0] is None, "alarmed early"
    alarm, _ = ladder.observe(Verdict.NAKED, 1031.0)
    assert alarm and "CONFIRMED" in alarm, "never alarmed after the window"


def __test_recovery_inside_the_window_disarms_the_ladder__():
    """Catches: a latching timer that alarms for a position which got its
    protection a few seconds later — the normal, healthy case."""
    ladder = core.AlarmLadder(window_s=30.0)
    ladder.observe(Verdict.NAKED, 1000.0)
    ladder.observe(Verdict.OK, 1010.0)
    assert ladder.observe(Verdict.NAKED, 1015.0)[0] is None, (
        "the confirm window did not restart after the position was covered")


def __test_confirm_window_is_at_least_one_bar_period__():
    """Catches: a flat 30s constant (too short at 15m, where the legitimate
    arming gap is 900s) and, in the other direction, a cadence-derived window
    (broker.py:301-305 records a 5x-cadence rule producing a MEASURED false
    cancel; residue_detector answered it with a flat floor)."""
    assert core.confirm_window_s(0.0) == 30.0
    assert core.confirm_window_s(60.0) == 60.0
    assert core.confirm_window_s(900.0) == 900.0


def __test_alarm_outranks_could_not_determine_in_loop_exit__():
    """Catches: ``max()`` precedence copied from ``venue.py``, under which a
    run that ALARMED reports as merely inconclusive because a later read
    failed — the loudest finding demoted by the quietest."""
    assert core.worst_exit_code([Verdict.OK, Verdict.NAKED,
                                 Verdict.UNDETERMINED]) == core.EXIT_NEGATIVE
    assert core.worst_exit_code([Verdict.OK, Verdict.UNDETERMINED]) == core.EXIT_UNKNOWN
    assert core.worst_exit_code([Verdict.OK, Verdict.HOLD]) == core.EXIT_OK


def __test_exit_codes_match_the_venue_toolkit__():
    """Drift guard. The pure core redefines these rather than importing
    ``venue`` (which would drag config loading into it), so nothing but this
    pin stops the two definitions from diverging — at which point every
    wrapper grading this tool silently misreads it."""
    venue = _load("venue")
    assert (core.EXIT_OK, core.EXIT_NEGATIVE, core.EXIT_UNKNOWN) == (
        venue.EXIT_OK, venue.EXIT_NEGATIVE, venue.EXIT_UNKNOWN)


# === the shell: sight, journal, phantoms, heartbeat ========================

def _broker(fake_client, tmp_path, **responses):
    base = {"get_security_definition": (200, [{"ceilingPrice": "2100",
                                               "floorPrice": "1800",
                                               "securityGroupId": "FU"}]),
            "get_loan_packages": (200, {"loanPackages": [{"id": 42}]})}
    base.update(responses)
    config = bmod.DNSEBrokerConfig(
        api_key="k", api_secret="s", account_no="ACC001", trading_token="tok",
        token_file=str(tmp_path / "missing_token.json"))
    instance = bmod.DNSEBroker(symbol="VN30F1M", timeframe="1", config=config)
    instance._client = fake_client(**base)
    return instance


_CATALOGUE = (200, {"data": [{"symbolType": "VN30F1M", "symbol": "41I1G9000"},
                             {"symbolType": "VN30F2M", "symbol": "41I1GA000"}]})


def __test_sight_is_NOT_proven_when_the_alias_resolves_to_itself__(
        fake_client, tmp_path):
    """The #145 signature, and the reason this check exists at all.

    ``resolve_contract`` caches the UNRESOLVED alias on a FAILED instruments
    read (provider.py:212-219), and that is also the legitimate passthrough
    for stocks — so the error and success paths are byte-identical.
    ``get_position`` then filters on the alias, matches nothing, and answers
    None = FLAT forever, with no exception and no log. Two agreeing reads
    cannot catch it: both agree, both are blind.

    Catches: trusting ``get_position`` without proving sight — a watchdog
    that reports OK for the life of the process over a real position.
    """
    broker = _broker(fake_client, tmp_path, get_instruments=(500, {}))
    proven, detail = watch.prove_sight(broker)
    assert not proven, (
        "sight was declared PROVEN after a FAILED instruments read — the "
        "alias would be cached and every position read would answer FLAT (#145)")
    assert "UNRESOLVED" in detail, (
        f"the refusal must name WHAT failed; got {detail!r}. W0 is the first "
        f"consumer of require_contract's guarantee, so its message is the one "
        f"an operator reads when the contract cannot be established.")


def __test_sight_IS_proven_against_a_healthy_catalogue__(fake_client, tmp_path):
    """The over-block control: a proof that can never succeed would pass the
    pin above while making the watchdog permanently BLIND and useless."""
    broker = _broker(fake_client, tmp_path, get_instruments=_CATALOGUE)
    proven, detail = watch.prove_sight(broker)
    assert proven, f"a healthy catalogue failed the sight proof: {detail}"


def __test_journal_never_creates_the_store_and_unknown_account_is_UNAVAILABLE__(
        tmp_path):
    """Catches: instantiating ``BrokerStore`` (its constructor CREATES the
    file, so absence would read as a clean account) and treating an unmatched
    account key as an empty owned set."""
    missing = tmp_path / "nope" / "broker.sqlite"
    assert watch.journal_attribution(missing, "ACC001") == (None, None, None)
    assert not missing.exists(), "the attribution read CREATED the store file"


def __test_exposure_is_not_day_scoped_so_an_overnight_position_stays_owned__(
        tmp_path):
    """The deliberate scope SPLIT, and the direction it was chosen in.

    Ownership of an ORDER id is day-scoped (DNSE reuses NORMAL ids across
    days, #96 — a stale row must not claim the operator's order). EXPOSURE is
    NOT, because day-scoping it drops the entry row of a position held
    overnight and answers 'we own nothing' — silence over a real naked
    position.

    Catches: applying the day filter to the exposure sum as well, which reads
    as the tidier implementation and is the unsafe one.
    """
    import sqlite3
    db = tmp_path / "broker.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE orders (exchange_order_id TEXT, symbol TEXT,"
                 " side TEXT, qty REAL, filled_qty REAL, from_entry TEXT,"
                 " extras TEXT, closed_ts_ms INTEGER, created_ts_ms INTEGER,"
                 " updated_ts_ms INTEGER, run_instance_id INTEGER)")
    conn.execute("CREATE TABLE runs (run_instance_id INTEGER, account_id TEXT,"
                 " plugin_name TEXT)")
    yesterday = int((time.time() - 26 * 3600) * 1000)
    today = int(time.time() * 1000)
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?)", [
        # an entry filled YESTERDAY, position still open overnight
        ("501", "41I1G9000", "buy", 1.0, 1.0, None, '{"leg_kind":"ENTRY"}',
         None, yesterday, yesterday, 1),
        # today's protective exit
        ("daf-cond-1", "41I1G9000", "sell", 1.0, 0.0, "E",
         '{"leg_kind":"STOP_LOSS"}', None, today, today, 1),
    ])
    conn.execute("INSERT INTO runs VALUES (1, 'ACC001', 'DNSE Broker')")
    conn.commit()
    conn.close()

    owned, exposure, per_id = watch.journal_attribution(db, "ACC001")
    assert exposure == 1.0, (
        f"overnight exposure was dropped (got {exposure}) — the watchdog "
        f"would report 'nothing owned' over a real position")
    assert "501" not in owned, (
        "a prior-day NUMERIC id stayed in the owned set — the #96 cross-day "
        "reuse trap, where it can claim the operator's order")
    assert per_id["daf-cond-1"]["from_entry"] == "E"


def __test_phantom_classification_is_memoized_positively_only__(
        fake_client, tmp_path):
    """#41 shells accumulate all day and each costs detail reads every cycle;
    unmemoized, the watchdog walks into a self-inflicted 429. ``Activated`` is
    terminal so caching it is sound — but a NEGATIVE result must NOT be
    cached, because a ``New`` conditional can still become ``Activated``.

    Catches: no memoization (cost grows monotonically) and over-memoization
    (a live order frozen as 'not a phantom' forever).
    """
    rows = {"shell-1": {"orderStatus": "Activated", "externalOrderId": "999"},
            "live-1": {"orderStatus": "New"}}

    def _detail(_acct, oid, _mkt, order_category=None):
        row = rows.get(str(oid))
        return (200, row) if row else (200, {})

    broker = _broker(fake_client, tmp_path, get_order_detail=_detail)
    cache = watch._PhantomCache()

    class _O:
        def __init__(self, oid):
            self.id = oid

    assert cache.classify(broker, _O("shell-1"))[0] is True
    before = broker._client.count("get_order_detail")
    assert cache.classify(broker, _O("shell-1"))[0] is True
    assert broker._client.count("get_order_detail") == before, (
        "a known #41 shell was re-read from the venue — the per-cycle detail "
        "cost grows monotonically as shells accumulate")
    assert cache.classify(broker, _O("live-1"))[0] is False
    rows["live-1"] = {"orderStatus": "Activated", "externalOrderId": "1000"}
    assert cache.classify(broker, _O("live-1"))[0] is True, (
        "a NEGATIVE classification was cached — an order that later triggered "
        "would be counted as live cover forever")


def __test_heartbeat_reports_AGE_of_last_evaluation_not_mere_liveness__():
    """Catches: a heartbeat emitted by the evaluation loop itself.

    There is no asyncio deadline on the venue detail reads (urllib3 defaults
    connect=30/read=60), so a worst-case cycle runs for minutes. A loop-emitted
    heartbeat simply STOPS when the loop hangs, and silence is ambiguous
    between 'hung' and 'never started'. A rising age is unambiguous.
    """
    seen = []
    beat = watch.Heartbeat(0.01, lambda seq, age: seen.append((seq, age)))
    beat.start()
    deadline = time.time() + 2.0
    while len(seen) < 3 and time.time() < deadline:
        time.sleep(0.01)
    beat.stop()
    assert len(seen) >= 3, f"heartbeat did not tick (got {seen})"
    assert [s for s, _ in seen] == sorted(s for s, _ in seen), "sequence not monotonic"
    ages = [age for _, age in seen]
    assert ages[-1] > ages[0], (
        "the heartbeat's age did not RISE while no evaluation completed — it "
        "is reporting liveness, which a hung loop also reports right up until "
        "it stops")


# === review findings F1/F2/F3/N1 (Fable's round on 9bb50003) ===============

def __test_venue_position_with_no_journalled_exposure_is_UNATTRIBUTED__():
    """F1 — the review's most serious finding, and the one my docstring's
    "independent of the belief it audits" claim hid.

    Ownership is 100% the engine's journalled `filled_qty`, so an UNJOURNALLED
    fill reads as foreign. The measured shape: a STOP entry journals the
    umbrella id and its normal-book child is adopted only at the Activated poll
    — so if the engine DIES between trigger and adoption (#39/#120, the exact
    case this sidecar exists for) `filled_qty` stays 0 while the account holds
    a real position.

    Catches: collapsing (venue != 0, owned == 0) into exposure == 0 and
    printing [OK] every cycle over a live naked position — which is what the
    shipped 9bb50003 did.

    Mutation: return the OK assessment for exposure == 0 before this branch ->
    verdict becomes OK and this reddens.
    """
    assessment = core.evaluate(_obs(position_signed=1.0, owned_exposure=0.0,
                                    resting=()))
    assert assessment.verdict is Verdict.UNATTRIBUTED, assessment.reason
    assert assessment.verdict.exit_code == core.EXIT_UNKNOWN, (
        "UNATTRIBUTED must be could-not-determine: on a shared netting account "
        "we cannot tell the operator's position from our own unjournalled one")
    assert assessment.verdict.value != "OK", (
        "an operator grepping [OK] must never be reassured by this state")


def __test_unrecognised_session_phase_is_UNDETERMINED_not_HOLD__():
    """F2 — my own law, broken in my own first gate.

    `venue.session_phase` answers "UNKNOWN (ImportError)" when the L0 import
    fails (venue.py:81-82). A `not in CONTINUOUS_PHASES` test routed that FAILED
    READ to HOLD: exit 0, silent, for an entire session.

    Catches: any "everything that is not continuous is off-session" test.
    Mutation: `if obs.phase not in CONTINUOUS_PHASES: return HOLD` -> reddens.
    """
    assessment = core.evaluate(_obs(phase="UNKNOWN (ImportError)"))
    assert assessment.verdict is Verdict.UNDETERMINED, assessment.reason
    assert assessment.verdict.exit_code == core.EXIT_UNKNOWN


def __test_known_off_session_phases_still_HOLD__():
    """The over-block control for F2: if every non-continuous phase became
    UNDETERMINED, the tool would report could-not-determine all night and the
    F2 pin above would still pass. Catches a fix that forgot the allowlist —
    including the holiday ANNOTATION form venue.py appends.

    This pin ALSO listed lunch and atc until review round 2 refuted the premise
    (see `__test_lunch_and_atc_are_EVALUATED_not_held__`). It was the SECOND
    pin orphaned by that one contract change, and the suite found it rather
    than a sweep — which is why the sweep is now the rule: when a change
    inverts a documented behaviour, grep the phase/verdict tokens across the
    test file BEFORE running, because `-x` only surfaces them one at a time.
    """
    for phase in ("closed", "closed (exchange holiday)"):
        assessment = core.evaluate(_obs(phase=phase))
        assert assessment.verdict is Verdict.HOLD, (
            f"phase {phase!r} graded {assessment.verdict} — a known "
            f"off-session window must FREEZE, not alarm and not puzzle")


def __test_stale_numeric_id_matching_the_journal_row_is_still_ours__():
    """N1, face 1 — the OVERNIGHT bracket.

    A position held overnight keeps its protective order resting with a NUMERIC
    id whose journal row is from yesterday. Day-scoping that id out of the
    owned set made real cover stop counting, so a PROTECTED position graded
    NAKED and re-warned all morning until a live run refreshed updated_ts_ms.

    Catches: dropping prior-day numeric ids outright (the shipped behaviour).
    """
    assert core.stale_numeric_id_verdict("sell", 1.0, "sell", 1.0) == "owned"


def __test_stale_numeric_id_with_mismatched_side_or_qty_is_unclassifiable__():
    """N1, face 2 — the REISSUED id (#96).

    DNSE reuses NORMAL ids across days, so a prior-day row's id may belong to
    the operator's order today. Counting it as our cover is false SILENCE.

    Catches: the blunt fix (day-scope ownership IN unconditionally), which
    trades the morning false alarm for a silent naked position. Note the
    verdict is UNCLASSIFIABLE, not "foreign": calling it foreign would quietly
    restore the false alarm instead of admitting we cannot tell.
    """
    assert core.stale_numeric_id_verdict("sell", 1.0, "buy", 1.0) == "unclassifiable"
    assert core.stale_numeric_id_verdict("sell", 1.0, "sell", 3.0) == "unclassifiable"
    assert core.stale_numeric_id_verdict(None, 1.0, "sell", 1.0) == "unclassifiable"


def __test_stale_id_mismatch_poisons_the_cycle_rather_than_alarming__(
        fake_client, tmp_path):
    """N1 wiring, through `read_resting` — which had ZERO test references
    before this round, and that absence is why F1 and F3 shipped.

    A mismatched prior-day id must come back owned=True, classifiable=False so
    the core's poison rule fires (UNDETERMINED). Catches leaving owned=False,
    which drops it to a plain foreign order and grades NAKED — a confident
    alarm built on evidence we just admitted we cannot read.
    """
    rows = {"501": {"orderStatus": "New"}}

    def _detail(_acct, oid, _mkt, order_category=None):
        return (200, rows.get(str(oid), {}))

    def _orders(_acct, _mkt, order_category=None, **_k):
        if order_category != "NORMAL":
            return (200, {"orders": [], "totalPages": 1})
        return (200, {"orders": [{"id": "501", "symbol": "VN30F1M", "side": "NB",
                                  "quantity": 3, "fillQuantity": 0,
                                  "orderStatus": "New"}], "totalPages": 1})

    broker = _broker(fake_client, tmp_path, get_orders=_orders,
                     get_order_detail=_detail)
    per_id = {"501": {"from_entry": "E", "leg_kind": "STOP_LOSS",
                      "side": "sell", "qty": 1.0, "stale_numeric": True}}
    resting = watch.read_resting(broker, "VN30F1M", set(), per_id,
                                 watch._PhantomCache())

    assert resting and len(resting) == 1
    order = resting[0]
    assert order.owned is True and order.classifiable is False, (
        f"a mismatched prior-day id came back owned={order.owned} "
        f"classifiable={order.classifiable}; it must poison the verdict, not "
        f"fall through to a NAKED alarm")
    assert core.evaluate(_obs(resting=resting)).verdict is Verdict.UNDETERMINED


def __test_sight_is_reproved_every_cycle__(fake_client, tmp_path, monkeypatch):
    """F3 — `prove_sight` ran ONCE before the loop and was passed in forever,
    so the catalogue-membership arm (the #113 ROLL case) could never fire: a
    process running across the roll boundary reads an expired dated code, gets
    None, and prints OK for the rest of the day.

    Catches: hoisting the sight proof out of the cycle. Pinned through
    `evaluate_once`, which had zero test references before this round.
    """
    monkeypatch.setattr(watch, "_read_position_size_confirmed",
                        lambda *_a, **_k: 0.0)
    broker = _broker(fake_client, tmp_path, get_instruments=_CATALOGUE)

    watch.evaluate_once(broker, "VN30F1M", tmp_path / "none.sqlite", "ACC001")
    after_first = broker._client.count("get_instruments")
    watch.evaluate_once(broker, "VN30F1M", tmp_path / "none.sqlite", "ACC001")
    after_second = broker._client.count("get_instruments")

    assert after_first >= 1, "sight was never proved at all"
    assert after_second > after_first, (
        "the instruments catalogue was not re-read on the second cycle — a "
        "stale cache across the roll (#113) could never be detected mid-run")


def __test_heartbeat_stall_EXITS_rather_than_only_flagging__():
    """A stall must terminate the process, not merely set a flag.

    Two shipped behaviours this catches, in order of discovery:
    (a) print-only — a warning no wrapper can act on;
    (b) flag-only — the flag was consulted at LOOP END, which a genuinely hung
        cycle never reaches, so the process printed STALLED forever and exited
        2 only if someone pressed Ctrl-C.

    The action is injected because the default is `os._exit`: wired in
    unconditionally it killed the pytest process itself (exit 2, no summary,
    the whole suite gone). The default here is still the real one — this pin
    asserts the EXIT CODE the watchdog would hand its supervisor.
    """
    exits = []
    beat = watch.Heartbeat(0.01, lambda seq, age: None, stall_after_s=0.02,
                           on_stall=exits.append)
    beat.start()
    deadline = time.time() + 2.0
    while not exits and time.time() < deadline:
        time.sleep(0.01)
    beat.stop()
    assert exits, (
        "no completed evaluation for well past the threshold and the heartbeat "
        "neither exited nor flagged — a watchdog that has stopped watching "
        "must stop loudly enough for a supervisor to restart it")
    assert exits[0] == core.EXIT_UNKNOWN, (
        f"stalled with exit code {exits[0]}; a hung watchdog has not observed "
        f"anything, so its verdict is could-not-determine")
    assert beat.stalled


# === review round 2 (Fable on e9b14434) — four more false-silence paths =====

def __test_loop_exit_code_covers_UNATTRIBUTED__():
    """R2-F1, unconditional and unpinned before this.

    `Verdict.exit_code` said UNATTRIBUTED == 2 while `worst_exit_code` tested a
    hardcoded tuple that omitted it — so `--once` answered 2 and LOOP MODE,
    which is the mode Friday runs, exited 0 after an unattributed cycle. The
    same policy was encoded twice and only one copy was updated.

    Catches: any re-hardcoding of the verdict list in worst_exit_code.
    """
    assert core.worst_exit_code([Verdict.UNATTRIBUTED]) == core.EXIT_UNKNOWN
    assert core.worst_exit_code(
        [Verdict.OK, Verdict.UNATTRIBUTED, Verdict.OK]) == core.EXIT_UNKNOWN, (
        "a loop that saw an UNATTRIBUTED cycle exited 0 — the wrapper is told "
        "the invariant held")
    assert core.worst_exit_code(
        [Verdict.UNATTRIBUTED, Verdict.NAKED]) == core.EXIT_NEGATIVE, (
        "an alarm must still outrank could-not-determine")


def __test_partially_unjournalled_position_is_UNATTRIBUTED__():
    """R2-F2 — the residual of the first UNATTRIBUTED fix, which tested
    `owned == 0.0` EXACTLY.

    The same mechanism (#39/#120: a stop entry's normal-book child never
    adopted because the engine died between trigger and adoption) produces a
    NONZERO remainder whenever only part of the position is journalled — a
    pyramiding stop entry, or the #105 frozen-2 flip. Venue +2 with journal +1
    graded [OK] exposure +1 covered, silently carrying an unaccounted contract.

    Catches: `owned == 0.0` instead of a magnitude comparison.
    """
    assessment = core.evaluate(_obs(position_signed=2.0, owned_exposure=1.0,
                                    resting=(_order(side="sell"),)))
    assert assessment.verdict is Verdict.UNATTRIBUTED, (
        f"venue +2 / journal +1 graded {assessment.verdict.value}: "
        f"{assessment.reason}")
    assert "unattributed" in assessment.reason


def __test_fully_journalled_position_is_not_flagged_unattributed__():
    """Over-block control for R2-F2: if any position triggered UNATTRIBUTED the
    tool would be permanently exit 2 and the pin above would still pass.
    Venue +2 fully journalled with cover is a normal, healthy cycle."""
    assessment = core.evaluate(_obs(position_signed=2.0, owned_exposure=2.0,
                                    resting=(_order(side="sell", qty=2.0),)))
    assert assessment.verdict is Verdict.OK, assessment.reason


def __test_lunch_and_atc_are_EVALUATED_not_held__():
    """R2-F4. HOLD at lunch/atc rested on a measurement taken on a WEEKEND
    (live_test/README.md:387); T9 (README:179) shows a PendingCancel row served
    THROUGH lunch, so those books are populated. Holding meant exit 0 and
    silence for 90 minutes at lunch and 15 at ATC — the latter being exactly
    when an unprotected position meets the auction print.

    Catches: the original {closed, lunch, atc} freeze set.
    """
    for phase in ("lunch", "atc"):
        assessment = core.evaluate(_obs(phase=phase, resting=()))
        assert assessment.verdict is Verdict.NAKED, (
            f"phase {phase!r} graded {assessment.verdict.value} over an "
            f"uncovered position — 105 minutes a day of silence")


# The `closed`-still-holds control that belonged here is NOT duplicated: it is
# `__test_known_off_session_phases_still_HOLD__` above, which already asserts
# exactly that over both the plain and holiday-annotated forms. Three pins were
# converging on one fact after this round, and a fact asserted in three places
# is three places to drift — the same duplication that produced R2-F1.


def __test_a_venue_record_created_TODAY_under_a_prior_day_id_is_a_reissue__():
    """R2-N1 — the coincidence side+qty alone cannot catch.

    Derivative quantity is almost always 1, so on an id hit the corroboration
    degrades to side-only: "our overnight `501 sell 1` was cancelled with the
    engine down, the venue reissued `501` to the operator's `sell 1`" reads as
    our cover over a naked position. A venue record CREATED TODAY under an id
    our journal recorded on a PRIOR day is a reissue by definition — the order
    we journalled cannot have been created after we wrote it down.

    Catches: side+qty corroboration without the date.
    """
    day_start = 1_789_400_000_000
    assert core.stale_numeric_id_verdict(
        "sell", 1.0, "sell", 1.0,
        venue_created_ms=day_start + 3_600_000,   # created TODAY
        day_start_ms=day_start) == "unclassifiable"
    assert core.stale_numeric_id_verdict(
        "sell", 1.0, "sell", 1.0,
        venue_created_ms=day_start - 86_400_000,  # created YESTERDAY: ours
        day_start_ms=day_start) == "owned"


def __test_unparsable_or_absent_venue_date_does_not_pass_the_reissue_check__():
    """R3 finding: this pin's NAME claimed a guarantee it never tested.

    It asserted only that `_epoch_ms` returns None — which was true and
    irrelevant, because `stale_numeric_id_verdict` then ran its date clause
    only `if venue_created_ms is not None` and fell through to side+qty,
    returning "owned". So an ABSENT or UNPARSABLE createdDate passed the
    reissue check while a pin named `does_not_silently_pass_the_reissue_check`
    stood over it. A pin that asserts a PRECONDITION and names a CONSEQUENCE is
    worse than no pin: its name is what the next reader trusts.

    The harm state needs nothing to go wrong — the doc marks createdDate
    OPTIONAL (dnse-get-order-detail.md:208, required=false). Our overnight
    `501 sell 1` journalled yesterday, cancelled with the engine down, DNSE
    reissues 501 to the operator's `sell 1` (#96), the record omits the date,
    and that order reads as OUR cover over a naked long: [OK], exit 0.

    Now asserts the VERDICT for both shapes. Red against 48f7ee42's body.
    """
    day_start = 1_789_400_000_000
    for absent_or_unparsable in (None,):
        assert core.stale_numeric_id_verdict(
            "sell", 1.0, "sell", 1.0,
            venue_created_ms=absent_or_unparsable,
            day_start_ms=day_start) == "unclassifiable", (
            "a prior-day id whose venue record carries NO usable createdDate "
            "was accepted as our cover — side+qty is effectively side-only on "
            "derivatives, where quantity is almost always 1")

    # The parsing half, still worth pinning: a date we cannot read must become
    # None, never 0 (which would read as 'created long ago' and pass).
    assert watch._epoch_ms(None) is None
    assert watch._epoch_ms("not-a-date") is None
    assert watch._epoch_ms(1789456800000) == 1789456800000.0
    assert watch._epoch_ms(1789456800) == 1789456800000.0

    # The over-block control: a genuine prior-day record still counts as ours,
    # so the refusal cannot be satisfied by rejecting every stale id.
    assert core.stale_numeric_id_verdict(
        "sell", 1.0, "sell", 1.0,
        venue_created_ms=day_start - 86_400_000,
        day_start_ms=day_start) == "owned"


def __test_sight_fails_when_the_alias_has_repointed_across_the_roll__(
        fake_client, tmp_path):
    """R2-F3, and tomorrow IS roll morning.

    The provider caches a RESOLVED code permanently, and after the repoint the
    catalogue lists BOTH the expired and the new contract for a while — so a
    membership test (`resolved in codes`) still passes, get_position then
    filters on a contract the account no longer holds, answers None, and the
    sidecar reports "nothing bot-owned is open" over a position held in the NEW
    code.

    Catches: membership instead of "is this the CURRENT mapping for our alias?".
    """
    rolled = (200, {"data": [
        {"symbolType": "VN30F1M", "symbol": "41I1GA000"},   # the NEW front month
        {"symbolType": "", "symbol": "41I1G9000"},          # expired, still listed
    ]})
    broker = _broker(fake_client, tmp_path, get_instruments=rolled)
    # Stub the PUBLIC method, not the private cache. The first cut seeded
    # `broker._contract_cache = {"VN30F1M": "41I1G9000"}` — and when #145's fix
    # legitimately changed that cache to a 3-tuple carrying a read timestamp,
    # this pin broke with `ValueError: too many values to unpack` while the
    # code under test was fine. A pin reaching into a private shape makes a
    # colleague's correct refactor look like a regression; pin the interface.
    broker.require_contract = lambda *a, **k: "41I1G9000"    # cached pre-roll
    proven, detail = watch.prove_sight(broker)
    assert not proven, (
        "a stale pre-roll contract passed the sight proof because it was still "
        "listed — every position read would answer FLAT")
    assert "REPOINTED" in detail or "#113" in detail


def __test_empty_instruments_catalogue_is_not_proof_of_sight__(
        fake_client, tmp_path):
    """A 200 with `data: []` used to PROVE sight, because the membership check
    was guarded by `if codes and ...` and an empty set skipped it entirely.
    Empty is not an answer."""
    broker = _broker(fake_client, tmp_path, get_instruments=(200, {"data": []}))
    broker.require_contract = lambda *a, **k: "41I1G9000"
    proven, detail = watch.prove_sight(broker)
    assert not proven, "an EMPTY catalogue proved sight"
    assert "EMPTY" in detail
