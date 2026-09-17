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


core = _load("naked_position")
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
    """Off session the books answer 200 with ZERO rows, so every order
    legitimately vanishes. Catches: evaluating across the lunch break or
    after 14:45, which alarms on every held position."""
    assessment = core.evaluate(_obs(phase="lunch", resting=()))
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

watch = _load("naked_watch")


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
        "alias is now cached and every position read will answer FLAT (#145)")
    assert "#145" in detail or "cached" in detail


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
