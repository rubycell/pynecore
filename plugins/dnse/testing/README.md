# The offline fake DNSE venue (#157)

Run a real strategy through the real engine and the real plugin against a venue that never
touches the network, needs no credentials, and reproduces the conditional-order behaviour the
DNSE sandbox cannot.

```bash
FAKE_VENUE_DAY=plugins/dnse/testing/fixtures/venue_day/SYNTHETIC_vn30f1m_1m.json.gz \
FAKE_VENUE_LIVE_BARS=60 \
.venv/bin/pyne run plugins/dnse/testing/live_test/l2b_fill_protect_flatten.py \
    dnse_fake:VN30F1M@1 --broker
```

## What it is

One state machine, two adapters, one replayable day.

| piece | file | what it does |
|---|---|---|
| state machine | `venue_core.py` | the venue itself: two books, activation, fills, refusal codes. Pure Python, no socket, no clock |
| socket adapter | `venue_http.py` | serves that machine over loopback HTTP so the real client, the real vendored SDK and real urllib3 run unmodified |
| the day | `venue_day.py` | the per-print stream plus 1m bars for one session, RECORDED from the venue, SYNTHETIC from tracked bars, or DERIVED-FROM-1M from downloaded history |
| the downloader | `venue_day_from_history.py` | builds DERIVED-FROM-1M days from the venue's 1m history, read-only |
| the broker | `../pynecore_dnse/fake_broker.py` | entry point `dnse_fake`; replays the day as data while routing orders to the machine |
| the controls | `venue_mutants.py` | a deliberately wrong venue per pinned fact, so the pins are proven able to fail |

Nothing here is reached by a production run. `dnse_fake` is opt-in by NAME, like `dnse_event`
and `dnse_replay_sandbox`, and it reads its own `dnse_fake.toml`, so the live config and the real
trading token are never involved.

## Why it exists

Before this, the conditional lifecycle could only be exercised against production. The sandbox
rejects the STOP and OCO categories outright and has no price simulation; the pytest fake-client
seam stops at the client wrapper, with no wire, no state and no clock. So activation, the
normal-book child, partial fills by traded volume and the measured refusal codes cost market
hours and real money to test. Here they cost nothing.

## The rules this fake follows

**It reproduces the venue, not the documentation.** Where the two disagree, the measured
behaviour wins. The status vocabulary is the venue's own (`New`, `Filled`, `Canceled`,
`Activated`, `PendingReplace`) rather than the uppercase variant an older fake synthesised — the
plugin's status map absorbs the difference, so only a venue-level pin can catch it.

**It reproduces awkward truths rather than tidying them.** A spent OCO umbrella still reads
`Activated` with its stop price populated on a flat account, indistinguishable from an armed
one. That ambiguity is real and a caller must resolve it through the child, so the fake keeps it.

**An OCO stop leg AMENDS its child in place; it never spawns.** Measured 2026-09-18: one id goes
`New → PendingReplace → New → Filled`. There is no far leg, so one-cancels-other never appears as
a sibling being cancelled. An earlier draft of the conformance suite pinned the spawn model, and
building to it would have taught the engine a venue that does not exist.

**Endpoints the plugin does not call answer 404, not a plausible fiction.** Executions answers
404 specifically because production does on this account, which is why the plugin books at
average fill price.

**A day states its provenance or it is refused.** A missing or unknown label raises rather than
defaulting: assuming RECORDED fabricates a measurement, assuming SYNTHETIC discards a real one.

## Safety

- The server binds loopback only. A fake venue on a routable interface is an unauthenticated
  order endpoint.
- A production-looking endpoint is refused from BOTH sides: the server's bind address and the
  config's `base_url`/`ws_url`, checked before anything starts. The broker overwrites the address
  with its own port moments later, so a production host in the config would otherwise be harmless
  only by accident.
- The config's credentials are nonsense and its `token_file` points at a path that does not
  exist, so a fake run cannot read or disturb the real trading token.

## Proving the pins can fail

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python plugins/dnse/testing/venue_mutants.py
```

Each mutant makes the venue wrong in one specific way and names the ONE test that must catch it.
Mutants are runtime patches, never source edits, because a file-copy restore can leave stale
bytecode running the mutant while the source looks correct.

Read the CONTROL row first. It applies a change that alters nothing and must be reported
ESCAPED. A harness whose fixture crashes reports every mutant as caught and shows a perfect
score, so a table of uniform CAUGHT rows means nothing without that row. This already earned its
keep: a mutant escaped because the test it targeted had never been collected — its name lacked
the trailing underscores pytest requires — and the suite had been reporting six of seven as a
clean pass. `test_collection_sweep.py` now pins that directly.

## Building a day

```python
from venue_day import synthesise_day_from_ohlcv, save_day
day = synthesise_day_from_ohlcv("workdir/data/dnse_VN30F1M_1.ohlcv", symbol="41I1G9000")
save_day(day, "plugins/dnse/testing/fixtures/venue_day/SYNTHETIC_vn30f1m_1m.json.gz")
```

A synthetic day emits open, high, low and close per bar and conserves the bar's volume, because
fills are matched against traded volume. Its stated limitation: intrabar SEQUENCE is unknowable
from bar data, so only a RECORDED day can answer questions about the order trades arrived in.

`DayRecorder` builds a RECORDED day from a live feed; its frame source is injected, so it is
testable without the venue. Joining a session late marks the day PARTIAL and records where it
joined, so a late start is never read as a whole session.

## Building days from the venue's own 1m history

```bash
.venv/bin/python plugins/dnse/testing/venue_day_from_history.py --symbol VN30F1M --sessions 5
```

One request, `GET /price/ohlc`, read-only: no orders, no trading token, no account endpoint. It
writes one `DERIVED-FROM-1M_<symbol>_<date>.json.gz` per session under `fixtures/venue_day/`,
with prints reconstructed from each bar's OHLC path exactly as a synthetic day's are.

It deliberately does NOT use the provider's own `download_ohlcv`, because that persists through
`save_ohlcv_data`, which truncates and rewrites the shared `.ohlcv` file for that (provider,
symbol, timeframe) — the same damage `pyne run --from` does. Everything stays in memory, and any
destination inside a `workdir/data` directory is refused outright.

**A derived day is a third provenance, not a synonym for SYNTHETIC.** Its prices are the venue's
own and it is dated, like a RECORDED day; its intrabar sequence is reconstructed from bar data,
like a SYNTHETIC one. So it carries SYNTHETIC's limitation unchanged — no day of this kind can
answer questions about the order in which trades arrived — and it must state where it came from.
A file carrying the label with no provenance behind it is refused rather than loaded.

### Three things measured against production while building this (2026-09-18)

| what was asked | what the venue answered |
|---|---|
| `/price/ohlc` for `VN30F1M` | 200, full sessions of 241 1m bars |
| `/price/ohlc` for the dated contract `41I1GA000` | **200 with every array empty** and no error field |
| `/price/ohlc` for the retired contract `41I1G9000` | 400 |

So the endpoint serves the rolling ALIASES only, and its way of saying "not a symbol I serve" is
a silent empty rather than an error. That is why a 200 with no bars is refused here: read as a
quiet day, it would write files for sessions that never happened.

**The alias series splices at the roll, so a day cannot claim a dated contract.** Asked over the
same fortnight, `VN30F1M` and `VN30F2M` return different closes for the same past sessions,
because each alias means whichever contract was front or next AT THE TIME. On 2026-09-18
`VN30F1M` resolved to `41I1GA000` and `VN30F2M` to `41I1GB000` — the roll had happened that
morning — so four of the five most recent `VN30F1M` sessions belong to the contract that expired
the day before. Each day therefore records the alias it was fetched under, the dated contract
that alias resolved to, and the date that resolution was taken, and asserts nothing about which
contract a past session belonged to.

Two sanity checks worth knowing, both visible in the files built on 2026-09-18: a derivative day
runs 09:00 to 14:45 while a stock day runs 09:15 to 14:45 (the two ATO windows differ), and every
full session reports exactly **2 gaps** — the lunch break and the 14:30-to-14:45 auction pause.
The gap count is why the resolution check tests the DOMINANT step rather than every step.

## The replay is re-stamped onto the wall clock

The day is served SHIFTED so the first live bar lands on the current minute, with one offset
applied to bars, derived prints, the history endpoint and the catalogue's final trade date. The
day file keeps its recorded timestamps; the shift happens at serve time, and every run logs
`replay offset = +N s (day YYYY-MM-DD served as today)`.

This is not cosmetic. The engine anchors its live path to the wall clock in TWO independent
places — a feed-staleness watchdog and a synthesiser that emits one bar per MISSED timeframe
boundary — so a day stamped in the past misses a boundary every real minute and the engine
substitutes its own flat bars forever. Measured before the shift: 65 synthetic bars, no fills,
nothing to compare. After it: zero synthetic bars and a real trade list.

**Consequence for any vehicle with an absolute time window.** Those windows are wall-clock
milliseconds, so they must be set past LAUNCH PLUS WARMUP, not merely "around now". Warmup
replays up to 500 bars and takes tens of seconds, and a window that opens during it burns the
vehicle's stages against the backtest engine, which routes no orders. The round-2 checker saw
stages T1 to T4 fire inside warmup with no broker line at all. A window on the recorded day's
dates never opens.

That was the real cause of the staged no-fill probe appearing to place nothing.

**Measured end to end, 2026-09-18**, window at launch plus 100 s, 300 live bars, 0.25 s pacing:

| observation | result |
|---|---|
| idle-bar synth lines | 0 |
| states reached | 0 through 15 — every state |
| dispatches | 20 |
| venue events | 23 CREATED, 23 CANCELLED |
| parked modifies | 0 |

An earlier run of this table said states 0 to 13 with a park at T6, and that was an artefact of a
bug in this fake, not a property of the plugin. The fake had gated the amend on ASSET TYPE and
answered 500 for every derivative amend; the plugin then parked on a refusal the real venue would
never have sent. With the amend gated on the BOOK — which is what the registry says — T6 amends
in place and the probe reaches every state. A fake that refuses where the venue accepts teaches
the engine to take a recovery path it does not need, and it is worth knowing that the wrong
version looked entirely plausible: a 500 on amend IS a measured fact, it just belongs to the
conditional book.

## Amends are asymmetric by asset type

The discriminator is the BOOK, not the asset — which is the opposite of what this fake assumed
for one revision.

| what is amended | outcome | source |
|---|---|---|
| CONDITIONAL book (STOP/OCO), any asset | HTTP 500 | `Live-L1-T07-AmendConditional500` (#18) |
| NORMAL book, derivative | 200, amended IN PLACE, same id | `Live-L1-T06-AmendNormal`, PASS 08-14, re-verified 08-17 |
| NORMAL book, stock | 200 with a NEW id, the old one `Canceled` | #117, prod 2026-09-15 |

The conditional 500 is why the plugin routes a conditional modify away from a PUT entirely: a
conditional entry becomes its own cancel-and-replace, a conditional exit becomes a park. On the
normal book none of that applies and the venue simply accepts the change. After a stock amend,
anything still tracking the old id goes blind — the same failure family as a stop entry's child.

## Trade-list parity

Run the parity script under `fixtures/offline/`. It runs the parity vehicle twice over the same
day — once as a file-mode backtest, once through the fake — and compares the closed-trade lists.
Four known differences are stated beside the assertion, including the one that matters most:
trade TIMES are not comparable, because a backtest stamps bar time and a live run stamps wall
clock, so trades are matched by ORDER. A mismatch is a finding about the fake or the plugin,
never an accepted difference.

The day and its bar store are chosen together, and `venue_day_dataset.dataset_from_day` builds
the store FROM the day so that "the same bars" is a fact rather than a belief about how a local
file was once made. That builder may write only names under the `fakeparity` prefix; a shared
store name is refused outright.

```bash
FAKE_VENUE_PARITY_DAY=plugins/dnse/testing/fixtures/venue_day/DERIVED-FROM-1M_VN30F1M_2026-09-18.json.gz \
FAKE_VENUE_PARITY_DATASET=fakeparity_derived_VN30F1M_1 \
FAKE_VENUE_LIVE_BARS=25 \
.venv/bin/python plugins/dnse/testing/fixtures/offline/trade_list_parity.py
```

### The window was wrong, and the synthetic day had been hiding it (measured 2026-09-18)

The comparison window used to be derived as *the fake's first trade time minus the replay
offset*. That is unsound by this suite's own rule: a backtest stamps a trade with its BAR time
and a live run stamps the WALL CLOCK at which it closed, so no offset converts one into the
other. Run against the derived day for 2026-09-18 the derived start landed at 14:06:39 while the
first live bar was 14:06:00; the 39 seconds crossed a bar boundary, dropped one backtest entry,
and the acceptance test reported `backtest 3 vs fake 4` when both engines had produced 4.

The synthetic day passed only because its first trade happened not to straddle a boundary, so
the green was luck. The same flaw could equally have TRIMMED a window until a real difference
disappeared, which is the worse direction. The fake now reports its first live bar directly and
the harness uses it; both days pass, and a mutant restores the old derivation to prove the pins
can fail.

## Grading a fake run from the venue, not from its log

A live DNSE result is graded from the venue's order record. Set `FAKE_VENUE_RECORD_FILE` and a
fake run is graded the same way:

```bash
FAKE_VENUE_DAY=plugins/dnse/testing/fixtures/venue_day/DERIVED-FROM-1M_VN30F1M_2026-09-18.json.gz \
FAKE_VENUE_LIVE_BARS=60 \
FAKE_VENUE_RECORD_FILE=/tmp/record.json \
.venv/bin/pyne run plugins/dnse/testing/live_test/l2b_fill_protect_flatten.py \
    dnse_fake:VN30F1M@1 --broker
```

The file is rewritten after EVERY state transition, not at exit. A supervised run is ended by the
operator and a run stopped with a signal reaches no exit hook, which is how the parity harness
once compared a backtest against itself.

**The l2b vehicle against the derived day for 2026-09-18**, 181 warmup bars and 60 live bars:

| observation | result |
|---|---|
| idle-bar synth lines | 0 |
| entry conditional activated, child tracked | 3 |
| OCO umbrella + TP child created | 3 |
| complete round trips | 3 |
| position at the end | flat |

The chain the vehicle exists to exercise is visible end to end: a conditional entry rests on the
STOP book with a string id, a print through the trigger activates it and spawns an integer-id
NORMAL child, the fill lands on the CHILD rather than on the id we placed, the bracket goes out
as an umbrella plus a TP child, and the max-two-candles rule cancels the child and closes the
position.

**Parked modifies are expected here and are NOT a finding.** The vehicle re-issues its entry on
every flat bar to chase the breakout, so the plugin repeatedly tries to cancel a conditional that
has already activated; the venue answers `CO-ORD-013 order is done` and the plugin parks the
modify, keeping the old id as possibly-live. That is the documented behaviour of a conditional
entry modify. It is also not specific to derived days: the same vehicle against the SYNTHETIC day
parks 38 times in 110 seconds. Whether the plugin should keep re-attempting a cancel on a
conditional whose child it has already adopted is a separate question, for its own card.

## A fake run does not end when its bars run out — it blocks

Measured 2026-09-18 on both new fill-tier vehicles, and it is not specific to them.

`FAKE_VENUE_LIVE_BARS=40` replayed all forty live bars in about **thirty seconds** — and then the
process stayed alive with the log not growing, blocked in the live loop waiting for a next bar
from a day that is exhausted. F12 sat that way for six minutes until it was killed (exit 144,
expected after your own kill). F09 did the same and was killed by its own `timeout 600` wrapper
(exit 124).

**Three consequences for anyone driving this fake:**

1. **`FAKE_VENUE_LIVE_BARS` bounds what is REPLAYED, not how long the process lives.** Every fake
   run needs an explicit stop — a `timeout`, or a kill once the record file stops changing. Do
   not wait for it to finish; it will not.
2. **The record file, not process exit, is the completion signal.** It is rewritten after every
   state transition, so it is complete long before the process is stopped. Grades stand despite a
   run being killed or timing out — which is exactly the property the record file was built for.
3. **Read the status of the RUN, not of the wrapper.** `timeout N pyne run … ; echo $?` leaves the
   real number in the echo and **zero** as the command's status, because the shell reports the
   last command. That shape produced a reported "exit code 0" when the truth was 124. Capture to
   a variable, or put nothing after the command. Same family as the pipe-exit trap already
   written into the plan docs.

### Is the block a defect? No — but its SILENCE is

The blocking is deliberate and it is right. `watch_ohlcv` ends with `await self._exhausted.wait()`
on an `asyncio.Event`, so a replay that has served its last bar parks rather than returning.
That models the venue correctly: a real feed never ends either, and the engine's live path has no
"the data is finished" state. Giving it one would teach the engine a shutdown path that
production never takes, which is the same mistake as a fake that refuses where the venue accepts.
An exhausted replay is a quiet feed, and a quiet feed is a thing that really happens.

What is wrong is that it goes quiet **without saying so**, and two things combine to make that
worse than it looks:

- **Nothing is logged when the last bar is handed over.** From outside, "finished replaying" and
  "hung" are the same observation: a live process and a log that stopped growing. Every consumer
  is therefore forced to guess with a timeout.
- **The engine's own staleness watchdog is switched off here** (`feed_timeout_bars = None`), for a
  good and documented reason: a paced replay trips it and the engine then substitutes its own
  flat bars forever. But the consequence is that the fake has *no* staleness signal at all, so
  the one mechanism that would otherwise surface the silence is the one we removed.

So the missing piece is an **end-of-replay signal**, not a change to the loop and not an exit.
The seam already exists and is simply never used: `self._exhausted` is created and waited on, and
nothing ever sets it. Setting it at the last bar, with a log line naming how many bars were
served, would let a runner wait on a signal instead of on a timeout and let a human tell the two
states apart at a glance — while leaving the parking behaviour, which is the correct model,
exactly as it is.

Filed as a #157 follow-up rather than changed here.

## The vendor's own examples are part of this suite

`examples/` runs DNSE's published SDK examples against this fake, unmodified. They are worth the
trouble for one reason, measured on 2026-09-18.

Six defects were found in the fake that day. Three came from our own pins and three from the
vendor's examples, and the two sets are not the same kind of thing.

The three our pins found were **shape mismatches that crashed a reader** — a payload the consuming
code could not parse, so it raised or fell through a guard. Loud, once something exercised the
path.

The three the examples found **answered cleanly and were simply not answers to the question**: an
account payload keyed on the wrong field, a security definition missing the field that decides
whether an instrument classification is authoritative or a guess, and four routes that accepted a
parameter and never read it. Nothing crashed. Every response was well-formed. All of them wrong.

**Our own tests cannot find that class, because we write both the question and the answer.** When
the same understanding produces the fake and the test of the fake, a shared misreading is
invisible from the inside: the test agrees with the fake precisely because both came from the same
head. The vendor's examples are the only input in this tree written by the people who run the
venue, which makes them the only place where a misreading of ours shows up as a failure instead of
as agreement.

## Known gaps

- **The WebSocket half is not served over the socket.** The vendored connection passes an SSL
  context unconditionally and the pinned websockets version refuses that against a `ws://`
  address, so no local plain-WS server can ever be reached by the real client. That is also why
  `fake_dnse_ws.py` stopped working the day the SDK was vendored. The path off production is the
  injected client factory added under #160.
