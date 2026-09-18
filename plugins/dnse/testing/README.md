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
| the day | `venue_day.py` | the per-print stream plus 1m bars for one session, RECORDED from the venue or SYNTHETIC from tracked bars |
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

## Known gaps

- **The L1 staged no-fill probe runs but places no orders.** It walks every state against the
  fake and dispatches nothing; the cause is not yet identified. The `l2b` vehicle does place,
  fill and flatten correctly, so this is specific to that probe rather than to the fake.
- **The #162 park replay is not built yet.**
- **The WebSocket half is not served over the socket.** The vendored connection passes an SSL
  context unconditionally and the pinned websockets version refuses that against a `ws://`
  address, so no local plain-WS server can ever be reached by the real client. That is also why
  `fake_dnse_ws.py` stopped working the day the SDK was vendored. The path off production is the
  injected client factory added under #160.
