# Friday 2026-09-18 runbook — the roll morning

Fire-and-grade. Everything here was built and verified Wed 09-16; nothing is
improvised at the bell. Two items only: **the roll-probe grade (at open)** and
**Live-L3-F13** (the WS-vs-poll latency A/B).

Order is forced, not chosen: #113 exists only at the roll, and F13's WS arm is
only meaningful with #129 applied (it is, since 58787aaa).

---

## Thursday 09-17 (expiry day) — ONE action, or Friday measures nothing

**Launch the roll probe Thursday afternoon, detached:**

```bash
cd /home/mike/workspace/github/pynecore
setsid nohup .venv/bin/python plugins/dnse/testing/live_test/probe_113_roll_cache.py \
    --hours 20 > plugins/dnse/testing/live_test/logs/roll113_$(date +%m%d).log 2>&1 < /dev/null &
echo "pid=$!  # record it, and record `git rev-parse --short HEAD` as the launch fingerprint"
```

Why Thursday and not Friday morning: the probe's AGED arm only tests anything
if it cached **before** the repoint. A Friday start risks caching the
already-rolled value, after which both arms agree for an uninteresting reason.
The probe says exactly that in its no-divergence verdict rather than reporting
a pass — but a wasted night is still a wasted night.

- Read-only, no orders, **no trading token needed** (instruments is an api-key
  read), so an overnight run is immune to token expiry and needs **no L0 gate**.
- `setsid nohup` is required: a plain background job dies with the session
  (~10 min), and this must survive overnight. Pattern verified 09-16.
- Sanity before walking away: `tail -3` the log and confirm cycles are ticking
  with all three arms populated.

**WHAT HAPPENED 2026-09-17/18 (measured): the roll went unmeasured.** The only probe run
had been started 09-16 16:27 and died 09-17 08:04 — before Thursday's session — and nobody
relaunched it Thursday afternoon (the live session took the day). At 06:40 Friday the alias
already resolved to the NEW code (VN30F1M → 41I1GA000, VN30F2M → 41I1GB000): the mechanic
("the morning after expiry") is confirmed, the timestamp is lost inside a ~22 h gap. Rule from
it, applied by Worker3's proposal: **on the day BEFORE expiry, the pre-flight has a row
"roll probe PROCESS alive" — check the process (`pgrep -af probe_113_roll_cache`), not the log
tail, because a tail cannot tell a probe that stopped a minute ago from one that stopped a day
ago — and relaunch it if dead.** Next chance: the October expiry.

---

## Friday, at open — 1. Grade the roll probe (#113)

(2026-09-18: no probe log to grade — see above. Today's step 1 is instead: re-run
`resolve_contract` inside 08:45–09:05 and record the code; then W0's `sight:` line must name
`41I1GA000` — a stale `41I1G9000` from the per-instance cache is the #113 hazard observed.)

The probe prints its own verdict; grade from that, not from impressions.

| Outcome | Meaning |
|---|---|
| `CONFIRMED — first divergence at <ts>` | AGED (cached) ≠ FRESH (uncached). #113 is real: a long-lived process holds the pre-roll contract. Fix direction is already named in the verdict — the `_secdef_cache` TTL next door is the in-repo pattern. |
| `NO DIVERGENCE across N cycles spanning the boundary` | Evidence for **that window only**. Check the AGED start value against the venue's current mapping: if they were already equal at start, the probe began after the repoint and the arms agreed for an uninteresting reason — **that is not a pass**. |
| `INCONCLUSIVE` | The watch never crossed a trading-day boundary. Nothing measured. |

**Report the bracketing cycle timestamps when divergence is caught** — the
repoint's wall-clock window (to the 30 s hot-poll grid) is itself a new venue
fact for CLAUDE.md, and this is the only day of the month it can be observed.

Also note any `venue read FAILED` lines: those cycles proved nothing either
way, and the verdict counts them separately rather than folding them into a
pass.

---

## Friday — 2. Live-L3-F13 (WS vs poll fill latency)

**Gates first (the runner enforces all of these and aborts itself):**
token VERDICT GOOD · `venue.py flat` exit 0 · L0 exit 0 · session `continuous`.

```bash
bash plugins/dnse/testing/live_test/run_f13_latency.sh --arm ws   [--fills N]
bash plugins/dnse/testing/live_test/run_f13_latency.sh --arm poll [--fills N]
```

One arm per invocation, by design: the engine dedups a fill across transports,
so the transport arriving second produces no event and its timing is invisible
in a combined run. Vehicle is `l2_fill_flatten` — one market entry per run
(NORMAL-book, as #130 requires), then flatten.

### Sample size — OPERATOR'S CALL, still pending as of Wed 09-16
`--fills N` covers either answer with no redesign.
- **n=1**: the poll arm's latency is ~uniform over its 0.5 s grid, so one sample
  lands anywhere in [0, 0.5 s] + RTT. Report it as an **anecdote with that bound
  stated** — never as a mean. The runner prints this caveat itself when N=1.
- **n=3 per arm** (6 small fills total) makes the comparison meaningful.

### Grading — DELIVERY EVIDENCE ONLY (#134, formal)

| Arm | Must be present | Must be ABSENT |
|---|---|---|
| `ws` | `[BROKER] WS ORDER SOURCE FIRST LIVE FRAME` | — |
| `poll` | `WS order feed disabled by config — poll-only` | any `order frame via WS` line |

**The subscribe line is not evidence.** `WS order feed subscribe REQUESTED for:
…` will very likely list the broker channel even though #131 measured the venue
refusing it asynchronously with `SUBSCRIBE_FAILED` *after* a clean subscribe. If
the FIRST LIVE FRAME line is absent, the WS transport did not work — whatever
the subscribe line says. No milestone, no claim.

Expect a `WS server error frame: …` warning on the ws arm: that is the broker
channel being refused, and it is EXPECTED, not a failure. It cannot be attributed
to a specific channel (the vendored client drops the code and channel name), so
read it alongside the milestone, never instead of it.

**Latency** = `T(event FILLED … leg=entry)` − `T(venue fill)`, venue fill time
from `venue.py order <id>`. The ws arm splits further using the existing
`order frame via WS` line: venue→frame is delivery, frame→event is our own
queue/drain cost — which says whose fault a disappointing number is.

---

## Standing rules (unchanged, non-negotiable)

- **Exit 2 from any venue tool = COULD NOT DETERMINE.** Never "no", never "flat".
- **Never `venue.py sweep`** — shared netting account.
- **No position into the 14:30 ATC.** Anything naked or unclear: flatten in the
  app FIRST, diagnose second.
- Grade from the **venue record**, never the run log alone.
- Raw logs carry identifiers → park in `backup/deleteable/`, commit stripped
  evidence only, and scrub-gate every publish. Note the gate only checks
  identifiers listed in `scrub_identifiers.txt` — a pass is evidence about
  those, not about everything.
- Teardown always: `venue.py status` + `venue.py flat` exit 0.

## Known-and-expected — do not re-investigate on the day

- Broker WS channel refused (`SUBSCRIBE_FAILED`) — #131, reproduced twice.
- Conditional-book events never stream — #130. Only NORMAL-book activity appears.
- **CORRECTION 2026-09-17 (measured at the source, Worker1 + Worker2):** `token_status.py` has ALWAYS
  exited non-zero on a NOT GOOD verdict (`return 0 if good else 1`, both return paths). The earlier
  line here claiming it "exits 0" was wrong and was inherited, not measured — and it was the stated
  reason the F13 runner did not gate on it. As of #146 (d2d48fc4+) the runner DOES gate on the exit
  status (stdin from /dev/null, `^VERDICT: GOOD` also required); L0 remains a second backstop, not the
  only one. Callers must capture `$?` — never pipe the tool into `tail` and read the pipe's status.
- The toml carries a commented `#enable_ws_order_events = true` template line.
  It is inert; the runner's guard is line-start anchored and ignores it.

## Added 2026-09-17 evening (from the day's reviews) — passive captures, no extra orders

- **Capture one prod `/positions` payload** while a position is open (during any F13 run):
  `venue.py status` output into the day's evidence — the #135 review found NO captured
  prod positions frame under logs/; the `NB`/`NS` label evidence is docs + fixtures + a
  prod ORDER frame only. One capture closes that.
- **#135 flatten caveat**: `flatten.py` now derives the sign from `.side` (short 1 → BUY at
  the ceiling). The BUY arm is newly reachable: a known ceiling reject answers 200-then-
  `Rejected`, which flatten would misreport as "NOT FLAT after 25 s" (rc 1, protection left
  armed — safe direction). If flatten says NOT FLAT, read the venue app before acting.
- **#135 id-reuse measurement**: note the FIRST venue order id of the day against Thursday's
  id range — reuse vs monotonic settles whether the fill watermark needs day-scoping (the
  "ids are reused per day" premise is documented nowhere; it is a code comment).
- **F13 vehicle reminder**: `l2_fill_flatten` flattens IN-SCRIPT (`strategy.close("E")`,
  engine path); `flatten.py` is teardown only. `--fills N` = runs per arm (operator's n).
