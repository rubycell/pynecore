# Three venue facts measured on production, Friday 2026-09-18 evening

Measured between 21:36 and 21:49 ICT, after the close, on a flat account with a freshly minted
token. Two probes, each placing exactly one order, each cancelling it and **verifying the cancel
by reading the order back**. Account left flat with no live orders.

Two of the three **correct facts already recorded in this repo**, and one of those corrections
points at a test whose premise is wrong.

## 1. A stop order's SIZE cannot be changed (the operator's question)

```
PLACE stop qty=1        http=201  id=damkpiavfqkc7397pnig
AMEND qty 1 -> 2        http=500  {"code": "REMOTE_SERVER_ERROR",
                                   "message": "Error in backend service"}
READ BACK after amend   http=200  quantity=1     <- unchanged
CANCEL                  http=200  -> Canceled, verified
```

**First measurement of the QUANTITY case.** #18 (`Live-L1-T07-AmendConditional500`, re-measured
2026-09-08) established the 500 for a *price* amend. The quantity had never been sent, because the
plugin adds a leg instead of amending (#123), so this half rested on inference until now.

It also refines #18: the 500 carries a **structured code**, `REMOTE_SERVER_ERROR`, which is in
DNSE's published System Errors list. The record said only "HTTP 500".

The plugin was already correct — a conditional exit whose quantity grows adds a leg, and
everything else, including a shrink, parks. No route sends a stop size change to the wire.

## 2. A CLOSED session does not reject writes — and that contradicts what we have written down

Measured post-close on a Friday evening:

| write | result |
|---|---|
| place a conditional STOP | **HTTP 201**, rests on the conditional book |
| place a NORMAL limit | **HTTP 200**, `Pending` then `PendingNew` |
| amend that order | HTTP 400 `CAN_NOT_MODIFY_ORDER_IN_PENDING_NEW_STATUS` |
| cancel that order | **HTTP 200**, `Canceled`, verified |

So placement is **accepted on both books** while closed, and cancel works. Only the amend is
refused — and for a **state** reason rather than a session one: an order placed while closed sits
in `PendingNew`, queued for the next session and never having reached the exchange, and an order
in that state cannot be modified.

`CAN_NOT_MODIFY_ORDER_IN_PENDING_NEW_STATUS` is a new code, not in the published error list (the
list carries the cancel-side `CAN_NOT_CANCEL_PENDINGNEW_ORDER_IN_OPEN_SESSION`).

### What this corrects

- **"closed rejects everything"** is wrong. Pre-open was already measured as accepting; post-close
  now measures as accepting too. Both regimes accept.
- **`t33_closed_hours.py`'s premise is refuted.** It asserts that placement is REFUSED while
  closed and grades every accepted order as a failure. That probe had never been run live, and
  `venue_core.py` says so in as many words: the closed-session refusal code is labelled a DESIGN
  CHOICE, "not evidence", with "settled by: running the L1-T33 post-close probe once and
  recording which code the venue sends". That is what these measurements do, and the answer is
  that the venue sends no refusal at all.
- **The fake venue is now wrong in the dangerous direction.** `_CLOSED_PHASES` refuses placement
  when the phase is `closed`, so the fake refuses what the venue accepts. That is the same shape
  as the amend model corrected in round 1: the engine learns a recovery path it does not need, and
  every offline pin agrees with the fake because both came from the same belief.

**Not changed here.** Inverting the fake's closed-session model flips `t33_closed_hours.py` from
passing to failing, and that file belongs to another session. The measurement is recorded; the
change is a decision for its owner, and the direction is clear.

## 3. `finalTradeDate` is absent for EVERY VN30F contract, on both surfaces

```
secdef 41I1GA000 board G1   finalTradeDate = None   listingDate = 2026-08-21
secdef 41I1GA000 board T1   finalTradeDate = None
/market/instruments         VN30F1M 41I1GA000  None
                            VN30F2M 41I1GB000  None
                            VN30F1Q 41I1GC000  None
                            VN30F2Q 41I1H3000  None
```

The plugin logged it and fell back:

```
[BROKER] no venue finalTradeDate for 41I1GA000 — using COMPUTED 2026-10-15
         (3rd Thursday, walked back off weekends/holidays) (#118)
```

The computed value is right — the third Thursday of October 2026 is the 15th — so nothing is
broken today. That is the concern rather than the comfort: **the venue silently stopped serving a
field the GTD clamp depends on, and the only signal was one log line.** It was served before
(measured 2026-08-14: VN30F1M carried `2026-08-20`). The contracts are not new either; this one
listed on 2026-08-21.

#118 exists because a wrong final trade date makes every conditional unplaceable, and the fallback
is a computation rather than a measurement. It will be wrong the first time a third Thursday is a
holiday and the walk-back disagrees with the exchange.

## Probe hygiene, recorded because both probes had bugs

- The first refused to run because `get_position` is async and was called without `await`, so the
  flat check tested a coroutine object. **It failed safe** — refusing rather than proceeding — but
  for the wrong reason, which is the `fetch_position` trap in CLAUDE.md one layer along.
- The second version's hand-written conditional payload omitted `durationType` and was refused
  `400 BAD_REQUEST`. Taking the payload shape from the plugin (`broker.py:1478-1492`) fixed it.

Both are the reason the house rule says drive the plugin's own path rather than improvise one. A
probe is as capable of a false reading as the code it examines.

**A hazard worth naming for anyone repeating this:** a conditional placed now carries a GTD a week
out, and a day order placed post-close is queued for the next session. Either survives the weekend
if cleanup fails. Both probes cancelled in a `finally` block and verified the cancel by reading the
order back; an assumed cancel is not a cancel.
