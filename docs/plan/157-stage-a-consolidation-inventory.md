# #157 stage A — consolidation inventory of the fakes that already exist

Measured 2026-09-18 in worktree `pynecore-worker4`. This document exists because the card's
original baseline was wrong: it asserted that nothing offline served the venue surface, and in
fact three fakes already existed. Nothing here is retired until the leader reviews stage A;
retirement then means a move into `backup/deleteable/` plus the untrack commit, never a delete.

## The three existing fakes

| file | size | shape | alive? | what it models |
|---|---|---|---|---|
| `plugins/dnse/testing/fake_venue.py` | 5.6 KB | in-process, stands in for the client wrapper | **YES** — imported by `plugins/dnse/tests/test_fixes_end_to_end.py:23` | card #10's measured quirks: cancel-is-an-ACK with a configurable `ack_lag` (#20), NO cascade on cancel (#19), the two books with their id shapes, a wrong-book cancel answering `RESOURCE_NOT_FOUND` |
| `plugins/dnse/testing/fake_dnse.py` | 17 KB | a real HTTP server on a port | no test imports it | golden fixtures replayed by path match, plus a synthesised stateful order book and a `session_open` switch |
| `plugins/dnse/testing/fake_dnse_ws.py` | 7.4 KB | the WS half of that server | no test imports it | the WS side of the same |

Supporting data already in the tree: `record_fixtures.py` (4.7 KB) and `dnse_fixtures.json`
(31 KB, a 22-entry list of captured responses).

## Measured versus invented — the distinction that decides what can be reused

This is the part that matters. A fake is only evidence about the real venue where its behaviour
was measured from it; everywhere else it is a story that will be believed.

**`fake_venue.py` — measured.** Every quirk in its docstring is traceable to a dated live
observation (cancel-ACK measured on a named order, the no-cascade case from a named test pair).
Its status vocabulary is the venue's own: `New`, `Canceled`. This is the file whose content
should survive consolidation.

**`fake_dnse.py` — mixed, and the invented part is the order lifecycle.** Its header claims
responses are "1-1 with production" because they are replayed from `dnse_fixtures.json`. That
claim holds for the READ endpoints that were captured; it does NOT hold for order state, because
the captured fixtures contain **no `orderStatus` field at all** (`grep '"orderStatus"'` over the
fixture file returns nothing). The order lifecycle is synthesised by the server itself, and it
synthesises a vocabulary the venue does not use: `NEW`, `FILLED`, `CANCELLED` (uppercase, and
`CANCELLED` with the double L), against the venue's `New`, `Filled`, `Canceled`, `Activated`,
`PendingNew`, `PartiallyFilled`.

**Why that divergence is invisible today, and why it still matters.** The plugin normalises
before it branches: `_STATUS_MAP` (`broker.py:162-171`) is keyed on uppercased strings and maps
`"CANCELLED"` and `"CANCELED"` to the same value. So the plugin tolerates either spelling and no
test would ever catch the difference. That is benign in its effect and instructive in its shape:
it is precisely the silent-wrongness the panel's correctness lens warned about — a fake that is
wrong in a direction the code happens to absorb teaches nothing, and passes.

**Decision:** the consolidated state machine emits the MEASURED vocabulary only. A pin asserts
it, so a future drift back to the invented one fails loudly rather than being absorbed by the
status map.

## Why the socket server is dead, and what that implies for reviving it

The panel's maintainability lens reported, and the dates support, that `fake_dnse.py` was written
on 2026-08-04 and stopped working when the SDK was vendored two days later: the vendored
connection always supplies an SSL context, which the pinned websockets version refuses against a
plain `ws://` URL, and the server's auth reply does not match the token the vendored client
accepts. **UNVERIFIED by me** — I have not run it or read those two sites; it is recorded here as
the reviewer's claim with the reason it is plausible (nothing imports it, so nothing would have
noticed), and it must be checked before any revival is scheduled.

The implication either way is the same: reviving it is not "start the old server". It is porting
it onto the consolidated state machine as a second adapter, which is what the adjudication
decided and what keeps one venue truth rather than two.

## Consolidation plan

1. `venue_core.py` (the stub committed for the test-first suite) becomes the single state machine
   and **absorbs `fake_venue.py`'s measured quirks rather than re-deriving them** — the ack-lag
   cancel, the no-cascade rule, the id shapes per book, the wrong-book 404.
2. `test_fixes_end_to_end.py` keeps passing throughout, unchanged. It is the only existing test
   that pins any of this behaviour, so it is the regression that matters; it is re-pointed at the
   consolidated machine only once that machine reproduces every quirk it relies on.
3. `fake_dnse.py` and `fake_dnse_ws.py` become the socket ADAPTER over the same machine when
   stage F needs `pyne run --broker` unchanged. Until then they stay where they are, untouched
   and unretired.
4. `dnse_fixtures.json` and `record_fixtures.py` are the payload SOURCE, extended by the stage B
   recorder rather than replaced. Their 22 captured responses stay authoritative for the shapes
   they cover.
5. Nothing is retired before the leader's stage A review; then by move into `backup/deleteable/`
   with the untrack commit.

## Reproduce

```bash
grep -rn "fake_dnse\|fake_venue" --include='*.py' plugins/ | grep -v _vendor
grep -ohE '"(New|Filled|Canceled|CANCELLED|FILLED|NEW)"' plugins/dnse/testing/fake_venue.py plugins/dnse/testing/fake_dnse.py | sort | uniq -c
grep -n "_STATUS_MAP" -A 10 plugins/dnse/pynecore_dnse/broker.py
grep -c '"orderStatus"' plugins/dnse/testing/dnse_fixtures.json
```
