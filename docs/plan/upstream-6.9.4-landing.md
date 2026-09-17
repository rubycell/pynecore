# Landing upstream 6.9.2 → 6.9.4 on dnse-broker-v2 — diff review and re-verification plan

Read-only Fable review of `git diff v6.9.2..v6.9.4` against the fork (2026-09-17, during the
Thursday live session; nothing executed). Fork base: `merge-base v6.9.2 dnse-broker-v2 = v6.9.2`.
Fork on `sync_engine.py`: 43 commits, +1883/−116. Upstream on `sync_engine.py`: +105 in 8 hunks,
all from 45bc8103 and f8c6a1ca. Fork does not touch `one_way_emulator.py` / `storage.py`.

## Verdict per live-measured fact (CLAUDE.md)

| Fact | Verdict | Why |
|---|---|---|
| #121 same-bar arm for a PRE-PLACED exit (live) | UNAFFECTED | live arm = sync_engine wake + script_runner WAKE drain; upstream touches neither region; SimPosition changes never run live (`BrokerPosition`) |
| reactive exit arms a bar late (live) | UNAFFECTED | same mechanism |
| #124 re-arm on external cancel | AT RISK (indirect) | 45bc8103's restart sweep can make the ENGINE cancel live protection; an engine-issued cancel is "ours", so #124 does not re-arm |
| qty=2 flip fence (test_130) | UNAFFECTED (unverified-by-run) | `flip_extra` freeze identical; the new same-level batch rule needs two orders at ONE price; test_130 uses distinct levels |
| pyramiding enforced | UNAFFECTED | gate line identical; diff mentions are comments |
| `max_position_size` live vs backtest | UNAFFECTED | 0 diff mentions |
| #73 clamp / absence-proof | **MECHANISM MOVED** | new restart sweep reads the CLAMPED `_position.size` after `reconcile()` and, through the fork's rewritten `_retire_orphan_exits_on_flat_book`, would issue venue cancels with `venue_flattened_externally=True` |
| two-read position rule | UNAFFECTED | f8c6a1ca reads only reversal markers |
| #122 unconfirmed-flat preservation | AT RISK | the sweep is a second flat-evidence path not routed through `_unconfirmed_flat_pending` |

## The one real collision — 45bc8103 (spent entries / orphan exits on restart)

Upstream inserts, inside `_reconstruct_pine_bracket_state` after `_reconstruct_pine_entry_orders()`:
`if self._position.size == 0.0: self._retire_orphan_exits_on_flat_book(journal_only=True)`, and adds
the `journal_only` parameter. In the fork, `_retire_orphan_exits_on_flat_book` (~:9295–9377) was
rewritten to call `_cleanup_position_tracking(pid, venue_flattened_externally=True)` on the documented
premise "AUTHORITATIVE: reached only via a closing-leg FILL OF OURS (single caller)". Upstream adds a
SECOND caller whose flat evidence is the in-memory position at restart — and on the fork the ordering is
`engine.reconcile()` (startup adoption, #73 clamp) BEFORE `settle_restart_state`, so that size is the
clamped-adopted size. Whenever the clamp adopts 0 while the venue holds exposure (foreign legs,
`ADOPTED_STARTUP_EXTRA`, journal without our fill cursors) or #122 preserved tracking on an
unconfirmed flat, the sweep would CANCEL the journal's protective legs — the "#122 stale FLAT nearly
retired protection" hazard on a path #122 does not guard. The hunks are ~17 lines apart, so git will
likely merge WITHOUT a conflict marker: the dangerous case. **Must be re-derived deliberately at the
rebase** — the restart caller should route through the fork's `flat_evidence_unconfirmed=True` /
`_unconfirmed_flat_pending` path (preserve, confirm later), never `venue_flattened_externally=True`.

## Clean applies

- f8c6a1ca (duplicate reversal closes): fork region byte-identical to 6.9.2; reads only reversal
  markers — no `ExchangePosition` sign, no account net.
- 8bc93c8c (one-way fan close-leg skips): `one_way_emulator.py` only, fork untouched; DNSE is a
  netting venue and does not implement `close_leg` (verify with a grep at rebase time).
- `storage.py reopen_order` now resets `created_ts_ms` — note for any duplicate guard anchored on it.
- `live_runner.py` and `cli/commands/run.py`: upstream did not touch either → #84 halt untouched.
- `script_runner.py`: upstream adds a late-closed-bar drop (0ba96e85, `candle.is_closed and
  candle.timestamp < last_bar_timestamp` → dropped with a warning) ~70 lines below the fork's #121
  WAKE hunk — adjacent, not overlapping; check it against the #100 sub-minute WS synthesis feed
  (could a synthesized forming bar precede a late venue candle?). Trivial `__slots__`/import
  collisions in `script_runner.py` and `lib/strategy/__init__.py` (fork P5 drawdown slots).

## Backtest-oracle effect (lib/strategy, SimPosition only)

88d346c2 (same-price-level batches by activation order), aa12a5e3 (an intrabar-activated exit leg
gets a wrong-side instant fill and the extreme→open stretch of the bar), 959f1fe9 (pyramid adds
inherit live exit legs; trailing fills in path order). For a stop entry with a pre-placed SL that
lies between the fill bar's open and its extreme, the SL can now fill ON the entry bar where 6.9.2
carried it to the next bar. l2b has no backtest oracle (entries gated on `barstate.isrealtime`); the
staged fill test's F3/F4 (SL = `low[1]`/`high[1]`) can shift one bar earlier in rare gap bars. No
grader keys on the fill bar index; grading is from the venue record.

## Landing sequence (operator executes the reset/force-push)

1. Not before Friday 09-18 ~15:00 ICT; never while a `--broker` run or the W0 sidecar runs.
2. Scratch worktree per the fork policy; full suites vs a pristine-upstream control; land by reset +
   force-with-lease.
3. At the 45bc8103 hunk: re-derive the restart caller as above; add a pin: restart with a journal
   protective leg + clamped-to-0 adoption must NOT issue a venue cancel (red-first on the naive merge).
4. After landing, before any fill-tier run: **L0 gate + a restart no-fill case** (live position +
   journal legs; once with normal adoption, once with a foreign leg so the #73 clamp adopts 0) —
   look for the new "flat book left exit tracking under parent … retiring" startup line and any
   unexpected venue cancel. Closest existing shapes: the T-tier restart-adoption cases / T10.
5. NOT required: fill-tier re-measurement of #121 arm timing (path untouched); live re-measurement of
   the qty-2 / pyramiding fences (backtest-only, pinned by test_130 — run pytest after the rebase).
   Oracle side: one staged-fill backtest-mode run, diff F3/F4 exit bars against the saved oracle — a
   one-bar-earlier SL is aa12a5e3, not a regression.
6. CLAUDE.md: after step 4 passes, no banner needed for #121/#124 (untouched); add the aa12a5e3
   oracle note to the staged-fill section.
