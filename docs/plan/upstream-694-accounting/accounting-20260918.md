# #151 commit accounting — v6.9.2..01e78834 (source, 350) → v6.9.4..land-694 (c0f079ab)

Join key: stable patch-id (`git patch-id --stable`) of the source commit = `backup/fork-commits-land-694-source-20260918.patchids`. Rows are position-matched (rebase preserves order) and subject-checked. Classes: REPLAYED-EQUIVALENT (patch-id equal) / REPLAYED-EQUIVALENT-CONTEXT (patch-id differs, context-stripped +/- lines identical) / REPLAYED-MODIFIED (+/- delta shown) / SUPERSEDED-BY-UPSTREAM / DROPPED-BY-DECISION (none). No row = lost.

| # | source | source patch-id | landed | landed patch-id | class | note | subject |
|---|---|---|---|---|---|---|---|
| 1 | 3487260f | 9e3ad50befd3 | ed076933 | 9e3ad50befd3 | REPLAYED-EQUIVALENT |  | fix(matrix): make Matrix iterable and sized (for row in m, len) |
| 2 | c130c316 | 149951797f8c | f3c3e8a3 | a3099ace3367 | REPLAYED-EQUIVALENT-CONTEXT | +/- lines identical; context differs | feat(optimize): port `pyne optimize` onto upstream (sequential path) |
| 3 | c2d158db | 2bc7adc36665 | 0b3ce58f | 2bc7adc36665 | REPLAYED-EQUIVALENT |  | feat(optimize): port the parallel worker path to inputs= (drop var cache) |
| 4 | 16efaa39 | 331b477a7e63 | fc9bd1ec | 331b477a7e63 | REPLAYED-EQUIVALENT |  | feat(stats): port fork-only drawdown + P&L metric family onto upstream (P5) |
| 5 | f8d6014c | 5075b91d50ea | f38cf5e2 | 5075b91d50ea | REPLAYED-EQUIVALENT |  | refactor(stats): rename our intrabar DD to unrealized_max_drawdown, drop dead close DD |
| 6 | 6d1916d3 | 2fb9bec7c460 | 8dba3e26 | aa1d70a2d9a1 | REPLAYED-MODIFIED | 4 differing +/- lines: -+version = "6.9.2" # PineVersion.Major.Minor — tracks upstream \| ++version = "6.9.4" # PineVersion.Major.Minor — tracks upstream \| --version = "6.9.2" # PineVersion.Major.Minor \| +-version = "6.9.4" # PineVersion.Major.Minor | chore: publish fork as opencode-pyneruntime dist (import stays pynecore) |
| 7 | dd0ac4df | c516710d1137 | 66de60d6 | c516710d1137 | REPLAYED-EQUIVALENT |  | fix(optimize): adapt fork-only code to upstream 6.7.1 APIs |
| 8 | 4247b0aa | cc94e96edf09 | dc01c796 | cc94e96edf09 | REPLAYED-EQUIVALENT |  | chore: ignore backup/ |
| 9 | 54fea8b5 | 07d74b1dc497 | d63aa156 | 07d74b1dc497 | REPLAYED-EQUIVALENT |  | feat(plugins): add DNSE provider + broker plugins with an offline fake venue |
| 10 | 4b389948 | 76f822b9bb6d | 492b410c | 76f822b9bb6d | REPLAYED-EQUIVALENT |  | fix(dnse-broker): poll REST for fills and bars; live-verified transport layer |
| 11 | 63d00d0d | 0af395add612 | 277d4b74 | 0af395add612 | REPLAYED-EQUIVALENT |  | test(plugin): derive the owning dist name instead of hardcoding upstream's |
| 12 | d73008c7 | c19c6877b91f | 87b6bbc5 | c19c6877b91f | REPLAYED-EQUIVALENT |  | chore(dnse): remove v1 plugin + vendored dnse-py |
| 13 | 98e7bf41 | c3930910ab34 | 150572a9 | c3930910ab34 | REPLAYED-EQUIVALENT |  | docs(dnse): mirror DNSE OpenAPI docs + specs with a sync/verify script |
| 14 | ad20325f | b2c0bff4403f | 30a248e6 | b2c0bff4403f | REPLAYED-EQUIVALENT |  | feat(dnse): vendor official openapi-sdk 2.0.0 for v2 |
| 15 | d1eafb55 | a21f772b1afa | 42dcd20b | a21f772b1afa | REPLAYED-EQUIVALENT |  | docs(dnse): add DNSE broker v2 plan |
| 16 | d8329b62 | 5f045cc9e1a1 | 05a2ff03 | 5f045cc9e1a1 | REPLAYED-EQUIVALENT |  | docs(dnse): trim plan to native-conditional + add doc-sync + version-pin gotcha |
| 17 | 9bc86313 | cfa2679c45bc | 638e2d75 | cfa2679c45bc | REPLAYED-EQUIVALENT |  | feat(dnse): v2 client wrapper + data provider on openapi-sdk |
| 18 | a4e35b5b | e781d3c58654 | 327a1cfd | e781d3c58654 | REPLAYED-EQUIVALENT |  | feat(dnse): v2 native-conditional broker (first cut) |
| 19 | def27f73 | 240456b1e380 | e5cdb2c4 | 240456b1e380 | REPLAYED-EQUIVALENT |  | fix(dnse): bracket = NORMAL TP + native STOP SL, not native OCO |
| 20 | 9e6e6164 | 9595e43a404f | 2620833b | 9595e43a404f | REPLAYED-EQUIVALENT |  | feat(dnse): bracket via native OCO, track its working LO (externalOrderId) |
| 21 | e69080ad | 365357ab7a18 | e86e1fdc | 365357ab7a18 | REPLAYED-EQUIVALENT |  | docs(dnse): sync plan to implemented native-OCO design |
| 22 | a14acd5b | 1f3d93214083 | fb0b28fa | 1f3d93214083 | REPLAYED-EQUIVALENT |  | fix(dnse-broker): cancel conditional entries by their own book |
| 23 | fe468fe8 | 35417c2ccaa2 | c019ce40 | 35417c2ccaa2 | REPLAYED-EQUIVALENT |  | test(dnse): order-lifecycle strategies t7/t8 + -0.1% flatten guard |
| 24 | 149bff6d | e322e9c3ddd7 | 8e526d90 | e322e9c3ddd7 | REPLAYED-EQUIVALENT |  | feat(dnse-broker): classify every DNSE error into a deliberate action |
| 25 | 3192332d | f0c2a77c06f9 | 5cd94f78 | f0c2a77c06f9 | REPLAYED-EQUIVALENT |  | test(dnse): unit suite for the plugin — 97% first-party coverage |
| 26 | 9f1ff9f5 | 350104dc43bd | 8e8b449c | 350104dc43bd | REPLAYED-EQUIVALENT |  | fix(dnse-broker): graceful exit-skip + clean account-resolution error |
| 27 | eaeb67d6 | 586a04827c29 | c61b61b4 | 586a04827c29 | REPLAYED-EQUIVALENT |  | feat(dnse): expose get_expected_price + flag non-breaking changelog additions |
| 28 | c8c1f5d8 | c947b02d7352 | 6aecdc0e | c947b02d7352 | REPLAYED-EQUIVALENT |  | feat(dnse): trading-token minter (task #7) — cron + manual OTP refresh |
| 29 | bc3b75ca | 97110db22a15 | 42ed0d80 | 97110db22a15 | REPLAYED-EQUIVALENT |  | docs(dnse): trading-token minter setup — cron + Gmail app-password + manual fallback |
| 30 | 8e25f18d | 56ac57b775a8 | 8c0634cf | 56ac57b775a8 | REPLAYED-EQUIVALENT |  | docs: document the fork's broker plugins + reference plugin repos |
| 31 | bf0da8d2 | e4cf0eb2d9b1 | cf6b9186 | e4cf0eb2d9b1 | REPLAYED-EQUIVALENT |  | feat(dnse): token_status — manual 08:05 status check + live probe + OTP refresh |
| 32 | 484e0410 | cc7b4ad3100e | 124e1ab5 | cc7b4ad3100e | REPLAYED-EQUIVALENT |  | feat(dnse): Gmail creds via .env — template + minter auto-load |
| 33 | 7ece307a | 900d392130db | 963a7983 | 900d392130db | REPLAYED-EQUIVALENT |  | test(dnse): contract probe — validate_plugin_contract passes clean (task #5) |
| 34 | d62406f0 | 9c0e7d816456 | a5e60f89 | 9c0e7d816456 | REPLAYED-EQUIVALENT |  | test(dnse): 3-level graded live-test suite + first live L1 pass |
| 35 | 5bf79425 | b9fbacc9a7ac | 07fc85b2 | b9fbacc9a7ac | REPLAYED-EQUIVALENT |  | docs(dnse): drop --from from all 3 live-test plans |
| 36 | 88dd78fd | 8332fb2949d9 | c402acb1 | 8332fb2949d9 | REPLAYED-EQUIVALENT |  | test(dnse): Level 0 venue-semantics gate — mandatory pre-flight for L1/L2/L3 |
| 37 | 194bc6a1 | 91f079cabbce | 5b2ca0f0 | 91f079cabbce | REPLAYED-EQUIVALENT |  | fix(dnse): price a triggered stop THROUGH its trigger, not at it |
| 38 | c63e9635 | 0d6f6a70dde5 | 1e4687e1 | 0d6f6a70dde5 | REPLAYED-EQUIVALENT |  | fix(dnse): freeze the L3 stop-loss level — a moving stop amends, DNSE 500s |
| 39 | d0b7bfe7 | 01cd471c008a | 734604d2 | 01cd471c008a | REPLAYED-EQUIVALENT |  | test(dnse): L1 complete live; L2 partial; retune stop slippage + explicit cancel |
| 40 | 6ad3dec4 | dc7f5e754218 | a9f9656d | dc7f5e754218 | REPLAYED-EQUIVALENT |  | fix(dnse): drop the phantom oca_type from l2b's strategy.exit |
| 41 | f801d926 | ff6d7e7bbd30 | ab1df9bf | ff6d7e7bbd30 | REPLAYED-EQUIVALENT |  | test(dnse): staged 4-test place/cancel probe with full live logging |
| 42 | e19a335d | 314ee0e37b16 | 1a11d8b6 | 314ee0e37b16 | REPLAYED-EQUIVALENT |  | test(dnse): gate the staged probe on an explicit trade window, not isrealtime |
| 43 | f46db6fc | 9051246dede9 | 447116ba | 9051246dede9 | REPLAYED-EQUIVALENT |  | fix(dnse): a 2xx cancel is an ACK, not a completion — verify at the venue |
| 44 | 458b7f25 | 30658915194e | 08834077 | 30658915194e | REPLAYED-EQUIVALENT |  | fix(dnse): session-phase safety for L0, ATC cancel classification, corrected claims |
| 45 | aceb960e | 99bdd34f452e | 04543312 | 99bdd34f452e | REPLAYED-EQUIVALENT |  | fix(dnse): cascade a cancelled entry to its exit legs (#19) + live-trace regression |
| 46 | 9931a58f | 0305f5ef1d67 | 57d9ac7d | 0305f5ef1d67 | REPLAYED-EQUIVALENT |  | test(dnse): fake venue reproducing the measured quirks — verifies #19/#20 offline |
| 47 | 925e33c4 | c659a6220a2a | 37d343a5 | c659a6220a2a | REPLAYED-EQUIVALENT |  | fix(dnse): clamp GTD to the contract's final trade date — and #20 VERIFIED LIVE |
| 48 | 91f756ba | ec726fb4f6e0 | 199e4583 | ec726fb4f6e0 | REPLAYED-EQUIVALENT |  | test(dnse): #19 cascade VERIFIED LIVE — cancelling the entry now takes its exit leg |
| 49 | a4bfeca3 | 62faca27ca51 | 29b9cdb1 | 62faca27ca51 | REPLAYED-EQUIVALENT |  | test(dnse): extend the staged probe to nine no-fill cases (T5-T9) |
| 50 | 4fea9a83 | 713cc142c960 | 8e66ab27 | 713cc142c960 | REPLAYED-EQUIVALENT |  | revert(dnse): plugin-side exit cascade — T5 measured it breaking engine ownership |
| 51 | 9fb8cefe | 38a9ac8f64e3 | 5a7fc13d | 38a9ac8f64e3 | REPLAYED-EQUIVALENT |  | test(dnse): T6-T9 all PASS live; T5 evidence for the reverted cascade |
| 52 | 451d32c3 | 2b9c3c33f1fd | c73a79ab | 2b9c3c33f1fd | REPLAYED-EQUIVALENT |  | test(dnse): T10 dual-strategy isolation PASS — two concurrent engines, one account |
| 53 | 89096bf0 | d7954aed2ea2 | 3782ee38 | d7954aed2ea2 | REPLAYED-EQUIVALENT |  | test(dnse): reusable runner for the T10 dual-strategy isolation test |
| 54 | c99329f2 | f422f27af68a | ec8f00e9 | f422f27af68a | REPLAYED-EQUIVALENT |  | test(dnse): merge the backtest oracle into a staged Level-3 fill test; retire superseded files |
| 55 | 91aefcef | ad4a53b650c7 | a65bc3fd | ad4a53b650c7 | REPLAYED-EQUIVALENT |  | test(dnse): OCA-semantics cases T11-T13 in the staged place/cancel probe |
| 56 | 559d46f1 | e686cf731d0f | 11c6d647 | e686cf731d0f | REPLAYED-EQUIVALENT |  | docs(dnse): live-test suite README + link it from the repo README |
| 57 | fc3679bd | bebab4fcf41c | 65f85ddf | bebab4fcf41c | REPLAYED-EQUIVALENT |  | docs: force-track a sandbox-aware CLAUDE.md |
| 58 | 8853f936 | 71dc44a9e44f | 9f350de5 | 71dc44a9e44f | REPLAYED-EQUIVALENT |  | docs+data: sandbox runs transpile via PyPI and backtest via tracked OHLCV |
| 59 | 27b6546b | e9ed76fa9660 | e6592c13 | e9ed76fa9660 | REPLAYED-EQUIVALENT |  | docs: link the trading-token workflow from the DNSE testing section |
| 60 | 3323f395 | e880c9b4260e | 14f6ec65 | e880c9b4260e | REPLAYED-EQUIVALENT |  | docs: live-run session mechanics learned the hard way this week |
| 61 | da5682b4 | d7cda7c0fa8f | c8cd6be4 | d7cda7c0fa8f | REPLAYED-EQUIVALENT |  | docs(dnse): canonical test naming — Live-L<level>-<case> — one registry |
| 62 | fef8a013 | 9c07d6826f5c | 46774a3f | 9c07d6826f5c | REPLAYED-EQUIVALENT |  | test(dnse): full place/cancel regression on 6.8.5 — Live-L1-T01..T13 + T10 all PASS |
| 63 | 7245cdcf | c8be111eae8e | 24577dd9 | c8be111eae8e | REPLAYED-EQUIVALENT |  | test(dnse): Live-L4 bar parity + latency probe — first live PASS (#24) |
| 64 | b6ff77b6 | e5e066256b7f | 35f42456 | e5e066256b7f | REPLAYED-EQUIVALENT |  | docs(dnse): registry rows for Live-L4 (smoke PASS 08-17) |
| 65 | dbdffb17 | ef7df564e4ec | 4c6950c7 | ef7df564e4ec | REPLAYED-EQUIVALENT |  | chore(dnse): reset staged toml to fail-safe; refresh tracked 1m OHLCV snapshots |
| 66 | 26282062 | 9b10127fc40f | d824681d | 9b10127fc40f | REPLAYED-EQUIVALENT |  | docs: never use GitHub 'Sync fork' on this repo — rebase flow only |
| 67 | 26468e02 | 87f80401f039 | b45e444a | 87f80401f039 | REPLAYED-EQUIVALENT |  | test(dnse): double-check fixes to live suites + L4 reopen evidence + register Live-L4-T03 |
| 68 | 8c2d5505 | fef24e397149 | aa2648d8 | fef24e397149 | REPLAYED-EQUIVALENT |  | test(dnse): Live-L4-T03 measured — ATC bar delivery + params chunk A evidence |
| 69 | 13ecf516 | 4741f451fc99 | 08df6bbd | 4741f451fc99 | REPLAYED-EQUIVALENT |  | chore(dnse): runtime state after 08-17 PM live runs |
| 70 | aaa0caba | 796b3fd540f0 | 693ee6e7 | 796b3fd540f0 | REPLAYED-EQUIVALENT |  | fix(dnse): provider syminfo — Python weekday numbering + real stock sessions (#30) |
| 71 | 102dcc7b | 5540965fdc59 | e28482ec | 5540965fdc59 | REPLAYED-EQUIVALENT |  | fix(lib): session.islastbar fires on auction settlement prints (#29) |
| 72 | 2b0b9dee | f14441b62029 | c6ba0130 | f14441b62029 | REPLAYED-EQUIVALENT |  | test(dnse): direct probes for Live-L1-T14/T15/T17 + registry rows (card #22) |
| 73 | 1637b421 | 8d7e889d2f38 | 90ac81c2 | 8d7e889d2f38 | REPLAYED-EQUIVALENT |  | fix(dnse): oca_cancel NATIVE -> SOFTWARE — entry groups are venue-unlinked (#33) |
| 74 | 7c22bf0b | 1797bcd03f37 | b4257a96 | 1797bcd03f37 | REPLAYED-EQUIVALENT |  | feat(binance): spot broker plugin — testnet-proven via DNSE-style staged suite |
| 75 | bc50d4bb | a6eda997ba04 | fc322920 | a6eda997ba04 | REPLAYED-EQUIVALENT |  | fix(dnse): crossed stop at placement -> immediate marketable LO (#34) |
| 76 | 81f7c8f0 | b53279b56716 | cc3ee68f | b53279b56716 | REPLAYED-EQUIVALENT |  | docs(dnse): risk tiers in the canonical test plan — FILL cases are the gated top tier |
| 77 | 481af56f | 6f1e03eb620a | 146926ab | 6f1e03eb620a | REPLAYED-EQUIVALENT |  | test(dnse): morning no-fill ladder — T26b/T30/T15/T17 PASS, T16 bug found (#36) |
| 78 | 749ba9ec | b1e90718fb0a | fa59f6e0 | b1e90718fb0a | REPLAYED-EQUIVALENT |  | docs(dnse): correct T16/#36 root cause after store review — orders never journaled, not a missing seam |
| 79 | b12e9448 | fd8278d83068 | de0cfe6d | fd8278d83068 | REPLAYED-EQUIVALENT |  | fix(lib): halt on stale slice views and wrong-sized matrix add_row/add_col |
| 80 | 28784346 | fdac45510a8e | efb16580 | fdac45510a8e | REPLAYED-EQUIVALENT |  | chore(dnse): runtime state after 08-18 AM no-fill ladder |
| 81 | ef5dabfa | 74500f55081b | a02e6aa3 | 74500f55081b | REPLAYED-EQUIVALENT |  | test(dnse): Live-L1-T18-ImmediateCancel — no minimum-rest rule, PASS both books |
| 82 | 8a125354 | fa3eafb1bb0c | 954053b5 | fa3eafb1bb0c | REPLAYED-EQUIVALENT |  | docs(dnse): two-book venue mechanic + F10 PASS / F11 blocked by #39 |
| 83 | d4240605 | 3918d7c63f9a | c3c3c511 | 3918d7c63f9a | REPLAYED-EQUIVALENT |  | fix(dnse): adopt a triggered stop's NORMAL-book child — stop-entry fills were invisible (#39) |
| 84 | 46421f5d | e79d48cb10a3 | 83661c76 | e79d48cb10a3 | REPLAYED-EQUIVALENT |  | test(dnse): Live-L1-T14 provisional PASS — 8/8 ATC cancel refusals, orders left to expire |
| 85 | 6a1901d6 | 360836f1ade6 | 34ab8eaf | 360836f1ade6 | REPLAYED-EQUIVALENT |  | test(dnse): Live-L1-T14 FINAL PASS — both orders Expired at the close |
| 86 | 71db448d | c9aebe1962c5 | 815672d6 | c9aebe1962c5 | REPLAYED-EQUIVALENT |  | perf(dnse): order polling 2s -> 0.5s (config-driven) + F11 flattens on sight |
| 87 | 9b63d676 | 3d071d8664bd | 7cff21bb | 3d071d8664bd | REPLAYED-EQUIVALENT |  | docs(dnse): Live-L3-F11 PASS — #39 and #33 both live-verified in one run |
| 88 | c705db19 | d58260bab8ad | 5d59a8db | d58260bab8ad | REPLAYED-EQUIVALENT |  | test(dnse): fill ladder redesigned — the OPERATOR closes, the strategy never does |
| 89 | b278a5e4 | d77e48932180 | 69947a13 | d77e48932180 | REPLAYED-EQUIVALENT |  | docs(dnse): mandatory live-session run order — L0, then no-fill, then fills LAST |
| 90 | f9869fcf | 317d9c16d2f1 | 228f552c | 317d9c16d2f1 | REPLAYED-EQUIVALENT |  | docs(dnse): no-fill gate is the T01-T03 smoke, not the full ladder |
| 91 | b8179e22 | b9e935b9f86e | eabe63c6 | b9e935b9f86e | REPLAYED-EQUIVALENT |  | test(dnse): fill-test protocol fixes from the 2026-08-19 review (#42 B/C/D + protection timing) |
| 92 | 0e717179 | c9a51cf9a034 | 00d1ce98 | c9a51cf9a034 | REPLAYED-EQUIVALENT |  | fix(dnse): bound + retry stop-child adoption, escalate when it never resolves (#42-A) |
| 93 | bf6f30f5 | 78c33dbef4c3 | 66d7ddf0 | 78c33dbef4c3 | REPLAYED-EQUIVALENT |  | feat(dnse): venue toolkit — stop hand-writing probes for live state |
| 94 | e75a7cc2 | 8d4f553bb30c | 8acd39a4 | 8d4f553bb30c | REPLAYED-EQUIVALENT |  | refactor(dnse): venue toolkit uses the vendored SDK, not a private bypass |
| 95 | 1983c2b1 | 149e791c8f1f | c03f4d45 | 149e791c8f1f | REPLAYED-EQUIVALENT |  | feat(dnse): venue toolkit refuses writes on a bad token; token check no longer crashes on corrupt state |
| 96 | 7e1e192e | e6e09e861dd7 | db547fee | e6e09e861dd7 | REPLAYED-EQUIVALENT |  | test(dnse): contract roll validated — alias resolves new contract, GTD clamp moves, full L0 PASS |
| 97 | 146c9ffc | 28bda7c687c0 | 1916676f | 28bda7c687c0 | REPLAYED-EQUIVALENT |  | fix(dnse): #43 — unresolved OCO umbrella queued for poll-loop adoption (S1) |
| 98 | 5218b955 | bba549834908 | 8addf77e | bba549834908 | REPLAYED-EQUIVALENT |  | test(dnse): Live-L1 T01-T03 smoke PASS on 41I1G9000 — first staged probe post-roll |
| 99 | 30a55174 | 2c767dfa8ed6 | f4a8c817 | 2c767dfa8ed6 | REPLAYED-EQUIVALENT |  | fix(dnse): #45 #47 — cancel-path false-confirmations |
| 100 | aaa2c13f | 24289d3c0148 | e1337f86 | 24289d3c0148 | REPLAYED-EQUIVALENT |  | fix(dnse): #49 — get_position speaks the engine contract vocabulary (long/short) |
| 101 | e29a36be | 379e8509e44b | daf8ae05 | 379e8509e44b | REPLAYED-EQUIVALENT |  | feat(engine): #48 — periodic position-drift detector + executable divergence matrix |
| 102 | 2a3edc13 | b49993021dd0 | 78af73a0 | b49993021dd0 | REPLAYED-EQUIVALENT |  | feat(dnse): #37 — dual-mode feed (S1'): tick synthesis behind feed_mode dispatch |
| 103 | 20a0a8a2 | c91baa3adf97 | c8913a86 | c91baa3adf97 | REPLAYED-EQUIVALENT |  | fix(dnse): #37 — /trades/latest interleaves two boards with INCOMPARABLE counters |
| 104 | 0c0bb2f9 | d0848ea061ac | 3efd9f3f | d0848ea061ac | REPLAYED-EQUIVALENT |  | test(dnse): Live-L1-T32-AtoProbe PASS — first-ever ATO order-acceptance measurement |
| 105 | 5a39ccc3 | c32d37096551 | 34070ae5 | c32d37096551 | REPLAYED-EQUIVALENT |  | docs: INVALID_TRADING_TOKEN on conditional writes is a venue anomaly, not a token problem |
| 106 | cdbf11af | 5084d64e10f7 | a5f64d8d | 5084d64e10f7 | REPLAYED-EQUIVALENT |  | chore(dnse): data snapshot from the T32 live session (dnsebroker VN30F1M @1) |
| 107 | 048f6191 | 746b2b6650fa | 5eaa227b | 746b2b6650fa | REPLAYED-EQUIVALENT |  | docs(dnse): master test plan v3 — dual-system execution policy |
| 108 | b2360e2d | 41e436acf8df | e319814e | 41e436acf8df | REPLAYED-EQUIVALENT |  | docs(dnse): registry — F01-F08 park reason corrected (stale #39 block) |
| 109 | 85e8a5c9 | c997d0e7f4ee | 3aaa5e69 | c997d0e7f4ee | REPLAYED-EQUIVALENT |  | docs(dnse): backfill scope is T19-T31 (T31 scripts exist with no registry row) + annotation-rot sweep |
| 110 | 8bd9fb0d | e111d96ec68e | 78320441 | e111d96ec68e | REPLAYED-EQUIVALENT |  | docs(dnse): master plan — offline-suite map, red-first anchor index, evidence-class rules |
| 111 | 5ba9b73c | 1ef69de4c8de | 93bee8ce | 1ef69de4c8de | REPLAYED-EQUIVALENT |  | docs(dnse): correct tick-fleet ceiling (5 per key at the 2s default, not 2) + add T27 to the v3 queue |
| 112 | 5df29f0d | 41ac11241164 | 1ef98857 | 41ac11241164 | REPLAYED-EQUIVALENT |  | docs(dnse): dnse_v2_fix_plan complete — three reference-plugin audits folded in |
| 113 | f97b5213 | a5f8a42a6597 | 2d9d6832 | a5f8a42a6597 | REPLAYED-EQUIVALENT |  | fix(dnse-docs): Market-Data WS venue fact REFUTED — plan corrected + capture probe |
| 114 | c7c207b6 | 55f3e61561fd | 9f2829a3 | 55f3e61561fd | REPLAYED-EQUIVALENT |  | feat(dnse): trading-WS path fully discovered — probe --trading mode + plan update |
| 115 | 6ddf18a6 | c064fc2d07fd | d8cbf98d | c064fc2d07fd | REPLAYED-EQUIVALENT |  | docs: DNSE WS testing rules — how the silent-WS false verdict happened and how to never repeat it |
| 116 | 03e17814 | 4d6cce50c40c | 2bafaac3 | 4d6cce50c40c | REPLAYED-EQUIVALENT |  | docs(dnse): fix plan — executive summary tables (skip/change/new) folded into the plan doc |
| 117 | a472fb46 | c7c278dfa531 | b82b4af0 | c7c278dfa531 | REPLAYED-EQUIVALENT |  | docs(dnse): fix plan v2 — 11-agent adversarial review verdicts + resequenced execution order |
| 118 | d979985b | 86247bf9b8d7 | 9a7517e9 | 86247bf9b8d7 | REPLAYED-EQUIVALENT |  | docs(dnse): A/B comparison of two blind adversarial reviews of the fix plan |
| 119 | 21ba09ad | 1de9e2ecbec4 | a7cd407a | 1de9e2ecbec4 | REPLAYED-EQUIVALENT |  | docs(dnse): merge round-2 A/B corrections into fix plan v2 (operator-approved) |
| 120 | cbbd4c73 | 9c1ec94c2628 | e47d0e1c | 9c1ec94c2628 | REPLAYED-EQUIVALENT |  | fix(dnse): cancel disposition from positive observations only — never CANCEL_CONFIRMED on a fill-raced cancel (#55) |
| 121 | 748dbb39 | 42dae244cf59 | adef2f99 | 42dae244cf59 | REPLAYED-EQUIVALENT |  | fix(dnse): migrate operational scripts off the removed _cancel_one (#55 follow-up) |
| 122 | 593ac155 | e229b0727782 | 9d382a11 | e229b0727782 | REPLAYED-EQUIVALENT |  | fix(dnse): page completeness — a truncated read must never look flat or complete (#62, closes #57 #61) |
| 123 | f253d25d | ffb3eb0c528d | 48113d92 | ffb3eb0c528d | REPLAYED-EQUIVALENT |  | fix(dnse): feed-health ladder — a persistently failing poll can no longer silence the fill feed (#54) |
| 124 | 5b141a61 | 1b9a816faa28 | 7bfa5440 | 1b9a816faa28 | REPLAYED-EQUIVALENT |  | fix(dnse): retire the INVALID_TRADING_TOKEN auto-retry — one refusal, one write, complete guidance (#58) |
| 125 | 042abce6 | f37e691a40a1 | 9c200655 | f37e691a40a1 | REPLAYED-EQUIVALENT |  | refactor(dnse): Phase 0a — truthful contract: config extraction, generic provider, honest markers, validator stub check (#66) |
| 126 | 3ab5e69e | 3e884e0de6ea | 5c1ade17 | 3e884e0de6ea | REPLAYED-EQUIVALENT |  | fix(dnse): transport error split — no raw urllib3 exception reaches the engine (#67, Phase A1) |
| 127 | 63480cb8 | 30de126a7d9d | 6f7c1afe | 30de126a7d9d | REPLAYED-EQUIVALENT |  | fix(dnse): OA-400 with the Authorization message is an AUTH failure, not an order reject (#68) |
| 128 | 1b008821 | 51421e7d1796 | f65637ba | 51421e7d1796 | REPLAYED-EQUIVALENT |  | feat(dnse): persist-first journal + journal-rooted restart adoption — the A2 keystone (#36) |
| 129 | bc6f700c | dc6deb096c2a | 15fd8d65 | dc6deb096c2a | REPLAYED-EQUIVALENT |  | fix(dnse): fill slices book at their own price via the budget clamp; restart re-emission dies (#56, item 5) |
| 130 | a490e263 | c43cd858b5fc | da666387 | c43cd858b5fc | REPLAYED-EQUIVALENT |  | docs(dnse): Phase B loop-resumable runbook — per-card scope, red-first test plans, state detection for 5h-cadence resumption |
| 131 | 379a06a2 | c33f670a5b6e | 574459f2 | c33f670a5b6e | REPLAYED-EQUIVALENT |  | docs(dnse): runbook model tiering — opus panels, sonnet for mechanical card/doc shipping, leader for judgment only |
| 132 | 4fc8d673 | 88bdcf2c1329 | 824b65b1 | 88bdcf2c1329 | REPLAYED-EQUIVALENT |  | feat(dnse): recovery verdict ladder — loud lost-reply quarantine, sibling-strand report, matcher rejected by design (#71, Phase B1) |
| 133 | 1d1ea843 | bd5fdfa22a26 | 21e2a72a | bd5fdfa22a26 | REPLAYED-EQUIVALENT |  | docs(dnse): runbook B2 state — four consumer sites verified in today's engine, probe mechanism identified |
| 134 | 99789008 | ecae5a55181e | ffb80af3 | ecae5a55181e | REPLAYED-EQUIVALENT |  | fix(dnse): ownership clamp covers the replayed-close sibling; fills keep the journal exposure ledger live (#73, Phase B2) |
| 135 | 7b8a27c2 | 1798aa03ced3 | a5383f21 | 1798aa03ced3 | REPLAYED-EQUIVALENT |  | feat(dnse): external-cancel residue detector — history-only CANCELLED authority, paginated history reader (#74, Phase B3; fixes #69) |
| 136 | c9c311a8 | 204a30e0d9a7 | b85b0237 | 204a30e0d9a7 | REPLAYED-EQUIVALENT |  | fix(dnse): holiday-aware session phase — verification-dated VN holiday table (#70) |
| 137 | e7e23a7c | 28f76e38dfed | 19bc7921 | 28f76e38dfed | REPLAYED-EQUIVALENT |  | docs(dnse): measure the two open B3 premises read-only — /orders/history is NORMAL-book-only; closed-session books are readable-and-empty |
| 138 | 516c3c0a | b3ac4dcd8f00 | 9bfc211e | b3ac4dcd8f00 | REPLAYED-EQUIVALENT |  | test(dnse): close the three suite double-check gaps (#76) |
| 139 | 7043bad7 | 9f740e086c16 | 4276d01a | 9f740e086c16 | REPLAYED-EQUIVALENT |  | fix(engine): journal-rooted restart entry snapshot for SOFTWARE-idempotency venues; wire-symbol reconstruction join (#77) |
| 140 | f7ecb054 | 21995d327175 | 03e3f2a3 | 21995d327175 | REPLAYED-EQUIVALENT |  | diag(engine): TEMP #77 breadcrumb — settle_restart_state latch state (revert after live) |
| 141 | ebddde60 | e3aa91a73bbb | 66cadb54 | e3aa91a73bbb | REPLAYED-EQUIVALENT |  | fix(dnse): clear stale terminal markers when a deterministic coid row is reopened (#77) |
| 142 | 75e6b269 | 2b1d3a244c49 | c5e7e7d4 | 2b1d3a244c49 | REPLAYED-EQUIVALENT |  | docs(dnse): Live-L1-T16 restart adoption PASS live 2026-09-07 (#77) |
| 143 | 7135898c | 731795e953e8 | 3ac7c9b0 | 731795e953e8 | REPLAYED-EQUIVALENT |  | fix(dnse): derive order_type from stopPrice, not hardcoded LIMIT (#78) |
| 144 | f2ef7e60 | 3153c1295396 | e32538a8 | 3153c1295396 | REPLAYED-EQUIVALENT |  | test(dnse): full no-fill regression T1-T13 re-verified live @1m; candle-color gates removed (2026-09-07) |
| 145 | 71320e5d | 83335b2e0fe6 | 0a41313b | 83335b2e0fe6 | REPLAYED-EQUIVALENT |  | test(dnse): FILL tier first systematic run — F1,F3,F4 PASS, F2 partial, F5 CRITICAL #82 (naked marketable exit while flat); ladder stopped for safety |
| 146 | 01539d10 | 6a096a1cb8df | e37c7c3c | 6a096a1cb8df | REPLAYED-EQUIVALENT |  | docs(dnse): FILL registry rows graded from the 2026-09-07 systematic run — F01/F03/F04 PASS, F02 partial, F05 CRITICAL #82, F06-F08 parked |
| 147 | d0ffde90 | e32625f2b869 | 8d5240c0 | e32625f2b869 | REPLAYED-EQUIVALENT |  | chore(data): refresh tracked OHLCV snapshots from today's live runs |
| 148 | c39cf26c | 76ab2251f299 | e96f80b5 | 76ab2251f299 | REPLAYED-EQUIVALENT |  | fix(engine): marketable whole-row exit must not close a flat/wrong-side position (#82) |
| 149 | 673c1844 | 6f1dc59b0c61 | e9d82a3b | 6f1dc59b0c61 | REPLAYED-EQUIVALENT |  | docs(dnse): F05 registry — #82a fixed (guard held live), #82b open (native protection fills naked pre-entry-fill) |
| 150 | ee081ea0 | da04ab051603 | 03470f5a | da04ab051603 | REPLAYED-EQUIVALENT |  | fix(engine): withhold protective exits until the entry fills on standalone-exit venues; clamp exit qty to the live position (#82b) |
| 151 | 5375f93a | 1ab558418de4 | 79cb7651 | 1ab558418de4 | REPLAYED-EQUIVALENT |  | docs(plan): live-trading platform research + broker-plugin lessons |
| 152 | f2782213 | f9eb6e7dfe19 | 7e3b2366 | f9eb6e7dfe19 | REPLAYED-EQUIVALENT |  | chore(dnse): F5 re-grade toml window state + refreshed 1m broker data snapshot |
| 153 | bfe0a71e | 53f8d126d425 | 7ce3e588 | 53f8d126d425 | REPLAYED-EQUIVALENT |  | docs(dnse): archive session artifacts — 2026-08-31 read-only battery probe (masked, card-referenced) + the #59 A/B round's V1-under-review plan and item-6 netting review |
| 154 | 65d89909 | 07c1a6927efe | 70ff8ddd | 07c1a6927efe | REPLAYED-EQUIVALENT |  | fix(binance): implement min_sellable_base for the 6.8.9 spot-inventory protocol |
| 155 | fe241bad | f8e34d8762a3 | 353ccf04 | f8e34d8762a3 | REPLAYED-EQUIVALENT |  | fix(binance): declare exit_orders_execute_standalone — spot exits are naked-executable (#82b) |
| 156 | 8115a68a | f4bd7b7828b0 | eea2423e | f4bd7b7828b0 | REPLAYED-EQUIVALENT |  | docs: third venue-property axis (standalone vs attach exits) + #82b window cost |
| 157 | dc9c2205 | d6a8007a1ea0 | b035069c | d6a8007a1ea0 | REPLAYED-EQUIVALENT |  | test(binance): pyramid + scale-out probe (Live-B3) — multi-entry and partial exits, live on testnet |
| 158 | f2d1f08e | e72ea08701e1 | 18ba66bc | e72ea08701e1 | REPLAYED-EQUIVALENT |  | docs(binance): document level semantics and close the B3 tier gap |
| 159 | e9fffd60 | 47c1cbe8305f | 320bca1f | 47c1cbe8305f | REPLAYED-EQUIVALENT |  | fix(dnse): derivatives continuous starts 09:00, not 09:15 (operator-corrected 2026-09-08) |
| 160 | 394e3eb1 | 59073e14b121 | 90000569 | 59073e14b121 | REPLAYED-EQUIVALENT |  | test(binance): re-verify B4-B8 on 6.9.1+#82b; correct three stale registry rows |
| 161 | 65fd9916 | faeb18135e83 | 4093d94c | faeb18135e83 | REPLAYED-EQUIVALENT |  | docs(dnse): F05 #82 re-graded PASS live 2026-09-08 (#82b withhold proven); new blocker #83 (entry-stop self-cancel false-quarantine) parks F06-F08 |
| 162 | 9ebb843c | c643edd08c23 | 8995d1f1 | c643edd08c23 | REPLAYED-EQUIVALENT |  | fix(engine): own parked forced-cancel observed as CANCELLED is our cancel landing, never an external cancel (#83) |
| 163 | cc7fc78f | e6c07d68ae4c | 95227c6c | e6c07d68ae4c | REPLAYED-EQUIVALENT |  | fix(dnse): arm the core bar-feed staleness watchdog (40->16 bars) + instance-scoped failed-poll ladder (#81) |
| 164 | b8102601 | 18677ba9efed | f426bfbe | 18677ba9efed | REPLAYED-EQUIVALENT |  | docs(dnse): venue fact — NORMAL amend edits one field per call (both keys required); combined price+qty modify 400s (measured 2026-09-08) |
| 165 | 0a4d602e | bd2793819d32 | fce5c110 | bd2793819d32 | REPLAYED-EQUIVALENT |  | fix(dnse): conditional-book modifies reach the venue — outcome-gated cancel+replace for entries, loud park for exits (#85) |
| 166 | 1c5110f7 | a3ee583e46df | 8860d743 | a3ee583e46df | REPLAYED-EQUIVALENT |  | fix(dnse): NORMAL-book amend splits into one-changed-field-per-PUT legs, diffed against venue truth (#86) |
| 167 | 1ad4e261 | 46c55b340a1d | 854da6e4 | 46c55b340a1d | REPLAYED-EQUIVALENT |  | fix(dnse/engine): stale-fill cancel poisoning + dual both-set entry handlers (#87) |
| 168 | c438a029 | 7689cf2bfcd3 | 3df0cb00 | 7689cf2bfcd3 | REPLAYED-EQUIVALENT |  | test(dnse): no-fill probes T19-T21 for the #85/#86/#87 live grades (states 12-14) |
| 169 | 472a855b | c29ff8abbcd4 | 2d560972 | c29ff8abbcd4 | REPLAYED-EQUIVALENT |  | docs(dnse): Live-L1-T19/T20/T21 PASS live 2026-09-08 — #85/#86/#87 no-fill grades (venue-record verified) |
| 170 | 00739544 | 32319a2dff80 | 01e6ed03 | 32319a2dff80 | REPLAYED-EQUIVALENT |  | docs(dnse): Live-L1-T16 re-graded PASS (substance) 2026-09-08 under the #87 restore filter |
| 171 | 3e910870 | 2c0a83cae168 | 8ebead3f | 2c0a83cae168 | REPLAYED-EQUIVALENT |  | fix(dnse): close-first flatten with journal-rooted protection sweep + fill-tier runner (#91) |
| 172 | f56a3511 | fb3a4828d6e8 | 9e7d2280 | fb3a4828d6e8 | REPLAYED-EQUIVALENT |  | test(dnse): id-discriminating T16 grader (#90) + WS probe close reporting and --dual mode (#92) |
| 173 | 4c929579 | b2d212ad208c | 540762c8 | b2d212ad208c | REPLAYED-EQUIVALENT |  | docs(dnse): adversarial 10-commit review record + test-plan additions (T22-T24, F9-F12, L4 rows) + suite-command deselect |
| 174 | bfe15c6a | 66f67d8b4ad3 | ac249330 | 66f67d8b4ad3 | REPLAYED-EQUIVALENT |  | fix(dnse/engine): venue expiry is a lifecycle end, not an external cancel (#94); ALREADY_FILLED exempt from the cancel-scope prune (#95) |
| 175 | 145711e0 | 8499c24e2b8f | 30b05426 | 8499c24e2b8f | REPLAYED-EQUIVALENT |  | fix(dnse): bracket exit modifies never fabricate success — journal-rooted OCO-origin routing to the loud park (#93) |
| 176 | 2620a61a | 782e50e33ad2 | fc6bd2c4 | 782e50e33ad2 | REPLAYED-EQUIVALENT |  | feat(dnse): tick->bar synthesis foundation for sub-minute timeframes (#80 phase 1) |
| 177 | 9923a0de | 71eb8701e364 | ff00d7b5 | 71eb8701e364 | REPLAYED-EQUIVALENT |  | feat(dnse): sub-minute live feed — WS tick synthesis coexisting with venue candles (#100, closes-candidate #98) |
| 178 | 2c8424b5 | e6ce8be93d24 | 92fb0a8b | e6ce8be93d24 | REPLAYED-EQUIVALENT |  | feat(dnse): provider-mode warmup for synthesized timeframes reads the LTF store, never downloads (#100) |
| 179 | 42579d33 | 74e3dc541f91 | 4f85b6d5 | 74e3dc541f91 | REPLAYED-EQUIVALENT |  | fix(dnse): empty synthesized-timeframe warmup file no longer aborts a live launch (#100) |
| 180 | 84af8288 | 629e81b02d2f | 40341963 | 629e81b02d2f | REPLAYED-EQUIVALENT |  | docs(dnse): Live-L1 T19/20/21 @15S PASS live 2026-09-09 — #100 tick-synthesis feed graded live |
| 181 | e1dbeed4 | 870676ac9bb5 | 6d5acdda | 870676ac9bb5 | REPLAYED-EQUIVALENT |  | feat(dnse): event-clock probe pacing — TESTING-ONLY dnse_event entry point (#101) |
| 182 | 14a1af6c | 2df2ab977be6 | ecd8ad60 | 2df2ab977be6 | REPLAYED-EQUIVALENT |  | test(dnse): F9 bracket-trailing stage (oracle-smoked) + S4 umbrella probe + event-clock config note (#93/#101) |
| 183 | fcaa6d79 | dcdb76ab6f42 | fd9742e6 | dcdb76ab6f42 | REPLAYED-EQUIVALENT |  | fix(dnse): flatten attribution scoped by account + day for reusable NORMAL ids (#96) |
| 184 | d26c7af9 | 54c1719720aa | 1e205cae | 54c1719720aa | REPLAYED-EQUIVALENT |  | docs(dnse): T22 expiry-routing PASS live 2026-09-09; DAY expiry is a ~15:04 batch, not 14:45-sharp (#94, #98, #102) |
| 185 | fdfd7a31 | 4e588c832f2d | b843e675 | 4e588c832f2d | REPLAYED-EQUIVALENT |  | feat(dnse): route market indices via type=INDEX + clamp the venue's 14:45 ATC bars (#104) |
| 186 | ea8e99aa | 923360dc8409 | be3eabaf | 923360dc8409 | REPLAYED-EQUIVALENT |  | docs(dnse): record the two-window ATO + index 14:45 bad-bar venue facts (#104, #21) |
| 187 | de2fc5da | a346f5d55ab1 | 5a03ac8b | a346f5d55ab1 | REPLAYED-EQUIVALENT |  | docs: CLAUDE.md — correct the -500 warmup claim, add cross-symbol recipe + not-a-bug list |
| 188 | c0c6466b | d385de8a393f | f227585d | d385de8a393f | REPLAYED-EQUIVALENT |  | test: pin the flip-vs-armed-exit cross-engine behaviour + commit its probes |
| 189 | a9eb4064 | fac276965739 | afd95e16 | fac276965739 | REPLAYED-EQUIVALENT |  | docs: CLAUDE.md — the live 'no --security flags' claim was UNVERIFIED |
| 190 | 1b5739db | cf3ddd5b8656 | 818743b9 | cf3ddd5b8656 | REPLAYED-EQUIVALENT |  | test: correct the flip/exit mechanism and pin its order dependence (#105) |
| 191 | 06f02a62 | 607d8977c2f6 | 7372f8f5 | 607d8977c2f6 | REPLAYED-EQUIVALENT |  | test(strategy): cover the whole documented strategy surface (96 members) |
| 192 | 9e9eb64d | 1ab19acb3959 | d6154e7f | 1ab19acb3959 | REPLAYED-EQUIVALENT |  | fix(cli): converge the -N bars warmup instead of freezing on a market closure (#106) |
| 193 | 085a59f9 | 5be0a5732c75 | a59a8ba2 | 5be0a5732c75 | REPLAYED-EQUIVALENT |  | docs: correct the stale "REST-only — no WebSocket transport" claim (DNSE uses WS for #100 sub-minute bars) |
| 194 | 247da503 | bdb21e46cd33 | 888a0f99 | bdb21e46cd33 | REPLAYED-EQUIVALENT |  | docs(dnse): accept + document the ~1-bar unprotected fill window; fix WS probe channel case (#107) |
| 195 | e258c83f | 80ccf14fbeb4 | 46ab6e11 | 80ccf14fbeb4 | REPLAYED-EQUIVALENT |  | test(dnse): harden live-test backstops — manual PnL% + cancel/close ordering (#108) |
| 196 | 7e395820 | ff2e19ca32f4 | 65211a53 | ff2e19ca32f4 | REPLAYED-EQUIVALENT |  | feat(dnse): reusable Sandbox order-lifecycle + WS probe; document Sandbox (#109) |
| 197 | 77d9d959 | 11c65db04d77 | 344b21e3 | 11c65db04d77 | REPLAYED-EQUIVALENT |  | chore(dnse): bump vendored SDK v2.0.0 -> v2.2.0 (#110) |
| 198 | edb444ea | 4bfcd838a600 | 7244df32 | 4bfcd838a600 | REPLAYED-EQUIVALENT |  | docs: correct DNSE Sandbox precision — NORMAL category (all order types), no price simulation |
| 199 | e82f71e9 | f7fd18b153a9 | 274dd5ca | f7fd18b153a9 | REPLAYED-EQUIVALENT |  | test(#111): read-side baseline — a sandbox WS fill frame drives an engine position update |
| 200 | b5cce774 | c792237afdd9 | ca631273 | c792237afdd9 | REPLAYED-EQUIVALENT |  | feat(broker): arm protective exits on the fill event (opt-in, #111 legs 1-2) |
| 201 | dd0a4efd | 8d3eadf35464 | a9459ebd | 8d3eadf35464 | REPLAYED-EQUIVALENT |  | test(#111): sandbox integration probe — arm-on-fill fires against a REAL sandbox fill |
| 202 | b3441b84 | 4e421671096f | 5002621b | 4e421671096f | REPLAYED-EQUIVALENT |  | test(#111): sandbox arm-on-fill probe — cover BOTH a stock (HPG) and a derivative |
| 203 | 082df91e | 416f45d5a15b | a6dd4032 | 416f45d5a15b | REPLAYED-EQUIVALENT |  | docs: CLAUDE.md — sandbox supports engine-driven testing (non-netting deals model + catalog gotchas) |
| 204 | 943f8552 | 71f7a8d2a488 | b3714040 | 71f7a8d2a488 | REPLAYED-EQUIVALENT |  | docs: CLAUDE.md — how a --live/--broker test runs (one-plugin model) + sandbox testing via probes |
| 205 | fe328113 | cdb80dba0d82 | aa210a67 | cdb80dba0d82 | REPLAYED-EQUIVALENT |  | docs: CLAUDE.md — name the replay-bars + sandbox-orders test "Sandbox Replay E2E" |
| 206 | 647fbb0b | 80437210d067 | b176db16 | 80437210d067 | REPLAYED-EQUIVALENT |  | feat(dnse): Sandbox Replay E2E harness — replay data + sandbox orders drive a real strategy (#114) |
| 207 | 1bb458e4 | 7b820c79320e | 76dc499e | 7b820c79320e | REPLAYED-EQUIVALENT |  | docs: CLAUDE.md — DNSE positions are venue-derived from fills (confirmed); venue-read reliability UNCONFIRMED (#115) |
| 208 | 51fbd533 | 6ea839e3014a | b94b3e72 | 6ea839e3014a | REPLAYED-EQUIVALENT |  | docs: CLAUDE.md — VN30 futures symbol taxonomy (alias vs dated contract; an alias is NOT a "type") |
| 209 | 825f757a | 3a01ea57166c | c5532ea6 | 3a01ea57166c | REPLAYED-EQUIVALENT |  | docs: CLAUDE.md — VN30 alias resolution CONFIRMED on prod (VN30F1M->41I1G9000, VN30F2M->41I1GA000) |
| 210 | ac6b0ac9 | 9c9a1bf549e1 | 398b9b2a | 9c9a1bf549e1 | REPLAYED-EQUIVALENT |  | docs: CLAUDE.md — CORRECT position model: get_position is account-net, WRONG granularity for multi-strategy |
| 211 | e2f1a922 | 2eaa3a732288 | 0b3df78b | 2eaa3a732288 | REPLAYED-EQUIVALENT |  | docs+test: correct CLAUDE.md to the #73 per-strategy isolation design; pin get_position=account-net invariant |
| 212 | 74b15857 | 2b2907ba288b | 30e6f6a3 | 2b2907ba288b | REPLAYED-EQUIVALENT |  | test(dnse): settle the sandbox-e2e cancel-retry loop — message-blind classification (xfail repro) |
| 213 | 7d4b6d0f | d3ad7e4804e1 | e9450855 | d3ad7e4804e1 | REPLAYED-EQUIVALENT |  | test(dnse): Sandbox Replay E2E runner (.pine-driven, order-path grading) + /sandbox-e2e skill |
| 214 | b856ef73 | 4ff1c5d13f4d | 42e2f101 | 4ff1c5d13f4d | REPLAYED-EQUIVALENT |  | test(dnse): sandbox-e2e runner names the amend-405 / run-errored cause (learned from the SL/TP example) |
| 215 | c6da1a9e | 5585f75dcb28 | 4832f93d | 5585f75dcb28 | REPLAYED-EQUIVALENT |  | docs: CLAUDE.md — record measured sandbox order-ops + fill cadence (place/cancel ok, amend 405, timer-fill) |
| 216 | 2c95f114 | 92bade533cc3 | ce8b25d2 | 92bade533cc3 | REPLAYED-EQUIVALENT |  | docs: CLAUDE.md — operator-confirmed roll mechanics (Friday-morning repoint, holiday->preceding day, finalTradeDate compact+ISO forms) |
| 217 | 2ae15284 | 703fdccd66f5 | 193766d7 | 703fdccd66f5 | REPLAYED-EQUIVALENT |  | fix(dnse): #118 GTD expiry clamp (floor+computed fallback) + #119 stock price-unit codec (đồng wire, G1/G2 guards) |
| 218 | 017e2bd1 | 4bd683b5052c | b63d5489 | 4bd683b5052c | REPLAYED-EQUIVALENT |  | docs(dnse): correct the trading-WS claim — prod order events NEVER captured (wrong channel used); prod needs subscribe_broker_order_event + a main-thread wake (#121/#107) |
| 219 | 4637b27b | bda149b8165c | 4163ff10 | bda149b8165c | REPLAYED-EQUIVALENT |  | feat(dnse/engine): #121 arm-on-fill WAKE (main-thread, deadlock-safe) + dual poll/WS transport + default ON |
| 220 | 49bc12bb | 86ec3456bed1 | cd79f2f6 | 86ec3456bed1 | REPLAYED-EQUIVALENT |  | test(engine/dnse): pin the #121 partial-entry protection gap — later partial lots left naked (xfail repro) |
| 221 | e9d05326 | 0407016b55e3 | 25323049 | 0407016b55e3 | REPLAYED-EQUIVALENT |  | fix(engine/dnse): #123 extend partial-entry protection via ADD-A-LEG per slice (no naked window) |
| 222 | 479a88f3 | 1629a35f1ff7 | 4dc8d699 | 1629a35f1ff7 | REPLAYED-EQUIVALENT |  | test(engine)+probe(dnse): #124 red-first repro + 2026-09-15 live session — cancel is EXTERNAL, prod WS order events captured, STOCK amend=cancel+replace |
| 223 | 47a3b575 | 6a68a25af1f2 | bec914e1 | 6a68a25af1f2 | REPLAYED-EQUIVALENT |  | fix(dnse/engine): #117 amend id re-map, #121 dual WS channels, #124 instrumentation, #125 venue.py book routing, #126 extend hard-reject degrade |
| 224 | 340b5791 | dd29ef611f1f | 431159a9 | dd29ef611f1f | REPLAYED-EQUIVALENT |  | fix(engine/dnse): #124 protective-exit cancel — bounded re-arm + venue-evidence classification (panel-adjudicated); acceptance xfail flips green |
| 225 | 41bac15a | d4ff3ec5e0d2 | fc273380 | d4ff3ec5e0d2 | REPLAYED-EQUIVALENT |  | docs+test(dnse): 2026-09-15 live session — same-bar SL/TP arm with a PRE-PLACED bracket; #124 re-arm confirmed live; #18 closed stale |
| 226 | 50950cd0 | d7a8b02b34ed | f5649b34 | d7a8b02b34ed | REPLAYED-EQUIVALENT |  | feat(dnse/testing): 2026-09-16 session prep — #128-OBS child-metadata instrument, WS per-socket channel discrimination, bare-stop lifecycle probe, session runners + handoff plan |
| 227 | 58787aaa | 928384ec692a | 9162bde4 | 928384ec692a | REPLAYED-EQUIVALENT |  | fix(dnse): #129 investorId parse + #130/#131 WS book facts measured live |
| 228 | f830a622 | 7608c2695ff1 | 87ac15cf | 7608c2695ff1 | REPLAYED-EQUIVALENT |  | security: redact the live account number from 37 tracked files + mask the auth banner |
| 229 | 9fce3ffd | bb5a13622f7b | def061b4 | bb5a13622f7b | REPLAYED-EQUIVALENT |  | fix(dnse/testing): gate run_ws_book_discriminator on a provably flat account |
| 230 | 059d7b3b | 77bd1ece7c9f | 450f84d5 | 77bd1ece7c9f | REPLAYED-EQUIVALENT |  | test(engine)+docs: sep16 wrap-up — #120 CONFIRMED (3 paths) + #122 split-verdict repros; measured WS facts into the docs mirror; day records |
| 231 | 9b4ae8bb | 146bfdc65e52 | e3228e77 | 146bfdc65e52 | REPLAYED-EQUIVALENT |  | fix(dnse): #134 grade the WS order transport on DELIVERY, not on a subscribe ACK |
| 232 | a72d3638 | a7bce0ea18b8 | cc00d80e | a7bce0ea18b8 | REPLAYED-EQUIVALENT |  | test(dnse): Live-L3-F13 latency runner — one arm per run, config restored on every exit |
| 233 | 9c149a84 | 5ad36f0b9df1 | f66dea6d | 5ad36f0b9df1 | REPLAYED-EQUIVALENT |  | docs+test(dnse): Friday 09-18 runbook + the #113 roll-cache probe |
| 234 | 099bf3fe | c9d1d33f80af | ef4e4598 | c9d1d33f80af | REPLAYED-EQUIVALENT |  | chore(hooks): mechanical code-review gate (operator law 2026-09-16) |
| 235 | 1700a8ff | b7666d975463 | 65d0a904 | b7666d975463 | REPLAYED-EQUIVALENT |  | chore(hooks): card-read gate + card.sh helper (operator rule: agents must read comments) |
| 236 | fe216927 | 5b61b74b121f | 5591a83a | 5b61b74b121f | REPLAYED-EQUIVALENT |  | docs: sep16 card audit — 13 open cards verified at HEAD (1 quietly-done closed, 2 outdated retitled, 2 part-done shrunk, 8 still-real incl. 2 raised); one genuine venue item (#116, needs a fill, Friday piggyback); 3 new findings (Binance EXPIRED exemption -> #79 retitle, pytest collection hole -> new card, unexplained 09-08 duplicate -> #135 hypothesis) |
| 237 | 28e48baf | 38d7c79a0cf3 | 45fbe744 | 38d7c79a0cf3 | REPLAYED-EQUIVALENT |  | fix(dnse): #135 ratchet the fill watermark + honour the configured WS endpoint |
| 238 | abd0df2f | 8c0d0b9ad303 | 66209d96 | 8c0d0b9ad303 | REPLAYED-EQUIVALENT |  | fix(engine): #120 — a refused protective-exit PLACE degrades instead of killing the process naked (panel-adjudicated S3-prime) |
| 239 | bf5096e3 | 73e2ecfac1a3 | 8f917f2d | 73e2ecfac1a3 | REPLAYED-EQUIVALENT |  | fix(dnse/tools): flatten refuses to act on an unproven position sign |
| 240 | deab273a | f2d91cb858e8 | 3b6c5dd2 | f2d91cb858e8 | REPLAYED-EQUIVALENT |  | fix(engine): #122 venue-gate the protective re-arm and make its bound stop dispatch |
| 241 | 6abe04c6 | 1ae33c0f4e8b | 884a0320 | 1ae33c0f4e8b | REPLAYED-EQUIVALENT |  | fix(engine): #122 follow-up — one flat snapshot may not retire protection |
| 242 | f6f6c24d | 3b9f8fabde07 | fd7df38c | 3b9f8fabde07 | REPLAYED-EQUIVALENT |  | docs: venue fact — a DNSE position read is not authoritative alone; two agreeing reads before anything irreversible (measured 2x 2026-09-16; enforced bf5096e3 + 6abe04c6) |
| 243 | 785dbd56 | 2a7065864c00 | 60b0fd6d | 2a7065864c00 | REPLAYED-EQUIVALENT |  | fix(engine): #122 — confirm EVERY retire verdict, and separate the reads in time |
| 244 | 1a5dc6d0 | 192e90755008 | d1c2aa0a | 192e90755008 | REPLAYED-EQUIVALENT |  | fix(engine): #122 — reconcile alone may retire protection; the inline path may not |
| 245 | cecf4d8e | c8386de05e2e | 8fcb8239 | c8386de05e2e | REPLAYED-EQUIVALENT |  | fix(engine): #122 — an external flatten must cancel our exits on native-OCA venues too |
| 246 | f56782aa | c58e1b886b8f | ada60328 | c58e1b886b8f | REPLAYED-EQUIVALENT |  | fix(engine): #122 — external-flatten cleanup must honour the cancel disposition |
| 247 | 55472a36 | c165c8914b14 | 0597b9ec | c165c8914b14 | REPLAYED-EQUIVALENT |  | fix(engine): #122 — an unresolved cancel may never cost us the order's tracking |
| 248 | a1463ead | a2b2ae843ca2 | d455fe1a | a2b2ae843ca2 | REPLAYED-EQUIVALENT |  | fix(engine): #122 — an unresolved external-flatten cancel must be durably parked |
| 249 | 79e7d97c | 1e7579c4fbed | bb915bf1 | 1e7579c4fbed | REPLAYED-EQUIVALENT |  | fix(engine): #122 — park the cancel BEFORE the round-trip, and pin the re-drive |
| 250 | 2a2f54dd | 2cf28d56afbe | 2ee4d6ae | 2cf28d56afbe | REPLAYED-EQUIVALENT |  | fix(engine): #122 — release a cancel park only where the cancel was proven |
| 251 | 4600cf39 | 312191c4132e | 6e6e0864 | 312191c4132e | REPLAYED-EQUIVALENT |  | test(engine): #122/#139 — a labelled tripwire where a real guard cannot exist |
| 252 | d64aad5c | ea808106eae6 | c81ec71d | ea808106eae6 | REPLAYED-EQUIVALENT |  | test(engine): #139 tripwire must fail LOUDLY, not xfail silently |
| 253 | 33ad115d | 7a22c8a4b9ac | 1710cc2d | 7a22c8a4b9ac | REPLAYED-EQUIVALENT |  | test(engine): #139 characterization must PROVE the ambiguous branch ran |
| 254 | 22ec32fa | 6db4e08d0181 | 85f71b8e | 6db4e08d0181 | REPLAYED-EQUIVALENT |  | fix(engine): #122 — a connection error on one exit must not abort the sweep |
| 255 | 6af337c4 | 36530f37917c | e1b463b8 | 36530f37917c | REPLAYED-EQUIVALENT |  | fix(engine): #122 — the flat-book orphan retire has external-flatten semantics too |
| 256 | e232425d | 0137b6b9f605 | ebc22198 | 0137b6b9f605 | REPLAYED-EQUIVALENT |  | fix(engine): #122 — journal-only exits must be SEEN before they can be cancelled |
| 257 | 9ba409e4 | 4083a990e300 | 5a0a4405 | dff05d7e2937 | REPLAYED-EQUIVALENT-CONTEXT | +/- lines identical; context differs | fix(engine): #122 — the parent_flat_snapshot cascade is external-flatten too |
| 258 | d966bad2 | 6185f5c8952a | 3fe27e4d | 7e3e258d633d | REPLAYED-EQUIVALENT-CONTEXT | +/- lines identical; context differs | fix(engine): #122 — separate "position vanished" from "we can PROVE it vanished" |
| 259 | f11897b3 | 5fe039d4229d | 2c78ecb4 | 5fe039d4229d | REPLAYED-EQUIVALENT |  | fix(engine): #122 — unconfirmed flat evidence must preserve EVERY tracking layer, not just the exits loop |
| 260 | 122f8f50 | 6da87cd3851b | b37d0c39 | bffed02fa6bc | REPLAYED-EQUIVALENT-CONTEXT | +/- lines identical; context differs | fix(engine): #122 — preserved-unconfirmed state gets a resolver; reject evidence gets discriminated |
| 261 | 074be297 | 30d0c0b19bbe | 6a7d1df4 | 30d0c0b19bbe | REPLAYED-EQUIVALENT |  | fix(engine): #122 — the pending-flat sweep must be reachable when the local book is already flat |
| 262 | 3ee06a57 | 108b37386cfb | 69d77259 | 108b37386cfb | REPLAYED-EQUIVALENT |  | fix(engine): #122 — a pending-only flat state must also serve the 120s confirmation grace |
| 263 | 3a79bcb9 | 1e85e0a51f0c | c18b4260 | 1e85e0a51f0c | REPLAYED-EQUIVALENT |  | fix(engine): #122 — a pending-flat marker must die with its episode |
| 264 | 9b740554 | 08c5335186f2 | ae8bd5c7 | 08c5335186f2 | REPLAYED-EQUIVALENT |  | fix(engine): #122 — a partially filled entry's remainder must be cancelled, not disowned |
| 265 | 3b338e1e | 6ed6a82df3e7 | 7db1b449 | 6ed6a82df3e7 | REPLAYED-EQUIVALENT |  | fix(engine): #122 — entry-remainder cancel: tolerant compare, every retiring path, and a moot-park fixed point |
| 266 | 735bdb4e | ac00bde955dd | c8d48790 | ac00bde955dd | REPLAYED-EQUIVALENT |  | fix(engine): #122 — a moot entry park must survive the loss of its live intent |
| 267 | 1965ceb7 | 5e768bd91231 | 47945d17 | 5e768bd91231 | REPLAYED-EQUIVALENT |  | fix(engine): #122 — a parked cancel must not be hijacked by an ordinary re-emission |
| 268 | b6e8dddb | c46b1a4f4fdf | 8029c86a | c46b1a4f4fdf | REPLAYED-EQUIVALENT |  | fix(engine): #84 — deliver a feed-liveness HALT when a blind run is exposed |
| 269 | 1f182cb1 | a0b1ff4b536b | e7aa322e | a0b1ff4b536b | REPLAYED-EQUIVALENT |  | fix(dnse): #135 — flattening a SHORT sent a SELL and DOUBLED the position |
| 270 | b35f616a | 52d2110e1512 | 403b744f | 52d2110e1512 | REPLAYED-EQUIVALENT |  | docs(live-test): Friday runbook — three passive captures from the 09-17 reviews (#135 positions frame, flatten BUY-arm caveat, id-reuse measurement) |
| 271 | 1e66749e | 1758f7fc4860 | 8aec3a93 | 1758f7fc4860 | REPLAYED-EQUIVALENT |  | test(dnse): #135 — replace a vacuous assertion in the stale-flat pin |
| 272 | b1d6ff52 | 0743fec960d5 | 2f682fd0 | 0743fec960d5 | REPLAYED-EQUIVALENT |  | chore(review): approve #132-W0 sidecar files for execution (pre-test manifest entries) |
| 273 | ca0ee117 | 67d864d1cbe5 | 7c394ca8 | 67d864d1cbe5 | REPLAYED-EQUIVALENT |  | chore(review): re-approve #132-W0 test file (import-time fix: register path-loaded module in sys.modules) |
| 274 | 9bb50003 | 605cfae5ae2f | aeb5fbd8 | 605cfae5ae2f | REPLAYED-EQUIVALENT |  | feat(dnse/tools): #132 W0 — read-only naked-position sidecar, alarm-only |
| 275 | c08528cd | 4fe2bd7dd12a | 521bcdce | 4fe2bd7dd12a | REPLAYED-EQUIVALENT |  | docs(plan): Friday 09-18 F13 on the l2b vehicle — plan + executor prompt (l2 fallback, per-run window, arm-chain grading) |
| 276 | 4c2a92ea | 4510738a8f99 | 6236ebc9 | 4510738a8f99 | REPLAYED-EQUIVALENT |  | docs(engine): #122 — the finding-28 opt-in contract was overstated; state it precisely |
| 277 | cb54ebd6 | 1b0f2ee2f41c | e8386cd7 | 1b0f2ee2f41c | REPLAYED-EQUIVALENT |  | docs(plan): F13 clock correction — log stamps are bar time; venue record + wall-clock tee prefix; grader refuses bar-stamped logs |
| 278 | f6dafc96 | 6bc94e2d9be7 | 1997b1ff | 6bc94e2d9be7 | REPLAYED-EQUIVALENT |  | chore(review): approve #146 runner+grader hashes; close the pytest bypass in the code-review hook |
| 279 | 37890d1f | 517e809015f2 | 1881f19f | 517e809015f2 | REPLAYED-EQUIVALENT |  | docs(plan): F13 Friday timing — 45-min worst-case runs at 5m; ws arm morning, poll arm afternoon; NO-SAMPLE cancel never guesses an id |
| 280 | 0e804193 | f406111ce453 | cfe36a64 | f406111ce453 | REPLAYED-EQUIVALENT |  | feat(dnse/live_test): #146 — F13 on the l2b vehicle, with grading split out |
| 281 | a25a97fa | e9f5106376f9 | e43c2a1d | e9f5106376f9 | REPLAYED-EQUIVALENT |  | fix(engine): #84 review round — pause the blindness clock on EVERY path |
| 282 | b4239a14 | ec3e750b844e | 5fe536c6 | ec3e750b844e | REPLAYED-EQUIVALENT |  | docs(hooks): code-review hook — --deselect still imports the file; use --ignore or -k to exclude a foreign unapproved test |
| 283 | 661c8873 | 2fe1231ea287 | d84fcdbc | 2fe1231ea287 | REPLAYED-EQUIVALENT |  | chore(review): approve test_025 with the #147 placement pin for the mutant run |
| 284 | 6e7603a9 | d405b0d6e265 | f06c6da4 | d405b0d6e265 | REPLAYED-EQUIVALENT |  | chore(review): approve #132-W0 fix hashes (F1/F2/F3/N1, stall threshold, interval 15) for verification |
| 285 | e8ee8fa9 | 074648867182 | 6a9450e2 | 074648867182 | REPLAYED-EQUIVALENT |  | test(engine): #147 — pin finding 28's guard placement, and correct the reason it gave |
| 286 | e9b14434 | 96307667275d | 71c6b751 | 96307667275d | REPLAYED-EQUIVALENT |  | fix(dnse/tools): #132 W0 — close three false-silence paths found in review |
| 287 | f4fe88e1 | 4e15c9acb1e0 | f6624fdf | 4e15c9acb1e0 | REPLAYED-EQUIVALENT |  | docs(plan): F13 — 45-min runs unless the terminator lands; deadline/session gates; operator flatten duties; venue.py --json for T(venue fill) |
| 288 | d86f9ff1 | c17774e0ac0a | bc2b3bf5 | c17774e0ac0a | REPLAYED-EQUIVALENT |  | chore(review): approve #146 fix set (guard helper, venue.py --json, terminator) for verification |
| 289 | 4690c015 | 0ed7d54f5228 | db30e1d9 | 0ed7d54f5228 | REPLAYED-EQUIVALENT |  | chore(review): approve test_025 (#147 docstring correction) for the gate run |
| 290 | 355d1da4 | 0af53e642fac | 739817b7 | 0af53e642fac | REPLAYED-EQUIVALENT |  | chore(review): approve #84 closing-pin files for verification |
| 291 | 3d55e8cc | a66a5083049c | 3b326473 | a66a5083049c | REPLAYED-EQUIVALENT |  | chore(review): re-approve f13_grade.py (child-frame anchor corrected) |
| 292 | 121bc8fe | a20ba347b1c9 | 248c994c | a20ba347b1c9 | REPLAYED-EQUIVALENT |  | chore(review): approve test_025 (#147 assertion wording) for the gate run |
| 293 | d2d48fc4 | 99b26960ff2c | ed6e8081 | 99b26960ff2c | REPLAYED-EQUIVALENT |  | fix(dnse/live_test): #146 — close the F13 order-path and gate findings |
| 294 | fed3b079 | 4d758aea32fa | 8b0c1100 | 4d758aea32fa | REPLAYED-EQUIVALENT |  | docs(engine): #147 — the placement pin's stated mechanism was refuted; say what it proves |
| 295 | 48a9437b | 27210b454df2 | e868f85d | 27210b454df2 | REPLAYED-EQUIVALENT |  | docs(plan,hooks): Friday W0 shadow row (conditional, after the repoint, --bar-period); hook limit marked MEASURED |
| 296 | dbcb4e8e | f3ec211aa85f | 43a742e8 | f3ec211aa85f | REPLAYED-EQUIVALENT |  | test(engine): #84 closing pins — make the budget rows and the exit order discriminate |
| 297 | 836fb962 | 2e8b7bf267ac | 79aa89ff | 2e8b7bf267ac | REPLAYED-EQUIVALENT |  | chore(review): approve #145 stage-1 files for the gate run |
| 298 | 9eec7843 | 65f60eadd992 | fdaa48e7 | 65f60eadd992 | REPLAYED-EQUIVALENT |  | chore(review): approve #132-W0 round-2 fix hashes for verification |
| 299 | 9d524a85 | c3e437173755 | 8ceae6d4 | c3e437173755 | REPLAYED-EQUIVALENT |  | chore(review): re-approve test_naked_watch.py (orphaned HOLD pin re-pointed to closed) |
| 300 | 614881eb | fa2f3fbb8761 | f9f399e6 | fa2f3fbb8761 | REPLAYED-EQUIVALENT |  | fix(dnse): #145 STAGE 1 — a failed instruments read no longer poisons the contract cache forever |
| 301 | 9056c1b7 | 95fbd391f80c | 385b8948 | 95fbd391f80c | REPLAYED-EQUIVALENT |  | chore(review): re-approve test_naked_watch.py after the orphan sweep |
| 302 | b68f2a6c | 86124554a84d | e18d40d3 | 86124554a84d | REPLAYED-EQUIVALENT |  | chore(review): approve W0 on_stall seam files for verification |
| 303 | 185bc6d9 | 23342ac24565 | 2799fecf | 23342ac24565 | REPLAYED-EQUIVALENT |  | docs(live-test): runbook — token_status.py exit code IS reliable (measured); the F13 runner now gates on it (#146) |
| 304 | 0741a4d1 | c28ffd90c2c7 | 609ccda0 | c28ffd90c2c7 | REPLAYED-EQUIVALENT |  | chore(review): approve #146 round-3 fix hashes for verification |
| 305 | bf6f4ae6 | a19f6e9522ce | 4c205c61 | a19f6e9522ce | REPLAYED-EQUIVALENT |  | chore(review): re-approve test_f13_grade.py (loud fixture mutation helper) |
| 306 | 8ab74e3b | aec3bdea6105 | 8260c7df | aec3bdea6105 | REPLAYED-EQUIVALENT |  | chore(review): approve #145 stage-1b files for the gate run |
| 307 | 594f6de4 | 80f85605445d | ed7fe55d | 80f85605445d | REPLAYED-EQUIVALENT |  | docs(plan): Friday row 1 — 08:00 auto-mint cron is a TEST; manual mint at 08:20 if no GOOD (#133) |
| 308 | 4879aad6 | 750078a80789 | 3a4e7263 | 750078a80789 | REPLAYED-EQUIVALENT |  | chore(review): re-approve provider.py (#145 stage 1b, edit applied) |
| 309 | 5455ef45 | 16be0cd04a25 | 85a4c55d | 16be0cd04a25 | REPLAYED-EQUIVALENT |  | fix(dnse/live_test): #146 round 3 — grade the real log shape, not an invented one |
| 310 | 63d80af9 | eb7769b7eec4 | d76fb262 | eb7769b7eec4 | REPLAYED-EQUIVALENT |  | chore(review): approve #133 read-only IMAP headers probe |
| 311 | a68b907d | 57e36ac41a50 | 099671bd | 57e36ac41a50 | REPLAYED-EQUIVALENT |  | chore(review): approve #145 stage-1b final hashes |
| 312 | e50e49dd | 39c649f89ae2 | 6022b85b | 39c649f89ae2 | REPLAYED-EQUIVALENT |  | chore(review): approve #132-W0 round-2 final hashes |
| 313 | 48f7ee42 | 9682c5a518b2 | 6d402f9e | 9682c5a518b2 | REPLAYED-EQUIVALENT |  | fix(dnse/tools): #132 W0 round 2 — four more false-silence paths, and one policy deleted |
| 314 | 3b85a80d | e04565b401cb | a212cf8e | e04565b401cb | REPLAYED-EQUIVALENT |  | fix(dnse): #145 STAGE 1b — the passthrough heuristic was refuted; resolve it against the real catalogue |
| 315 | 4db5914f | 47a94c3c33b4 | c4bb549a | 47a94c3c33b4 | REPLAYED-EQUIVALENT |  | docs(plan): sep18 F13 — venue dates are ISO strings, cleanup cancels every entry id, arm latency as graded |
| 316 | 307fe977 | 845d2df05a1c | 8977a297 | 845d2df05a1c | REPLAYED-EQUIVALENT |  | fix(dnse/live_test): #146 — venue dates are ISO-8601, and a naive one is refused |
| 317 | c1b655b4 | b8c15e8c77c4 | d0d2b233 | b8c15e8c77c4 | REPLAYED-EQUIVALENT |  | docs(plan): sep18 row 10 — W0 exit-code semantics, one store per symbol, UNSTOPPED is healthy for OCO |
| 318 | 29a3f3ec | 2575703f64d9 | 3f7a5aed | 2575703f64d9 | REPLAYED-EQUIVALENT |  | fix(dnse): #145 STAGE 1c — an absent `total` was the same poison through a door I opened |
| 319 | 4316fd7a | 941b078a7058 | 0f979bbd | 941b078a7058 | REPLAYED-EQUIVALENT |  | fix(dnse/live_test): #146 — define flat_confirmed before the exit trap can call it |
| 320 | b2cbb6c5 | d44f2a144b3f | 51474f44 | d44f2a144b3f | REPLAYED-EQUIVALENT |  | fix(dnse/tools): #132 W0 — an absent createdDate is could-not-determine, not cover |
| 321 | 623f3183 | c80b30b3703c | 78b5c533 | c80b30b3703c | REPLAYED-EQUIVALENT |  | docs(plan): sep18 row 10 — W0 shadow run is on (b2cbb6c5 reviewed CLEAN) |
| 322 | 1d55ca30 | 566ea8c5ee88 | 6b558342 | 566ea8c5ee88 | REPLAYED-EQUIVALENT |  | fix(dnse): #133 — make the OTP automation survivable, and the dead-cron visible |
| 323 | 8ade6de5 | 7eafbd90b1d0 | 0539dc93 | 7eafbd90b1d0 | REPLAYED-EQUIVALENT |  | fix(dnse): #133 — the status line said "cron" while measuring token age |
| 324 | fb6b0ab5 | 942bade16fef | 184e157a | 942bade16fef | REPLAYED-EQUIVALENT |  | docs(plan): sep18 step 1 — the 08:20 gate is token_status --require-cron; either outcome is a #133 measurement |
| 325 | 3a1cf478 | f1032eb35067 | 67f72c21 | f1032eb35067 | REPLAYED-EQUIVALENT |  | docs(plan): Thursday 09-17 PM ws arm on expiry day — window-bars 3 deviation and why |
| 326 | 3cf9aeb2 | 705f330e6c43 | 1d9490fd | 705f330e6c43 | REPLAYED-EQUIVALENT |  | docs(plan): upstream 6.9.4 landing — hunk-level diff review vs live-measured facts; one real collision (45bc8103 restart sweep) |
| 327 | 15156887 | 60cda2a0d8e9 | 12372fe0 | 60cda2a0d8e9 | REPLAYED-EQUIVALENT |  | docs(plan): upstream 6.9.4 — 45bc8103 resolution plan, _build_envelope second silent merge, control counts |
| 328 | 2d628cac | 289268456480 | f233fe39 | 289268456480 | REPLAYED-EQUIVALENT |  | docs(plan): upstream 6.9.4 — trial rebase confirms outcome A (silent cancel); bare cleanup call cancels on DNSE |
| 329 | d4621d78 | a331b32e8dc1 | 0071329a | a331b32e8dc1 | REPLAYED-EQUIVALENT |  | docs(plan): Thursday attempt result — L0 gate FAIL on #118, no conditional order on expiry day; Friday runs both arms |
| 330 | 5bbb1bd2 | a93c1202eb8c | 761119ec | a93c1202eb8c | REPLAYED-EQUIVALENT |  | docs(plan): correction — expiry-day CO-ORD-006 was our midnight-UTC GTD ceiling, not a venue impossibility (#118) |
| 331 | 3b15004c | 43d9ca3cfe4b | 81f05881 | 43d9ca3cfe4b | REPLAYED-EQUIVALENT |  | fix(dnse): #118 — the GTD ceiling is a TIME, and midnight UTC is the wrong one |
| 332 | d258a44b | 3523587f4fc4 | fcc25fd3 | 3523587f4fc4 | REPLAYED-EQUIVALENT |  | docs(plan): upstream 6.9.4 — review deltas (P5 silent merge, pin retirement step, trial is a proof, _build_envelope criterion) |
| 333 | d05c294a | cd6a691c28f7 | 5f798e21 | cd6a691c28f7 | REPLAYED-EQUIVALENT |  | fix(dnse): #118 — the final trading day refuses conditionals outright; use next month |
| 334 | 6e75ddb0 | ac0e08a4904f | 302a897d | ac0e08a4904f | REPLAYED-EQUIVALENT |  | docs(plan): expiry-day conditionals — measured boundary is 14:30 ICT (07:30Z), operator's resting app stop; VN30F2M also passes |
| 335 | 32293b40 | 44bdaf581daa | d42299e9 | 44bdaf581daa | REPLAYED-EQUIVALENT |  | docs(plan): #118 confirmed — L0 PASS on the expiring contract with the 07:30Z GTD ceiling |
| 336 | d5bab205 | a4ef9ce4b576 | f8cef967 | a4ef9ce4b576 | REPLAYED-EQUIVALENT |  | fix(dnse): #118 — the ceiling is 14:30 ICT (continuous-session end), not 14:45 |
| 337 | 42c7c2d2 | 4b1be8bda737 | e85a2152 | 4b1be8bda737 | REPLAYED-EQUIVALENT |  | docs(plan): 0c heading and wording match the confirmed #118 outcome |
| 338 | 2a6fa0bb | feb94015da25 | f7694183 | feb94015da25 | REPLAYED-EQUIVALENT |  | test(dnse): #132 W1 baseline — the slipped-over scenario pinned on REAL prices + mocked order records |
| 339 | 0ef9781c | 5ca0e03b6882 | 70ad8c86 | 5ca0e03b6882 | REPLAYED-EQUIVALENT |  | feat(dnse/tools): #132 W1 — price_through.py, the slipped-over detector (pure, injected clock) |
| 340 | 0138a7f0 | b95c91854a55 | d53ce369 | b95c91854a55 | REPLAYED-EQUIVALENT |  | feat(tooling): #148 — per-worker worktrees + an ownership map, written but NOT enabled |
| 341 | d29180ee | df7e9af71aac | 1d14d1d5 | df7e9af71aac | REPLAYED-EQUIVALENT |  | chore(review): manifest entries for the #132 W1 baseline/detector, #148 rev 2-3, and the step-4 logger |
| 342 | 55799040 | 8b86e3c2a099 | f12b8044 | 8b86e3c2a099 | REPLAYED-EQUIVALENT |  | test(dnse): #133 — the suite was logging into the operator's real Gmail |
| 343 | 4cdbf263 | cb608afeb1b3 | a38db8c0 | cb608afeb1b3 | REPLAYED-EQUIVALENT |  | chore(#148): ownership map — engine paths leader-owned by consent, venue.py by consent, Worker2/Worker3/Fable files added |
| 344 | 07165c2d | d11b32558b36 | 9f997651 | d11b32558b36 | REPLAYED-EQUIVALENT |  | chore(#148): register the ownership hook; the hook file itself belongs to the leader |
| 345 | 49656181 | 0baa58bb3acf | 065aa64f | 0baa58bb3acf | REPLAYED-EQUIVALENT |  | fix(#148): a QUOTED literal redirect target no longer bypasses the ownership hook; variable targets warn |
| 346 | fb9e492b | 98874554e6f7 | df26568f | 98874554e6f7 | REPLAYED-EQUIVALENT |  | fix(hooks): the code-review gate refuses an execution path spelled through a shell variable |
| 347 | f0a0b9b0 | 8821d4528c74 | c5b8cfb7 | 8821d4528c74 | REPLAYED-EQUIVALENT |  | fix(hooks): the code-review gate also refuses substitution and backtick code paths that contain whitespace |
| 348 | 553cbbff | 4a88d90cb0ca | 025ff49b | 4a88d90cb0ca | REPLAYED-EQUIVALENT |  | test(dnse): #153 — the unit suite must not read the operator's real trading token |
| 349 | 8fae0490 | 921408abc500 | 7eaa3756 | 921408abc500 | REPLAYED-EQUIVALENT |  | fix(dnse): #133 — a status line must not encode the caller's timetable |
| 350 | 01e78834 | 6eb2c176b97b | c0f079ab | 6eb2c176b97b | REPLAYED-EQUIVALENT |  | docs(plan): morning gate is the 06:20 cron + --require-cron after ~06:40 (unattended mint proven 09-17 19:00) |

## Totals
- REPLAYED-EQUIVALENT: 345
- REPLAYED-EQUIVALENT-CONTEXT: 4
- REPLAYED-MODIFIED: 1
- DROPPED-BY-DECISION: 0
- SUPERSEDED-BY-UPSTREAM: 0
- rows: 350
