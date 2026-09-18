# How two false claims travelled on 2026-09-18, and the one rule that would have stopped both

Both were about a LIVE EVENT. Both came from a trusted source with confirming detail. Both were repeated without the one cheap check that refuted them. Neither was caught by a reviewer of the work; both were caught by someone re-reading the primary record days-of-hours later.

## Claim 1 — "the sidecar raised the first true NAKED alarm of its life"

| | |
|---|---|
| First stated | Worker3ForDNSEPlugin, ~10:47, reporting the naked position |
| From what evidence | `[NAKED]` lines in W0's own log at 10:46:31 and 10:46:48, real and correctly detected |
| Repeated by | Me, without checking, four times: to the operator, to DNSEPlugin, to Worker3 in reply, and into the #163 card body as a stated premise |
| Published to | The operator's record, the leader's record, #163's review post |
| Caught by | The #163 operational reviewer, by reading the NEXT LINE of the same log |
| The line that refuted it | `NAKED observed but WITHHELD for up to 300s: a reactively placed exit arms a bar late by design, so this is the expected post-entry window.` |
| Second refutation | `workdir/output/logs/naked_watch_alarms.log` does not exist — the ladder has never armed |
| Elapsed | ~3 hours, and it reached a card body before anyone looked |

**What was true**: the sidecar DETECTED the naked exposure, promptly and correctly.
**What was false**: that it ALARMED. It withheld escalation by design, the position cleared inside the grace window, and nobody was paged. Both layers were silent.

The failure was not credulity about a stranger. It was that "detected" and "alarmed" are two claims and the evidence offered supported only the first. I did not notice the join because the stronger reading fitted the story I was already telling — that the sidecar had earned its place that morning.

## Claim 2 — "three spent umbrellas" — RESOLVED, and it is the counter-example

| | |
|---|---|
| First stated | Worker3ForDNSEPlugin, ~11:14, in its #152 step-1 report — loose wording, wanting to smoke-test the new `flat` against them |
| Repeated by | DNSEPlugin, once, unchecked, in the reply instructing the smoke test |
| Caught by | **Worker3 itself**, ~11:18, when the live smoke printed 2 umbrellas and 3 phantom conditionals: "I had said three spent umbrellas loosely earlier and that was wrong" |
| Published to | Nowhere. It never reached a card |
| Elapsed | **~4 minutes** |

Chain supplied by DNSEPlugin from its own transcript. By the rule below that is TESTIMONY — a participant reporting what was said in a conversation they were in — not a measurement, and testimony of that kind is the reliable sort. I could not reconstruct it myself because I cannot read other sessions' transcripts.

**This is the counter-example, and it is more instructive than claim 1.** Same shape: loose wording from a trusted source, repeated once without checking. Different outcome, for one reason — **the person who made it measured before anyone repeated it further.** Three hours and a card body versus four minutes and nothing published, and the only variable is whether a measurement happened early.

The facts it got wrong, measured:

```
working : 0 live, 3 phantom (consumed conditionals, #41)
   phantom damb7fqvfqkc7397o3fg -> child 95206 did the work
   phantom damb54avfqkc7397o3c0 -> child 92146 did the work
   phantom damadmqvfqkc7397o0t0 -> child 39336 did the work
oco     : 2 umbrella(s), 0 ARMED
   damb5qivfqkc7397o3dg   SPENT   -> child 92176
   damadq2vfqkc7397o0tg   SPENT   -> child 39356
```

**Three phantoms and two umbrellas, on different books.** The phantoms are consumed ENTRY conditionals on the STOP book; the umbrellas are BRACKETS on the OCO book. "Three spent umbrellas" takes the phantom COUNT and attaches the umbrella LABEL — a category merge that produces a number which is simply wrong, and which no one would catch by sense-check because three of something spent is entirely plausible.

Note the timing: `venue.py status` could not see the OCO book AT ALL until #152 step 1 (`ce60a62c`, ~11:00). The loose claim was made at ~11:14 and measured at ~11:18 — so it was checkable only because the tool to check it had landed fourteen minutes earlier. Before that, anyone counting umbrellas was counting something else and the toolkit offered no way to notice.

## The pattern, stated once

Both claims were **about a live event**, both arrived **with confirming detail attached**, and in both cases the detail supported a WEAKER claim than the one made. The strong version travelled because the evidence looked like evidence: a timestamp, a log excerpt, a count. Nobody lied; everyone rounded up, once.

**What separates the two outcomes is not care, seniority, or suspicion — it is when someone measured.**

| | claim 1 | claim 2 |
|---|---|---|
| source | trusted peer, real supporting lines | trusted peer, loose wording |
| repeated unchecked | 4 times, by me | once, by the leader |
| who caught it | a reviewer, hours later, re-reading the primary record | the author, by measuring |
| elapsed | ~3 hours | ~4 minutes |
| published | the operator's record, the leader's record, a card body | nowhere |

Claim 2 was made just as loosely as claim 1. It cost nothing because the measurement came before the third repetition. That is the whole mechanism, and it is why the rule below asks for a line rather than for more scepticism.

## THE RULE — proposed for the handoff template and CLAUDE.md

> **A claim about a live event travels with the log line that shows it, or it is labelled UNVERIFIED.**
>
> The line must show the claim itself, not its neighbourhood. `[NAKED]` shows detection; it does not show an alarm. If the quoted line supports a weaker claim than the sentence above it, the weaker claim is the one you may make.
>
> When repeating someone else's claim about a live event, quote THEIR line or run the check yourself. A peer's claim about what THEY DID is testimony and is usually reliable. A peer's claim about what the SYSTEM DID is a measurement, and measurements are checked no matter who took them.

**Why this one and not "verify everything".** It is testable: a reader can look at any live-event claim and ask whether a log line is attached and whether that line says the same thing. It fails loudly — an unlabelled bare claim is visibly non-compliant. And it is cheap: both refutations above took under a minute once someone looked.

**What it would have cost today**: Claim 1 would have been written as "the sidecar DETECTED the naked position at 10:46:31 (`[NAKED] ... exposure +1 is OPEN and UNPROTECTED`); whether it escalated is UNVERIFIED" — which is true, useful, and would have prompted the next line to be read.
