# Worker handoff — standing rules (#148)

Every rule below is here because it cost something on 2026-09-17, across three
concurrent executor sessions and one leader. None of it is style. Where a rule
has a measurement, the measurement is given, because a rule whose evidence has
been dropped is the first thing a future reader talks themselves out of.

---

## 1. The seven measured incidents

All seven came from sessions sharing ONE working tree. Not one was a colliding
edit — which file ownership already prevents. Every one was **one session's
STATE reaching another session's PROCESS**.

1. **An uncollectable untracked test file stopped everyone's suite.** A module
   that raised at import made `pytest plugins/dnse/tests/` report a
   COLLECTION ERROR with zero tests executed, for every session, because
   `pytest.ini` carries `-x`. An uncollectable file is indistinguishable from a
   red suite.
2. **An intentional red-first pin sat in the shared tree**, red for everybody
   else's gate, while its author was mid-adjudication on whether to keep it.
3. **A stale mutant `.pyc` survived a source restore.** After a file-copy
   restore the tests still failed with the mutant's symptoms; the source on
   disk was correct and the `__pycache__` entry was not. `inspect.getsource()`
   reported the correct source, because getsource reads the `.py` while the
   code object came from the `.pyc` — the check that feels most authoritative
   is the one blind to this failure.
4. **A full-suite run over a mid-edit tree produced a phantom red** whose
   report carried three incompatible identities: a test defined at one line,
   cited at another, quoting a third test's assertion. That shape means "read
   during someone's edit", not "ordering bug".
5. **`active_task.json` is shared `.git` state.** With three sessions in
   flight it had already moved to another card by the time one session closed
   its own; the handoff failed with a JSON decode error. It was loud ONLY BY
   LUCK — a valid entry for a different card would have transitioned the WRONG
   card silently, because the `commit` flow's close step reads that same file.
6. **Two sessions' mid-edit files produced foreign reds** that had to be
   excluded from the gate and disclosed as foreign in every commit body that
   quoted a suite figure while they were present.
7. **A PreToolUse block DISCARDS the work bundled with it, and its refusal
   names a DIFFERENT file than the one that silently failed to exist.** Twice
   in one session: once a fix was reported as applied that the hook had eaten
   (edit and verification were one command); once a mutant plugin was never
   written while the refusal named an unrelated test file.

---

## 2. Ownership has two relations, not one

- **`owner`** — this session writes the path; others are blocked.
- **`needs_consent`** — another session PROPOSES a change; the owner applies it.

The second relation exists because of a real case. An operator-facing runbook
stated as fact that a tool exits 0 on a bad verdict. It never had — the claim
was inherited, not measured — and that false claim was cited as the REASON a
working exit-code gate had been left disabled. Fixing it was one card's work,
in another card's file, hours before a live run. Both sessions were right to
refuse to edit the other's file, and with only a hard owner the correct fix
stalls at the boundary.

**Unowned is not forbidden.** The map names owners, not permissions. A path
nobody claimed is open.

**Fail open, always.** An unmapped session is warned, never blocked. A hook
that wedges a session mid-card is worse than the collisions it prevents.

---

## 3. Hook composition

A second blocking hook does not simply add: it composes into states that
neither hook's message describes.

- **A block discards the bundled work.** Never put an edit and a gated run in
  one command. The edit will not happen, and the refusal may name a different
  file.
- **Confirm the artefact EXISTS before reasoning about its results.** Not "I
  composed it and saw a hook message" — check the file, or its hash.
- **Prefer a dedicated write over a shell heredoc for NEW files.** A heredoc
  can be refused as a unit and take the rest of the command with it.
- **Every refusal names its own hook and what unblocks it.** A refusal that
  does not costs more than the collision it prevented.
- **Do not double-gate — but check that the other gate actually covers the
  operation.** An earlier draft of the ownership hook exempted files listed in
  the review manifest, on the argument that the review hook already gated
  them. It does not: that hook is registered on **Bash only** and gates
  EXECUTION, so it never sees an Edit. The exemption let any mapped session
  edit another owner's approved file, which changed the hash, after which the
  OWNER's next run was refused by the review hook naming their own file — the
  exemption manufactured the wedge it was meant to prevent. "Another gate has
  this" is a claim to verify, not to assume.

---

## 4. Gates: read what the check says, not that it ran

Three failures in one day shared one shape — a check that ran, printed
something reassuring, and reported on a DIFFERENT OBJECT than the claim it was
used to support: source text vs a stale `.pyc`; a two-agreeing-reads rule vs an
unsigned magnitude; a pipeline's exit status vs pytest's.

- **Read the REAL exit status.** `cmd > f 2>&1; rc=$?`, or `set -o pipefail`,
  or `${PIPESTATUS[0]}` — never the status of `| tail`. A commit was made on an
  errored suite because a `&&` chain read `tail`'s 0.
- **Read the summary LINE too**, so "0 tests executed" cannot pass as green.
- **Pipe what you READ, capture what you ACT ON.** The convention already
  exists in-tree and is applied inconsistently: in one runner, one tool's `$?`
  is captured and gated while the tool on the line above is piped into `tail`
  and discarded.
- **Before trusting a check, name one broken state it would CATCH and one it
  would MISS.**

---

## 5. Mutation testing

**Mutate by RUNTIME PATCH, never by editing source, once hashes are pinned.** A
throwaway pytest plugin on `PYTHONPATH`, loaded with `-p`, that monkeypatches
the constant or replaces the function. It leaves every approved hash
byte-identical (report the sha256 before and after), and it avoids the
file-copy restore where the stale-`.pyc` trap lives.

**Run with `PYTHONDONTWRITEBYTECODE=1`** on anything path-loaded, and read the
colour from BEHAVIOUR — a returned value, a dispatched call, a wire payload —
never from source text.

### The broken-harness signature

Suspect the HARNESS, not the code, when:

- **Every mutant returns the same colour.** Uniform RED means the harness is
  failing before it reaches the code; uniform PASS means it is not reaching the
  code at all.
- **A mutant that should be impossible still passes.** Then the mutant is not
  being applied — verify the patch took by asserting the mutated behaviour
  directly, before trusting any test result derived from it.

**The harness must RAISE on a missing key**, never silently no-op. A
monkeypatch that quietly does nothing when the target moves produces a full
green run that means nothing at all.

**Every mutant needs a targeting control**: name the pin it must turn RED, and
confirm the OTHER pins stay green. A mutant that reddens everything has proved
nothing about the pin you were testing.

**Pins must assert what a wrong implementation gets wrong.** Measured failures
of this, all from one day: an implementation returning only a position's SIGN
passed every pin in its file, because every fixture held quantity 1 — the one
magnitude at which a sign and a size are the same number; a pin asserting a
DIRECTION across a loop (`> before`) that waved through five oversized closes;
a pin asserting on a store API that returns `[]` for the row it meant to check;
a "stock passthrough" control built on a 2-row fixture while the real catalogue
has ~3298 rows and a page that is always full, so it excluded nothing.

**Choose fixture values that make the wrong answers DIFFERENT NUMBERS**, and
assert what was SENT (side, quantity, category, price direction), not the call
sequence. **Cost is a correctness property** on a rate-limited venue: assert
request counts where a second request is only earned on one branch.

---

## 6. Machine-level state has no repo owner

Ownership by path cannot see it. Name who may write each piece — crontab,
systemd units, `.env`, credential config — and hold one rule:

**Every installer MERGES (append-if-absent) and NEVER replaces. Snapshot
first.**

The worked example: the user crontab (117 lines) is SHARED with a sibling
project whose own docs told agents to run `crontab deploy/crontab` in THREE
places. That file contained ZERO entries for this repo, and
`crontab <file>` is a whole-file REPLACE with no merge mode — so following one
repo's documented routine would silently delete this repo's 08:00 trading-token
refresh and `@reboot` guard. **Neither repo's tests would notice.** Fixed at
the source with an installer that merges inside a marker block, backs up first,
and offers `--dry-run`.

---

## 7. Working in a shared tree, until per-worker worktrees land

- **Announce before any full-suite run.**
- **Report foreign reds as foreign** — with HEAD and `git status` — never as
  facts about your change.
- **No intentional reds left in a shared tree.** Shelve a red-first pin to
  `backup/handover/` with its diagnosis until its card is adjudicated.
- **New test files importable within a minute.** Land a stub rather than an
  uncollectable module.
- **Explicit-path commits.** Never `git add -A` while others hold the tree.
- **`git pull --rebase --autostash origin <branch>`** with explicit refs.
- **Re-query card ids before any Done transition** rather than trusting
  `get-active` — until each worktree has its own.

---

## 8. Reporting

- **A claim ships with its evidence in the same message**, or is labelled
  UNVERIFIED.
- **A check must be red-first**: show it detecting the BROKEN state, or say why
  that cannot be shown.
- **Empty is suspicious, not conclusive.** `0` / `None` / absent may be the
  answer, or may mean the question never ran.
- **A filename-level grep is not an import check.** `grep -rl pine2pyne tests/`
  matches docstrings; `grep -rnE '^\s*(import|from)\s+pine2pyne'` answers the
  question that was asked.
- **Correct your own record when it turns out wrong**, in the same place the
  wrong version lives — a commit body, a card comment, and the code comment
  itself. A wrong "why" on a money path is what produces the next finding.
- **Extend a guard only after saying what it PROTECTS and what it COSTS when it
  fires.** Three fixes in one day each broke the thing their own design was
  meant to protect; each time the original guard was right, the extension was
  wrong, and the catch came from a test written by someone else for another
  reason.
