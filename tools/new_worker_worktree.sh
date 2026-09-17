#!/usr/bin/env bash
# Create an isolated worktree + its OWN venv for one executor session.
#
# WHY THIS EXISTS (#148). Seven incidents in one session came from three
# sessions sharing ONE working tree — not from colliding edits, which file
# ownership already prevents, but from one session's STATE reaching another
# session's PROCESS: an uncollectable file stopping everyone's suite under
# `-x`, an intentional red-first pin sitting in the shared tree, a stale
# mutant `.pyc` surviving a source restore, a full-suite run over a mid-edit
# tree reporting a phantom failure, and a SHARED `active_task.json` that can
# make the commit flow close the WRONG card.
#
# THE VENV IS THE LOAD-BEARING PART, NOT THE WORKTREE. The repo's shared
# `.venv` is an EDITABLE install of the MAIN checkout, so a worktree without
# its own venv silently imports the main tree's code: the
# verify-you-are-testing-new-code trap, structurally guaranteed rather than
# accidental. This script REFUSES TO FINISH unless `import pynecore` resolves
# INSIDE the new worktree.
#
# WHAT A WORKTREE CAN AND CANNOT DO
#   CAN: the code, and `pytest` (the DNSE suites use a fake-client seam, so
#        they need no credentials).
#   CANNOT: anything LIVE. `.env`, `workdir/config/plugins/*.toml` and
#        `workdir/state/` are gitignored, so they do not exist here. venue.py,
#        token_status.py, refresh_token.py, any `--broker` run: MAIN CHECKOUT
#        ONLY. A live tool run from a worktree fails on missing credentials, or
#        worse, reads a default and answers confidently about nothing.
#   CARD OPS: fine. `backlog_index.py` resolves the file→issue index via
#        `--git-common-dir` (one copy per repo) and `active_task.json` via
#        `--git-dir` (PER WORKTREE) — so each session owns its own active card,
#        which is the property the shared-file incident was about.
#   REVIEW MANIFEST: `.claude/fable-approved.manifest` is TRACKED, so a
#        worktree carries the copy from its branch point. Approvals recorded on
#        the base branch afterwards are not visible here until a rebase. The
#        fix is for the review hook to read the manifest from the MAIN checkout
#        (`git rev-parse --git-common-dir`'s parent) — it is the leader's
#        state, not the branch's. Until that lands: rebase before a gated run.
#
# Usage:
#   tools/new_worker_worktree.sh <name>            create
#   tools/new_worker_worktree.sh --check <name>    validate preconditions only
#
# Never deletes: a stale worktree is MOVED to backup/deleteable/.

set -euo pipefail

CHECK=0
if [ "${1:-}" = "--check" ]; then CHECK=1; shift; fi

NAME="${1:-}"
if [ -z "$NAME" ]; then
    echo "usage: $0 [--check] <name>   (e.g. worker2)" >&2
    exit 2
fi
case "$NAME" in
    *[!a-zA-Z0-9_-]*)
        echo "name must be [A-Za-z0-9_-] only: $NAME" >&2
        exit 2
        ;;
esac

ROOT="$(git rev-parse --show-toplevel)"
BASE_BRANCH="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"
WT="$(dirname "$ROOT")/$(basename "$ROOT")-$NAME"
BRANCH="${BASE_BRANCH}-${NAME}"
PY_VERSION="3.13"

say() { printf '%s\n' "$*"; }

# --- preconditions, checked the same way in both modes -----------------------
# Cheap and local only. `--check` exists to catch the setup mistakes that would
# otherwise fail HALFWAY THROUGH, leaving a worktree with no usable venv — a
# state that looks created and is not.
fail=0
note() { echo "  FAIL: $*" >&2; fail=1; }

# A detached HEAD makes `--abbrev-ref` print the literal "HEAD", which would
# create a branch named "HEAD-<name>" off nothing meaningful.
if [ "$BASE_BRANCH" = "HEAD" ]; then
    note "HEAD is DETACHED — check out a branch first (would create 'HEAD-$NAME')"
fi
command -v uv >/dev/null 2>&1 || note "uv is not on PATH — the per-worktree venv cannot be built"
[ -e "$WT" ] && note "$WT already exists — retire it first (git worktree move, never delete)"
# The retire path (`git worktree move`) leaves the BRANCH behind, so the second
# attempt at the same name fails at `worktree add -b` after the path check has
# already passed — halfway, with nothing usable. Catch it here instead.
if git -C "$ROOT" show-ref --verify --quiet "refs/heads/$BRANCH"; then
    note "branch $BRANCH already exists (a retired worktree leaves its branch) — delete or reuse it deliberately"
fi

say "worktree : $WT"
say "branch   : $BRANCH   (from $BASE_BRANCH)"
say "venv     : $WT/.venv  (python $PY_VERSION, editable: repo + plugins/dnse)"
say "assert   : import pynecore MUST resolve under $WT"
say "scope    : code + pytest only — live tools run from $ROOT"

if [ "$fail" = "1" ]; then
    echo "" >&2
    echo "preconditions failed; nothing created." >&2
    exit 1
fi

if [ "$CHECK" = "1" ]; then
    say ""
    say "--check: preconditions OK, nothing created."
    exit 0
fi

git -C "$ROOT" worktree add -b "$BRANCH" "$WT" "$BASE_BRANCH"

cd "$WT"
uv venv --python "$PY_VERSION" .venv
uv pip install --python "$WT/.venv/bin/python" -e '.[all,dev]' -e plugins/dnse
uv pip install --python "$WT/.venv/bin/python" pytest pytest-cov

# THE ASSERT. A worktree whose interpreter imports the MAIN tree is worse than
# no worktree at all: every test result it produces is about somebody else's
# code while looking like it is about yours.
RESOLVED="$("$WT/.venv/bin/python" -c 'import pynecore, sys; sys.stdout.write(pynecore.__file__)')"
case "$RESOLVED" in
    "$WT"/*)
        say ""
        say "OK: import pynecore -> $RESOLVED"
        ;;
    *)
        echo "" >&2
        echo "FAILED: import pynecore -> $RESOLVED" >&2
        echo "That is OUTSIDE $WT, so this worktree would run another tree's" >&2
        echo "code and every test result would be about the wrong source. The" >&2
        echo "editable install did not take. Do not use this worktree until it" >&2
        echo "resolves inside itself." >&2
        exit 1
        ;;
esac

say ""
say "SCOPE: code + pytest here. LIVE TOOLS (venue.py, token_status.py,"
say "refresh_token.py, any --broker run) MUST run from $ROOT — .env,"
say "workdir/config/plugins/*.toml and workdir/state/ are gitignored and"
say "absent here."
say ""
say "INTEGRATION (leader gates the merge):"
say "  cd $WT"
say "  git rebase $BASE_BRANCH"
say "  git push -u origin $BRANCH     # leader then fast-forwards $BASE_BRANCH"
say ""
say "GATES — read the REAL exit status, never a pipe's:"
say "  .venv/bin/python -m pytest tests/ -q > /tmp/g.txt 2>&1; echo \"status: \$?\"; tail -1 /tmp/g.txt"
say ""
say "REVIEW MANIFEST: .claude/ is gitignored, so .claude/ownership.toml, the"
say "hooks and the approval manifest travel here only because they are"
say "FORCE-TRACKED. This worktree carries the copies from its branch point —"
say "approvals recorded on $BASE_BRANCH afterwards are not visible until you"
say "rebase. Rebase before a gated run, or the review hook will refuse a file"
say "the leader has already approved."
say ""
say "RETIRE (never delete):"
say "  git worktree move $WT $ROOT/backup/deleteable/$(basename "$WT").TIMESTAMP"
say ""
say "NOTE: this venv records ABSOLUTE paths. Moving or retiring the worktree"
say "does not move the venv with it — a parked copy keeps pointing at the old"
say "location and will import from a directory that no longer exists. Rebuild"
say "rather than relocate."
