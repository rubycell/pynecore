#!/usr/bin/env python3
"""PreToolUse hook: block a session from WRITING another session's files (#148).

NOT REGISTERED. This file is inert until it is listed in settings.json, and it
is deliberately not introduced during a live trading session: a new PreToolUse
hook is the one change that can block every session at once.

WHAT IT IS FOR, and what it is not. Ownership stops two sessions EDITING one
file. It does nothing about one session's STATE reaching another session's
PROCESS — that is the per-worker worktree's job
(tools/new_worker_worktree.sh). Both are needed; neither substitutes.

Scope: Edit / Write / MultiEdit / NotebookEdit by their file path, AND the Bash
shapes that write a file without going through those tools — `> f`, `>> f`,
`tee f`, `sed -i … f`, `cp/mv … f`, `truncate`, `dd of=f`. Enforcement that
only watches the edit tools is trivially bypassed by a heredoc, and a rule that
can be bypassed by accident is not a rule.

FAIL OPEN, DELIBERATELY AND IN THREE PLACES:
  * no ownership.toml, or it does not parse  -> allow (visible warning)
  * no session-names.json, or this session is unmapped -> allow (visible)
  * the path has no owner                    -> allow (unowned is not forbidden)

THERE IS NO MANIFEST EXEMPTION, and the reason is worth keeping. An earlier
draft allowed any edit to a file whose sha256 was in
`.claude/fable-approved.manifest`, on the argument that the review hook already
gated it. That argument is FALSE FOR THIS OPERATION: `enforce_code_review` is
registered on **Bash only** and gates EXECUTION — it never sees Edit/Write.
Measured against the real map and manifest: a large share of owned files are
manifest-approved at any moment, so the exemption let any mapped session edit
them, the edit changed the hash, and the OWNER's next run was then refused by
the review hook naming their own file, with no record of who had edited it.
The fail-open manufactured exactly the wedge it was meant to prevent.

Warnings are emitted as a `systemMessage` on stdout, not to stderr: on exit 0
stderr reaches the debug log only, so a stderr-only warning about an unmapped
session would reach nobody and the hook would be silently allow-everything.

EVERY REFUSAL NAMES THIS HOOK AND WHAT UNBLOCKS IT. A refusal that does not say
which gate issued it, and how to clear it, costs more time than the collision
it prevented.

Exit codes per the hooks contract: 0 allow; 2 block (stderr shown to the model).
"""
from __future__ import annotations

import fnmatch
import json
import subprocess
import re
import sys
from pathlib import Path

def _repo_root() -> Path:
    """The toplevel of the tree being WRITTEN, not of the file being run.

    `Path(__file__).parents[2]` is the MAIN checkout even when the session is
    editing inside a worktree, so a main-root hook silently ignores every
    worktree path — the ownership map would stop applying in exactly the
    isolated trees #148 creates. Ask git instead; fall back to the file's own
    root only when git cannot answer.
    """
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip()).resolve()
    except Exception:
        pass
    return Path(__file__).resolve().parents[2]


REPO = _repo_root()
OWNERSHIP = REPO / ".claude" / "ownership.toml"
SESSION_NAMES = REPO / ".claude" / "session-names.json"

# Bash shapes that WRITE a path (group 1 = the target).
_WRITE_PATTERNS = [
    re.compile(r">>?\s*([^\s;&|>]+)"),
    re.compile(r"\btee\s+(?:-a\s+)?([^\s;&|]+)"),
    # No `$` anchor: `sed -i … f && …` and `sed -i … f; …` are ordinary.
    re.compile(r"\bsed\s+[^;&|]*-i[^;&|]*?\s([^\s;&|]+)"),
    re.compile(r"\bperl\s+[^;&|]*-p?i[^;&|]*?\s([^\s;&|]+)"),
    # Restoring a file DESTROYS another session's uncommitted work as surely
    # as writing it — and silently, which is worse.
    re.compile(r"\bgit\s+checkout\s+(?:--\s+)?([^\s;&|-][^\s;&|]*)"),
    re.compile(r"\bgit\s+restore\s+(?:--\S+\s+)*([^\s;&|-][^\s;&|]*)"),
    re.compile(r"\b(?:cp|mv)\s+[^;&|]*\s([^\s;&|]+)\s*(?:$|[;&|])"),
    re.compile(r"\btruncate\s+[^;&|]*\s([^\s;&|]+)"),
    re.compile(r"\bdd\s+[^;&|]*of=([^\s;&|]+)"),
]


def _warn(message: str) -> None:
    """Surface an allow-with-warning where a human will actually see it.

    On exit 0 stderr goes to the debug log only, so a stderr-only warning is
    indistinguishable from silence — and every fail-open path here is a state
    where the hook is enforcing NOTHING. A hook that has quietly stopped
    working must say so on the same channel as its refusals.
    """
    sys.stdout.write(json.dumps({
        "systemMessage": f"enforce_ownership is NOT enforcing: {message}"
    }) + "\n")
    sys.stderr.write(f"enforce_ownership (ALLOWING): {message}\n")


def _load_ownership() -> dict | None:
    try:
        import tomllib
        with OWNERSHIP.open("rb") as handle:
            return tomllib.load(handle)
    except Exception:
        return None


def _session_name(session_id: str | None) -> str | None:
    """Map this session's id to its name, or None when unmapped."""
    if not session_id:
        return None
    try:
        return json.loads(SESSION_NAMES.read_text()).get(session_id)
    except Exception:
        return None


def _targets(payload: dict) -> list[Path]:
    tool = payload.get("tool_name")
    tool_input = payload.get("tool_input") or {}
    raw: list[str] = []

    if tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        value = tool_input.get("file_path") or tool_input.get("notebook_path")
        if value:
            raw.append(value)
    elif tool == "Bash":
        command = tool_input.get("command") or ""
        # Strip QUOTED spans first. A `>` inside a quoted argument is text,
        # not a redirect — `gh issue comment --body "… -> path"` was matching
        # as a write to `path`. The trade-off is explicit: a redirect whose
        # TARGET is quoted (`> "a b.txt"`) is missed. Owned paths in this repo
        # have no spaces, and a false BLOCK on ordinary prose costs more than
        # a missed exotic write, which the edit tools still catch.
        #
        # IT IS ALSO LOAD-BEARING FOR `sed -i`, which was not obvious and was
        # found by a mutant's collateral: in `sed -i 's/a/b/' path` the SCRIPT
        # is a quoted span, and without stripping it the pattern captures
        # `'s/a/b/'` as the target instead of `path` — so the sed shapes stop
        # matching entirely. Removing this line does not merely re-admit a
        # false positive; it silently disables sed detection too.
        command = re.sub(r"'[^']*'|\"[^\"]*\"", " ", command)
        for pattern in _WRITE_PATTERNS:
            for match in pattern.finditer(command):
                raw.append(match.group(1).strip("'\""))

    out: list[Path] = []
    for item in raw:
        path = Path(item)
        if not path.is_absolute():
            path = (REPO / item)
        try:
            path = path.resolve()
            path.relative_to(REPO)
        except Exception:
            continue  # outside the repo -> not ours to gate
        out.append(path)
    return out


def _owner_of(rel: str, owners: dict) -> str | None:
    """Longest-glob match wins, so a specific file beats a directory glob."""
    best: tuple[int, str] | None = None
    for pattern, owner in owners.items():
        if fnmatch.fnmatch(rel, pattern):
            if best is None or len(pattern) > best[0]:
                best = (len(pattern), owner)
    return best[1] if best else None


def main(payload: dict) -> int:
    targets = _targets(payload)
    if not targets:
        return 0

    config = _load_ownership()
    if not config:
        _warn(f"no readable {OWNERSHIP.name}; ownership not enforced")
        return 0

    me = _session_name(payload.get("session_id"))
    if not me:
        _warn("this session is not mapped in session-names.json; "
              "ownership not enforced for it")
        return 0

    owners = config.get("owner") or {}
    consent = config.get("needs_consent") or {}

    blocked: list[tuple[str, str, bool]] = []
    for path in targets:
        rel = str(path.relative_to(REPO))
        owner = _owner_of(rel, owners)
        if owner is None or owner == me:
            continue
        # NO manifest exemption — see the module docstring. The review hook
        # gates Bash EXECUTION and never sees an Edit, so exempting an
        # approved file here would leave the owner's own file editable by
        # anyone and then refused to its owner on the changed hash.
        blocked.append((rel, owner, _owner_of(rel, consent) is not None))

    if not blocked:
        return 0

    lines = ["BLOCKED by enforce_ownership (#148): these paths belong to "
             f"another session, and you are {me}.\n"]
    for rel, owner, may_propose in blocked:
        lines.append(f"  - {rel}  ->  owned by {owner}\n")
        if may_propose:
            lines.append(
                f"      this path accepts PROPOSALS: send {owner} the exact "
                f"change and let them apply it.\n"
            )
    lines.append(
        "To clear: ask the owner to make the change, or ask the leader to "
        "reassign the path in .claude/ownership.toml. Editing another "
        "session's file mid-card is how two sessions end up with different "
        "beliefs about the same tree.\n"
    )
    sys.stderr.write("".join(lines))
    return 2


if __name__ == "__main__":
    try:
        data = json.load(sys.stdin)
    except Exception:
        raise SystemExit(0)
    try:
        raise SystemExit(main(data))
    except SystemExit:
        raise
    except Exception as exc:  # fail-open on OUR bug, loudly
        sys.stderr.write(
            f"enforce_ownership hook internal error (ALLOWING): {exc}\n")
        raise SystemExit(0)
