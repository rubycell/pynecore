#!/usr/bin/env python3
"""PreToolUse hook (Bash): block execution of UNREVIEWED generated code.

Operator law 2026-09-16: new/changed code files (sh/py) must be reviewed by the
leader session (Fable) BEFORE anything runs them. This hook makes the law
mechanical for every session in this project (executor sessions included).

A file is RUNNABLE when either:
  * it is TRACKED and CLEAN at HEAD (committed code went through review), or
  * its sha256 is listed in .claude/fable-approved.manifest
    (lines: "<sha256>  <repo-relative-path>", written by Fable on approval —
    any edit after approval changes the hash and re-blocks the file).

Scope: .py/.sh files INSIDE this repo referenced as things the command runs
(bash X, sh X, python X, ./X, or a bare executable path). Read-only mentions
(cat/grep/ls arguments) are, unavoidably, matched too when they look like an
execution — the hook errs toward blocking; Fable's approval is one manifest
line away. Files outside the repo, tracked-clean files, and non-code files
pass untouched. Fail-open ONLY on hook-internal errors (never block the whole
session on our own bug) — but parse problems in the COMMAND fall through to
allow, since the harness's own permission layer still applies.

pytest note (2026-09-17): any .py path in a pytest invocation is treated as an
execution — INCLUDING `--deselect path::name`, because pytest still IMPORTS a
deselected file at collection (module-level code runs). To exclude a foreign,
unapproved test file use `--ignore=path` (not imported) or `-k "not <name>"`.

Exit codes per the hooks contract: 0 allow; 2 block (stderr shown to the model).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / ".claude" / "fable-approved.manifest"

# command shapes that EXECUTE a file (group 1 = the path)
_EXEC_PATTERNS = [
    re.compile(r"(?:^|[;&|]\s*|\b)(?:bash|sh|source)\s+([^\s;&|]+\.sh)\b"),
    re.compile(r"(?:^|[;&|]\s*|\b)(?:[^\s;&|]*python[0-9.]*)\s+([^\s;&|]+\.py)\b"),
    re.compile(r"(?:^|[;&|]\s*)(\./[^\s;&|]+\.(?:sh|py))\b"),
    re.compile(r"(?:^|[;&|]\s*)((?:/|[A-Za-z0-9_./-]*/)?plugins/[^\s;&|]+\.(?:sh|py))(?=\s|$|[;&|])"),
    # pytest executes the named test file (and whatever it imports): every .py
    # path argument in a pytest / py.test / -m pytest invocation is an execution.
    # Known limit (MEASURED 2026-09-17: happened within the hour): an APPROVED test
    # importing an unapproved tool is not visible to a path matcher — the tool ran once
    # before re-approval. Workers must re-submit an edited tool's hash BEFORE running
    # any test that imports it.
    re.compile(r"(?:pytest|py\.test)\b[^;&|]*?\s([^\s;&|]+\.py)(?=\s|$|[;&|:])"),
    # `pyne run <script>.py …` EXECUTES the strategy, and it is the shape EVERY
    # live order-routing run uses (`--broker`). Neither pattern above sees it: the
    # path is an ARGUMENT, not the command head, so the plugins/ rule misses it,
    # and the interpreter rule needs "python" in the executable name, which
    # `.venv/bin/pyne` does not contain. MEASURED 2026-09-18 (Worker1 disclosed,
    # Fable reproduced red-first): two unapproved vehicles ran six times, no block.
    # Anchored on the SUBCOMMAND so a grep for the string "pyne" is not a false
    # positive; `pynecore` in a path does not match (no word boundary after "pyne").
    re.compile(r"(?:^|[;&|]\s*|\b)[^\s;&|]*\bpyne\s+(?:run|compile)\b[^;&|]*?\s([^\s;&|]+\.py)(?=\s|$|[;&|])"),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _approved_hashes() -> set[str]:
    if not MANIFEST.exists():
        return set()
    hashes = set()
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            hashes.add(line.split()[0])
    return hashes


def _tracked_and_clean(rel: str) -> bool:
    """True when the file is committed and byte-identical to HEAD."""
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "--error-unmatch", rel],
        capture_output=True,
    ).returncode == 0
    if not tracked:
        return False
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", rel],
        capture_output=True,
    ).returncode != 0
    return not dirty


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    if payload.get("tool_name") != "Bash":
        return 0
    command = (payload.get("tool_input") or {}).get("command") or ""

    # A code path spelled through a shell variable or command substitution
    # cannot be resolved without evaluating the shell. Measured 2026-09-17
    # (#153): `pytest "$R/plugins/…/test_x.py"` ran an edited, unapproved test
    # file THROUGH this gate — the `$R/…` form was either not captured by the
    # execution patterns at all or, when captured, joined onto REPO as a
    # literal `$R/…` that does not exist and was dropped by the exists() filter
    # below. Either way the file was never judged; silence read as "clean".
    # This is a hard gate on executing unreviewed code, so "cannot tell" must
    # REFUSE, not allow. Checked on the raw command text, not on the captured
    # candidates, so the pattern set cannot hide it again.
    # Three spellings, because the tokenizer is where the blind spot moved
    # (measured, Worker1, same day): a plain variable; a `$( … )` substitution
    # whose INSIDE may contain whitespace — `"$(git rev-parse --show-toplevel)
    # /plugins/…/test_x.py"` is the idiom this repo's own CLAUDE.md teaches,
    # and a char class that stops at the first space never reached its suffix;
    # and the backtick form of the same.
    unresolvable = (re.search(r"\$\([^)]*\)[^\s\"';&|]*\.(?:py|sh)\b", command)
                    or re.search(r"`[^`]*`[^\s\"';&|]*\.(?:py|sh)\b", command)
                    or re.search(r"[`$][^\s\"';&|]*\.(?:py|sh)\b", command))
    if unresolvable:
        sys.stderr.write(
            "BLOCKED by the code-review law: the code path "
            f"{unresolvable.group(0)!r} is spelled through a shell variable or "
            "command substitution and cannot be resolved by the hook, so it "
            "cannot be judged approved or not. Use a literal path (relative "
            "to the repo root or absolute).\n")
        return 2

    candidates: set[Path] = set()
    for pattern in _EXEC_PATTERNS:
        for match in pattern.finditer(command):
            raw = match.group(1).strip("'\"")
            p = Path(raw)
            if not p.is_absolute():
                p = (REPO / raw).resolve()
            try:
                p.relative_to(REPO)
            except ValueError:
                continue  # outside the repo -> not ours to gate
            if p.suffix in (".py", ".sh") and p.exists():
                candidates.add(p)

    if not candidates:
        return 0

    approved = _approved_hashes()
    blocked: list[str] = []
    for p in sorted(candidates):
        rel = str(p.relative_to(REPO))
        if _tracked_and_clean(rel):
            continue
        if _sha256(p) in approved:
            continue
        blocked.append(rel)

    if not blocked:
        return 0

    sys.stderr.write(
        "BLOCKED by the code-review law (operator, 2026-09-16): the following "
        "file(s) are neither committed-clean at HEAD nor Fable-approved:\n"
        + "".join(f"  - {b}\n" for b in blocked)
        + "Submit the file to Fable for review (path + purpose + verification). "
        "On approval Fable records its sha256 in .claude/fable-approved.manifest, "
        "after which this command will pass. Editing an approved file re-blocks it.\n"
    )
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # fail-open on OUR bug, loudly
        sys.stderr.write(f"enforce_code_review hook internal error (ALLOWING): {exc}\n")
        raise SystemExit(0)
