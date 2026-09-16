#!/usr/bin/env python3
"""PreToolUse hook (Bash): a card read must include the COMMENTS.

Operator rule 2026-09-16: agents kept reading only a card's body (`gh issue
view` shows the body alone unless comments are requested), missing panel
verdicts and adjudications that live in the thread. This hook blocks any
`gh issue view` whose command does not request comments, pointing at the
correct invocation (or the `.claude/card.sh` helper, which prints body +
all comments in one stream).

Allowed without comments: `gh issue list`, `gh issue edit/close/comment`,
and any `gh issue view` that DOES carry "comments" anywhere in the command.
Exit 0 allow; exit 2 block (stderr shown to the model). Fail-open on our
own internal errors only.
"""
from __future__ import annotations

import json
import re
import sys

_VIEW = re.compile(r"\bgh\s+issue\s+view\b")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    if payload.get("tool_name") != "Bash":
        return 0
    command = (payload.get("tool_input") or {}).get("command") or ""
    if not _VIEW.search(command):
        return 0
    if "comments" in command:
        return 0
    sys.stderr.write(
        "BLOCKED (card-read rule, operator 2026-09-16): `gh issue view` without "
        "comments reads only the BODY — panel verdicts and adjudications live in "
        "the thread. Re-run with `--json number,title,state,body,comments`, or "
        "use the helper: `bash .claude/card.sh <N>` (prints body + ALL comments).\n"
    )
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        sys.stderr.write(f"enforce_card_reads hook internal error (ALLOWING): {exc}\n")
        raise SystemExit(0)
