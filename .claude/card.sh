#!/usr/bin/env bash
# card.sh <N> — print a card's BODY + ALL COMMENTS in one readable stream.
# The standing way to read a card (operator rule 2026-09-16): the body is the
# plan, the comments are the history; reading only the body misses verdicts.
set -u
[ $# -ge 1 ] || { echo "usage: card.sh <issue-number>"; exit 2; }
gh issue view "$1" --repo rubycell/pynecore \
    --json number,title,state,body,comments \
| python3 -c '
import sys, json
d = json.load(sys.stdin)
print("# #{} [{}] {}".format(d["number"], d["state"], d["title"]))
print()
print("=" * 78)
print("BODY:")
print(d["body"] or "(empty)")
comments = d.get("comments") or []
for i, c in enumerate(comments, 1):
    who = (c.get("author") or {}).get("login", "?")
    print("=" * 78)
    print("COMMENT {}/{} by {} @ {}:".format(i, len(comments), who, c.get("createdAt", "")[:16]))
    print(c.get("body") or "")
'
