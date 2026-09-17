"""Unit tests for the #148 ownership hook — PreToolUse payloads in, verdicts out.

NOT RUN YET. Written during a live trading session under an explicit hold:
nothing from #148 executes or registers until the account is flat. These feed
the hook's `main()` directly with payload dicts, so they touch no venue, no
credentials and no settings.

MUTATION-TESTING NOTE, because this module is PATH-LOADED (it is not importable
as a package): a mutant run leaves a `__pycache__` entry that survives
restoring the source, and `inspect.getsource()` cannot see the difference —
it reads the `.py` while the code object comes from the `.pyc`. Verify
BEHAVIOURALLY (call it, assert the return), run mutants with
`PYTHONDONTWRITEBYTECODE=1`, and prefer a runtime patch over editing source.

Run explicitly: `pytest .claude/hooks/ -q`. They are not part of the core or
plugin suites; nothing in `tests/` or `plugins/dnse/tests/` collects them.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

_HOOK_PATH = pathlib.Path(__file__).resolve().parent / "enforce_ownership.py"
_spec = importlib.util.spec_from_file_location("enforce_ownership_uut", _HOOK_PATH)
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)

_OWNERSHIP = """
[owner]
"plugins/dnse/tools/venue.py" = "Worker3"
"plugins/dnse/testing/live_test/**" = "Worker3"
"tools/mine.sh" = "Worker1"

[needs_consent]
"plugins/dnse/testing/live_test/**" = "Worker3"
"""

_SESSIONS = {"sid-worker1": "Worker1", "sid-worker3": "Worker3"}


@pytest.fixture()
def wired(tmp_path, monkeypatch):
    """Point the hook at throwaway config, and at a repo root we control."""
    repo = tmp_path
    claude = repo / ".claude"
    claude.mkdir()
    (claude / "ownership.toml").write_text(_OWNERSHIP)
    (claude / "session-names.json").write_text(json.dumps(_SESSIONS))
    (claude / "fable-approved.manifest").write_text("")
    for rel in ("plugins/dnse/tools", "plugins/dnse/testing/live_test", "tools"):
        (repo / rel).mkdir(parents=True, exist_ok=True)
    (repo / "plugins/dnse/tools/venue.py").write_text("# owned by Worker3\n")
    (repo / "tools/mine.sh").write_text("# owned by Worker1\n")

    # No MANIFEST patch: the hook no longer reads the manifest at all
    # (review finding F-1 — a manifest entry must not exempt an EDIT).
    # `monkeypatch.setattr` on a removed attribute would raise, which is the
    # right behaviour: a fixture that silently patches nothing is how a test
    # ends up asserting against a module it never configured.
    monkeypatch.setattr(hook, "REPO", repo)
    monkeypatch.setattr(hook, "OWNERSHIP", claude / "ownership.toml")
    monkeypatch.setattr(hook, "SESSION_NAMES", claude / "session-names.json")
    return repo


def _edit(path, session="sid-worker1"):
    return {"tool_name": "Edit", "session_id": session,
            "tool_input": {"file_path": str(path)}}


def _bash(command, session="sid-worker1"):
    return {"tool_name": "Bash", "session_id": session,
            "tool_input": {"command": command}}


def __test_another_owners_path_is_blocked__(wired, capsys):
    rc = hook.main(_edit(wired / "plugins/dnse/tools/venue.py"))
    assert rc == 2
    assert "venue.py" in capsys.readouterr().err


def __test_my_own_path_is_allowed__(wired):
    assert hook.main(_edit(wired / "tools/mine.sh")) == 0


def __test_an_unmapped_session_is_allowed_with_a_warning__(wired, capsys):
    """Fail OPEN. A hook that wedges an unknown session is worse than the
    collision it prevents — and an unmapped session is a leader-bookkeeping
    gap, not a policy violation by that session."""
    rc = hook.main(_edit(wired / "plugins/dnse/tools/venue.py",
                         session="sid-nobody"))
    assert rc == 0
    out = capsys.readouterr()
    assert "not mapped" in out.err
    # Review finding F-3: on exit 0 stderr reaches the debug log only, so the
    # warning must ALSO go out as a systemMessage — otherwise a hook that has
    # silently stopped enforcing looks exactly like one that is working.
    assert "systemMessage" in out.out
    assert "NOT enforcing" in out.out


def __test_an_unowned_path_is_allowed__(wired):
    """Unowned is not forbidden. The map names owners, not permissions."""
    assert hook.main(_edit(wired / "README.md")) == 0


@pytest.mark.parametrize("command", [
    "sed -i 's/a/b/' plugins/dnse/tools/venue.py",
    "echo x > plugins/dnse/tools/venue.py",
    "echo x >> plugins/dnse/tools/venue.py",
    "cat foo | tee plugins/dnse/tools/venue.py",
])
def __test_bash_write_shapes_are_blocked__(wired, command):
    """Enforcement that only watches Edit/Write is bypassed by a shell
    redirect, and a rule breakable by accident is not a rule."""
    assert hook.main(_bash(command)) == 2


def __test_a_bash_heredoc_write_is_blocked__(wired):
    """`cat > path <<EOF` is the shape that would otherwise walk straight
    past an edit-tool-only gate."""
    assert hook.main(_bash(
        "cat > plugins/dnse/tools/venue.py <<'EOF'\nx\nEOF")) == 2


def __test_a_read_only_bash_on_an_owned_path_is_allowed__(wired):
    """Reading someone's file is not editing it. A gate that blocks `grep`
    teaches people to work around the gate."""
    assert hook.main(_bash("grep -n foo plugins/dnse/tools/venue.py")) == 0


def __test_an_approved_file_is_STILL_owner_gated__(wired):
    """A manifest entry must NOT exempt an EDIT. Review finding F-1.

    An earlier draft allowed any edit to a manifest-approved file, arguing the
    review hook already gated it. That hook is registered on **Bash only** and
    gates EXECUTION — it never sees an Edit. So the exemption let any mapped
    session edit another owner's approved file; the edit changed the hash; and
    the OWNER's next run was refused by the review hook naming their own file,
    with no record of who had touched it. The fail-open manufactured the wedge
    it was meant to prevent.
    """
    import hashlib
    target = wired / "plugins/dnse/tools/venue.py"
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (wired / ".claude/fable-approved.manifest").write_text(
        f"{digest}  plugins/dnse/tools/venue.py\n")
    assert hook.main(_edit(target)) == 2, (
        "an approved file was editable by a non-owner — the owner's next run "
        "would then be refused on a hash they did not change"
    )


def __test_a_quoted_redirect_is_not_a_write__(wired):
    """`>` inside a quoted argument is TEXT, not a redirect.

    `gh issue comment --body "... -> plugins/dnse/tools/venue.py"` matched as
    a write and would have blocked ordinary prose. A false BLOCK on a comment
    costs more than the exotic write this trade-off misses, which the edit
    tools still catch.
    """
    assert hook.main(_bash(
        'gh issue comment 1 --body "see -> plugins/dnse/tools/venue.py"')) == 0


@pytest.mark.parametrize("command", [
    'echo x > "plugins/dnse/tools/venue.py"',
    "echo x >> 'plugins/dnse/tools/venue.py'",
    'cat <<EOF > "plugins/dnse/tools/venue.py"\nx\nEOF',
])
def __test_a_QUOTED_literal_redirect_target_is_blocked__(wired, command):
    """Measured at enabling (2026-09-17): `echo x > "plugins/dnse/tools/venue.py"`
    produced NO target — the quote-strip removed the path before the redirect
    pattern saw it — so a habit-quoted literal path bypassed the hook with one
    keystroke. The disclosure comment had called that case exotic. The wrong
    implementation this pins against is exactly the shipped one: strip quotes
    first, scan second, with no quoted-target pattern on the raw command."""
    assert hook.main(_bash(command)) == 2


def __test_a_variable_redirect_target_is_allowed_but_WARNED__(wired, capsys):
    """`F=plugins/dnse/tools/venue.py; echo x >> "$F"` cannot be resolved without
    evaluating the shell, which a PreToolUse hook must not do. It is allowed
    (`> "$LOG"` to scratch is ordinary) but NEVER silent: a systemMessage on
    stdout names the unresolved target. The wrong implementation is the silent
    allow, which is what let two sessions believe the hook was not loaded."""
    rc = hook.main(_bash('F=plugins/dnse/tools/venue.py; echo x >> "$F"'))
    out = capsys.readouterr().out
    assert rc == 0
    assert "systemMessage" in out and "cannot resolve" in out and "$F" in out


@pytest.mark.parametrize("command", [
    "sed -i 's/a/b/' plugins/dnse/tools/venue.py && echo done",
    "git checkout -- plugins/dnse/tools/venue.py",
    "git restore plugins/dnse/tools/venue.py",
])
def __test_more_write_shapes_are_blocked__(wired, command):
    """`sed -i … f && …` is ordinary, so the pattern cannot anchor on `$`.
    And RESTORING a file destroys another session's uncommitted work as surely
    as writing it — more quietly, which is worse."""
    assert hook.main(_bash(command)) == 2


def __test_a_refusal_names_the_hook_and_the_remedy__(wired, capsys):
    """A refusal that does not say which gate issued it, and how to clear it,
    costs more time than the collision it prevented."""
    hook.main(_edit(wired / "plugins/dnse/tools/venue.py"))
    err = capsys.readouterr().err
    assert "enforce_ownership" in err
    assert "ownership.toml" in err


def __test_a_needs_consent_refusal_names_who_applies_it__(wired, capsys):
    """The second relation exists so a correct fix in another session's file
    does not stall at the boundary — so the refusal must point at the route,
    not just close the door."""
    hook.main(_edit(wired / "plugins/dnse/testing/live_test/run_x.sh"))
    err = capsys.readouterr().err
    assert "PROPOSALS" in err
    assert "Worker3" in err


def __test_a_missing_ownership_file_allows__(wired, monkeypatch, capsys):
    """Fail open on our own missing config, loudly."""
    monkeypatch.setattr(hook, "OWNERSHIP", wired / ".claude/nope.toml")
    assert hook.main(_edit(wired / "plugins/dnse/tools/venue.py")) == 0
    assert "ownership" in capsys.readouterr().err
