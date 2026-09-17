"""Pin for `tools/new_worker_worktree.sh --check` (#148).

RUNS THE SCRIPT IN A THROWAWAY GIT REPO, never this one. The script derives its
root from `git rev-parse --show-toplevel` in the CWD, so running it from a
temporary repo confines every branch and worktree question to that repo. A pin
that created branches in the real tree to test a refusal would be doing the
thing this card exists to stop.

`--check` only. The creating path builds a venv and installs editable packages;
that is not a unit test, and the assert it exists for (`import pynecore` must
resolve INSIDE the worktree) can only be proved by really creating one — which
happens on the operator's word, not in a test.

Run explicitly: `pytest tools/ -q`. Nothing in `tests/` or `plugins/dnse/tests/`
collects this.
"""
from __future__ import annotations

import pathlib
import subprocess

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parent / "new_worker_worktree.sh"


def _repo(tmp_path):
    """A throwaway git repo with one commit and a branch to base off."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "base"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "f").write_text("x\n")
    subprocess.run(["git", "add", "f"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "c"], cwd=tmp_path, check=True)
    return tmp_path


def _run(cwd, *args):
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=cwd,
                          capture_output=True, text=True)


def _state(cwd):
    def out(*args):
        return subprocess.run(args, cwd=cwd, capture_output=True,
                              text=True).stdout
    return (out("git", "worktree", "list"),
            out("git", "branch", "--list"),
            sorted(p.name for p in pathlib.Path(cwd).parent.iterdir()))


def __test_check_creates_nothing__(tmp_path):
    """The whole point of a dry run: it must be inert.

    Asserted on the WORLD (worktrees, branches, sibling directories), not on
    the script's own say-so — a dry run that printed "nothing created" while
    creating something is exactly the failure this mode exists to rule out.
    """
    repo = _repo(tmp_path / "r")
    before = _state(repo)
    result = _run(repo, "--check", "probe")
    assert result.returncode == 0, result.stderr
    assert "nothing created" in result.stdout
    assert _state(repo) == before


def __test_check_refuses_a_pre_existing_branch__(tmp_path):
    """The retire path (`git worktree move`) leaves the BRANCH behind.

    Without this the second attempt at a name passes the path check and then
    fails inside `git worktree add` — halfway, with a partially built tree and
    no venv. Refusing up front is the difference between "declined" and
    "broken".
    """
    repo = _repo(tmp_path / "r")
    subprocess.run(["git", "branch", "base-probe"], cwd=repo, check=True)
    before = _state(repo)
    result = _run(repo, "--check", "probe")
    assert result.returncode == 1
    assert "already exists" in result.stderr
    assert _state(repo) == before


def __test_check_refuses_a_detached_head__(tmp_path):
    """`git rev-parse --abbrev-ref HEAD` prints the literal "HEAD" when
    detached, so the branch would have been `HEAD-<name>`. Measured on a real
    detached worktree before this guard existed."""
    repo = _repo(tmp_path / "r")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                          capture_output=True, text=True).stdout.strip()
    subprocess.run(["git", "checkout", "-q", "--detach", head], cwd=repo,
                   check=True)
    result = _run(repo, "--check", "probe")
    assert result.returncode == 1
    assert "DETACHED" in result.stderr


@pytest.mark.parametrize("args,expected", [
    (("--check", "bad name"), "name must be"),
    (("--check",), "usage:"),
])
def __test_bad_input_is_refused_before_anything_else__(tmp_path, args, expected):
    """Argument validation exits 2 (usage), distinct from 1 (preconditions
    failed) — the caller can tell "you asked wrongly" from "the world is not
    ready"."""
    repo = _repo(tmp_path / "r")
    result = _run(repo, *args)
    assert result.returncode == 2
    assert expected in result.stderr
