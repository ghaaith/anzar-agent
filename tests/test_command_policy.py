"""Tests for the command execution policy (pure decision logic)."""

import os
import tempfile

import pytest

from anzar.agent.policy import CommandPolicy, Verdict


@pytest.fixture
def policy():
    with tempfile.TemporaryDirectory() as ws:
        yield CommandPolicy(ws)


# -- ALLOW -------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        "echo hello",
        "ls",
        "cat README.md",
        "grep foo src/main.py",
        "head -n 20 file.txt",
        "python main.py",
        "python -m pytest",
        "python -m pytest --tb=short",
        "python3 script.py",
        "node script.js",
        "py --version",
        "pip list",
        "pip show requests",
        "git status",
        "git diff",
        "git log --oneline",
        "git checkout main",
        "git branch feature-x",
        "git commit -m wip",
        "find . -name '*.py'",
        "find src -type f",
        "sed 's/a/b/' file.txt",
        "npm test",
        "npm run test",
        "npm run build",
        "go test ./...",
        "cargo test",
        "pytest -k '~Async'",
        "bash script.sh",
        "grep -E 'a' file",
    ],
)
def test_allowed_commands(policy, cmd):
    assert policy.evaluate(cmd).verdict is Verdict.ALLOW, cmd


# -- BLOCK -------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        "sudo rm -rf /",
        "dd if=/dev/zero of=/dev/sda",
        "mkfs.ext4 /dev/sdb1",
        "shutdown now",
        "reboot",
        "killall python",
        "xargs rm",
        "eval rm -rf /",
        "rm -rf /",
        "rm -rf .",
        "rm -rf ..",
        "rm -rf *",
        "rm -rf ~",
        "rm -rf ../etc",
        "python -c 'import os; os.remove(\"x\")'",
        "py -c 'print(1)'",
        "node -e 'require(\"fs\").rmSync(\"/\")'",
        "bash -c 'echo hi'",
        "sh -c 'echo hi'",
        "cmd /c del /f /q file",
        "powershell -Command Remove-Item / -Recurse",
        "git push --force origin main",
        "git push --force-with-lease",
        "git reset --hard HEAD",
        "git clean -f",
        "git checkout -- file",
        "git branch -D old",
        "git stash drop",
        "git stash clear",
        "git rm file.txt",
        "find . -delete",
        "find . -exec rm {} ;",
        "find . -execdir rm -rf .",
        "echo hi > file",
        "echo hi >> file",
        "grep foo < file",
        "echo a; ls",
        "echo a && ls",
        "echo a || ls",
        "echo a | grep a",
        "echo `whoami`",
        "echo $(whoami)",
        "echo ${HOME}",
        "cd ..",
        "cd..",
        "cat ../etc/passwd",
        "ls ~",
        "ls ~/docs",
        "echo $HOME",
        "echo %USERPROFILE%",
        "cat /etc/passwd",
        "python C:\\evil.py",
        "cat //server/share/file",
        "",
        "   ",
    ],
)
def test_blocked_commands(policy, cmd):
    assert policy.evaluate(cmd).verdict is Verdict.BLOCK, repr(cmd)


# -- APPROVE -----------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        "pip install requests",
        "pip uninstall requests",
        "npm install",
        "npm install requests",
        "npm run dev",
        "cd src",
        "cp a b",
        "mv a b",
        "touch file",
        "mkdir dir",
        "rm -r dir",
        "git push origin main",
        "git clone https://github.com/x/y",
        "git fetch",
        "git stash",
        "git remote add origin url",
        "git tag v1.0",
        "git tag -d v1.0",
        "make build",
        "cargo run",
        "mvn package",
        "unknown-tool --flag",
        "somebinary",
        "python --some-flag",
    ],
)
def test_approval_commands(policy, cmd):
    assert policy.evaluate(cmd).verdict is Verdict.APPROVE, cmd


# -- non-false-positive guards ------------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        "pytest -k '~Async'",
        "grep -v foo file.txt",
        "python -m py_compile src/main.py",
        "node --version",
        "git log -p -- file.txt",
        "git show HEAD",
        "git ls-files",
    ],
)
def test_not_blocked_by_heuristics(policy, cmd):
    assert policy.evaluate(cmd).verdict is not Verdict.BLOCK, cmd


# -- traversal details ---------------------------------------------------------


def test_tilde_glued_to_word_is_not_home(policy):
    assert policy.evaluate("pytest -k ~runner").verdict is not Verdict.BLOCK


def test_relative_in_workspace_path_allowed(policy):
    decision = policy.evaluate("python main.py")
    assert decision.verdict is Verdict.ALLOW


def test_workspace_basename_is_not_traversal(policy):
    # Even if the workspace happens to be named like a flag value.
    decision = policy.evaluate("ls")
    assert decision.verdict is Verdict.ALLOW


def test_evaluate_empty_returns_block(policy):
    assert policy.evaluate("").verdict is Verdict.BLOCK
    assert policy.evaluate(None).verdict is Verdict.BLOCK


def test_injection_detection_reports_reason(policy):
    decision = policy.evaluate("echo hi; ls")
    assert decision.verdict is Verdict.BLOCK
    assert "injection" in decision.reason.lower()
