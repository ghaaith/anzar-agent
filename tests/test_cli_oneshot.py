"""Tests for one-shot task parsing (``anzar "<task>"``).

``_extract_task_arg`` decides whether the first bare CLI argument is a task or
the start of a subcommand (``checkpoint``, ``diff``, ...). It must leave flags
and subcommand arguments untouched.
"""

from __future__ import annotations

import pytest

from anzar.cli import _extract_task_arg


def test_no_args():
    assert _extract_task_arg([]) == ([], None)


def test_task_is_rest_of_command_line():
    filtered, task = _extract_task_arg(["build", "the", "checkout", "module"])
    assert filtered == []
    assert task == "build the checkout module"


def test_task_with_leading_flags_keeps_flags():
    filtered, task = _extract_task_arg(["--new", "say", "hello"])
    assert filtered == ["--new"]
    assert task == "say hello"


def test_task_preserves_flag_replaced_after_task():
    filtered, task = _extract_task_arg(["refactor", "api", "--new"])
    assert filtered == []
    assert task == "refactor api --new"


def test_workspace_flag_value_preserved():
    filtered, task = _extract_task_arg(["-w", "C:\\src", "add", "tests"])
    assert filtered == ["-w", "C:\\src"]
    assert task == "add tests"


def test_load_flag_value_preserved():
    filtered, task = _extract_task_arg(["--load", "3", "say hi"])
    assert filtered == ["--load", "3"]
    assert task == "say hi"


@pytest.mark.parametrize(
    "argv",
    [
        ["checkpoint", "--desc", "snapshot"],
        ["diff", "abc..def", "-w", "proj"],
        ["rollback", "xyz", "--yes"],
        ["history", "-n", "5"],
        ["doctor"],
    ],
)
def test_known_subcommands_are_not_treated_as_tasks(argv):
    filtered, task = _extract_task_arg(argv)
    assert task is None
    assert filtered == argv
