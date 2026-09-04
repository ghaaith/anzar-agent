"""Command execution policy for the Anzar agent.

A code-enforced policy layer sits between the LLM's ``run_command`` tool call
and the sandbox executor. Every command is classified as one of:

- ``ALLOW``      — execute immediately (safe, read-only, or workspace-local).
- ``APPROVE``    — ask the user first (mutating / side-effecting / unknown).
- ``BLOCK``      — refuse outright (command injection, workspace traversal,
                   destructive system-level operations).

The verdict is computed statically from the raw command string. It is a
heuristic security control, NOT a real sandbox: when Docker is unavailable the
command still runs on the host, so this policy is best-effort defense rather
than a hard isolation boundary.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Verdict(str, Enum):
    ALLOW = "allow"
    APPROVE = "approve"
    BLOCK = "block"


@dataclass(frozen=True)
class CommandDecision:
    verdict: Verdict
    reason: str


@dataclass(frozen=True)
class CommandPolicy:
    """Static command policy bound to one workspace directory.

    Args:
        workspace_path: Absolute path of the workspace. Commands that try to
            escape it (``..``, absolute paths, drive letters) are blocked.
        timeout: Per-command execution timeout in seconds.
        max_stdout_chars: Hard cap on captured stdout.
        max_stderr_chars: Hard cap on captured stderr.
    """

    workspace_path: str
    timeout: int = 30
    max_stdout_chars: int = 4000
    max_stderr_chars: int = 2000

    def __post_init__(self) -> None:
        ws = os.path.normpath(os.path.abspath(str(self.workspace_path)))
        if os.name == "nt":
            ws = ws.lower()
        object.__setattr__(self, "_workspace", Path(ws))
        object.__setattr__(self, "_workspace_str", str(self._workspace))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, command: str) -> CommandDecision:
        """Classify a raw command string. Empty/whitespace commands are blocked."""
        if command is None or not command.strip():
            return CommandDecision(Verdict.BLOCK, "empty command")

        if self._injection_violation(command):
            return CommandDecision(Verdict.BLOCK, "command injection tokens detected")

        if self._traversal_violation(command):
            return CommandDecision(Verdict.BLOCK, "workspace traversal or absolute path")

        return self.classify(command)

    def classify(self, command: str) -> CommandDecision:
        """Classify a (non-injected, non-traversal) command by its verb."""
        try:
            tokens = shlex.split(command, posix=(os.name != "nt"))
        except ValueError:
            return CommandDecision(
                Verdict.APPROVE, "unparseable quoting — require human approval"
            )

        if not tokens:
            return CommandDecision(Verdict.BLOCK, "empty command")

        verb = os.path.basename(tokens[0]).lower()
        if verb.endswith(".exe"):
            verb = verb[:-4]

        if not verb:
            return CommandDecision(Verdict.BLOCK, "missing command verb")

        decision = self._classify_verb(verb, tokens)
        if decision is not None:
            return decision

        return CommandDecision(
            Verdict.APPROVE, f"unrecognized command '{verb}' — require approval"
        )

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def _injection_violation(self, command: str) -> bool:
        """Detect shell metacharacters used for chaining/piping/redirect."""
        tokens = (";", "&&", "||", "|", "`", "$(", "${", ">", "<", "\n", "\r")
        return any(tok in command for tok in tokens)

    def _traversal_violation(self, command: str) -> bool:
        """Detect attempts to reference paths outside the workspace."""
        # Parent-directory references: "..", "../x", "cd.." at token boundaries.
        if re.search(r"(^|[\s=\"'])(\.\.(?:[\\/]|$)|cd\.\.)", command):
            return True

        # Bare home reference: "~" (or "~/...") as its own token. A tilde glued
        # to a word (e.g. `pytest -k "~Async"`) is not a home reference.
        if re.search(r"(^|[\s=\"'])(~)(?=[\s/\\\"']|$)", command):
            return True

        # Environment-variable home references.
        for var in ("$HOME", "$USERPROFILE", "%HOME%", "%USERPROFILE%"):
            if var in command:
                return True

        # Absolute paths: bare root slash, unix absolute, drive letters, UNC.
        if re.search(r"(^|[\s=\"'])/(?:[^\s=\"']|$)", command):
            return True
        if re.search(r"(^|[\s=\"'])[A-Za-z]:[\\/]", command):
            return True
        if re.search(r"(^|[\s=\"'])\\\\(?!\\)", command):
            return True
        return False

    # ------------------------------------------------------------------
    # Verb classification
    # ------------------------------------------------------------------

    def _classify_verb(self, verb: str, tokens: list[str]) -> CommandDecision | None:
        """Return a decision for a known verb, or None to fall through to default."""
        if verb.startswith("mkfs") or verb in self._BLOCKED:
            return CommandDecision(Verdict.BLOCK, f"'{verb}' is blocked")
        if verb in self._PATH_VERBS:
            return CommandDecision(Verdict.APPROVE, f"'{verb}' acts on a path — require approval")
        if verb in ("python", "py", "python3", "node"):
            return self._interpreter_decision(tokens)
        if verb in ("bash", "sh", "zsh", "dash", "ksh", "fish", "cmd", "powershell", "pwsh"):
            return self._inline_code_decision(verb, tokens)
        if verb == "git":
            return self._git_decision(tokens)
        if verb == "rm":
            return self._rm_decision(tokens)
        if verb == "find":
            return self._find_decision(tokens)
        if verb == "sed":
            return self._sed_decision(tokens)
        if verb in ("pytest", "pytest-3"):
            return CommandDecision(Verdict.ALLOW, "pytest test run")
        if verb in self._LINTERS:
            return CommandDecision(Verdict.ALLOW, f"'{verb}' is a linter/formatter")
        if verb in ("npm", "npx", "yarn", "pnpm", "bun"):
            return self._node_decision(tokens)
        if verb == "pip":
            return self._pip_decision(tokens)
        if verb in ("go", "cargo", "mvn", "gradle", "deno"):
            return self._lang_tool_decision(tokens)
        if verb in ("make", "cmake", "ninja", "bazel", "meson"):
            return CommandDecision(Verdict.APPROVE, f"'{verb}' is a build step — require approval")
        if verb in self._READONLY:
            return CommandDecision(Verdict.ALLOW, f"'{verb}' is read-only")
        return None

    # -- interpreters & shells --------------------------------------------

    def _interpreter_decision(self, tokens: list[str]) -> CommandDecision:
        """python/py/python3/node: script execution is allowed, inline code is not."""
        args = tokens[1:]
        if not args:
            return CommandDecision(Verdict.BLOCK, "bare interpreter opens an interactive shell")
        for tok in args:
            if tok in ("-c", "-e", "-p", "-i"):
                return CommandDecision(
                    Verdict.BLOCK, f"'{tokens[0]} {tok}' executes inline code — blocked"
                )
            if tok == "-m":
                return CommandDecision(Verdict.ALLOW, "interpreter module execution")
        if args[0] in ("-V", "--version", "-h", "--help"):
            return CommandDecision(Verdict.ALLOW, "interpreter info flag")
        if args[0].startswith("-"):
            return CommandDecision(Verdict.APPROVE, "interpreter flag invocation — require approval")
        return CommandDecision(Verdict.ALLOW, "interpreter script execution")

    def _inline_code_decision(self, verb: str, tokens: list[str]) -> CommandDecision:
        """Shells / cmd / powershell: any inline-code flag is blocked outright."""
        flags = self._INLINE_CODE_FLAGS[verb]
        for tok in tokens[1:]:
            if tok in flags:
                return CommandDecision(
                    Verdict.BLOCK, f"'{verb} {tok}' executes inline code — blocked"
                )
            if tok.startswith("-") and len(tok) > 1:
                return CommandDecision(
                    Verdict.APPROVE, f"'{verb}' with unknown flag — require approval"
                )
        if len(tokens) > 1:
            return CommandDecision(Verdict.ALLOW, f"'{verb}' runs a script file")
        return CommandDecision(Verdict.BLOCK, f"bare '{verb}' opens an interactive shell")

    # -- git ------------------------------------------------------------------

    def _git_decision(self, tokens: list[str]) -> CommandDecision:
        sub = self._git_subcommand(tokens)
        if sub is None:
            return CommandDecision(Verdict.APPROVE, "git without a clear subcommand — require approval")
        sub = sub.lower()
        if sub in ("push",) and self._git_force_push(tokens):
            return CommandDecision(Verdict.BLOCK, "git force push is blocked")
        if sub in ("push", "pull", "fetch", "clone"):
            return CommandDecision(Verdict.APPROVE, "git network operation — require approval")
        if sub in ("reset",) and "--hard" in tokens:
            return CommandDecision(Verdict.BLOCK, "git reset --hard is blocked")
        if sub in ("clean",) and "-f" in tokens:
            return CommandDecision(Verdict.BLOCK, "git clean -f is blocked")
        if sub == "checkout" and "--" in tokens:
            return CommandDecision(Verdict.BLOCK, "git checkout -- discards work — blocked")
        if sub in ("branch",) and any(t in tokens for t in ("-d", "-D")):
            return CommandDecision(Verdict.BLOCK, "git branch -d/-D is blocked")
        if sub == "stash" and any(t in tokens for t in ("drop", "clear")):
            return CommandDecision(Verdict.BLOCK, "git stash drop/clear is blocked")
        if sub == "rm":
            return CommandDecision(Verdict.BLOCK, "git rm is blocked")
        if sub == "stash":
            return CommandDecision(Verdict.APPROVE, "git stash — require approval")
        if sub == "tag":
            tag_args = tokens[self._sub_index(tokens, "tag") + 1 :]
            if any(not t.startswith("-") for t in tag_args):
                return CommandDecision(Verdict.APPROVE, "git tag mutation — require approval")
            return CommandDecision(Verdict.ALLOW, "git tag list")
        if sub in ("remote", "tag") and any(
            t in tokens for t in ("add", "remove", "delete", "set-url", "rm")
        ):
            return CommandDecision(Verdict.APPROVE, "git remote/tag mutation — require approval")
        return CommandDecision(Verdict.ALLOW, f"git {sub}")

    def _sub_index(self, tokens: list[str], sub: str) -> int:
        return next(i for i, t in enumerate(tokens) if t == sub)

    def _git_subcommand(self, tokens: list[str]) -> str | None:
        skip = ("-C", "-c", "--git-dir", "--work-tree")
        it = iter(tokens[1:])
        for tok in it:
            if tok in skip:
                next(it, None)
                continue
            if tok.startswith("-"):
                continue
            return tok
        return None

    def _git_force_push(self, tokens: list[str]) -> bool:
        return any(t in tokens for t in ("-f", "--force", "--force-with-lease", "--force-if-includes"))

    # -- rm -------------------------------------------------------------------

    def _rm_decision(self, tokens: list[str]) -> CommandDecision:
        short = [t for t in tokens[1:] if t.startswith("-") and not t.startswith("--")]
        recursive = (
            any(t in ("-r", "-R", "--recursive") for t in tokens[1:])
            or any("r" in t[1:] for t in short)
        )
        force = "--force" in tokens or any("f" in t[1:] for t in short)
        if not recursive:
            return CommandDecision(Verdict.ALLOW, "rm on a single file — allow")
        targets = [t for t in tokens[1:] if not t.startswith("-")]
        for target in targets:
            if target in (".", "..", "*", "~", "/"):
                return CommandDecision(Verdict.BLOCK, f"rm -rf target '{target}' is blocked")
            if self._is_outside_workspace(target):
                return CommandDecision(Verdict.BLOCK, "rm -rf escapes the workspace — blocked")
        if force or recursive:
            return CommandDecision(Verdict.APPROVE, "recursive rm — require approval")
        return CommandDecision(Verdict.ALLOW, "rm single path — allow")

    # -- find ------------------------------------------------------------------

    def _find_decision(self, tokens: list[str]) -> CommandDecision:
        for flag in ("-delete", "-exec", "-execdir", "-ok"):
            if flag in tokens:
                return CommandDecision(Verdict.BLOCK, "find with destructive action is blocked")
        return CommandDecision(Verdict.ALLOW, "find search — allow")

    # -- sed ------------------------------------------------------------------

    def _sed_decision(self, tokens: list[str]) -> CommandDecision:
        if "-i" in tokens or "--in-place" in tokens:
            return CommandDecision(Verdict.APPROVE, "sed -i modifies files — require approval")
        return CommandDecision(Verdict.ALLOW, "sed stream — allow")

    # -- node tooling ------------------------------------------------------------

    def _node_decision(self, tokens: list[str]) -> CommandDecision:
        args = [t for t in tokens[1:] if not t.startswith("-")]
        script = args[0] if args else None
        if script == "run":
            sub = args[1] if len(args) > 1 else None
            if sub in ("test", "build", "lint", "format", "typecheck", "check", "tsc", "eslint"):
                return CommandDecision(Verdict.ALLOW, "npm run <safe script>")
            return CommandDecision(Verdict.APPROVE, "npm run <arbitrary script> — require approval")
        if script in ("test", "build", "lint", "format", "typecheck", "check", "tsc", "eslint", "prettier"):
            return CommandDecision(Verdict.ALLOW, f"npm {script}")
        return CommandDecision(Verdict.APPROVE, "npm install/etc — require approval")

    # -- pip -------------------------------------------------------------------

    def _pip_decision(self, tokens: list[str]) -> CommandDecision:
        args = [t for t in tokens[1:] if not t.startswith("-")]
        cmd = args[0] if args else None
        if cmd in ("list", "freeze", "show", "check", "debug", "help"):
            return CommandDecision(Verdict.ALLOW, f"pip {cmd} is read-only")
        return CommandDecision(Verdict.APPROVE, "pip install/uninstall — require approval")

    # -- generic language toolchains ---------------------------------------------

    def _lang_tool_decision(self, tokens: list[str]) -> CommandDecision:
        args = [t for t in tokens[1:] if not t.startswith("-")]
        cmd = args[0] if args else None
        if cmd in ("test", "build", "fmt", "lint", "check", "clippy", "doc", "vet", "clean"):
            return CommandDecision(Verdict.ALLOW, f"{tokens[0]} {cmd}")
        return CommandDecision(Verdict.APPROVE, f"{tokens[0]} — require approval")

    # -- helpers -------------------------------------------------------------------

    def _is_outside_workspace(self, target: str) -> bool:
        """True if a path target refers outside the workspace root."""
        if not target.startswith((".", "/", "\\")) and not os.path.isabs(target):
            return False
        try:
            resolved = (self._workspace / target).resolve()
        except (OSError, ValueError):
            return True
        try:
            resolved.relative_to(self._workspace)
            return False
        except ValueError:
            return True

    # -- classification tables -------------------------------------------------------

    _BLOCKED = {
        "sudo", "su", "dd", "fdisk", "shutdown", "reboot",
        "halt", "poweroff", "kill", "killall", "pkill", "xargs", "mount",
        "umount", "useradd", "usermod", "passwd", "crontab", "systemctl",
        "service", "eval", "source", "export", "unset",
    }

    _PATH_VERBS = {"cd", "cp", "mv", "ln", "mkdir", "rmdir", "touch", "chmod", "chown", "tee"}

    _READONLY = {
        "ls", "ll", "cat", "head", "tail", "less", "more", "grep", "rg",
        "awk", "echo", "pwd", "which", "where", "dir", "ps", "top",
        "htop", "env", "history", "file", "stat", "wc", "du", "df",
        "printf", "basename", "dirname",
    }

    _LINTERS = {
        "ruff", "black", "isort", "flake8", "mypy", "eslint", "prettier",
        "tsc", "py_compile", "pylint",
    }

    _INLINE_CODE_FLAGS = {
        "bash": ("-c",),
        "sh": ("-c",),
        "zsh": ("-c",),
        "dash": ("-c",),
        "ksh": ("-c",),
        "fish": ("-c",),
        "cmd": ("/c", "/k"),
        "powershell": ("-Command", "-EncodedCommand"),
        "pwsh": ("-Command", "-EncodedCommand"),
    }
