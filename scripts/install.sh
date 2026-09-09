#!/bin/sh
# Anzar — one-line installer for macOS and Linux.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/ghaaith/anzar-agent/main/scripts/install.sh | sh
#
# Installs Python 3 (required), bootstraps pipx, and installs the `anzar`
# CLI. Safe to re-run: it upgrades an existing install.

set -eu

PACKAGE="anzar-agent"

log()  { printf '\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!!  %s\033[0m\n' "$*"; }

# --- 1. Find Python 3 -------------------------------------------------------
PY=""
for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        PY="$cand"
        break
    fi
done

if [ -z "$PY" ]; then
    warn "Python 3 is required but was not found."
    printf '\nInstall it from https://www.python.org/downloads/ then re-run this script.\n' >&2
    exit 1
fi

if ! "$PY" -m pip --version >/dev/null 2>&1; then
    warn "pip is missing for $PY. Install pip (https://pip.pypa.io/) and re-run."
    exit 1
fi

# --- 2. Bootstrap pipx (run as a module to avoid PATH gymnastics) -----------
if ! "$PY" -m pipx --version >/dev/null 2>&1; then
    log "Installing pipx..."
    "$PY" -m pip install --quiet --user pipx
    "$PY" -m pipx ensurepath >/dev/null 2>&1 || true
fi

if ! "$PY" -m pipx --version >/dev/null 2>&1; then
    warn "Could not install pipx."
    exit 1
fi

# --- 3. Install/upgrade anzar -----------------------------------------------
if "$PY" -m pipx list --short 2>/dev/null | grep -qx "$PACKAGE"; then
    log "Upgrading $PACKAGE..."
    "$PY" -m pipx upgrade "$PACKAGE" --force
else
    log "Installing $PACKAGE..."
    "$PY" -m pipx install "$PACKAGE" --force
fi

# --- 4. Help the user find the binary ---------------------------------------
BIN_DIR="$("$PY" -m pipx environment --value PIPX_BIN_DIR 2>/dev/null || echo "$HOME/.local/bin")"

if command -v anzar >/dev/null 2>&1; then
    printf '\n\033[1;32mDone! Anzar is installed.\033[0m\n'
    printf '  Run:    \033[1manzar\033[0m\n'
    printf '  One-shot: \033[1manzar "explain this codebase"\033[0m\n'
else
    printf '\n\033[1;32mInstalled.\033[0m Add \033[1m%s\033[0m to your PATH, e.g. in your shell rc:\n' "$BIN_DIR"
    printf '    export PATH="%s:$PATH"\n' "$BIN_DIR"
    printf '  Then run \033[1manzar\033[0m to get started.\n'
fi