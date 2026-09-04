"""CLI entry point for Anzar — DB-backed conversations with persistence."""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style as PtStyle
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.markdown import Markdown

from anzar import __version__
from anzar.agent import create_agent
from anzar.agent.core import AnzarAgent
from anzar.agent.providers import (
    SUPPORTED_PROVIDERS,
    provider_models,
    provider_name,
    provider_names,
)
from anzar.config import AnzarConfig
from anzar.db.auth import hash_password
from anzar.db.base import SessionLocal, init_db
from anzar.db.models import Checkpoint, Conversation, Message, Settings, Task, User, Workspace
from sqlalchemy.orm import Session

CLI_USER_EMAIL = "cli@local.anzar"
console = Console()
HISTORY_DIR = Path.home() / ".anzar"
HISTORY_FILE = HISTORY_DIR / "history.txt"
PT_STYLE = PtStyle.from_dict({"prompt": "bold cyan", "continuation": "dim"})


# --- Key bindings ---

kb = KeyBindings()


@kb.add("escape", "enter")
def _accept(event):
    event.current_buffer.validate_and_handle()


@kb.add("c-c")
def _cancel(event):
    event.current_buffer.reset()


# --- Helpers ---


def _ensure_cli_user(db, workspace_path: str) -> User:
    user = db.query(User).filter(User.email == CLI_USER_EMAIL).first()
    if not user:
        user = User(
            email=CLI_USER_EMAIL,
            name="CLI User",
            plan="pro",
            auth_provider="cli",
        )
        user.password_hash = hash_password("cli-no-password")
        db.add(user)
        db.flush()
        ws = Workspace(
            user_id=user.id,
            name="CLI Workspace",
            disk_path=workspace_path,
            disk_limit_mb=5000,
        )
        db.add(ws)
        config = AnzarConfig.load()
        settings = Settings(
            user_id=user.id,
            provider=config.provider,
            model=config.model,
        )
        db.add(settings)
        db.commit()
        db.refresh(user)
    else:
        ws = db.query(Workspace).filter(Workspace.user_id == user.id).first()
        if ws and ws.disk_path != workspace_path:
            ws.disk_path = workspace_path
            db.commit()
    return user


def _create_conv(db, user_id: uuid.UUID, title: str = "New Chat") -> Conversation:
    conv = Conversation(user_id=user_id, title=title)
    db.add(conv)
    db.commit()
    return conv


def _get_conv_by_index(db, user_id: uuid.UUID, index: int) -> Optional[Conversation]:
    convs = (
        db.query(Conversation)
        .filter(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )
    if 1 <= index <= len(convs):
        return convs[index - 1]
    return None


def _resume_conv(db, user: User, last_conversation_id: Optional[str]) -> Optional[Conversation]:
    """Load the saved last conversation for this user (None if unusable)."""
    if not last_conversation_id:
        return None
    try:
        cid = uuid.UUID(last_conversation_id)
    except (ValueError, AttributeError):
        return None
    return (
        db.query(Conversation)
        .filter(Conversation.id == cid, Conversation.user_id == user.id)
        .first()
    )


def _select_conversation(
    db,
    user: User,
    known,
    config: AnzarConfig,
    workspace_path: str,
) -> tuple[Optional[Conversation], bool]:
    """Choose the conversation to open.

    Precedence:
      1. ``--load N``     → conversation by index
      2. ``--continue``   → last conversation (regardless of workspace)
      3. ``--new``        → fresh conversation
      4. default          → auto-resume the last conversation when the
                           workspace matches the one it was last used in,
                           otherwise start a fresh conversation.

    Returns ``(conv, is_new)``. ``conv`` is None only when ``--load``
    references a missing index (the caller reports the error).
    """
    if known.load:
        return _get_conv_by_index(db, user.id, known.load), False

    if known.resume and config.last_conversation_id:
        conv = _resume_conv(db, user, config.last_conversation_id)
        if conv is not None:
            return conv, False

    if not known.new and config.last_conversation_id and config.last_workspace == workspace_path:
        conv = _resume_conv(db, user, config.last_conversation_id)
        if conv is not None:
            return conv, False

    return _create_conv(db, user.id), True


def _count_messages(db, conv_id: uuid.UUID) -> int:
    return db.query(Message).filter(Message.conversation_id == conv_id).count()


def _touch_conv(db, conv_id: uuid.UUID):
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if conv:
        conv.updated_at = datetime.now(timezone.utc)
        db.flush()


def _auto_title(db, conv_id: uuid.UUID, message: str):
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if conv and conv.title in ("New Chat", "CLI Session"):
        conv.title = message[:60].strip() or "New Chat"
        db.commit()


def _build_agent(
    db,
    user: User,
    conv_id: uuid.UUID,
    workspace_path: str,
    config: AnzarConfig,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> AnzarAgent:
    ws = db.query(Workspace).filter(Workspace.user_id == user.id).first()
    return create_agent(
        db=db,
        user_id=user.id,
        conversation_id=conv_id,
        workspace_id=ws.id if ws else None,
        provider_override=provider_override or config.provider,
        model_override=model_override or config.model,
        max_steps=config.max_steps,
    )


_PROVIDER_ENV_VARS = {
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def _provider_has_key(
    provider: str, db: Optional[Session] = None, user: Optional[User] = None
) -> bool:
    """True if an API key is available for the provider.

    Checks env vars (auto-loaded from .env), the provider-agnostic
    ANZAR_API_KEY, and — when a db/user pair is passed — a key persisted by the
    user in the local DB ``settings`` row.
    """
    if provider == "ollama":
        return True
    if os.environ.get("ANZAR_API_KEY"):
        return True
    env_var = _PROVIDER_ENV_VARS.get(provider)
    if env_var and os.environ.get(env_var):
        return True
    if db is not None and user is not None:
        settings = db.query(Settings).filter(Settings.user_id == user.id).first()
        if (
            settings
            and settings.provider == provider
            and settings.api_key_encrypted
        ):
            return True
    return False


def _mask_api_key(key: Optional[str]) -> str:
    """Mask an API key for display (first 4 + last 4 characters)."""
    if not key:
        return "****"
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def _set_api_key(
    db: Session,
    user: User,
    conv: Conversation,
    workspace_path: str,
    config: AnzarConfig,
    provider: str,
    key: Optional[str],
    toolbar_text: dict,
    reporter=None,
    model: Optional[str] = None,
) -> tuple[Optional[AnzarAgent], str, Optional[str]]:
    """Store or clear a provider API key for the CLI user, then rebuild agent.

    Mirrors ``_apply_model``'s return shape: (new_agent_or_None, provider, model).
    Uses the existing single-key-per-user ``Settings`` row (plaintext, matching
    the server's current behavior). Setting a key also selects that provider so
    the agent runs immediately. ``model`` (if given) is applied to the rebuilt
    agent; otherwise the provider's default model is used.
    """
    def _say(message: str):
        if reporter is not None:
            reporter(message)
        else:
            console.print(message)

    if provider not in SUPPORTED_PROVIDERS:
        _say(
            f"[red]Unknown provider:[/red] {provider}. "
            f"Choices: {', '.join(SUPPORTED_PROVIDERS)}"
        )
        return None, config.provider, config.model

    if provider == "ollama":
        _say("[yellow]Ollama runs locally — no API key is needed.[/yellow]")
        return None, config.provider, config.model

    settings = db.query(Settings).filter(Settings.user_id == user.id).first()
    if settings is None:
        settings = Settings(user_id=user.id, provider=provider)
        db.add(settings)

    if key in (None, "", "clear"):
        settings.api_key_encrypted = None
        db.commit()
        _say(f"[green]Key cleared[/green] for {provider}.")
        return None, config.provider, config.model

    settings.api_key_encrypted = key
    settings.provider = provider
    db.commit()
    _say(
        f"[green]Key stored[/green] for {provider} "
        f"(masked: {_mask_api_key(key)})."
    )
    # Rebuild the agent with this provider (and the requested model, if any).
    # create_agent reads the key we just persisted from the DB settings row.
    return _apply_model(
        db,
        user,
        conv,
        workspace_path,
        config,
        provider,
        model,
        toolbar_text,
        reporter=reporter,
    )


def _cmd_list_keys(db, user: User, reporter=None) -> None:
    """Show which providers have a usable key (DB-stored or env)."""
    settings = db.query(Settings).filter(Settings.user_id == user.id).first()
    lines: list[str] = []
    any_key = False
    for pid in provider_names():
        var = _PROVIDER_ENV_VARS.get(pid)
        has_env = bool(var and os.environ.get(var))
        has_db = bool(
            settings and settings.provider == pid and settings.api_key_encrypted
        )
        if has_db:
            any_key = True
            lines.append(
                f"  {provider_name(pid)} [{pid}]  stored  "
                f"{_mask_api_key(settings.api_key_encrypted)}"
            )
        elif has_env:
            any_key = True
            lines.append(f"  {provider_name(pid)} [{pid}]  env ({var})")
    if not any_key:
        lines.append("No API keys found. Use /apikey <provider> <key> to add one.")

    if reporter is not None:
        reporter("API keys:\n" + "\n".join(lines))
        return
    console.print("[bold]API keys:[/bold]")
    console.print("\n".join(lines))


def _apply_model(
    db,
    user: User,
    conv: Conversation,
    workspace_path: str,
    config: AnzarConfig,
    provider: str,
    model: Optional[str],
    toolbar_text: dict,
    reporter=None,
) -> tuple[Optional[AnzarAgent], str, Optional[str]]:
    """Switch provider/model: rebuild agent, persist config, refresh toolbar.

    Returns (new_agent, provider, model); on failure (e.g. missing API key)
    returns (None, current_provider, current_model) and nothing is changed.

    ``reporter`` is an optional callable(str) used instead of the Rich console
    (the TUI passes one so output goes to a widget instead of the screen).
    """
    def _say(message: str):
        if reporter is not None:
            reporter(message)
        else:
            console.print(message)

    try:
        agent = _build_agent(db, user, conv.id, workspace_path, config, provider, model)
    except Exception as e:
        _say(
            f"[red]Failed to switch[/red] to {provider}"
            f"{' / ' + model if model else ' / (default)'}: {e}"
        )
        return None, config.provider, config.model
    config.provider = provider
    config.model = model
    config.api_key = None  # clear stale key — auto-detect from .env on next load
    config.save()
    resolved = model or config.get_default_model()
    toolbar_text["provider"] = provider
    toolbar_text["model"] = resolved
    _say(f"[green]Switched[/green] to {provider} / {resolved}")
    return agent, provider, model


# --- Command handlers ---


def _cmd_help():
    console.print("""
[bold]Commands:[/bold]
  /help                  Show this help
  /quit                  Exit Anzar
  /clear                 Clear current conversation messages
  /list                  List saved conversations
  /load <N>              Load conversation #N from /list
  /rename <title>        Rename current conversation
  /save <name>           Alias for /rename
  /export [file.md]      Export conversation as markdown
  /search <query>        Search messages across conversations
  /workspace [path]      Show or change workspace directory
    /model                 Open interactive model picker
   /model set <p> [m]     Set provider and optionally model (e.g. /model set openai gpt-4o)
   /step_limit [n]        Show or set the per-turn step budget (default 50, max 500)
   /apikey <p> <key>      Store an API key for a provider (masked; auto-selects provider). /apikey <p> clear to delete
   /list-keys             Show providers with a usable key (stored or env)
   /paste                 Enter multi-line mode (Ctrl+D to send)
   /graph                 Write the agent graph (Mermaid) to a temp file

[bold]Input:[/bold]
  Enter         New line
  Alt+Enter     Send message
  Up/Down       Cycle through history

[bold]Examples:[/bold]
  /list
  /load 2
  /rename "Fix login bug"
  /search error
  /model set openai gpt-4o
  /apikey anthropic sk-ant-***
  /list-keys
""")


def _cmd_model_menu(
    db,
    user: User,
    conv: Conversation,
    workspace_path: str,
    config: AnzarConfig,
    current_provider: str,
    current_model: Optional[str],
    current_agent: AnzarAgent,
    toolbar_text: dict,
) -> tuple[AnzarAgent, str, Optional[str]]:
    """Interactive model toggle: pick a provider, then a model."""
    keep = (current_agent, current_provider, current_model)
    console.print()
    console.print(f"  [dim]Current:[/dim] {current_provider} / {current_model or config.get_default_model()}")

    providers = provider_names()
    console.print("  [bold]Provider:[/bold]")
    for i, pid in enumerate(providers, 1):
        mark = " [green](current)[/green]" if pid == current_provider else ""
        warn = "" if _provider_has_key(pid, db, user) else " [yellow](no API key — may fail)[/yellow]"
        console.print(f"    {i}. {provider_name(pid)} [{pid}]{mark}{warn}")
    console.print("    0. Keep current")
    console.print("    [dim]Press Enter to keep current.[/dim]")

    try:
        choice = console.input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        return keep
    if choice == "0" or not choice:
        return keep
    try:
        idx = int(choice)
    except ValueError:
        console.print("[red]Invalid choice.[/red]")
        return keep
    if not (1 <= idx <= len(providers)):
        console.print("[red]Invalid choice.[/red]")
        return keep

    pid = providers[idx - 1]
    models = provider_models(pid)
    console.print(f"  [bold]Model for {provider_name(pid)}:[/bold]")
    for i, m in enumerate(models, 1):
        is_current = pid == current_provider and m == current_model
        mark = " [green](current)[/green]" if is_current else ""
        default_mark = " [dim](default)[/dim]" if i == 1 and not is_current else ""
        console.print(f"    {i}. {m}{mark}{default_mark}")
    console.print("    0. Use default")
    console.print("    [dim]Press Enter to use the default.[/dim]")

    model: Optional[str] = None
    try:
        model_choice = console.input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        return keep
    if model_choice:
        try:
            midx = int(model_choice)
        except ValueError:
            midx = -1
        if midx == 0:
            model = None
        elif 1 <= midx <= len(models):
            model = models[midx - 1]
        else:
            console.print("[red]Invalid choice — using default.[/red]")

    # Model has been chosen. If the provider needs a key and we don't have one,
    # prompt for it now, then store it and build the agent with this model.
    if pid != "ollama" and not _provider_has_key(pid, db, user):
        from rich.prompt import Prompt

        console.print(
            f"  [yellow]No API key stored for {provider_name(pid)}.[/yellow]"
        )
        key = Prompt.ask(
            f"  Enter API key for {provider_name(pid)}",
            console=console,
            password=True,
        ).strip()
        if not key:
            console.print("  [dim]Keeping previous provider (no key set).[/dim]")
            return keep
        return _set_api_key(
            db, user, conv, workspace_path, config, pid, key, toolbar_text, model=model
        )

    result = _apply_model(db, user, conv, workspace_path, config, pid, model, toolbar_text)
    if result[0] is None:
        return keep  # switch failed — keep current agent/selection
    return result


def _cmd_list(db, user: User):
    rows = _list_conversations(db, user)
    if not rows:
        console.print("[dim]No conversations yet.[/dim]")
        return
    table = Table(box=box.SIMPLE, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Date", width=12)
    table.add_column("Title")
    table.add_column("Msg", justify="right")
    for i, d, title, n in rows:
        table.add_row(str(i), d, title, str(n))
    console.print(table)


def _list_conversations(db, user: User) -> list[tuple[int, str, str, int]]:
    """Return conversation rows (index, date, title, msg count) newest first."""
    convs = (
        db.query(Conversation)
        .filter(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )
    rows: list[tuple[int, str, str, int]] = []
    for i, conv in enumerate(convs, 1):
        n = _count_messages(db, conv.id)
        d = (
            conv.updated_at.strftime("%m/%d %H:%M")
            if conv.updated_at
            else conv.created_at.strftime("%m/%d %H:%M")
        )
        rows.append((i, d, conv.title[:60], n))
    return rows


def _cmd_search(db, user: User, query: str, conv_id: Optional[uuid.UUID] = None):
    results = _search_messages(db, user, query, conv_id)
    if not results:
        console.print("[dim]No results found.[/dim]")
        return
    table = Table(box=box.SIMPLE, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Conversation", width=30)
    table.add_column("Date", width=12)
    table.add_column("Snippet")
    for i, (msg, title, _) in enumerate(results, 1):
        snippet = msg.content[:80].replace("\n", " ")
        d = msg.created_at.strftime("%m/%d %H:%M")
        table.add_row(str(i), title[:30], d, snippet)
    console.print(table)


def _search_messages(db, user: User, query: str, conv_id: Optional[uuid.UUID] = None):
    """Return (Message, Conversation.title, Conversation.id) matches, newest first."""
    base = (
        db.query(Message, Conversation.title, Conversation.id.label("cid"))
        .join(Conversation, Message.conversation_id == Conversation.id)
        .filter(Conversation.user_id == user.id)
    )
    if conv_id:
        base = base.filter(Message.conversation_id == conv_id)
    return (
        base.filter(Message.content.ilike(f"%{query}%"))
        .order_by(Message.created_at.desc())
        .limit(20)
        .all()
    )


def _cmd_export(db, user: User, conv_id: uuid.UUID, filepath: Optional[str] = None, reporter=None):
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if not conv:
        (reporter or console.print)("[red]Conversation not found.[/red]")
        return None
    msgs = (
        db.query(Message)
        .filter(Message.conversation_id == conv_id)
        .order_by(Message.created_at)
        .all()
    )
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in conv.title)
    if not filepath:
        filepath = f"anzar-{safe_title[:40].strip()}.md"
    lines = [
        f"# {conv.title}",
        f"Date: {conv.created_at.strftime('%Y-%m-%d %H:%M')}",
        f"Messages: {len(msgs)}",
        "",
    ]
    for msg in msgs:
        role = "## You" if msg.role == "user" else "## Anzar"
        lines.append(role)
        lines.append("")
        lines.append(msg.content)
        lines.append("")
    Path(filepath).write_text("\n".join(lines), encoding="utf-8")
    (reporter or console.print)(f"[green]Exported[/green] to {filepath}")
    return filepath


# --- Rendering ---


def _render_user_message(message: str):
    console.print()
    console.print(
        Panel(
            Text(message),
            title="You",
            title_align="left",
            border_style="cyan",
            padding=(1, 2),
        )
    )


def _stream_agent_response(agent: AnzarAgent, prompt: str) -> str:
    full = ""
    last_render = 0.0
    render_interval = 0.1

    def _panel(text: str) -> Panel:
        return Panel(
            text.rstrip(),
            title="Anzar",
            title_align="left",
            border_style="green",
            padding=(1, 2),
        )

    try:
        with Live(_panel(""), refresh_per_second=10, transient=False) as live:
            for chunk in agent.stream(prompt):
                full += chunk
                now = time.monotonic()
                if now - last_render >= render_interval:
                    last_render = now
                    live.update(_panel(full))
            live.update(_panel(full))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
    return full


def _run_pipe_mode(agent: AnzarAgent):
    pipe_input = sys.stdin.read()
    if not pipe_input.strip():
        return
    try:
        for chunk in agent.stream(pipe_input.strip()):
            console.print(chunk, end="")
        console.print()
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")


# --- Main ---


def _force_utf8_output():
    """Ensure streaming output doesn't crash on non-cp1252 chars (e.g. emoji)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _run_legacy_repl(
    db,
    config: AnzarConfig,
    user: User,
    conv: Conversation,
    workspace_path: str,
    agent: AnzarAgent,
    msg_count: int,
    is_new: bool,
):
    """Original prompt_toolkit REPL (kept behind --legacy / non-TUI contexts)."""
    # --- Session state ---
    current_provider = config.provider
    current_model = config.model
    toolbar_text = {
        "provider": current_provider,
        "model": current_model or config.get_default_model(),
        "msgs": str(msg_count),
    }

    def _toolbar():
        d = toolbar_text
        return HTML(
            f" <b>{d['provider']}</b>/<b>{d['model']}</b>"
            f"  |  <b>{d['msgs']}</b> msgs"
            f"  |  /help"
        )

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    session = PromptSession(
        multiline=True,
        key_bindings=kb,
        history=FileHistory(str(HISTORY_FILE)),
        style=PT_STYLE,
        prompt_continuation=lambda width, line_no, wrap_count: HTML("<b><style color='ansiyellow'>...</style></b> "),
    )

    # --- Welcome ---
    welcome = Panel(
        Text.assemble(
            (f"Anzar v{__version__}", "bold cyan"),
            (" \u2022 AI Software Engineer Agent", "dim"),
            "\n\n",
            (f"Workspace: ", "bold"),
            (workspace_path, ""),
            "\n",
            (f"Provider:  ", "bold"),
            (f"{current_provider}", "cyan"),
            "\n",
            (f"Model:     ", "bold"),
            (f"{current_model or config.get_default_model()}", "cyan"),
        ),
        border_style="blue",
        padding=(1, 2),
    )
    if not is_new:
        welcome.subtitle = f'"{conv.title}" ({msg_count} messages)'
    console.print()
    console.print(welcome)
    console.print("[dim]  Alt+Enter to send | /help for commands[/dim]\n")

    # --- REPL loop ---
    while True:
        try:
            user_input = session.prompt(
                HTML("<b>></b> "),
                bottom_toolbar=_toolbar,
            )
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        raw = user_input.strip()
        if not raw:
            continue

        # --- Commands ---
        if raw == "/quit":
            break

        if raw == "/help":
            _cmd_help()
            continue

        if raw == "/clear":
            agent.memory.clear()
            console.print("[dim]Conversation cleared.[/dim]")
            continue

        if raw == "/list":
            _cmd_list(db, user)
            continue

        if raw == "/model":
            agent, current_provider, current_model = _cmd_model_menu(
                db, user, conv, workspace_path, config, current_provider, current_model, agent, toolbar_text
            )
            continue

        if raw.startswith("/model set"):
            parts = raw.split()
            if len(parts) >= 3:
                new_agent, current_provider, current_model = _apply_model(
                    db,
                    user,
                    conv,
                    workspace_path,
                    config,
                    parts[2],
                    parts[3] if len(parts) >= 4 else None,
                    toolbar_text,
                )
                if new_agent is not None:
                    agent = new_agent
            else:
                console.print("[red]Usage:[/red] /model set <provider> [model]")
            continue

        if raw.startswith("/apikey"):
            parts = raw.split()
            if len(parts) < 2:
                console.print("[red]Usage:[/red] /apikey <provider> <key> | /apikey <provider> clear")
            else:
                provider = parts[1]
                key = parts[2] if len(parts) > 2 else None
                toolbar_text = {"provider": current_provider, "model": current_model, "msgs": ""}
                new_agent, current_provider, current_model = _set_api_key(
                    db, user, conv, workspace_path, config, provider, key, toolbar_text
                )
            continue

        if raw.startswith("/list-keys"):
            _cmd_list_keys(db, user)
            continue

        if raw.startswith("/search"):
            parts = raw.split(maxsplit=1)
            if len(parts) >= 2:
                query = parts[1].strip()
                current_only = "--current" in query
                if current_only:
                    query = query.replace("--current", "").strip()
                _cmd_search(db, user, query, conv.id if current_only else None)
            else:
                console.print("[red]Usage:[/red] /search <query> [--current]")
            continue

        if raw.startswith("/workspace"):
            parts = raw.split(maxsplit=1)
            new_path = parts[1].strip() if len(parts) > 1 else None
            if new_path:
                abs_path = os.path.abspath(new_path)
                if not os.path.isdir(abs_path):
                    console.print(f"[red]Directory not found:[/red] {abs_path}")
                    continue
                workspace_path = abs_path
                ws = db.query(Workspace).filter(Workspace.user_id == user.id).first()
                if ws:
                    ws.disk_path = workspace_path
                    db.commit()
                agent = _build_agent(
                    db, user, conv.id, workspace_path, config, current_provider, current_model
                )
                console.print(f"[green]Workspace changed[/green] to {workspace_path}")
            else:
                console.print(f"  [dim]Workspace:[/dim] {workspace_path}")
            continue

        if raw.startswith("/load"):
            parts = raw.split()
            if len(parts) >= 2:
                try:
                    idx = int(parts[1])
                    target = _get_conv_by_index(db, user.id, idx)
                    if target:
                        conv = target
                        agent = _build_agent(
                            db, user, conv.id, workspace_path, config, current_provider, current_model
                        )
                        n = _count_messages(db, conv.id)
                        msg_count = n
                        toolbar_text["msgs"] = str(n)
                        console.print(f'[green]Loaded[/green] "{conv.title}" ({n} messages)')
                    else:
                        console.print(f"[red]Conversation #{idx} not found.[/red]")
                except ValueError:
                    console.print("[red]Usage:[/red] /load <number>")
            else:
                console.print("[red]Usage:[/red] /load <number> (from /list)")
            continue

        if raw.startswith("/rename") or raw.startswith("/save"):
            parts = raw.split(maxsplit=1)
            if len(parts) >= 2:
                new_title = parts[1].strip()
                conv.title = new_title[:255]
                db.commit()
                console.print(f'[green]Renamed[/green] to "{new_title}"')
            else:
                console.print("[red]Usage:[/red] /rename <title>")
            continue

        if raw.startswith("/export"):
            parts = raw.split(maxsplit=1)
            filepath = parts[1].strip() if len(parts) > 1 else None
            _cmd_export(db, user, conv.id, filepath)
            continue

        if raw == "/paste":
            console.print("[dim]Enter multi-line (Ctrl+D to send, Ctrl+C to cancel):[/dim]")
            lines = []
            try:
                while True:
                    line = console.input("[bold yellow]... [/bold yellow] ")
                    lines.append(line)
            except EOFError:
                console.print()
            except KeyboardInterrupt:
                console.print("\n[dim]Cancelled.[/dim]")
                continue
            multi_input = "\n".join(lines)
            if not multi_input.strip():
                continue
            _render_user_message(multi_input)
            _auto_title(db, conv.id, multi_input)
            _stream_agent_response(agent, multi_input)
            n = _count_messages(db, conv.id)
            toolbar_text["msgs"] = str(n)
            continue

        # --- Regular message ---
        _render_user_message(raw)
        _auto_title(db, conv.id, raw)
        _stream_agent_response(agent, raw)
        n = _count_messages(db, conv.id)
        toolbar_text["msgs"] = str(n)

    # Save state on exit
    config.last_conversation_id = str(conv.id)
    config.last_workspace = workspace_path
    config.save()
    db.close()
    console.print("[dim]Goodbye![/dim]")


def _cmd_graph(args) -> None:
    """``anzar graph`` — print the agent graphs as Mermaid (no DB required)."""
    from anzar.agent.core import create_agent_from_config

    try:
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env")
    except ImportError:
        pass

    config = AnzarConfig.load()
    workspace = os.path.abspath(args.workspace or config.last_workspace or os.getcwd())
    provider = args.provider or config.provider or "groq"
    model = args.model or config.model or None

    agent = create_agent_from_config(
        provider=provider, model=model, workspace_path=workspace
    )
    diagram = agent.mermaid_fast() if args.fast else agent.mermaid_all()
    print(diagram)


def _cmd_benchmark(args) -> None:
    """``anzar benchmark`` — run the agent latency benchmark via the script."""
    script = Path(__file__).resolve().parent.parent.parent / "scripts" / "benchmark_agent.py"
    cmd = [sys.executable, str(script), "--runs", str(args.runs)]
    if args.json:
        cmd.append("--json")
    import subprocess

    try:
        raise SystemExit(subprocess.call(cmd))
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted.[/yellow]")
        raise SystemExit(130)


def _cmd_checkpoint(known, db: Session, workspace_path: str) -> None:
    """``anzar checkpoint`` — create a recoverable workspace snapshot."""
    from anzar.checkpoint import create_checkpoint, format_task_history  # noqa: F401

    try:
        cp = create_checkpoint(workspace_path, db=db, description=known.desc)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Checkpoint failed: {e}[/red]")
        return
    mode = "git" if cp.is_git and cp.git_commit else "files"
    ref = cp.git_commit or cp.snapshot_path
    console.print(f"[bold]Workspace:[/bold] {workspace_path}")
    console.print(
        f"[green]Checkpoint {cp.id}[/green] created ({mode}, {cp.file_count} files)"
    )
    if ref:
        console.print(f"  ref: {ref}")
    if cp.description:
        console.print(f"  description: {cp.description}")
    console.print("\nUse [bold]anzar diff[/bold] to review or [bold]anzar rollback[/bold] to restore.")


def _cmd_diff(known, db: Session, workspace_path: str) -> None:
    """``anzar diff`` — show changes since a checkpoint."""
    from anzar.checkpoint import compute_diff, get_latest_checkpoint

    if known.id:
        from uuid import UUID as _UUID
        try:
            cp_id = _UUID(known.id)
        except ValueError:
            console.print("[red]Invalid checkpoint ID.[/red]")
            return
        cp = db.query(Checkpoint).filter(Checkpoint.id == cp_id).first()
    else:
        cp = get_latest_checkpoint(workspace_path, db)
    if cp is None:
        console.print("[yellow]No checkpoint found for this workspace.[/yellow]")
        return
    try:
        diff = compute_diff(cp.id, workspace_path, db)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Diff failed: {e}[/red]")
        return
    console.print(f"[bold]Workspace:[/bold] {workspace_path}")
    console.print(f"[bold]Checkpoint {cp.id}[/bold]  →  current workspace")
    console.print(f"  {diff.summary}")
    if diff.added:
        console.print("\n[green]Added:[/green]")
        for f in diff.added:
            console.print(f"  + {f}")
    if diff.modified:
        console.print("\n[yellow]Modified:[/yellow]")
        for f in diff.modified:
            console.print(f"  ~ {f}")
    if diff.deleted:
        console.print("\n[red]Deleted:[/red]")
        for f in diff.deleted:
            console.print(f"  - {f}")


def _cmd_rollback(known, db: Session, workspace_path: str) -> None:
    """``anzar rollback`` — restore the workspace to a checkpoint."""
    from anzar.checkpoint import get_latest_checkpoint, rollback_checkpoint

    if known.id:
        from uuid import UUID as _UUID
        try:
            cp_id = _UUID(known.id)
        except ValueError:
            console.print("[red]Invalid checkpoint ID.[/red]")
            return
        cp = db.query(Checkpoint).filter(Checkpoint.id == cp_id).first()
    else:
        cp = get_latest_checkpoint(workspace_path, db)
    if cp is None:
        console.print("[yellow]No checkpoint found for this workspace.[/yellow]")
        return
    from rich.prompt import Confirm
    console.print(f"[bold]Workspace:[/bold] {workspace_path}")
    if not known.yes and not Confirm.ask(
        f"Restore workspace to checkpoint {cp.id}? [y/N]", default=False
    ):
        console.print("Cancelled.")
        return
    try:
        restored = rollback_checkpoint(cp.id, workspace_path, db)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Rollback failed: {e}[/red]")
        return
    console.print(f"[green]Restored {len(restored)} files.[/green]")
    for f in restored[:20]:
        console.print(f"  ~ {f}")
    if len(restored) > 20:
        console.print(f"  ... and {len(restored) - 20} more")


def _cmd_history(known, db: Session, workspace_path: str) -> None:
    """``anzar history`` — show recent task history for the workspace."""
    from anzar.checkpoint import format_task_history, get_task_history

    tasks = get_task_history(workspace_path, db, limit=known.limit)
    console.print(f"[bold]Task history for {workspace_path}[/bold]")
    console.print(format_task_history(tasks))


def _resolve_subcommand_workspace(known, db: Session, config: "AnzarConfig") -> str:
    """Resolve the workspace for checkpoint/diff/rollback/history.

    Priority:
      1. An explicit ``-w/--workspace`` flag.
      2. The current working directory — but only if it already has
         checkpoints (i.e. anzar has worked there before), so ``anzar diff``
         in the folder you're working on shows the right result.
      3. ``config.last_workspace`` (anzar's previous session).
      4. ``os.getcwd()`` as a last resort.
    """
    from anzar.checkpoint import get_latest_checkpoint

    explicit = getattr(known, "workspace", None)
    if explicit:
        return os.path.abspath(explicit)

    cwd = os.path.abspath(os.getcwd())
    if get_latest_checkpoint(cwd, db) is not None:
        return cwd

    if config.last_workspace:
        return os.path.abspath(config.last_workspace)

    return cwd


def main():
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    _force_utf8_output()

    parser = argparse.ArgumentParser(
        description="Anzar — AI Software Engineer Agent",
        add_help=False,
    )
    parser.add_argument("--workspace", "-w", help="Workspace directory")
    parser.add_argument("--load", type=int, help="Load conversation number from /list")
    parser.add_argument(
        "--continue", dest="resume", action="store_true", help="Resume last conversation"
    )
    parser.add_argument(
        "--new", action="store_true", help="Start a fresh conversation (do not auto-resume)"
    )
    parser.add_argument("--list", action="store_true", help="List conversations and exit")
    parser.add_argument("--legacy", action="store_true", help="Use the legacy prompt_toolkit REPL instead of the TUI")
    parser.add_argument("--version", "-v", action="store_true", help="Show version")
    parser.add_argument("--help", "-h", action="store_true", help="Show help")
    sub = parser.add_subparsers(dest="subcommand")
    p_graph = sub.add_parser("graph", help="Print the agent graph as Mermaid", add_help=False)
    p_graph.add_argument("--fast", action="store_true", help="Show only the fast-path graph")
    p_graph.add_argument("--provider", help="Provider to build the graph for")
    p_graph.add_argument("--model", help="Model name")
    p_bench = sub.add_parser("benchmark", help="Run the agent latency benchmark", add_help=False)
    p_bench.add_argument("--runs", type=int, default=1, help="Repetitions per scenario")
    p_bench.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    p_serve = sub.add_parser("serve", help="Start the web server", add_help=False)
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p_cp = sub.add_parser("checkpoint", help="Create a workspace snapshot", add_help=False)
    p_cp.add_argument("--desc", default=None, help="Description of the checkpoint")
    p_cp.add_argument("--workspace", "-w", help="Workspace directory")
    p_diff = sub.add_parser("diff", help="Show changes since a checkpoint", add_help=False)
    p_diff.add_argument("id", nargs="?", default=None, help="Checkpoint ID (default: latest)")
    p_diff.add_argument("--workspace", "-w", help="Workspace directory")
    p_rb = sub.add_parser("rollback", help="Restore workspace to a checkpoint", add_help=False)
    p_rb.add_argument("id", nargs="?", default=None, help="Checkpoint ID (default: latest)")
    p_rb.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    p_rb.add_argument("--workspace", "-w", help="Workspace directory")
    p_hist = sub.add_parser("history", help="Show task history", add_help=False)
    p_hist.add_argument("-n", dest="limit", type=int, default=20, help="Number of entries (default: 20)")
    p_hist.add_argument("--workspace", "-w", help="Workspace directory")
    known, _ = parser.parse_known_args()

    if known.help:
        parser.print_help()
        console.print("\nRun [bold]anzar[/bold] to start the interactive REPL.")
        console.print("Inside the REPL, type [bold]/help[/bold] for commands.\n")
        console.print("Commands: [bold]anzar graph[/bold], [bold]anzar benchmark[/bold], [bold]anzar serve[/bold], [bold]anzar checkpoint[/bold], [bold]anzar diff[/bold], [bold]anzar rollback[/bold], and [bold]anzar history[/bold].")
        return

    if known.version:
        print(f"anzar-agent v{__version__}")
        return

    if known.subcommand == "graph":
        _cmd_graph(known)
        return

    if known.subcommand == "benchmark":
        _cmd_benchmark(known)
        return

    if known.subcommand == "serve":
        from anzar.server import run_server
        run_server(host=known.host, port=known.port)
        return

    init_db()
    db = SessionLocal()
    config = AnzarConfig.load()

    workspace_path = (
        getattr(known, "workspace", None)
        or config.last_workspace
        or os.getcwd()
    )
    workspace_path = os.path.abspath(workspace_path)

    if known.subcommand == "checkpoint":
        _cmd_checkpoint(known, db, _resolve_subcommand_workspace(known, db, config))
        db.close()
        return

    if known.subcommand == "diff":
        _cmd_diff(known, db, _resolve_subcommand_workspace(known, db, config))
        db.close()
        return

    if known.subcommand == "rollback":
        _cmd_rollback(known, db, _resolve_subcommand_workspace(known, db, config))
        db.close()
        return

    if known.subcommand == "history":
        _cmd_history(known, db, _resolve_subcommand_workspace(known, db, config))
        db.close()
        return

    user = _ensure_cli_user(db, workspace_path)

    if known.list:
        _cmd_list(db, user)
        db.close()
        return

    # Determine conversation
    conv, is_new = _select_conversation(db, user, known, config, workspace_path)
    if known.load and conv is None:
        console.print(f"[red]Conversation #{known.load} not found.[/red]")
        db.close()
        return

    agent = _build_agent(db, user, conv.id, workspace_path, config)
    msg_count = _count_messages(db, conv.id)

    # Pipe mode
    if not sys.stdin.isatty():
        _run_pipe_mode(agent)
        config.last_conversation_id = str(conv.id)
        config.last_workspace = workspace_path
        config.save()
        db.close()
        return

    # Interactive TUI (default) or legacy prompt_toolkit REPL (--legacy)
    if not known.legacy:
        from anzar.tui.app import AnzarTui

        try:
            AnzarTui(
                agent=agent,
                config=config,
                db=db,
                user=user,
                conv=conv,
                workspace_path=workspace_path,
                version=__version__,
            ).run()
        finally:
            # The TUI persists config + closes the DB on unmount (it may have
            # switched conversations); this guards against a startup failure.
            try:
                db.close()
            except Exception:
                pass
    else:
        _run_legacy_repl(db, config, user, conv, workspace_path, agent, msg_count, is_new)


if __name__ == "__main__":
    main()
