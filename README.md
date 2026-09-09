# Anzar — AI Software Engineer Agent

Local-first coding agent for your terminal. Install with one command, then run it
in any project — ask for a change, review diffs, and roll back. No login, no
server required.

## Install

One command, on any platform (bootstraps Python/pipx if needed):

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/ghaaith/anzar-agent/main/scripts/install.sh | sh

# Windows (PowerShell)
irm https://raw.githubusercontent.com/ghaaith/anzar-agent/main/scripts/install.ps1 | iex
```

Prefer plain pip?

```bash
pipx install anzar-agent
# or
pip install anzar-agent
```

Requires Python 3.10+.

## Quickstart

```bash
# start the interactive terminal UI
anzar

# or run a single task and exit
anzar "explain this codebase"
anzar "add tests for the checkout module"
```

On first run, anzar asks which AI provider to use and where to paste your API
key. You can also configure it manually (see below).

Create a free API key with any provider to get started:

| provider      | key                                            | free tier |
| ------------- | ---------------------------------------------- | --------- |
| openrouter    | https://openrouter.ai/keys                     | yes       |
| groq          | https://console.groq.com/keys                  | yes       |
| gemini        | https://aistudio.google.com/apikey             | yes       |
| ollama        | local, no key needed                           | yes       |
| openai / anthropic | provider console                          | no        |

## Everyday commands

| command                | what it does                                  |
| ---------------------- | --------------------------------------------- |
| `anzar`                | open the interactive terminal UI              |
| `anzar "task"`         | run one task and exit                         |
| `anzar checkpoint`     | snapshot the current workspace                |
| `anzar diff`           | show changes since the last checkpoint        |
| `anzar diff <id>`      | review changes versus a specific checkpoint   |
| `anzar diff A..B`      | diff between two checkpoints                  |
| `anzar rollback`       | restore the workspace to the last checkpoint  |
| `anzar rollback <id>`  | restore to a specific checkpoint              |
| `anzar history`        | recent tasks for this workspace               |
| `anzar doctor`         | show environment/configuration diagnostics    |

Checkpoints are per-workspace. Snapshot with `anzar checkpoint` before big
changes, then diff or roll back whenever you need to.

## Configuration

The first-run wizard writes your key into local settings; the rest lives in
`~/.anzar/config.yaml`:

```yaml
provider: groq
model: openai/gpt-oss-120b
```

Environment variables also work and override on launch:

```bash
export ANZAR_PROVIDER=groq
export ANZAR_API_KEY=your-key          # overrides provider-specific vars
export GROQ_API_KEY=your-key
```

Provider env vars: `OPENROUTER_API_KEY`, `GROQ_API_KEY`, `GOOGLE_API_KEY`,
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`.

Inside the UI, `/provider`, `/model`, `/apikey`, and `/help` adjust the session
without touching files.

## How it works

- Read-only gate: every shell command is checked against a command policy;
  destructive commands are blocked or require your approval before running
- Workspace checkpoints let you diff and roll back anything the agent changed
- Everything runs locally against your own project files — no uploads

## Development

```bash
git clone https://github.com/ghaaith/anzar-agent.git
cd anzar-agent
pip install -e ".[dev]"
python -m pytest -q      # full suite, no API key required
ruff check src tests     # lint
```

## License

MIT