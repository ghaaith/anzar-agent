# Anzar — AI Software Engineer Agent

Your AI-Powered Code Companion. Local-first, works with your real projects.

## Install

```bash
pip install anzar-agent
```

## Usage

```bash
# Terminal mode
anzar

# Web UI mode
anzar serve
```

## Configuration

Create `~/.anzar/config.yaml`:

```yaml
provider: gemini
api_key: your-api-key-here
model: gemini-2.5-pro
```

Or use environment variables:

```bash
export ANZAR_PROVIDER=gemini
export ANZAR_API_KEY=your-api-key
```
