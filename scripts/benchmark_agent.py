"""Benchmark AnzarAgent latency across representative scenarios.

Measures wall-clock latency, LLM call count, tool call count, and response
length for four scenarios:

  1. simple  : plain chat (should hit the fast path after the upgrade)
  2. tool    : single tool use (list_files)
  3. multi   : multi-tool / parallel tool use (list_files + get_file_info)
  4. choice  : agent needs user input (ask_user / human-in-the-loop)

Usage:
    python scripts/benchmark_agent.py            # uses ANZAR_PROVIDER / .env
    python scripts/benchmark_agent.py --runs 3
    python scripts/benchmark_agent.py --json     # machine-readable output

The script requires a real API key (default: GROQ_API_KEY in .env). If none
is found it exits with an intervention message.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys
import tempfile
import time

from dotenv import load_dotenv

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

SCENARIOS = {
    "simple": "Hello! In one short sentence, how does a software engineer agent work?",
    "tool": "Use a tool to list the files in this workspace.",
    "multi": (
        "Use tools to list the files in this workspace and get metadata for the "
        "README.md file. Report what you find."
    ),
    "choice": (
        "I'm building a small CLI utility. There are several good directions: "
        "A) a file organizer, B) a markdown linter, or C) a git helper. "
        "Use ask_user to let me pick which one you should build, then continue."
    ),
}


def _load_provider() -> tuple[str, str, str]:
    """Return (provider, model, api_key) from env/.env, else raise."""
    provider = os.environ.get("ANZAR_PROVIDER", "groq")
    api_key = os.environ.get("ANZAR_API_KEY")
    if not api_key:
        key_map = {
            "openrouter": "OPENROUTER_API_KEY",
            "groq": "GROQ_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        env_var = key_map.get(provider)
        api_key = os.environ.get(env_var) if env_var else None
    model = os.environ.get("ANZAR_MODEL")
    if not api_key:
        print(
            "I need your intervention.\n"
            f"Required: a real API key to benchmark against provider '{provider}'.\n"
            f"Where to configure: a .env file in the project root or your environment.\n"
            f"Example env var: {key_map.get(provider, 'ANZAR_API_KEY')}=sk-...\n"
            "Add it, then rerun the benchmark."
        )
        sys.exit(2)
    return provider, model, api_key


def _make_workspace() -> str:
    """Create a throwaway workspace with a couple of small files."""
    ws = tempfile.mkdtemp(prefix="anzar-bench-")
    (pathlib.Path(ws) / "README.md").write_text(
        "# Benchmark workspace\n\nUsed to measure Anzar agent latency.\n", encoding="utf-8"
    )
    (pathlib.Path(ws) / "app.py").write_text(
        "def main():\n    print('hello')\n\nif __name__ == '__main__':\n    main()\n",
        encoding="utf-8",
    )
    (pathlib.Path(ws) / "utils.py").write_text(
        "import os\n\nTOTAL = 0\n\n", encoding="utf-8"
    )
    return ws


def _count_tool_calls(agent) -> int:
    metrics = getattr(agent, "last_metrics", None)
    if metrics is not None:
        return getattr(metrics, "tool_calls", 0)
    return -1


def _count_llm_calls(agent) -> int:
    metrics = getattr(agent, "last_metrics", None)
    if metrics is not None:
        return getattr(metrics, "llm_calls", 0)
    return -1


def _run_scenario(agent, name: str, prompt: str) -> dict:
    start = time.perf_counter()
    try:
        text = agent.run(prompt)
        elapsed = time.perf_counter() - start
        return {
            "scenario": name,
            "elapsed_s": round(elapsed, 3),
            "response_chars": len(text),
            "llm_calls": _count_llm_calls(agent),
            "tool_calls": _count_tool_calls(agent),
            "path": getattr(agent, "last_path", None),
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "scenario": name,
            "elapsed_s": round(elapsed, 3),
            "response_chars": 0,
            "llm_calls": _count_llm_calls(agent),
            "tool_calls": _count_tool_calls(agent),
            "path": "error",
            "error": f"{type(e).__name__}: {e}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1, help="Repetitions per scenario")
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    args = parser.parse_args()

    provider, model, api_key = _load_provider()

    from anzar.agent.core import create_agent_from_config

    ws = _make_workspace()
    try:
        agent = create_agent_from_config(
            provider=provider, model=model, api_key=api_key, workspace_path=ws
        )

        rows: list[dict] = []
        for name, prompt in SCENARIOS.items():
            for i in range(max(1, args.runs)):
                row = _run_scenario(agent, name, prompt)
                if args.runs > 1:
                    row["run"] = i + 1
                rows.append(row)

        if args.json:
            print(json.dumps({"provider": provider, "model": model, "results": rows}, indent=2))
            return

        print(f"\nBenchmark: provider={provider} model={model or 'default'} runs={args.runs}\n")
        print(f"{'scenario':<8} {'run':<4} {'elapsed_s':<10} {'llm_calls':<10} {'tool_calls':<10} {'chars':<8} path")
        print("-" * 70)
        for r in rows:
            run_col = r.get("run", "-")
            err = r.get("error")
            print(
                f"{r['scenario']:<8} {run_col:<4} {r['elapsed_s']:<10.3f} "
                f"{r['llm_calls']:<10} {r['tool_calls']:<10} {r['response_chars']:<8} {r.get('path') or '-'}"
                + (f"  [ERROR] {err}" if err else "")
            )
    finally:
        shutil.rmtree(ws, ignore_errors=True)


if __name__ == "__main__":
    main()
