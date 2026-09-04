import os
from pathlib import Path

from anzar.agent.core import create_agent_from_config

STUDIO_WORKSPACE = Path(
    os.getenv("ANZAR_STUDIO_WORKSPACE", r"C:\Users\ghait\AppData\Local\Temp\opencode\anzar-studio")
)

_agent = create_agent_from_config(
    provider=os.getenv("ANZAR_STUDIO_PROVIDER", "groq"),
    model=os.getenv("ANZAR_STUDIO_MODEL", "openai/gpt-oss-120b"),
    api_key=os.getenv("GROQ_API_KEY"),
    workspace_path=str(STUDIO_WORKSPACE),
)

graph = _agent._build_graph(compile_graphs=False)[0]
