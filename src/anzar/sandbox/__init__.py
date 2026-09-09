"""Docker sandbox manager for Anzar workspaces."""

from anzar.sandbox.limits import PLAN_LIMITS
from anzar.sandbox.manager import DockerSandboxManager, get_manager

__all__ = ["DockerSandboxManager", "get_manager", "PLAN_LIMITS"]
