"""Resource limits per plan tier."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceLimits:
    memory: str        # Docker memory limit (e.g., "256m")
    cpu: float         # CPU quota (e.g., 0.5 = half a core)
    disk_mb: int       # Disk limit in MB
    idle_timeout: int  # Seconds before auto-stop


PLAN_LIMITS: dict[str, ResourceLimits] = {
    "free": ResourceLimits(
        memory="256m",
        cpu=0.5,
        disk_mb=500,
        idle_timeout=30 * 60,       # 30 minutes
    ),
    "pro": ResourceLimits(
        memory="1g",
        cpu=1.0,
        disk_mb=2048,
        idle_timeout=2 * 60 * 60,   # 2 hours
    ),
    "team": ResourceLimits(
        memory="2g",
        cpu=2.0,
        disk_mb=5120,
        idle_timeout=6 * 60 * 60,   # 6 hours
    ),
}


def get_limits(plan: str) -> ResourceLimits:
    """Get resource limits for a plan, falling back to free tier."""
    return PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])
