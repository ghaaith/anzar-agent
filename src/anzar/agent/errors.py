"""Structured error handling for the Anzar agent.

Errors are classified into stable categories so the graph can decide whether
to retry (transient provider faults) or fail fast (auth/config problems),
and so callers can present a consistent, actionable message to the user.
"""

from __future__ import annotations

from enum import Enum


class ErrorCategory(str, Enum):
    """Coarse buckets for agent/provider failures."""

    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    SERVER = "server"
    CONNECTION = "connection"
    MALFORMED_TOOL = "malformed_tool"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class AnzarConfigurationError(Exception):
    """Raised when the agent is misconfigured (e.g. missing API key)."""


# Categories we will retry with backoff before giving up.
RETRYABLE_CATEGORIES = {
    ErrorCategory.RATE_LIMIT,
    ErrorCategory.SERVER,
    ErrorCategory.CONNECTION,
    ErrorCategory.TIMEOUT,
}

_MAX_BACKOFF_S = 2.0


def backoff_seconds(attempt: int) -> float:
    """Exponential backoff for retry ``attempt`` (0-indexed): 0.5, 1.0, 2.0…"""
    return min(_MAX_BACKOFF_S, 0.5 * (2 ** max(attempt, 0)))


def classify_error(exc: Exception) -> ErrorCategory:
    """Classify an arbitrary exception into an :class:`ErrorCategory`."""
    status = getattr(exc, "status_code", None)

    if isinstance(status, int):
        if status in (401, 403):
            return ErrorCategory.AUTH
        if status == 404:
            return ErrorCategory.CONFIGURATION
        if status == 429:
            return ErrorCategory.RATE_LIMIT
        if status in (408, 524):
            return ErrorCategory.TIMEOUT
        if status >= 500:
            return ErrorCategory.SERVER
        return ErrorCategory.UNKNOWN

    name = type(exc).__name__
    if "tool_use_failed" in str(exc) or "failed_generation" in str(exc):
        return ErrorCategory.MALFORMED_TOOL
    if "parse tool call" in str(exc).lower():
        return ErrorCategory.MALFORMED_TOOL
    if "RateLimit" in name or "429" in str(exc):
        return ErrorCategory.RATE_LIMIT
    if "Timeout" in name or "timeout" in name.lower():
        return ErrorCategory.TIMEOUT
    if "Connection" in name or "ConnectionError" in type(exc).__module__:
        return ErrorCategory.CONNECTION

    try:
        import httpx
    except ImportError:  # pragma: no cover
        httpx = None

    if httpx is not None and isinstance(exc, httpx.HTTPError):
        if isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout)):
            return ErrorCategory.CONNECTION
        return ErrorCategory.SERVER

    return ErrorCategory.UNKNOWN


def is_retryable(exc: Exception) -> bool:
    """True if the failure is transient and worth retrying."""
    return classify_error(exc) in RETRYABLE_CATEGORIES


def intervention_message(*, what: str, why: str, where: str, example_env: str) -> str:
    """Build a message in the standard 'I need your intervention.' format."""
    return (
        "I need your intervention.\n"
        f"What: {what}\n"
        f"Why: {why}\n"
        f"Where to configure: {where}\n"
        f"Example env var: {example_env}"
    )
