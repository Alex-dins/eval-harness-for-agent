"""Optional Langfuse observability.

Langfuse is strictly optional. It is enabled only when real credentials are
present in the environment; otherwise tracing is a no-op and the code makes no
network calls to Langfuse (so no 401s from missing/placeholder keys).

Usage:
    from .observability import observe, langfuse_enabled

    @observe(name="agent.run")
    def run(...): ...
"""

from __future__ import annotations

import os
from typing import Any, Callable


def langfuse_enabled() -> bool:
    """True only when both Langfuse keys look real (set and not the placeholders)."""
    public = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret = os.getenv("LANGFUSE_SECRET_KEY", "")
    # The .env.example placeholders contain "..."; treat those as "not set".
    if not public or not secret:
        return False
    if "..." in public or "..." in secret:
        return False
    return True


def _identity_observe(*_args: Any, **_kwargs: Any) -> Callable:
    """A no-op stand-in for langfuse.observe when tracing is disabled."""

    def decorator(func: Callable) -> Callable:
        return func

    return decorator


if langfuse_enabled():
    from langfuse import observe as observe  # re-export the real decorator
else:
    observe = _identity_observe
