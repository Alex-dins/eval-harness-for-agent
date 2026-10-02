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


from contextlib import contextmanager


@contextmanager
def trace_case(name: str, user_input: str):
    """Open a Langfuse trace for one eval case (named by case id).

    The agent run executes inside this context, so `agent.run` and its model/tool
    spans nest under one trace per case. Yields the span (or None when disabled).
    """
    if not langfuse_enabled():
        yield None
        return
    from langfuse import get_client

    with get_client().start_as_current_observation(name=name) as span:
        try:
            span.set_trace_io(input=user_input)
        except Exception:
            pass
        yield span


def mark_case(span, passed: bool, final_text: str, failed_checks: list[str]) -> None:
    """Record the case outcome on its trace so it's filterable in Langfuse.

    Sets the observation level to ERROR on failure (shown as the trace Status) and
    attaches a numeric `passed` score. No-op when tracing is disabled.
    """
    if span is None:
        return
    try:
        span.update(
            level="DEFAULT" if passed else "ERROR",
            status_message=(
                "passed" if passed else "failed: " + ", ".join(failed_checks)
            ),
            output=final_text,
        )
        span.set_trace_io(output=final_text)
        span.score_trace(
            name="passed",
            value=1 if passed else 0,
            comment=None if passed else ", ".join(failed_checks),
        )
    except Exception:
        # Observability must never break scoring.
        pass


def flush() -> None:
    """Flush pending traces to Langfuse. No-op when tracing is disabled.

    Short-lived CLI processes can exit before the background exporter sends its
    batch, so call this before the program ends to guarantee traces are shipped.
    """
    if not langfuse_enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        # Observability must never break the actual run.
        pass
