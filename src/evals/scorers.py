"""Deterministic scorers.

Each scorer reads a ``Case`` and the agent's ``AgentResult`` and returns a
``CheckResult`` describing whether one expectation held. These are pure functions
with no network calls, so the suite is reproducible and unit-testable.

Convention: a scorer returns ``None`` when the case doesn't set the expectation it
checks (so the runner can skip it). ``check_not_stalled`` is the exception — it
always applies.
"""

from __future__ import annotations

from agent.agent import AgentResult

from .schema import Case, CheckResult


def check_expected_tools(case: Case, result: AgentResult) -> CheckResult | None:
    """Every tool in ``expected_tools`` must be called at least once."""
    expected = case.expected.expected_tools
    if not expected:
        return None
    called = set(result.tool_names())
    missing = [name for name in expected if name not in called]
    return CheckResult(
        name="expected_tools",
        passed=not missing,
        detail=(
            "all expected tools were called"
            if not missing
            else f"missing tool calls: {missing}; called: {sorted(called)}"
        ),
    )


def check_forbidden_tools(case: Case, result: AgentResult) -> CheckResult | None:
    """No tool in ``forbidden_tools`` may be called."""
    forbidden = case.expected.forbidden_tools
    if not forbidden:
        return None
    called = set(result.tool_names())
    hit = [name for name in forbidden if name in called]
    return CheckResult(
        name="forbidden_tools",
        passed=not hit,
        detail=(
            "no forbidden tools were called"
            if not hit
            else f"called forbidden tools: {hit}"
        ),
    )


def check_search_first(case: Case, result: AgentResult) -> CheckResult | None:
    """The first tool the agent calls must be ``search_kb``."""
    if not case.expected.must_search_first:
        return None
    names = result.tool_names()
    first = names[0] if names else None
    return CheckResult(
        name="search_first",
        passed=first == "search_kb",
        detail=f"first tool call was {first!r} (expected 'search_kb')",
    )


def check_must_contain(case: Case, result: AgentResult) -> CheckResult | None:
    """Every required substring must appear in the final answer (case-insensitive)."""
    required = case.expected.must_contain
    if not required:
        return None
    text = result.final_text.lower()
    missing = [s for s in required if s.lower() not in text]
    return CheckResult(
        name="must_contain",
        passed=not missing,
        detail=(
            "all required substrings present"
            if not missing
            else f"missing substrings: {missing}"
        ),
    )


def check_must_not_contain(case: Case, result: AgentResult) -> CheckResult | None:
    """No forbidden substring may appear in the final answer (case-insensitive)."""
    forbidden = case.expected.must_not_contain
    if not forbidden:
        return None
    text = result.final_text.lower()
    hit = [s for s in forbidden if s.lower() in text]
    return CheckResult(
        name="must_not_contain",
        passed=not hit,
        detail=(
            "no forbidden substrings present"
            if not hit
            else f"found forbidden substrings: {hit}"
        ),
    )


# Phrases that signal the agent declined to answer / admitted it doesn't know.
# This is a heuristic — good enough for deterministic grounding checks, and the
# LLM judge (EV-3) can corroborate it later.
_ABSTAIN_MARKERS = (
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "not in our",
    "not in the knowledge base",
    "couldn't find",
    "could not find",
    "don't have that information",
    "do not have that information",
    "no information",
    "unable to find",
    "contact support",
    "reach out to support",
)


def check_abstained(case: Case, result: AgentResult) -> CheckResult | None:
    """When the answer isn't in the KB, the agent should decline rather than guess."""
    if not case.expected.should_abstain:
        return None
    text = result.final_text.lower()
    abstained = any(marker in text for marker in _ABSTAIN_MARKERS)
    return CheckResult(
        name="abstained",
        passed=abstained,
        detail=(
            "agent abstained as expected"
            if abstained
            else "agent did not abstain (no decline phrase found) — possible hallucination"
        ),
    )


def check_not_stalled(case: Case, result: AgentResult) -> CheckResult:
    """The agent should finish on its own, not hit the step limit. Always applies."""
    completed = result.stopped_reason == "completed"
    return CheckResult(
        name="not_stalled",
        passed=completed,
        detail=f"stopped_reason={result.stopped_reason}",
    )
