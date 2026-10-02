"""The runner: execute the agent on each case and apply the scorers.

Flow for one case:
    agent.run(case.input) -> AgentResult
    apply every deterministic scorer (each opts out if not relevant)
    optionally apply the LLM judge for correctness (when a judge is provided)
    a case passes only if every applied check passed.
"""

from __future__ import annotations

from typing import Callable

from agent.agent import Agent, AgentResult

from . import scorers
from .schema import Case, CaseResult, CheckResult

# The deterministic scorers that opt out via `None`. `check_not_stalled` always
# applies and is added separately.
_OPT_IN_SCORERS: list[Callable[[Case, AgentResult], CheckResult | None]] = [
    scorers.check_expected_tools,
    scorers.check_forbidden_tools,
    scorers.check_search_first,
    scorers.check_must_contain,
    scorers.check_must_not_contain,
    scorers.check_abstained,
]

# A judge is any callable (case, result) -> CheckResult | None. We keep it as a
# plug-in so the runner has no hard dependency on the LLM judge (EV-3) and stays
# fully testable offline.
Judge = Callable[[Case, AgentResult], CheckResult | None]


def _tool_calls_as_dicts(result: AgentResult) -> list[dict]:
    return [{"name": tc.name, "arguments": tc.arguments} for tc in result.tool_calls]


def run_case(case: Case, agent: Agent, judge: Judge | None = None) -> CaseResult:
    """Run one case and score it. Never raises — agent errors become a failed check."""
    try:
        result = agent.run(case.input)
    except Exception as exc:
        return CaseResult(
            case_id=case.id,
            category=case.category,
            passed=False,
            checks=[CheckResult(name="agent_error", passed=False, detail=repr(exc))],
            final_text="",
            tool_calls=[],
            stopped_reason="error",
        )

    checks: list[CheckResult] = []
    for scorer in _OPT_IN_SCORERS:
        check = scorer(case, result)
        if check is not None:
            checks.append(check)

    if judge is not None:
        judged = judge(case, result)
        if judged is not None:
            checks.append(judged)

    checks.append(scorers.check_not_stalled(case, result))

    return CaseResult(
        case_id=case.id,
        category=case.category,
        passed=all(c.passed for c in checks),
        checks=checks,
        final_text=result.final_text,
        tool_calls=_tool_calls_as_dicts(result),
        stopped_reason=result.stopped_reason,
    )


def run_cases(
    cases: list[Case], agent: Agent, judge: Judge | None = None
) -> list[CaseResult]:
    """Run and score a list of cases, in order."""
    return [run_case(case, agent, judge=judge) for case in cases]
