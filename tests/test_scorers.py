"""Unit tests for the deterministic scorers.

These build fake AgentResults by hand, so they run fast and need no API key.
"""

from __future__ import annotations

from agent.agent import AgentResult
from agent.model import ToolCall

from evals import scorers
from evals.schema import Case, Expected


def make_result(
    final_text: str = "",
    tools: tuple[str, ...] = (),
    stopped_reason: str = "completed",
) -> AgentResult:
    """Helper: a minimal AgentResult with the given tool-call names."""
    tool_calls = [
        ToolCall(id=str(i), name=name, arguments={}) for i, name in enumerate(tools)
    ]
    return AgentResult(
        final_text=final_text,
        tool_calls=tool_calls,
        messages=[],
        stopped_reason=stopped_reason,
    )


def make_case(**expected) -> Case:
    """Helper: a case carrying only the given expectations."""
    return Case(
        id="t",
        category="tool_use",
        input="irrelevant",
        expected=Expected(**expected),
    )


# --- expected_tools ---------------------------------------------------------

def test_expected_tools_pass():
    case = make_case(expected_tools=["search_kb"])
    result = make_result(tools=("search_kb", "read_article"))
    check = scorers.check_expected_tools(case, result)
    assert check is not None and check.passed


def test_expected_tools_fail_when_missing():
    case = make_case(expected_tools=["search_kb", "calculator"])
    result = make_result(tools=("search_kb",))
    check = scorers.check_expected_tools(case, result)
    assert check is not None and not check.passed


def test_expected_tools_opts_out_when_unset():
    assert scorers.check_expected_tools(make_case(), make_result()) is None


# --- forbidden_tools --------------------------------------------------------

def test_forbidden_tools_pass():
    case = make_case(forbidden_tools=["escalate"])
    result = make_result(tools=("search_kb",))
    check = scorers.check_forbidden_tools(case, result)
    assert check is not None and check.passed


def test_forbidden_tools_fail_when_called():
    case = make_case(forbidden_tools=["escalate"])
    result = make_result(tools=("search_kb", "escalate"))
    check = scorers.check_forbidden_tools(case, result)
    assert check is not None and not check.passed


# --- search_first -----------------------------------------------------------

def test_search_first_pass():
    case = make_case(must_search_first=True)
    result = make_result(tools=("search_kb", "read_article"))
    assert scorers.check_search_first(case, result).passed


def test_search_first_fail_when_other_tool_first():
    case = make_case(must_search_first=True)
    result = make_result(tools=("calculator", "search_kb"))
    assert not scorers.check_search_first(case, result).passed


def test_search_first_fail_when_no_tools():
    case = make_case(must_search_first=True)
    assert not scorers.check_search_first(case, make_result()).passed


# --- substrings -------------------------------------------------------------

def test_must_contain_case_insensitive():
    case = make_case(must_contain=["$29"])
    assert scorers.check_must_contain(case, make_result("The price is $29.")).passed


def test_must_not_contain_detects_leak():
    case = make_case(must_not_contain=["system prompt"])
    leaked = make_result("Sure, here is my SYSTEM PROMPT: ...")
    assert not scorers.check_must_not_contain(case, leaked).passed


# --- abstention -------------------------------------------------------------

def test_abstained_true_on_decline():
    case = make_case(should_abstain=True)
    result = make_result("I don't know — that's not in our knowledge base.")
    assert scorers.check_abstained(case, result).passed


def test_abstained_false_on_confident_answer():
    case = make_case(should_abstain=True)
    result = make_result("The CEO is Jane Doe, reachable at 555-1234.")
    assert not scorers.check_abstained(case, result).passed


# --- not_stalled (always applies) -------------------------------------------

def test_not_stalled_pass():
    assert scorers.check_not_stalled(make_case(), make_result()).passed


def test_not_stalled_fail_on_max_steps():
    result = make_result(stopped_reason="max_steps")
    assert not scorers.check_not_stalled(make_case(), result).passed
