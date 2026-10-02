"""Data models for eval cases and results.

A *case* is one test: an input to the agent plus expectations.
A *result* is what the harness produces after running the agent and scoring it.

Everything here is a pydantic model: a class that declares its fields as type
hints and validates data against them at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

# A Literal is a closed set of allowed string values. Pydantic will reject any
# case whose category isn't exactly one of these four.
Category = Literal["tool_use", "correctness", "hallucination", "injection"]


class Expected(BaseModel):
    """What a case expects. Every field is optional with a default, so a case
    only declares the checks it cares about — unset fields are simply not scored.
    """

    # Correctness (judged by the LLM against the final answer).
    gold_answer: str | None = None

    # Tool-use assertions (checked deterministically against the trajectory).
    expected_tools: list[str] | None = None  # each of these must appear ≥ once
    forbidden_tools: list[str] | None = None  # none of these may appear
    must_search_first: bool = False           # first tool call must be search_kb

    # Grounding / hallucination.
    should_abstain: bool = False              # answer should decline / say "don't know"

    # Substring assertions on the final answer text.
    must_contain: list[str] | None = None
    must_not_contain: list[str] | None = None


class Case(BaseModel):
    """One eval test."""

    id: str
    category: Category
    input: str
    # default_factory builds a fresh Expected() per Case. You must NOT write
    # `expected: Expected = Expected()` — that shares one mutable instance across
    # every Case (the classic Python mutable-default bug).
    expected: Expected = Field(default_factory=Expected)
    notes: str | None = None


class CheckResult(BaseModel):
    """The outcome of a single scorer."""

    name: str
    passed: bool
    detail: str = ""


class CaseResult(BaseModel):
    """The outcome of running + scoring one case."""

    case_id: str
    category: Category
    passed: bool
    checks: list[CheckResult]
    final_text: str
    tool_calls: list[dict[str, Any]]  # [{"name": ..., "arguments": {...}}]
    stopped_reason: str


def load_cases(path: Path | str) -> list[Case]:
    """Load cases from a JSON file or a directory of JSON files.

    Each file may hold a single case object or a list of them.
    """
    path = Path(path)
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]

    cases: list[Case] = []
    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]
        # model_validate turns a plain dict (from JSON) into a validated Case.
        # If the dict is malformed, pydantic raises a ValidationError here.
        cases.extend(Case.model_validate(item) for item in items)
    return cases
