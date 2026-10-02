"""LLM-as-judge scorer for open-ended answer correctness.

Substring checks can't tell whether a free-form answer is *correct* — only whether
it contains some text. The judge asks a model to compare the agent's answer to a
gold answer and return a structured verdict.

Design:
- Structured JSON output (`{"verdict": "pass"|"fail", "reason": ...}`) so parsing
  is robust.
- temperature=0 for reproducibility.
- The prompt tells the judge to grade *meaning*, not wording.

The public entry point `make_correctness_judge()` returns a callable shaped like a
scorer — `(case, result) -> CheckResult | None` — so the runner treats it like any
other check. It opts out (returns None) for cases without a `gold_answer`.
"""

from __future__ import annotations

import json
import os

from agent.agent import AgentResult

from .schema import Case, CheckResult

_JUDGE_SYSTEM = """\
You are a strict but fair grader for a customer-support assistant. You compare the \
assistant's ANSWER to a reference GOLD answer for a QUESTION.

Grade on semantic correctness, not wording: the answer passes if it conveys the \
key facts of the gold answer and contains no contradicting or fabricated facts. \
Extra correct detail is fine. Missing a key fact, or stating a wrong fact, fails.

Respond ONLY with JSON of the form:
{"verdict": "pass" | "fail", "reason": "<one short sentence>"}\
"""


def judge_correctness(
    question: str,
    gold_answer: str,
    agent_answer: str,
    model: str | None = None,
) -> CheckResult:
    """Ask the judge model whether ``agent_answer`` is correct vs. ``gold_answer``."""
    from openai import OpenAI

    client = OpenAI()
    model = model or os.getenv("JUDGE_MODEL", "gpt-4o-mini")

    user = (
        f"QUESTION:\n{question}\n\n"
        f"GOLD answer:\n{gold_answer}\n\n"
        f"ASSISTANT answer:\n{agent_answer}"
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
    )

    content = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(content)
        verdict = str(parsed.get("verdict", "")).lower()
        reason = str(parsed.get("reason", "")).strip()
    except json.JSONDecodeError:
        verdict, reason = "fail", f"could not parse judge output: {content!r}"

    return CheckResult(
        name="correctness",
        passed=verdict == "pass",
        detail=reason or f"judge verdict: {verdict!r}",
    )


def make_correctness_judge(model: str | None = None):
    """Return a scorer-shaped judge: (case, result) -> CheckResult | None.

    Opts out for cases without a gold_answer so it can be passed to the runner for
    every case harmlessly.
    """

    def judge(case: Case, result: AgentResult) -> CheckResult | None:
        if not case.expected.gold_answer:
            return None
        return judge_correctness(
            question=case.input,
            gold_answer=case.expected.gold_answer,
            agent_answer=result.final_text,
            model=model,
        )

    return judge
