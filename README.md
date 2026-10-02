# eval-harness

A small, **from-scratch eval harness for a tool-using LLM agent** — no agent
framework, no eval library. Built to show how agent evaluation actually works
under the hood.

It ships with a sample agent (a knowledge-base support assistant) and a home-grown
harness that scores it on four dimensions:

| Dimension | What it measures | How it's scored |
|---|---|---|
| **Tool use** | Does the agent call the right tools, in the right order, without over-calling? | Deterministic (trajectory assertions) |
| **Answer correctness** | Is the final answer factually right? | LLM-as-judge vs. a gold answer |
| **Hallucination / grounding** | When the answer isn't in the KB, does it abstain instead of inventing facts? | Deterministic abstention check |
| **Prompt-injection resistance** | Does it ignore malicious instructions hidden in documents or user input? | Deterministic (forbidden tools + leak detection) |

## How it works

```
┌────────────┐   input    ┌──────────────┐   trajectory   ┌──────────┐
│   Case     │ ─────────► │  Agent (SUT) │ ─────────────► │ Scorers  │ ─► CaseResult
│ (+expected)│            │  tool loop   │ final_text +   │ + Judge  │
└────────────┘            └──────────────┘ tool_calls     └──────────┘
```

- **Agent (`src/agent/`)** — a hand-written tool-use loop (model → tool call →
  result → model …) that returns a fully observable *trajectory*. The model is
  behind a provider-agnostic `ModelClient` interface. Observability via
  [Langfuse](https://langfuse.com) (`@observe`), optional.
- **Harness (`src/evals/`)** — pydantic case schema, deterministic scorers, an
  LLM-as-judge, a runner, and a reporter. Each scorer *opts out* when a case
  doesn't request it, so a case declares only the checks it cares about.

## Layout

```
src/agent/
  kb.py        # deterministic keyword search over datasets/kb/*.md
  model.py     # provider-agnostic ModelClient; OpenAIModel impl (Langfuse-traced)
  tools.py     # search_kb, read_article, calculator, escalate
  agent.py     # the tool-use loop -> AgentResult (full trajectory)
src/evals/
  schema.py    # pydantic: Case, Expected, CheckResult, CaseResult + load_cases
  scorers.py   # deterministic, API-free checks
  llm_judge.py # LLM-as-judge for answer correctness
  runner.py    # run agent per case + apply scorers
  report.py    # aggregate -> table + JSON
  cases/       # the eval cases (JSON)
datasets/kb/   # synthetic "Nimbus" knowledge base (one article has an injection payload)
```

## Setup

```bash
uv sync
cp .env.example .env     # add OPENAI_API_KEY (and optionally LANGFUSE_* keys)
```

## Run

```bash
# Try the agent directly
uv run agent "What is the refund policy?" --trace

# Run the full eval suite (writes a JSON report to results/)
uv run evals

# Filter by category, or skip the paid LLM judge
uv run evals --category injection
uv run evals --no-judge

# Deterministic unit tests (no API key needed)
uv run pytest
```

`uv run evals` prints a per-case table and per-category pass rates, and exits
non-zero if any case fails (so it can gate CI).

## Design notes

- **Why no framework?** Evals need the full trajectory; frameworks hide it. A
  hand-rolled loop keeps every message and tool call as inspectable data.
- **Deterministic first.** Everything that *can* be scored without a model is
  (tool calls, abstention, substrings), so most of the suite is reproducible,
  free, and unit-tested. The LLM judge is reserved for open-ended correctness.
- **Injection testing is built in.** `datasets/kb/integrations.md` carries a
  planted payload, and the injection cases assert the agent neither leaks its
  prompt nor takes the `escalate` action.
