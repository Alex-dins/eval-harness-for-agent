# CLAUDE.md

Guidance for working in this repository.

## What this is

An **eval harness for a tool-using agent**, built from scratch (no agent
framework) for learning and as a portfolio piece. Two halves:

- **`src/agent/`** — the system under test: a KB support assistant ("Nimbus
  Support") that answers questions grounded in a local knowledge base using a
  hand-written tool-use loop.
- **`src/evals/`** — a home-grown eval harness that scores the agent on tool use,
  answer correctness, hallucination/grounding, and prompt-injection resistance.

## Layout

```
src/agent/
  kb.py        # deterministic keyword search over datasets/kb/*.md
  model.py     # provider-agnostic ModelClient; OpenAIModel impl (traced via Langfuse)
  tools.py     # search_kb, read_article, calculator, escalate
  agent.py     # the tool-use loop; returns an AgentResult (full trajectory)
  __main__.py  # `uv run agent "..." [--trace]`
src/evals/
  schema.py    # pydantic models: Case, Expected, CheckResult, CaseResult + load_cases
  scorers.py   # deterministic scorers (tool use, abstention, substrings)
  cases/       # eval cases as JSON
datasets/kb/   # synthetic Nimbus knowledge base (integrations.md has an injection payload)
```

## Commands

```bash
uv sync                               # install deps
uv run agent "What is the refund policy?" --trace
uv run evals                          # run the eval suite (once built)
uv run pytest                         # deterministic unit tests (no API key needed)
```

## Conventions

- **Package manager: `uv`.** Add deps with `uv add`, run with `uv run`. Don't edit
  `pyproject.toml` deps by hand.
- **Build backend: hatchling**, configured for two packages (`src/agent`,
  `src/evals`).
- **Provider-agnostic agent:** the loop talks to the LLM only through the
  `ModelClient` protocol in `model.py`. Don't import a vendor SDK in `agent.py`.
- **Observability:** Langfuse via the `@observe` decorator and the
  `langfuse.openai` wrapper. It is a no-op when `LANGFUSE_*` env vars are unset, so
  keep it optional — never make code require Langfuse credentials to run.
- **Determinism in evals:** scorers in `scorers.py` must be pure and API-free so
  the suite is reproducible and unit-testable. LLM-based scoring lives separately.
- **Secrets:** config comes from `.env` (see `.env.example`); never commit `.env`.

## The AgentResult contract (what evals score against)

`agent.run(text)` returns an `AgentResult` with: `final_text: str`,
`tool_calls: list[ToolCall]` (each `.name`, `.arguments`), `tool_names()`,
`stopped_reason` (`"completed"` | `"max_steps"`), and `messages` (full log).
