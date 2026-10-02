# eval-harness

An eval harness for a tool-using agent.

## Layout

```
src/
  agent/        # the agent under test (implementation TBD)
    __main__.py # `uv run agent`
  evals/        # the eval harness — runs the agent and scores results (uses OpenAI)
    __main__.py # `uv run evals`
    cases/      # eval cases
datasets/       # eval datasets / fixtures
```

## Setup

```bash
uv sync
cp .env.example .env   # add your OPENAI_API_KEY
```

## Run

```bash
uv run agent
uv run evals
```
