"""CLI for the eval harness.

Usage:
    uv run evals                        # run all cases, judge on, write JSON
    uv run evals --category injection   # only injection cases
    uv run evals --cases path/to/dir    # custom case source
    uv run evals --no-judge             # deterministic checks only (free, offline-ish)

Exits non-zero if any case failed, so CI can gate on it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_CASES_DIR = Path(__file__).resolve().parent / "cases"


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the agent eval suite.")
    parser.add_argument(
        "--cases",
        default=str(DEFAULT_CASES_DIR),
        help="Case file or directory (default: src/evals/cases).",
    )
    parser.add_argument(
        "--category",
        choices=["tool_use", "correctness", "hallucination", "injection"],
        help="Only run cases in this category.",
    )
    parser.add_argument(
        "--out",
        default="results",
        help="Directory to write the JSON report into (default: results/).",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the LLM judge (run deterministic checks only).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the agent model (defaults to $AGENT_MODEL).",
    )
    args = parser.parse_args()

    # Imported here so --help works without an API key or heavy imports.
    from agent.agent import Agent
    from agent.model import OpenAIModel

    from .llm_judge import make_correctness_judge
    from .report import all_passed, print_table, write_json
    from .runner import run_cases
    from .schema import load_cases

    cases = load_cases(args.cases)
    if args.category:
        cases = [c for c in cases if c.category == args.category]
    if not cases:
        print("No cases to run.", file=sys.stderr)
        sys.exit(1)

    agent = Agent(model=OpenAIModel(model=args.model))
    judge = None if args.no_judge else make_correctness_judge()

    results = run_cases(cases, agent, judge=judge)

    from agent.observability import flush
    flush()

    print_table(results)
    out_path = write_json(results, args.out)
    print(f"\nWrote {out_path}")

    sys.exit(0 if all_passed(results) else 1)


if __name__ == "__main__":
    main()
