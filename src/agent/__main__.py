"""CLI for the KB support assistant.

Usage:
    uv run agent "What is the refund policy?"
    uv run agent "How much is the Team plan?" --trace
"""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the Nimbus support agent.")
    parser.add_argument("question", nargs="+", help="The user's question.")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print the full trajectory (tool calls and messages).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the agent model (defaults to $AGENT_MODEL).",
    )
    args = parser.parse_args()

    # Imported here so `uv run agent` fails cleanly without an API key only when
    # actually invoked, not at import time.
    from .agent import Agent
    from .model import OpenAIModel

    agent = Agent(model=OpenAIModel(model=args.model))
    question = " ".join(args.question)

    result = agent.run(question)

    from .observability import flush
    flush()

    print(result.final_text)

    if args.trace:
        print("\n--- trajectory ---", file=sys.stderr)
        print(f"stopped_reason: {result.stopped_reason}", file=sys.stderr)
        for i, tc in enumerate(result.tool_calls, 1):
            print(
                f"{i}. {tc.name}({json.dumps(tc.arguments)})",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
