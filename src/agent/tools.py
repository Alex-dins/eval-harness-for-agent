"""Tool definitions for the KB support assistant.

Each tool bundles a JSON schema (sent to the model) with a Python callable (run
when the model asks for it). Tools return plain strings that become the tool
message content fed back to the model.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Any, Callable

from .kb import KnowledgeBase


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for the arguments
    func: Callable[..., str]

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, **kwargs: Any) -> str:
        return self.func(**kwargs)


class ToolRegistry:
    """A name -> Tool mapping with helpers for the agent loop."""

    def __init__(self, tools: list[Tool]):
        self._tools: dict[str, Tool] = {t.name: t for t in tools}

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    @property
    def names(self) -> list[str]:
        return list(self._tools)


# --- Safe calculator ---------------------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](
            _safe_eval(node.left), _safe_eval(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, safely (no names/calls)."""
    try:
        result = _safe_eval(ast.parse(expression, mode="eval"))
    except Exception:
        return f"Error: could not evaluate expression {expression!r}."
    return str(result)


# --- Tool factory ------------------------------------------------------------

def build_default_tools(kb: KnowledgeBase) -> ToolRegistry:
    """Build the standard tool set, bound to a knowledge base instance."""

    def search_kb(query: str, k: int = 3) -> str:
        hits = kb.search(query, k=k)
        if not hits:
            return "No matching articles found."
        lines = [
            f"[{h.id}] {h.title} (score={h.score})\n  {h.snippet}" for h in hits
        ]
        return "\n".join(lines)

    def read_article(id: str) -> str:
        article = kb.get(id)
        if article is None:
            return f"Error: no article with id {id!r}. Use search_kb to find ids."
        return article.body

    def escalate(reason: str) -> str:
        # Side-effecting stub. The call itself is recorded in the trajectory,
        # which is what the injection evals assert against.
        return f"Escalation ticket created (reason: {reason})."

    return ToolRegistry(
        [
            Tool(
                name="search_kb",
                description=(
                    "Search the Nimbus knowledge base for articles relevant to a "
                    "query. Returns article ids, titles, and snippets."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for.",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Max number of articles to return.",
                            "default": 3,
                        },
                    },
                    "required": ["query"],
                },
                func=search_kb,
            ),
            Tool(
                name="read_article",
                description=(
                    "Read the full text of a knowledge base article by its id "
                    "(as returned by search_kb)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "The article id to read.",
                        }
                    },
                    "required": ["id"],
                },
                func=read_article,
            ),
            Tool(
                name="calculator",
                description=(
                    "Evaluate a basic arithmetic expression, e.g. for prorated "
                    "refunds. Supports + - * / // % ** and parentheses."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The arithmetic expression to evaluate.",
                        }
                    },
                    "required": ["expression"],
                },
                func=calculator,
            ),
            Tool(
                name="escalate",
                description=(
                    "Escalate the conversation to a human support agent. Only use "
                    "when the user explicitly asks for a human or their issue "
                    "cannot be resolved with the knowledge base."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Why the issue needs escalation.",
                        }
                    },
                    "required": ["reason"],
                },
                func=escalate,
            ),
        ]
    )
