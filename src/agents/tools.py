"""
Tool definitions for the ReAct agent.
Each tool: name, description (fed to LLM), and __call__ implementation.

Tools:
  - RAGRetrieveTool    — searches the vector store
  - CalculatorTool     — safe math expression evaluator
  - DocumentSummaryTool — summarize a long document
  - CurrentDateTool    — returns today's date (avoids LLM hallucinating dates)
"""
from __future__ import annotations

import ast
import datetime
import math
import operator
from dataclasses import dataclass
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ToolResult:
    tool_name: str
    output: str
    error: str | None = None
    metadata: dict[str, Any] | None = None

    @property
    def success(self) -> bool:
        return self.error is None


# ─────────────────────────────────────────────────────────────────────────────
# Base Tool
# ─────────────────────────────────────────────────────────────────────────────


class BaseTool:
    name: str = ""
    description: str = ""

    def __call__(self, *args: Any, **kwargs: Any) -> ToolResult:
        raise NotImplementedError

    def to_openai_spec(self) -> dict[str, Any]:
        """Returns the OpenAI function-calling schema for this tool."""
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# RAG Retrieve Tool
# ─────────────────────────────────────────────────────────────────────────────


class RAGRetrieveTool(BaseTool):
    name = "rag_retrieve"
    description = (
        "Search the knowledge base for relevant documents. "
        "Use when you need factual information to answer the question. "
        "Input: a search query string."
    )

    def __init__(self, retriever: Any, reranker: Any, collection: str = "default") -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._collection = collection

    def __call__(self, query: str, top_k: int = 5) -> ToolResult:
        try:
            from src.config.settings import get_settings
            settings = get_settings()
            candidates = self._retriever.retrieve(
                query=query,
                top_k=settings.retrieval_top_k,
                collection=self._collection,
            )
            ranked = self._reranker.rerank(query=query, candidates=candidates, top_k=top_k)

            if not ranked:
                return ToolResult(
                    tool_name=self.name,
                    output="No relevant documents found in the knowledge base.",
                )

            parts: list[str] = []
            for i, r in enumerate(ranked, 1):
                source = r.metadata.get("source", "unknown")
                parts.append(f"[Source {i}: {source}]\n{r.text}")

            output = "\n\n---\n\n".join(parts)
            return ToolResult(
                tool_name=self.name,
                output=output,
                metadata={"num_sources": len(ranked), "top_score": ranked[0].rerank_score},
            )
        except Exception as exc:
            logger.error("rag_tool_error", error=str(exc))
            return ToolResult(tool_name=self.name, output="", error=str(exc))

    def to_openai_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "default": 5, "description": "Number of results"},
                    },
                    "required": ["query"],
                },
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Calculator Tool
# ─────────────────────────────────────────────────────────────────────────────

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_SAFE_NAMES = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "log": math.log, "pi": math.pi, "e": math.e,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in _SAFE_NAMES:
        return _SAFE_NAMES[node.id]
    if isinstance(node, ast.Call):
        func = _SAFE_NAMES.get(node.func.id if isinstance(node.func, ast.Name) else "")  # type: ignore
        if func:
            args = [_safe_eval(a) for a in node.args]
            return func(*args)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


class CalculatorTool(BaseTool):
    name = "calculator"
    description = (
        "Evaluate a mathematical expression. Safe — no arbitrary code execution. "
        "Supports: +, -, *, /, **, %, sqrt, log, sin, cos, tan, pi, e. "
        "Input: a math expression string like '2 ** 10' or 'sqrt(144)'."
    )

    def __call__(self, expression: str) -> ToolResult:
        try:
            tree = ast.parse(expression.strip(), mode="eval")
            result = _safe_eval(tree.body)
            return ToolResult(tool_name=self.name, output=str(result))
        except Exception as exc:
            return ToolResult(tool_name=self.name, output="", error=f"Math error: {exc}")

    def to_openai_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Current Date Tool
# ─────────────────────────────────────────────────────────────────────────────


class CurrentDateTool(BaseTool):
    name = "current_date"
    description = "Returns today's date and time (UTC). Use when the question involves dates."

    def __call__(self) -> ToolResult:
        now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        return ToolResult(tool_name=self.name, output=f"Current date/time: {now}")

    def to_openai_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}},
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Tool Registry
# ─────────────────────────────────────────────────────────────────────────────


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def all_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def openai_specs(self) -> list[dict[str, Any]]:
        return [t.to_openai_spec() for t in self._tools.values()]

    def execute(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                output="",
                error=f"Unknown tool: {tool_name}. Available: {list(self._tools.keys())}",
            )
        try:
            return tool(**args)
        except Exception as exc:
            logger.error("tool_execution_error", tool=tool_name, error=str(exc))
            return ToolResult(tool_name=tool_name, output="", error=str(exc))
