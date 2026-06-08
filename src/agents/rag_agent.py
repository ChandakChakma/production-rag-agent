"""
ReAct RAG Agent — Reason + Act + Observe loop.

Architecture:
  1. THINK  — LLM decides whether to use a tool or answer directly
  2. ACT    — Execute the chosen tool
  3. OBSERVE — Feed tool output back to LLM
  4. Repeat until the LLM produces a final answer or max_iterations reached

Uses OpenAI function calling for structured tool dispatch.
All turns are tracked for cost monitoring and evaluation.

Interview talking point: ReAct (Yao et al. 2022) outperforms chain-of-thought
on knowledge-intensive tasks because it grounds reasoning in retrieved facts.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

from src.agents.memory import ConversationMemory
from src.agents.tools import ToolRegistry, ToolResult
from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a helpful AI assistant with access to a knowledge base and tools.

Guidelines:
1. Use the `rag_retrieve` tool FIRST whenever the question requires factual knowledge.
2. Always cite your sources by referencing [Source N] from the retrieved context.
3. If retrieved context doesn't answer the question, say so clearly.
4. Use `calculator` for any mathematical computation.
5. Be concise and accurate. Do not hallucinate.
6. If you cannot find the answer, say "I don't have enough information."
"""


@dataclass
class AgentTurn:
    """Records one full Reason→Act→Observe cycle for logging and eval."""
    thought: str
    tool_name: str | None
    tool_args: dict[str, Any] | None
    observation: str | None
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResponse:
    answer: str
    turns: list[AgentTurn]
    sources: list[dict[str, Any]]
    total_tokens: int
    latency_seconds: float
    iterations: int


class RAGAgent:
    """
    ReAct agent with RAG retrieval, tool execution, and conversation memory.
    """

    def __init__(self, tool_registry: ToolRegistry) -> None:
        settings = get_settings()
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model
        self._max_iterations = settings.agent_max_iterations
        self._temperature = settings.agent_temperature
        self._tools = tool_registry
        self._memory = ConversationMemory(
            max_tokens=settings.memory_max_tokens,
            window_size=settings.memory_window_size,
            model=settings.openai_model,
        )
        self._memory.set_system(_SYSTEM_PROMPT)
        logger.info("rag_agent_initialized", model=self._model)

    # ── Public ────────────────────────────────────────────────────────

    def run(self, user_query: str) -> AgentResponse:
        """Execute the ReAct loop for a user query."""
        t_start = time.perf_counter()
        self._memory.add_user(user_query)

        turns: list[AgentTurn] = []
        total_tokens = 0
        all_sources: list[dict[str, Any]] = []

        for iteration in range(self._max_iterations):
            messages = self._memory.get_messages()

            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore
                tools=self._tools.openai_specs(),
                tool_choice="auto",
                temperature=self._temperature,
                max_tokens=1000,
            )

            usage = response.usage
            if usage:
                total_tokens += usage.total_tokens

            choice = response.choices[0]
            msg: ChatCompletionMessage = choice.message

            # ── Final answer (no tool call) ───────────────────────────
            if choice.finish_reason == "stop" or not msg.tool_calls:
                answer = msg.content or ""
                self._memory.add_assistant(answer)
                logger.info(
                    "agent_finished",
                    iterations=iteration + 1,
                    tokens=total_tokens,
                )
                return AgentResponse(
                    answer=answer,
                    turns=turns,
                    sources=all_sources,
                    total_tokens=total_tokens,
                    latency_seconds=round(time.perf_counter() - t_start, 3),
                    iterations=iteration + 1,
                )

            # ── Tool calls ────────────────────────────────────────────
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                logger.debug("agent_tool_call", tool=fn_name, args=fn_args)
                result: ToolResult = self._tools.execute(fn_name, fn_args)

                observation = result.output if result.success else f"Error: {result.error}"

                turn = AgentTurn(
                    thought=msg.content or "",
                    tool_name=fn_name,
                    tool_args=fn_args,
                    observation=observation,
                )
                turns.append(turn)

                if fn_name == "rag_retrieve" and result.metadata:
                    all_sources.append(result.metadata)

                # Inject observation back into message history
                self._memory._messages.append(  # type: ignore
                    __import__("src.agents.memory", fromlist=["Message"]).Message(
                        role="assistant",
                        content=f"[Tool: {fn_name}]\nArgs: {json.dumps(fn_args)}\nResult: {observation}",
                    )
                )

        # Max iterations reached — ask LLM to synthesize
        logger.warning("agent_max_iterations_reached", max=self._max_iterations)
        messages = self._memory.get_messages()
        messages.append({
            "role": "user",
            "content": "Based on the information gathered so far, provide your best answer.",
        })
        final = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore
            temperature=self._temperature,
            max_tokens=800,
        )
        answer = final.choices[0].message.content or "I was unable to fully answer the question."

        return AgentResponse(
            answer=answer,
            turns=turns,
            sources=all_sources,
            total_tokens=total_tokens,
            latency_seconds=round(time.perf_counter() - t_start, 3),
            iterations=self._max_iterations,
        )

    def reset_memory(self) -> None:
        self._memory.clear()
        self._memory.set_system(_SYSTEM_PROMPT)
